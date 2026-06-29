import asyncio
import copy
import json
import time
from contextlib import nullcontext
from threading import Lock
from typing import Any, Dict, Optional

import gem
import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from roll.pipeline.agentic.llm_proxy import create_llm_proxy, BaseLLMProxy
from roll.pipeline.agentic.env_manager.base_env_manager import RolloutCache, BaseEnvManager
from roll.utils.env_action_limiter import get_global_limiter
from roll.distributed.scheduler.rollout_scheduler import GroupQueueManager
from roll.pipeline.agentic.env_manager.token_mask_utils import custom_apply_chat_template, compute_conversation_end_token_id
from roll.pipeline.agentic.tools.tool_env_wrapper import tool_wrapper
from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, AgenticConfig
from roll.utils.constants import GenerateStopReason
from roll.utils.functionals import pad_to_length, aggregate_metrics
from roll.utils.logging import get_logger
from roll.utils.str_utils import contains_renderable_field
from roll.pipeline.agentic.trajectory_signals import compute_trajectory_signals
from roll.distributed.scheduler.resume_state import (
    get_scheduling_weight_snapshot,
    get_trajectory_scheduling_state,
)
from roll.distributed.scheduler.trajectory_value import plan_tool_suspend_lease


class TrajEnvManager(BaseEnvManager):
    def __init__(self,
                 worker_config: EnvManagerConfig,
                 pipeline_config: AgenticConfig,
                 env_config: DictConfig,
                 tokenizer: PreTrainedTokenizer,
                 generate_scheduler,
                 output_queue: GroupQueueManager,
                 thread_lock: Lock,
                 mode='train',
                 *args, **kwargs):
        """
        """
        super().__init__()
        self.logger = get_logger()
        self.worker_config: EnvManagerConfig = worker_config
        self.pipeline_config = pipeline_config
        self.env_config: DictConfig = env_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.output_queue = output_queue
        self.mode = mode
        self.generate_scheduler: RouterManager = generate_scheduler

        # EnvManager states
        self.rollout_cache: Optional[RolloutCache] = None
        self.group_seed = None
        self.episode_id = None
        self.trajectory_id = None
        self.running = False
        self._next_request_type = "normal"
        self._resume_generation = 0
        self._pause_ts = None
        self._last_backend_id = None
        # Baseline for GEM tool_wrapper metrics; used to detect tool-return boundaries (G1).
        self._prev_tool_use_counter: Optional[int] = None
        # Minimal verification counters for G1 runtime invariant.
        self._resume_request_count = 0
        self._resume_mismatch_count = 0
        self._resume_e2e_latency_samples: list[float] = []
        self._resume_infer_latency_samples: list[float] = []
        self._resume_prefill_tokens_samples: list[float] = []
        self._resume_actual_hit_samples: list[Optional[float]] = []
        self._resume_matched_prefix_tokens_samples: list[Optional[float]] = []
        self._resume_pinned_kv_gb_seconds_samples: list[Optional[float]] = []
        self._resume_prefill_ratio_samples: list[Optional[float]] = []
        self._resume_saved_prefill_ms_samples: list[Optional[float]] = []
        self._external_wait_samples: list[float] = []
        self._resume_queue_wait_samples: list[float] = []
        self._resume_client_submit_before_samples: list[float] = []
        self._resume_pre_router_samples: list[float] = []
        self._resume_router_lookup_samples: list[float] = []
        self._resume_router_priority_samples: list[float] = []
        self._resume_router_schedule_samples: list[float] = []
        self._resume_dispatch_to_engine_start_samples: list[float] = []
        self._resume_engine_ttft_samples: list[float] = []
        self._resume_decode_tail_samples: list[float] = []
        self._resume_router_return_overhead_samples: list[float] = []
        self._resume_post_router_overhead_samples: list[float] = []
        self.use_thread_lock = self.env_config.get("use_thread_lock", False) # 避免同时执行大量cpu操作, 可以通过env_config配置
        self.thread_lock = thread_lock if self.use_thread_lock else nullcontext()
        # Set environment step concurrency limit
        self.max_env_step_concurrent = self.env_config.get("max_env_step_concurrent", 0)
        self.env_step_limiter = nullcontext()
        if self.max_env_step_concurrent > 0:
            env_tag = self.env_config.get("tag", "default")
            self.env_step_limiter = get_global_limiter(tag=env_tag, max_concurrent_calls=self.max_env_step_concurrent)

        with self.thread_lock, self.env_step_limiter:
            if "seed" in self.env_config['config']:
                self.env_config['config']["seed"] = self.env_config['group_seed']
            self.env = gem.make(env_id=self.env_config["env_type"], **self.env_config['config'])
            if "tool_wrapper" in self.env_config:
                self.env = tool_wrapper(self.env,
                                        wrapper_args=self.env_config.tool_wrapper.wrapper_args,
                                        tool_configs=self.env_config.tool_wrapper.tool_configs)

        self.cfg_template = self.pipeline_config.custom_envs[self.env_config["tag"]]
        self.agent_system_template = self.cfg_template["agent_system_template"]
        self.agent_template = self.cfg_template["agent_template"]

        if self.env_config["env_id"] == 0:
            self.logger.info(f"agent_system_template: {self.agent_system_template}")
            self.logger.info(f"agent_template: {self.agent_template}")

        # TODO: add rewards_scheduler for local ray reward workers
        self.llm_proxy: BaseLLMProxy = create_llm_proxy(
            generate_scheduler=self.generate_scheduler,
            llm_proxy_config=self.worker_config.llm_proxy,
            tokenizer=self.tokenizer,
            env=self.env
        )

    def run_rollout_loop(self, data: DataProto):
        """
        1. Each time run_rollout_loop is called,
           it will continuously play episodes until it receives a command that data collection is complete.
           The seed needs to be reset to ensure consistency across all groups.

        Seed update logic:
           group_seed = base_seed + group_id
           episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info['seed'] + self.env_config['group_seed']
        rollout_cache: RolloutCache = self.reset()
        start_step = self.current_step

        log_stats = {"generate_time": [], "step_time": [], "current_step": []}
        consecutive_abort_count = 0

        while self.running and rollout_cache is not None:

            with Timer(name="generate", logger=None) as generate_timer:
                lm_output: DataProto = self.make_decision(rollout_cache)
                stop_reason = lm_output.meta_info.pop("stop_reason")
            log_stats["current_step"].append(self.current_step)
            log_stats["generate_time"].append(generate_timer.last)

            if stop_reason == GenerateStopReason.ABORT:
                consecutive_abort_count += 1
                self.logger.warning(
                    "generation aborted group_id=%s env_id=%s episode_id=%s "
                    "consecutive_abort_count=%s",
                    self.env_config["group_id"],
                    self.env_config["env_id"],
                    self.episode_id,
                    consecutive_abort_count,
                )
                if consecutive_abort_count >= 3:
                    # Make progress instead of retrying the same prompt forever.
                    content = self.rollout_cache.history[-1]
                    content.setdefault("messages", [])
                    content.setdefault("prompt_ids", [])
                    eos_token_id = self.tokenizer.eos_token_id
                    if eos_token_id is None:
                        eos_token_id = self.tokenizer.pad_token_id
                    content["response_ids"] = [eos_token_id]
                    content["reward"] = 0
                    content["llm_response"] = ""
                    content.setdefault("metrics", {})
                    content.setdefault("metrics_agg_mode", {})
                    content["metrics"]["generation_abort"] = 1
                    content["metrics_agg_mode"]["generation_abort"] = "sum"
                    rollout_cache.step += 1
                    rollout_cache.terminated = True
                    rollout_cache.truncated = True
                    rollout_cache.history.append({
                        "observation": "",
                        "actions_left": max(self.env_config.max_steps - rollout_cache.step, 0),
                        "messages": None,
                    })
            else:
                consecutive_abort_count = 0

            with Timer(name="step", logger=None) as step_timer:
                if stop_reason == GenerateStopReason.FINISH:
                    rollout_cache: RolloutCache = self.step(lm_output)
            log_stats["step_time"].append(step_timer.last)

            if self.running and (rollout_cache.terminated or stop_reason == GenerateStopReason.MAX_LENGTH):
                self.logger.debug(f"group_id: {self.env_config['group_id']} env_id: {self.env_config['env_id']} episode_id: {self.episode_id} start_step {start_step} gen_stats: {log_stats}")
                log_stats = {"generate_time": [], "step_time": [], "current_step": []}
                rollout: DataProto = self.formulate_rollouts(rollout_cache)
                traj_group_id = f"{self.rollout_cache.tag}_{self.rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
                traj_id = f"{traj_group_id}_{self.rollout_cache.env_id}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id] * rollout.batch.batch_size[0], dtype=object)
                rollout.non_tensor_batch["traj_id"] = np.array([traj_id] * rollout.batch.batch_size[0], dtype=object)
                ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, rollout, self.env_config['env_id']))

                rollout_cache = self.reset()
                start_step = self.current_step

        ray.get(self.output_queue.put.remote(self.env_config['group_id'], self.episode_id, start_step, None, self.env_config['env_id']))

    def reset(self) -> RolloutCache:
        if self.trajectory_id:
            self._delete_current_kv_lease()
            get_trajectory_scheduling_state().clear(self.trajectory_id)
        self.rollout_cache = RolloutCache(env_id=self.env_config['env_id'],
                                          group_id=self.env_config['group_id'],
                                          tag=self.env_config['tag'])

        self.episode_id = ray.get(self.output_queue.get_episode_id.remote(
            self.env_config['group_id'],
            self.env_config['env_id']
        ))
        if self.episode_id is None:
            assert not self.running
            return None
        seed = self.group_seed + self.episode_id
        self.trajectory_id = (
            f"{self.env_config['tag']}_{self.env_config['group_id']}_"
            f"{self.episode_id}_{self.group_seed}_{self.env_config['env_id']}"
        )
        self._next_request_type = "normal"
        self._resume_generation = 0
        self._pause_ts = None
        self._last_backend_id = None
        self._prev_tool_use_counter = None
        self._resume_request_count = 0
        self._resume_mismatch_count = 0
        self._resume_e2e_latency_samples = []
        self._resume_infer_latency_samples = []
        self._resume_prefill_tokens_samples = []
        self._resume_actual_hit_samples = []
        self._resume_matched_prefix_tokens_samples = []
        self._resume_pinned_kv_gb_seconds_samples = []
        self._resume_prefill_ratio_samples = []
        self._resume_saved_prefill_ms_samples = []
        self._external_wait_samples = []
        self._resume_queue_wait_samples = []
        self._resume_client_submit_before_samples = []
        self._resume_pre_router_samples = []
        self._resume_router_lookup_samples = []
        self._resume_router_priority_samples = []
        self._resume_router_schedule_samples = []
        self._resume_dispatch_to_engine_start_samples = []
        self._resume_engine_ttft_samples = []
        self._resume_decode_tail_samples = []
        self._resume_router_return_overhead_samples = []
        self._resume_post_router_overhead_samples = []
        self._pending_resume_lease_ttl_s: Optional[float] = None
        self._pending_resume_lease_score: Optional[float] = None

        with self.thread_lock, self.env_step_limiter:
            # `observation` describes the current game-state prompt;
            # `info["suffix"]` carries the current environment-specific state string.
            observation, info = self.env.reset(seed=seed)
            if observation is None:
                return None
        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None,     # agent input messages
            **info,
        })
        return self.rollout_cache

    def _call_router_control_sync(self, method_name: str, *args, timeout_s: Optional[float] = None) -> Optional[Any]:
        """Best-effort call into RouterManager control-plane methods from sync env code."""
        try:
            if isinstance(self.generate_scheduler, ray.actor.ActorHandle):
                obj_ref = getattr(self.generate_scheduler, method_name).remote(*args)
                return ray.get(obj_ref, timeout=timeout_s) if timeout_s is not None else ray.get(obj_ref)
            method = getattr(self.generate_scheduler, method_name, None)
            if method is None:
                return None
            result = method(*args)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                if loop.is_running():
                    self.logger.debug("Skip %s: event loop already running", method_name)
                    return None
                return loop.run_until_complete(result)
            return result
        except Exception as e:
            self.logger.debug("Router control call %s failed: %s", method_name, e)
            return None

    def _delete_current_kv_lease(self) -> None:
        if not self.trajectory_id:
            return
        result = self._call_router_control_sync("delete_kv_lease", self.trajectory_id, timeout_s=2.0)
        if isinstance(result, dict):
            self.logger.debug("delete_kv_lease result for %s: %s", self.trajectory_id, result)

    def _is_tool_return_resume_boundary(self, info: Optional[dict[str, Any]]) -> bool:
        """True only when this env step crossed an external tool-wait and returns tool observation (G1)."""
        if not info:
            return False
        if info.get("use_tool") is True:
            return True
        metrics = info.get("metrics")
        if not isinstance(metrics, dict):
            return False
        raw = metrics.get("tool_use_counter")
        if raw is None:
            return False
        try:
            cur = int(raw)
        except (TypeError, ValueError):
            return False
        prev = self._prev_tool_use_counter if self._prev_tool_use_counter is not None else 0
        return cur > prev

    def _update_tool_use_counter_baseline(self, info: Optional[dict[str, Any]]) -> None:
        if not info:
            return
        metrics = info.get("metrics")
        if not isinstance(metrics, dict):
            return
        raw = metrics.get("tool_use_counter")
        if raw is None:
            return
        try:
            self._prev_tool_use_counter = int(raw)
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _llm_response_may_invoke_tool(response_text: str) -> bool:
        """Heuristic: LLM output is likely to trigger an external tool call."""
        if not isinstance(response_text, str):
            return False
        text = response_text.strip()
        if not text:
            return False
        if "<tool_call>" in text:
            return True
        # python_code_tool (GSM8K toolcall benchmark) uses <code>...</code>, not <tool_call>.
        if "<code>" in text and "</code>" in text:
            return True
        if "```python" in text or "```\npython" in text.lower():
            return True
        # search_tool (HotpotQA + retrieval) uses <search>...</search>
        if "<search>" in text and "</search>" in text:
            return True
        return False

    def _maybe_set_pending_tool_suspend_lease(self) -> None:
        """Register tool-wait KV lease before external tool blocks in env.step (L1 suspend)."""
        snapshot = get_scheduling_weight_snapshot()
        if not self.trajectory_id or self._last_backend_id is None:
            return
        traj_signals = compute_trajectory_signals(
            history=self.rollout_cache.history,
            step=self.rollout_cache.step,
            max_steps=self.env_config.max_steps,
            terminated=bool(self.rollout_cache.terminated),
            truncated=bool(self.rollout_cache.truncated),
        )
        history_len_tokens = 0
        for entry in self.rollout_cache.history:
            if entry.get("prompt_ids") is not None:
                history_len_tokens += len(entry["prompt_ids"])
            if entry.get("response_ids") is not None:
                history_len_tokens += len(entry["response_ids"])
        state = get_trajectory_scheduling_state()
        t_tool = state.get_t_tool_s(self.trajectory_id)
        route_meta = {
            "trajectory_id": self.trajectory_id,
            "request_type": "resume",
            "last_backend_id": int(self._last_backend_id) if self._last_backend_id is not None else None,
            "global_step": int(self.current_step),
            "model_version": int(self.current_step),
            "weight_version": int(self.current_step),
            "kv_lease_model_version": int(self.current_step),
            "env_id": self.env_config["env_id"],
            "history_len_tokens": float(history_len_tokens),
            "scheduling_t_tool_s": float(t_tool),
            **traj_signals,
        }
        if snapshot is not None:
            bias = state.get_p_hit_bias(self.trajectory_id)
            ttl, score, _, _ = plan_tool_suspend_lease(
                route_meta,
                belief=snapshot.belief,
                force_migrate_age_s=snapshot.force_migrate_age_s,
                value_weights=snapshot.value_weights,
                penalty_weights=snapshot.penalty_weights,
                lease_weights=snapshot.lease_weights,
                t_tool_s=t_tool,
                p_hit_bias=bias,
                feedback_hot_downgrade_bias=snapshot.feedback_hot_downgrade_bias,
                use_system_cost=snapshot.enable_system_cost_resume_scheduling,
                system_cost_weights=snapshot.system_cost_weights,
            )
            route_meta["resume_lease_ttl_s"] = ttl
            route_meta["resume_lease_score"] = score
            state.set_pending_tool_lease(
                self.trajectory_id,
                ttl_s=ttl,
                lease_score=score,
                backend_id=self._last_backend_id,
            )
            self._pending_resume_lease_ttl_s = ttl
            self._pending_resume_lease_score = score
        result = self._call_router_control_sync("set_tool_suspend_lease", route_meta, timeout_s=2.0)
        if isinstance(result, dict):
            ttl = result.get("ttl_s")
            score = result.get("lease_score")
            if ttl is not None and score is not None:
                state.set_pending_tool_lease(
                    self.trajectory_id,
                    ttl_s=float(ttl),
                    lease_score=float(score),
                    backend_id=self._last_backend_id,
                )
                self._pending_resume_lease_ttl_s = float(ttl)
                self._pending_resume_lease_score = float(score)
            self.logger.debug("set_tool_suspend_lease result for %s: %s", self.trajectory_id, result)

    def _scheduling_fields_for_meta(self) -> Dict[str, float]:
        """Fields passed to Router via meta_info (cross Ray actor)."""
        out: Dict[str, float] = {}
        if not self.trajectory_id:
            return out
        state = get_trajectory_scheduling_state()
        out["scheduling_t_tool_s"] = float(state.get_t_tool_s(self.trajectory_id))
        if self._pending_resume_lease_ttl_s is not None:
            out["pending_resume_lease_ttl_s"] = float(self._pending_resume_lease_ttl_s)
        if self._pending_resume_lease_score is not None:
            out["pending_resume_lease_score"] = float(self._pending_resume_lease_score)
        return out

    def step(self, llm_output: DataProto):
        responses = self.tokenizer.batch_decode(llm_output.batch['responses'], skip_special_tokens=False)

        tool_call_start_ts = time.time()
        if self._llm_response_may_invoke_tool(responses[0]):
            self._maybe_set_pending_tool_suspend_lease()
        with self.thread_lock, self.env_step_limiter:
            observation, reward, terminated, truncated, info = self.env.step(action=responses[0])
        tool_return_ts = time.time()
        suffix = info.pop("suffix", None)

        is_tool_return = self._is_tool_return_resume_boundary(info)

        self.rollout_cache.step += 1
        self.rollout_cache.terminated = terminated
        self.rollout_cache.truncated = truncated
        if self.rollout_cache.step >= self.env_config.max_steps:
            self.rollout_cache.terminated = True
            if not terminated:
                self.rollout_cache.truncated = True
        self.rollout_cache.history[-1]['reward'] = reward
        self.rollout_cache.history[-1]['llm_response'] = responses[0]
        if info is not None:
            info = dict(info)
            step_metrics = info.pop("metrics", None)
            step_metrics_agg_mode = info.pop("metrics_agg_mode", None)
            self.rollout_cache.history[-1].update(info)
            if step_metrics is not None:
                content = self.rollout_cache.history[-1]
                if "metrics" not in content or not isinstance(content["metrics"], dict):
                    content["metrics"] = {}
                content["metrics"].update(step_metrics)
            if step_metrics_agg_mode is not None:
                content = self.rollout_cache.history[-1]
                if "metrics_agg_mode" not in content or not isinstance(content["metrics_agg_mode"], dict):
                    content["metrics_agg_mode"] = {}
                content["metrics_agg_mode"].update(step_metrics_agg_mode)
        self.rollout_cache.history[-1]["use_tool"] = is_tool_return

        self._update_tool_use_counter_baseline(info)

        self.rollout_cache.history.append({
            "observation": observation,
            "actions_left": self.env_config.max_steps - self.rollout_cache.step,
            "messages": None
        })
        if suffix is not None:
            self.rollout_cache.history[-1]["suffix"] = suffix

        # Resume only after external tool wait + tool-return observation (aligns with format_messages tool branch).
        if is_tool_return:
            external_wait_s = max(0.0, tool_return_ts - tool_call_start_ts)
            self._external_wait_samples.append(external_wait_s)
            content = self.rollout_cache.history[-2]
            if "metrics" not in content or not isinstance(content["metrics"], dict):
                content["metrics"] = {}
            if "metrics_agg_mode" not in content or not isinstance(content["metrics_agg_mode"], dict):
                content["metrics_agg_mode"] = {}
            content["metrics"].update({
                "tool_call_start_ts": tool_call_start_ts,
                "tool_return_ts": tool_return_ts,
                "external_wait_s": external_wait_s,
            })
            content["metrics_agg_mode"].update({
                "tool_call_start_ts": "last",
                "tool_return_ts": "last",
                "external_wait_s": "mean",
            })
            self._pause_ts = tool_return_ts
            self._next_request_type = "resume"
            if self.trajectory_id:
                get_trajectory_scheduling_state().update_tool_wait(
                    self.trajectory_id, external_wait_s
                )
        else:
            self._pause_ts = None
            self._next_request_type = "normal"
        if self.rollout_cache.terminated or self.rollout_cache.truncated:
            self._delete_current_kv_lease()
        return self.rollout_cache

    def make_decision(self, rollout_cache: RolloutCache):
        lm_input = self.format_messages(rollout_cache)
        input_ids = lm_input.batch["input_ids"]

        if input_ids.shape[1] >= self.pipeline_config.sequence_length:
            self.logger.warning(f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {input_ids.shape[1]},"
                                f"maybe you should increase the response_length")
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})

        max_new_tokens = min(self.env_config["max_tokens_per_step"],
                             self.worker_config.generating_args.max_new_tokens,
                             self.pipeline_config.sequence_length-input_ids.shape[1])
        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(max_new_tokens, self.pipeline_config.sequence_length)
        if generation_config["max_new_tokens"] <= 0:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.MAX_LENGTH})
        lm_input.meta_info["src_rank"] = self.env_config["env_id"]
        request_type = self._next_request_type
        pause_age_s = 0.0
        if request_type == "resume" and self._pause_ts is not None:
            pause_age_s = max(0.0, time.time() - self._pause_ts)
            self._resume_generation += 1

        expected_tool_return = bool(
            len(self.rollout_cache.history) > 1 and self.rollout_cache.history[-2].get("use_tool", False)
        )
        if request_type == "resume":
            self._resume_request_count += 1
            if not expected_tool_return:
                self._resume_mismatch_count += 1
                self.logger.warning(
                    "G1 invariant violated: request_type=resume but previous step is not tool-return. "
                    "trajectory_id=%s env_id=%s step=%s",
                    self.trajectory_id,
                    self.env_config.get("env_id"),
                    self.rollout_cache.step,
                )

        traj_signals = compute_trajectory_signals(
            history=self.rollout_cache.history,
            step=self.rollout_cache.step,
            max_steps=self.env_config.max_steps,
            terminated=bool(self.rollout_cache.terminated),
            truncated=bool(self.rollout_cache.truncated),
        )
        lm_input.meta_info.update({
            "trajectory_id": self.trajectory_id,
            "global_step": int(self.current_step),
            "model_version": int(self.current_step),
            "weight_version": int(self.current_step),
            "kv_lease_model_version": int(self.current_step),
            "env_id": self.env_config["env_id"],
            "request_type": request_type,
            "resume_generation": self._resume_generation,
            "pause_ts": self._pause_ts,
            "pause_age_s": pause_age_s,
            "history_len_tokens": int(input_ids.shape[1]),
            "last_backend_id": self._last_backend_id,
            # Minimal verification signals for G1.
            "resume_expected_tool_return": expected_tool_return,
            "resume_request_count": self._resume_request_count,
            "resume_mismatch_count": self._resume_mismatch_count,
            # Trajectory value scheduling (see docs/trajectory_value_scheduling.md).
            **traj_signals,
            **self._scheduling_fields_for_meta(),
        })
        if request_type == "resume":
            self._pending_resume_lease_ttl_s = None
            self._pending_resume_lease_score = None
        self._next_request_type = "normal"

        input_messages = [item for items in self.rollout_cache.history for item in items["messages"]]

        infer_start_ts = time.time()
        lm_output: DataProto = self.llm_proxy.generate(messages=input_messages,
                                                       lm_input=lm_input,
                                                       generation_config=generation_config)
        infer_end_ts = time.time()

        if lm_output is None:
            return DataProto(meta_info={"stop_reason": GenerateStopReason.ABORT})
        selected_backend_id = lm_output.meta_info.get("selected_backend_id")
        if selected_backend_id is not None:
            self._last_backend_id = int(selected_backend_id)

        response_ids = lm_output.batch['responses'][0]
        response_ids = response_ids.tolist()
        content = self.rollout_cache.history[-1]

        if "infer_logprobs" in lm_output.batch.keys():
            infer_logprobs = lm_output.batch['infer_logprobs'][0][-len(response_ids):]
            content["infer_logprobs"] = infer_logprobs.tolist()

        content["response_ids"] = response_ids
        content["messages"].append({"role": "assistant", "content": self.tokenizer.decode(response_ids, skip_special_tokens=True)})

        # Per-resume observability metrics:
        # - resume_latency_e2e_s: tool-return -> this generation completion
        # - resume_infer_latency_s: only generation RPC latency for this resume turn
        # - resume_prefill_tokens: proxy for resume prefill/reload cost
        if request_type == "resume":
            def _meta_float(key: str) -> Optional[float]:
                value = lm_output.meta_info.get(key)
                if value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            resume_infer_latency_s = max(0.0, infer_end_ts - infer_start_ts)
            self._resume_infer_latency_samples.append(resume_infer_latency_s)

            history_len_tokens = float(input_ids.shape[1])
            resume_prefill_tokens = _meta_float("resume_prefill_tokens")
            if resume_prefill_tokens is None:
                resume_prefill_tokens = history_len_tokens
            self._resume_prefill_tokens_samples.append(resume_prefill_tokens)
            self._resume_actual_hit_samples.append(_meta_float("actual_hit"))
            self._resume_matched_prefix_tokens_samples.append(_meta_float("matched_prefix_tokens"))
            self._resume_pinned_kv_gb_seconds_samples.append(_meta_float("pinned_kv_gb_seconds"))
            self._resume_prefill_ratio_samples.append(_meta_float("prefill_ratio"))
            self._resume_saved_prefill_ms_samples.append(_meta_float("saved_prefill_ms"))

            if self._pause_ts is not None:
                resume_latency_e2e_s = max(0.0, infer_end_ts - self._pause_ts)
                self._resume_e2e_latency_samples.append(resume_latency_e2e_s)

            client_submit_ts = _meta_float("client_submit_ts")
            router_handle_start_ts = _meta_float("router_handle_start_ts")
            router_resume_enter_ts = _meta_float("router_resume_enter_ts")
            router_after_lookup_ts = _meta_float("router_after_lookup_ts")
            router_after_priority_ts = _meta_float("router_after_priority_ts")
            router_after_schedule_ts = _meta_float("router_after_schedule_ts")
            gateway_post_start_ts = _meta_float("gateway_post_start_ts")
            gateway_response_headers_ts = _meta_float("gateway_response_headers_ts")
            gateway_body_done_ts = _meta_float("gateway_body_done_ts")
            resume_enqueue_ts = _meta_float("resume_enqueue_ts")
            resume_dispatch_ts = _meta_float("resume_dispatch_ts")
            router_return_ts = _meta_float("router_return_ts")
            engine_start_ts = _meta_float("engine_start_ts")
            engine_first_token_ts = _meta_float("engine_first_token_ts")
            engine_finish_ts = _meta_float("engine_finish_ts")
            worker_generator_done_ts = _meta_float("worker_generator_done_ts")
            worker_postprocess_done_ts = _meta_float("worker_postprocess_done_ts")
            worker_log_done_ts = _meta_float("worker_log_done_ts")
            router_worker_response_ts = _meta_float("router_worker_response_ts")
            router_observe_done_ts = _meta_float("router_observe_done_ts")
            policy_ray_submit_done_ts = _meta_float("policy_ray_submit_done_ts")

            client_submit_before_s = None
            if client_submit_ts is not None:
                client_submit_before_s = max(0.0, client_submit_ts - infer_start_ts)
                self._resume_client_submit_before_samples.append(client_submit_before_s)

            client_to_router_handle_s = None
            if client_submit_ts is not None and router_handle_start_ts is not None:
                client_to_router_handle_s = max(0.0, router_handle_start_ts - client_submit_ts)

            policy_ray_submit_overhead_s = None
            if client_submit_ts is not None and policy_ray_submit_done_ts is not None:
                policy_ray_submit_overhead_s = max(0.0, policy_ray_submit_done_ts - client_submit_ts)

            policy_ray_wait_to_router_s = None
            if policy_ray_submit_done_ts is not None and router_handle_start_ts is not None:
                policy_ray_wait_to_router_s = max(0.0, router_handle_start_ts - policy_ray_submit_done_ts)

            direct_worker_data_path = _meta_float("direct_worker_data_path")
            policy_route_submit_ts = _meta_float("policy_route_submit_ts")
            policy_route_submit_done_ts = _meta_float("policy_route_submit_done_ts")
            policy_route_return_ts = _meta_float("policy_route_return_ts")
            policy_worker_submit_ts = _meta_float("policy_worker_submit_ts")
            policy_worker_submit_done_ts = _meta_float("policy_worker_submit_done_ts")
            policy_worker_return_ts = _meta_float("policy_worker_return_ts")
            policy_observe_submit_ts = _meta_float("policy_observe_submit_ts")
            policy_observe_submit_done_ts = _meta_float("policy_observe_submit_done_ts")
            policy_observe_return_ts = _meta_float("policy_observe_return_ts")
            policy_observe_async = _meta_float("policy_observe_async")
            policy_slim_route_request = _meta_float("policy_slim_route_request")
            policy_local_route_hint = _meta_float("policy_local_route_hint")
            policy_local_route_hint_hit = _meta_float("policy_local_route_hint_hit")
            router_slim_route_request = _meta_float("router_slim_route_request")
            router_fast_route_path = _meta_float("router_fast_route_path")
            observe_in_critical_path = _meta_float("observe_in_critical_path")
            observe_pending_count = _meta_float("observe_pending_count")
            observe_drain_count = _meta_float("observe_drain_count")
            router_route_decision_done_ts = _meta_float("router_route_decision_done_ts")
            router_route_return_ts = _meta_float("router_route_return_ts")
            router_observe_recv_ts = _meta_float("router_observe_recv_ts")

            route_rpc_latency_s = None
            if policy_route_submit_ts is not None and policy_route_return_ts is not None:
                route_rpc_latency_s = max(0.0, policy_route_return_ts - policy_route_submit_ts)
            router_route_compute_s = None
            if router_handle_start_ts is not None and router_route_decision_done_ts is not None:
                router_route_compute_s = max(0.0, router_route_decision_done_ts - router_handle_start_ts)
            route_return_overhead_s = None
            if router_route_return_ts is not None and policy_route_return_ts is not None:
                route_return_overhead_s = max(0.0, policy_route_return_ts - router_route_return_ts)
            worker_rpc_latency_s = None
            if policy_worker_submit_ts is not None and policy_worker_return_ts is not None:
                worker_rpc_latency_s = max(0.0, policy_worker_return_ts - policy_worker_submit_ts)
            worker_return_overhead_s = None
            if worker_log_done_ts is not None and policy_worker_return_ts is not None:
                worker_return_overhead_s = max(0.0, policy_worker_return_ts - worker_log_done_ts)
            observe_rpc_latency_s = None
            if policy_observe_submit_ts is not None and policy_observe_return_ts is not None:
                observe_rpc_latency_s = max(0.0, policy_observe_return_ts - policy_observe_submit_ts)
            router_observe_compute_s = None
            if router_observe_recv_ts is not None and router_observe_done_ts is not None:
                router_observe_compute_s = max(0.0, router_observe_done_ts - router_observe_recv_ts)

            router_handle_to_enqueue_s = None
            if router_handle_start_ts is not None and resume_enqueue_ts is not None:
                router_handle_to_enqueue_s = max(0.0, resume_enqueue_ts - router_handle_start_ts)

            pre_router_s = None
            if client_submit_ts is not None and resume_enqueue_ts is not None:
                pre_router_s = max(0.0, resume_enqueue_ts - client_submit_ts)
                self._resume_pre_router_samples.append(pre_router_s)

            router_lookup_s = None
            if router_resume_enter_ts is not None and router_after_lookup_ts is not None:
                router_lookup_s = max(0.0, router_after_lookup_ts - router_resume_enter_ts)
                self._resume_router_lookup_samples.append(router_lookup_s)

            router_priority_s = None
            if router_after_lookup_ts is not None and router_after_priority_ts is not None:
                router_priority_s = max(0.0, router_after_priority_ts - router_after_lookup_ts)
                self._resume_router_priority_samples.append(router_priority_s)

            router_schedule_s = None
            if router_after_priority_ts is not None and router_after_schedule_ts is not None:
                router_schedule_s = max(0.0, router_after_schedule_ts - router_after_priority_ts)
                self._resume_router_schedule_samples.append(router_schedule_s)

            dispatch_to_engine_start_s = None
            if resume_dispatch_ts is not None and engine_start_ts is not None:
                dispatch_to_engine_start_s = max(0.0, engine_start_ts - resume_dispatch_ts)
                self._resume_dispatch_to_engine_start_samples.append(dispatch_to_engine_start_s)

            engine_ttft_s = None
            if engine_start_ts is not None and engine_first_token_ts is not None:
                engine_ttft_s = max(0.0, engine_first_token_ts - engine_start_ts)
                self._resume_engine_ttft_samples.append(engine_ttft_s)

            decode_tail_s = None
            if engine_first_token_ts is not None and engine_finish_ts is not None:
                decode_tail_s = max(0.0, engine_finish_ts - engine_first_token_ts)
                self._resume_decode_tail_samples.append(decode_tail_s)

            gateway_post_to_headers_s = None
            if gateway_post_start_ts is not None and gateway_response_headers_ts is not None:
                gateway_post_to_headers_s = max(0.0, gateway_response_headers_ts - gateway_post_start_ts)

            gateway_body_parse_s = None
            if gateway_response_headers_ts is not None and gateway_body_done_ts is not None:
                gateway_body_parse_s = max(0.0, gateway_body_done_ts - gateway_response_headers_ts)

            router_tail_after_body_s = None
            if gateway_body_done_ts is not None and router_return_ts is not None:
                router_tail_after_body_s = max(0.0, router_return_ts - gateway_body_done_ts)

            router_return_overhead_s = None
            if engine_finish_ts is not None and router_return_ts is not None:
                router_return_overhead_s = max(0.0, router_return_ts - engine_finish_ts)
                self._resume_router_return_overhead_samples.append(router_return_overhead_s)

            post_router_overhead_s = None
            if router_return_ts is not None:
                post_router_overhead_s = max(0.0, infer_end_ts - router_return_ts)
                self._resume_post_router_overhead_samples.append(post_router_overhead_s)

            if "metrics" not in content or not isinstance(content["metrics"], dict):
                content["metrics"] = {}
            if "metrics_agg_mode" not in content or not isinstance(content["metrics_agg_mode"], dict):
                content["metrics_agg_mode"] = {}
            if self._resume_e2e_latency_samples:
                content["metrics"]["resume_latency_e2e_s"] = self._resume_e2e_latency_samples[-1]
            content["metrics"]["resume_infer_start_ts"] = infer_start_ts
            content["metrics"]["resume_client_submit_ts"] = client_submit_ts
            content["metrics"]["router_handle_start_ts"] = router_handle_start_ts
            content["metrics"]["router_resume_enter_ts"] = router_resume_enter_ts
            content["metrics"]["router_after_lookup_ts"] = router_after_lookup_ts
            content["metrics"]["router_after_priority_ts"] = router_after_priority_ts
            content["metrics"]["router_after_schedule_ts"] = router_after_schedule_ts
            content["metrics"]["gateway_post_start_ts"] = gateway_post_start_ts
            content["metrics"]["gateway_response_headers_ts"] = gateway_response_headers_ts
            content["metrics"]["gateway_body_done_ts"] = gateway_body_done_ts
            content["metrics"]["resume_enqueue_ts"] = resume_enqueue_ts
            content["metrics"]["resume_dispatch_ts"] = resume_dispatch_ts
            content["metrics"]["engine_start_ts"] = engine_start_ts
            content["metrics"]["engine_first_token_ts"] = engine_first_token_ts
            content["metrics"]["resume_first_token_ts"] = engine_first_token_ts if engine_first_token_ts is not None else infer_end_ts
            content["metrics"]["engine_finish_ts"] = engine_finish_ts
            content["metrics"]["router_return_ts"] = router_return_ts
            content["metrics"]["resume_infer_end_ts"] = infer_end_ts
            content["metrics"]["resume_infer_latency_s"] = resume_infer_latency_s
            content["metrics"]["resume_prefill_tokens"] = resume_prefill_tokens
            content["metrics"]["resume_history_len_tokens"] = history_len_tokens
            if client_submit_before_s is not None:
                content["metrics"]["resume_client_submit_before_s"] = client_submit_before_s
            if client_to_router_handle_s is not None:
                content["metrics"]["resume_client_to_router_handle_s"] = client_to_router_handle_s
            if router_handle_to_enqueue_s is not None:
                content["metrics"]["resume_router_handle_to_enqueue_s"] = router_handle_to_enqueue_s
            if pre_router_s is not None:
                content["metrics"]["resume_pre_router_s"] = pre_router_s
            if router_lookup_s is not None:
                content["metrics"]["resume_router_lookup_s"] = router_lookup_s
            if router_priority_s is not None:
                content["metrics"]["resume_router_priority_s"] = router_priority_s
            if router_schedule_s is not None:
                content["metrics"]["resume_router_schedule_s"] = router_schedule_s
            if dispatch_to_engine_start_s is not None:
                content["metrics"]["resume_dispatch_to_engine_start_s"] = dispatch_to_engine_start_s
            if engine_ttft_s is not None:
                content["metrics"]["resume_engine_ttft_s"] = engine_ttft_s
            if decode_tail_s is not None:
                content["metrics"]["resume_decode_tail_s"] = decode_tail_s
            if gateway_post_to_headers_s is not None:
                content["metrics"]["resume_gateway_post_to_headers_s"] = gateway_post_to_headers_s
            if gateway_body_parse_s is not None:
                content["metrics"]["resume_gateway_body_parse_s"] = gateway_body_parse_s
            if router_tail_after_body_s is not None:
                content["metrics"]["resume_router_tail_after_body_s"] = router_tail_after_body_s
            if router_return_overhead_s is not None:
                content["metrics"]["resume_router_return_overhead_s"] = router_return_overhead_s
            if post_router_overhead_s is not None:
                content["metrics"]["resume_post_router_overhead_s"] = post_router_overhead_s

            content["metrics"]["worker_generator_done_ts"] = worker_generator_done_ts
            content["metrics"]["worker_postprocess_done_ts"] = worker_postprocess_done_ts
            content["metrics"]["worker_log_done_ts"] = worker_log_done_ts
            content["metrics"]["router_worker_response_ts"] = router_worker_response_ts
            content["metrics"]["router_observe_done_ts"] = router_observe_done_ts
            content["metrics"]["policy_ray_submit_done_ts"] = policy_ray_submit_done_ts
            content["metrics"]["direct_worker_data_path"] = direct_worker_data_path
            content["metrics"]["policy_route_submit_ts"] = policy_route_submit_ts
            content["metrics"]["policy_route_submit_done_ts"] = policy_route_submit_done_ts
            content["metrics"]["policy_route_return_ts"] = policy_route_return_ts
            content["metrics"]["policy_worker_submit_ts"] = policy_worker_submit_ts
            content["metrics"]["policy_worker_submit_done_ts"] = policy_worker_submit_done_ts
            content["metrics"]["policy_worker_return_ts"] = policy_worker_return_ts
            content["metrics"]["policy_observe_submit_ts"] = policy_observe_submit_ts
            content["metrics"]["policy_observe_submit_done_ts"] = policy_observe_submit_done_ts
            content["metrics"]["policy_observe_return_ts"] = policy_observe_return_ts
            content["metrics"]["policy_observe_async"] = policy_observe_async
            content["metrics"]["policy_slim_route_request"] = policy_slim_route_request
            for key in (
                "resume_dispatch_value",
                "resume_dispatch_expected_saved_tokens",
                "resume_dispatch_queue_cost_tokens",
                "resume_dispatch_memory_pressure_cost_tokens",
                "resume_dispatch_inflight",
                "resume_dispatch_inflight_ratio",
                "resume_dispatch_memory_pressure",
                "resume_dispatch_history_len_for_value",
                "resume_dispatch_matched_tokens_for_value",
                "resume_dispatch_p_hit_for_value",
                "resume_dispatch_value_source_matched",
                "resume_dispatch_value_source_p_hit",
                "resume_dispatch_value_source_prior",
                "resume_dispatch_value_min",
                "resume_admission_admitted",
                "route_model_version",
                "kv_lease_model_version",
                "kv_lease_model_version_match",
                "kv_lease_stale_version_blocked",
                "kv_hit_same_version",
                "kv_hit_stale_version_blocked",
                "engine_kv_pinned_tokens",
                "engine_kv_evicted_tokens",
                "engine_kv_evicted_pinned_tokens",
                "engine_kv_lease_hit",
                "engine_kv_lease_miss",
                "engine_kv_lease_stale_version_blocked",
            ):
                value = _meta_float(key)
                if value is not None:
                    content["metrics"][key] = value
            content["metrics"]["policy_local_route_hint"] = policy_local_route_hint
            content["metrics"]["policy_local_route_hint_hit"] = policy_local_route_hint_hit
            content["metrics"]["router_slim_route_request"] = router_slim_route_request
            content["metrics"]["router_fast_route_path"] = router_fast_route_path
            content["metrics"]["observe_in_critical_path"] = observe_in_critical_path
            content["metrics"]["observe_pending_count"] = observe_pending_count
            content["metrics"]["observe_drain_count"] = observe_drain_count
            content["metrics"]["router_route_decision_done_ts"] = router_route_decision_done_ts
            content["metrics"]["router_route_return_ts"] = router_route_return_ts
            content["metrics"]["router_observe_recv_ts"] = router_observe_recv_ts
            if route_rpc_latency_s is not None:
                content["metrics"]["resume_route_rpc_latency_s"] = route_rpc_latency_s
            if router_route_compute_s is not None:
                content["metrics"]["resume_router_route_compute_s"] = router_route_compute_s
            if route_return_overhead_s is not None:
                content["metrics"]["resume_route_return_overhead_s"] = route_return_overhead_s
            if worker_rpc_latency_s is not None:
                content["metrics"]["resume_worker_rpc_latency_s"] = worker_rpc_latency_s
            if worker_return_overhead_s is not None:
                content["metrics"]["resume_worker_return_overhead_s"] = worker_return_overhead_s
            if observe_rpc_latency_s is not None:
                content["metrics"]["resume_observe_rpc_latency_s"] = observe_rpc_latency_s
            if router_observe_compute_s is not None:
                content["metrics"]["resume_router_observe_compute_s"] = router_observe_compute_s
            if policy_ray_submit_overhead_s is not None:
                content["metrics"]["resume_policy_ray_submit_overhead_s"] = policy_ray_submit_overhead_s
            if policy_ray_wait_to_router_s is not None:
                content["metrics"]["resume_policy_ray_wait_to_router_s"] = policy_ray_wait_to_router_s
            if engine_finish_ts is not None and worker_generator_done_ts is not None:
                content["metrics"]["resume_worker_generator_tail_s"] = max(0.0, worker_generator_done_ts - engine_finish_ts)
            if worker_generator_done_ts is not None and worker_postprocess_done_ts is not None:
                content["metrics"]["resume_worker_postprocess_s"] = max(0.0, worker_postprocess_done_ts - worker_generator_done_ts)
            if worker_postprocess_done_ts is not None and worker_log_done_ts is not None:
                content["metrics"]["resume_worker_log_s"] = max(0.0, worker_log_done_ts - worker_postprocess_done_ts)
            if worker_log_done_ts is not None and router_worker_response_ts is not None:
                content["metrics"]["resume_worker_to_router_return_s"] = max(0.0, router_worker_response_ts - worker_log_done_ts)
            if router_worker_response_ts is not None and router_observe_done_ts is not None:
                content["metrics"]["resume_router_observe_s"] = max(0.0, router_observe_done_ts - router_worker_response_ts)
            if router_observe_done_ts is not None and router_return_ts is not None:
                content["metrics"]["resume_router_finalize_s"] = max(0.0, router_return_ts - router_observe_done_ts)
            worker_log_skipped = _meta_float("worker_log_skipped")
            if worker_log_skipped is not None:
                content["metrics"]["worker_log_skipped"] = worker_log_skipped

            for key in (
                "resume_fast_path",
                "resume_queue_wait_s",
                "context_class_gpu_hit",
                "context_class_cpu_reload",
                "context_class_full_prefill",
                "selected_backend_affinity_hit",
                "selected_backend_migration",
                "worker_load_skew_at_dispatch",
                "selected_worker_load_at_dispatch",
                "remaining_steps",
                "max_steps",
                "remaining_steps_ratio",
                "trajectory_value",
                "order_score",
                "dispatch_score",
                "system_dispatch_score",
                "system_delay_regret",
                "expected_prefill_saved",
                "belief_p_hit",
                "resume_lease_ttl_s",
                "resume_lease_score",
                "kv_bytes_proxy",
                "memory_pressure",
                "pending_resume_lease_ttl_s",
                "pending_resume_lease_score",
                "belief_estimated_hit_tokens",
                "belief_estimated_prefill_tokens",
                "lookup_resume_found",
                "lookup_hit_tokens",
                "lookup_cache_confidence",
                "lookup_estimated_prefill_tokens",
                "lookup_lease_remaining_s",
                "ttl_remaining_s",
                "actual_hit",
                "matched_prefix_tokens",
                "estimated_prefill_tokens",
                "prefill_time_ms",
                "cache_confidence",
                "prefill_ratio",
                "engine_cache_confidence",
                "p_hit_measured",
                "p_hit_effective",
                "policy_local_route_hint_lease_remaining_s",
                "policy_local_route_hint_lease_score",
                "policy_local_route_hint_p_hit",
                "policy_local_route_hint_cache_age_s",
                "policy_local_route_hint_use_dispatch_value",
                "policy_local_route_hint_dispatch_value",
                "policy_local_route_hint_expected_saved_tokens",
                "policy_local_route_hint_expected_source_matched",
                "policy_local_route_hint_expected_source_p_hit",
                "policy_local_route_hint_expected_source_prior",
                "policy_local_route_hint_history_len_for_value",
                "policy_local_route_hint_matched_tokens_for_value",
                "policy_local_route_hint_p_hit_for_value",
                "policy_local_route_hint_default_p_hit",
                "policy_local_route_hint_queue_cost_tokens",
                "policy_local_route_hint_memory_pressure_cost_tokens",
                "policy_local_route_hint_inflight",
                "policy_local_route_hint_inflight_ratio",
                "policy_local_route_hint_memory_pressure",
                "saved_prefill_tokens",
                "saved_prefill_ms",
                "saved_prefill_ms_per_gb_second",
                "pinned_kv_gb_seconds",
                "avoidable_reprefill_tokens",
                "dead_pinned_kv_gb_seconds",
                "hot_resume_miss_ratio",
                "locality_mismatch_count",
                "queue_decay_loss_ms",
                "queue_decay_loss_proxy",
                "kv_lease_effective_ttl_s",
            ):
                if key in lm_output.meta_info:
                    value = lm_output.meta_info[key]
                    try:
                        content["metrics"][key] = float(value)
                    except (TypeError, ValueError):
                        content["metrics"][key] = value
            context_class = lm_output.meta_info.get("context_class")
            if isinstance(context_class, str):
                content["resume_context_class"] = context_class
            if "resume_queue_wait_s" in lm_output.meta_info:
                self._resume_queue_wait_samples.append(float(lm_output.meta_info["resume_queue_wait_s"]))
            content["metrics_agg_mode"].update({
                "resume_latency_e2e_s": "mean",
                "resume_infer_start_ts": "last",
                "resume_client_submit_ts": "last",
                "router_handle_start_ts": "last",
                "gateway_post_start_ts": "last",
                "gateway_response_headers_ts": "last",
                "gateway_body_done_ts": "last",
                "resume_enqueue_ts": "last",
                "resume_dispatch_ts": "last",
                "engine_start_ts": "last",
                "engine_first_token_ts": "last",
                "resume_first_token_ts": "last",
                "engine_finish_ts": "last",
                "router_return_ts": "last",
                "worker_generator_done_ts": "last",
                "worker_postprocess_done_ts": "last",
                "worker_log_done_ts": "last",
                "router_worker_response_ts": "last",
                "router_observe_done_ts": "last",
                "policy_ray_submit_done_ts": "last",
                "direct_worker_data_path": "mean",
                "policy_route_submit_ts": "last",
                "policy_route_submit_done_ts": "last",
                "policy_route_return_ts": "last",
                "policy_worker_submit_ts": "last",
                "policy_worker_submit_done_ts": "last",
                "policy_worker_return_ts": "last",
                "policy_observe_submit_ts": "last",
                "policy_observe_submit_done_ts": "last",
                "policy_observe_return_ts": "last",
                "policy_observe_async": "mean",
                "policy_slim_route_request": "mean",
                "resume_dispatch_value": "mean",
                "resume_dispatch_expected_saved_tokens": "mean",
                "resume_dispatch_queue_cost_tokens": "mean",
                "resume_dispatch_memory_pressure_cost_tokens": "mean",
                "resume_dispatch_inflight": "mean",
                "resume_dispatch_inflight_ratio": "mean",
                "resume_dispatch_memory_pressure": "mean",
                "resume_dispatch_history_len_for_value": "mean",
                "resume_dispatch_matched_tokens_for_value": "mean",
                "resume_dispatch_p_hit_for_value": "mean",
                "resume_dispatch_value_source_matched": "mean",
                "resume_dispatch_value_source_p_hit": "mean",
                "resume_dispatch_value_source_prior": "mean",
                "resume_dispatch_value_min": "mean",
                "resume_admission_admitted": "mean",
                "route_model_version": "last",
                "kv_lease_model_version": "last",
                "kv_lease_model_version_match": "mean",
                "kv_lease_stale_version_blocked": "sum",
                "kv_hit_same_version": "mean",
                "kv_hit_stale_version_blocked": "sum",
                "engine_kv_pinned_tokens": "last",
                "engine_kv_evicted_tokens": "last",
                "engine_kv_evicted_pinned_tokens": "last",
                "engine_kv_lease_hit": "sum",
                "engine_kv_lease_miss": "sum",
                "engine_kv_lease_stale_version_blocked": "sum",
                "kv_lease_state_code": "mean",
                "kv_lease_state_created": "mean",
                "kv_lease_state_active": "mean",
                "kv_lease_state_renewed": "mean",
                "kv_lease_state_expired": "mean",
                "kv_lease_state_released": "mean",
                "kv_lease_state_evicted": "mean",
                "kv_lease_version": "mean",
                "kv_lease_record_ttl_s": "mean",
                "kv_lease_record_score": "mean",
                "kv_lease_remaining_s": "mean",
                "kv_lease_backend_id": "mean",
                "policy_local_route_hint": "mean",
                "policy_local_route_hint_hit": "mean",
                "router_slim_route_request": "mean",
                "router_fast_route_path": "mean",
                "observe_in_critical_path": "mean",
                "observe_pending_count": "last",
                "observe_drain_count": "sum",
                "router_route_decision_done_ts": "last",
                "router_route_return_ts": "last",
                "router_observe_recv_ts": "last",
                "resume_route_rpc_latency_s": "mean",
                "resume_router_route_compute_s": "mean",
                "resume_route_return_overhead_s": "mean",
                "resume_worker_rpc_latency_s": "mean",
                "resume_worker_return_overhead_s": "mean",
                "resume_observe_rpc_latency_s": "mean",
                "resume_router_observe_compute_s": "mean",
                "resume_infer_end_ts": "last",
                "resume_infer_latency_s": "mean",
                "resume_prefill_tokens": "mean",
                "resume_history_len_tokens": "mean",
                "resume_fast_path": "mean",
                "resume_client_to_router_handle_s": "mean",
                "resume_router_handle_to_enqueue_s": "mean",
                "resume_gateway_post_to_headers_s": "mean",
                "resume_gateway_body_parse_s": "mean",
                "resume_router_tail_after_body_s": "mean",
                "resume_queue_wait_s": "mean",
                "resume_client_submit_before_s": "mean",
                "resume_pre_router_s": "mean",
                "resume_router_lookup_s": "mean",
                "resume_router_priority_s": "mean",
                "resume_router_schedule_s": "mean",
                "resume_dispatch_to_engine_start_s": "mean",
                "resume_engine_ttft_s": "mean",
                "resume_decode_tail_s": "mean",
                "resume_router_return_overhead_s": "mean",
                "resume_post_router_overhead_s": "mean",
                "resume_policy_ray_submit_overhead_s": "mean",
                "resume_policy_ray_wait_to_router_s": "mean",
                "resume_worker_generator_tail_s": "mean",
                "resume_worker_postprocess_s": "mean",
                "resume_worker_log_s": "mean",
                "resume_worker_to_router_return_s": "mean",
                "resume_router_observe_s": "mean",
                "resume_router_finalize_s": "mean",
                "worker_log_skipped": "mean",
                "context_class_gpu_hit": "sum",
                "context_class_cpu_reload": "sum",
                "context_class_full_prefill": "sum",
                "selected_backend_affinity_hit": "mean",
                "selected_backend_migration": "mean",
                "worker_load_skew_at_dispatch": "mean",
                "selected_worker_load_at_dispatch": "mean",
                "remaining_steps": "mean",
                "max_steps": "last",
                "remaining_steps_ratio": "mean",
                "trajectory_value": "mean",
                "order_score": "mean",
                "dispatch_score": "mean",
                "system_dispatch_score": "mean",
                "system_delay_regret": "mean",
                "expected_prefill_saved": "mean",
                "belief_p_hit": "mean",
                "resume_lease_ttl_s": "mean",
                "resume_lease_score": "mean",
                "kv_bytes_proxy": "mean",
                "memory_pressure": "mean",
                "pending_resume_lease_ttl_s": "mean",
                "pending_resume_lease_score": "mean",
                "belief_estimated_hit_tokens": "mean",
                "belief_estimated_prefill_tokens": "mean",
                "lookup_resume_found": "mean",
                "lookup_hit_tokens": "mean",
                "lookup_cache_confidence": "mean",
                "lookup_estimated_prefill_tokens": "mean",
                "lookup_lease_remaining_s": "mean",
                "ttl_remaining_s": "mean",
                "actual_hit": "mean",
                "matched_prefix_tokens": "mean",
                "estimated_prefill_tokens": "mean",
                "prefill_time_ms": "mean",
                "cache_confidence": "mean",
                "prefill_ratio": "mean",
                "engine_cache_confidence": "mean",
                "p_hit_measured": "mean",
                "p_hit_effective": "mean",
                "policy_local_route_hint_lease_remaining_s": "mean",
                "policy_local_route_hint_lease_score": "mean",
                "policy_local_route_hint_p_hit": "mean",
                "policy_local_route_hint_cache_age_s": "mean",
                "policy_local_route_hint_use_dispatch_value": "mean",
                "policy_local_route_hint_dispatch_value": "mean",
                "policy_local_route_hint_expected_saved_tokens": "mean",
                "policy_local_route_hint_expected_source_matched": "mean",
                "policy_local_route_hint_expected_source_p_hit": "mean",
                "policy_local_route_hint_expected_source_prior": "mean",
                "policy_local_route_hint_history_len_for_value": "mean",
                "policy_local_route_hint_matched_tokens_for_value": "mean",
                "policy_local_route_hint_p_hit_for_value": "mean",
                "policy_local_route_hint_default_p_hit": "mean",
                "policy_local_route_hint_queue_cost_tokens": "mean",
                "policy_local_route_hint_memory_pressure_cost_tokens": "mean",
                "policy_local_route_hint_inflight": "mean",
                "policy_local_route_hint_inflight_ratio": "mean",
                "policy_local_route_hint_memory_pressure": "mean",
                "saved_prefill_tokens": "mean",
                "saved_prefill_ms": "mean",
                "saved_prefill_ms_per_gb_second": "mean",
                "pinned_kv_gb_seconds": "sum",
                "avoidable_reprefill_tokens": "sum",
                "dead_pinned_kv_gb_seconds": "sum",
                "hot_resume_miss_ratio": "mean",
                "locality_mismatch_count": "sum",
                "queue_decay_loss_ms": "sum",
                "queue_decay_loss_proxy": "sum",
                "kv_lease_effective_ttl_s": "mean",
            })

        lm_output.meta_info["stop_reason"] = GenerateStopReason.FINISH
        return lm_output

    def format_messages(self, history: RolloutCache) -> DataProto:
        content = self.rollout_cache.history[-1]

        messages = []
        user_content = ""
        if content["actions_left"] == self.env_config.max_steps:
            messages.append({"role": "system", "content": self.agent_system_template})
            if "env_instruction" in history.history[0]:
                user_content =  f"{history.history[0]['env_instruction']}\n"
        if len(self.rollout_cache.history) > 1 and self.rollout_cache.history[-2].get("use_tool", False):
            messages.append({"role": "tool", "content": content["observation"]})
        else:
            render_dict = {"observation": content["observation"]}
            if contains_renderable_field(self.agent_template, "turn_idx"):
                render_dict["turn_idx"] = self.rollout_cache.step + 1
            if contains_renderable_field(self.agent_template, "suffix"):
                render_dict["suffix"] = content.get("suffix", "")
            if contains_renderable_field(self.agent_template, "actions_left"):
                render_dict["actions_left"] = content["actions_left"]
            if contains_renderable_field(self.agent_template, "max_response_length"):
                render_dict["max_response_length"] = self.env_config["max_tokens_per_step"]
            user_content += self.agent_template.format(**render_dict)
            messages.append({"role": "user", "content": user_content})

        prompt_ids = custom_apply_chat_template(messages=messages, tokenizer=self.tokenizer, add_generation_prompt=True, skip_mock_system_prompt=self.pipeline_config.skip_mock_system_prompt)
        history_token_ids = []
        for items in self.rollout_cache.history[:-1]:
            history_token_ids.extend(items["prompt_ids"])
            history_token_ids.extend(items["response_ids"])
        if len(history_token_ids):
            prompt_ids = compute_conversation_end_token_id(self.tokenizer) + prompt_ids
        input_ids = history_token_ids + prompt_ids

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        # Huggingface Transformers prefer position_ids to be 0-based.
        # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
        # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
        position_ids = attention_mask.cumsum(dim=-1) - 1
        lm_input = DataProto()
        lm_input.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])
        content["prompt_ids"] = prompt_ids
        content["messages"] = messages
        return lm_input

    def formulate_rollouts(self, rollout_cache: RolloutCache):
        """

        """
        # Drop only the trailing placeholder (tool-return observation before resume
        # generate). Keep completed turns that already have llm_response/metrics.
        last_hist = rollout_cache.history[-1]
        if "observation" in last_hist and not last_hist.get("llm_response"):
            rollout_cache.history.pop(-1)
        history = rollout_cache.history[:-1]
        last_cache = copy.deepcopy(rollout_cache.history[-1])
        last_cache.pop("reward", None)
        history.append(last_cache)

        scores = [i['reward'] for i in self.rollout_cache.history]
        episode_score = sum(scores)

        token_ids = []
        prompt_masks = []
        response_masks = []
        infer_logprobs = []
        for items in self.rollout_cache.history:
            token_ids.extend(items["prompt_ids"])
            token_ids.extend(items["response_ids"])
            prompt_masks.extend([1] * len(items["prompt_ids"]) + [0] * len(items["response_ids"]))
            response_masks.extend([0] * len(items["prompt_ids"]) + [1] * len(items["response_ids"]))
            if "infer_logprobs" in items:
                infer_logprobs.extend([0] * len(items["prompt_ids"]) + items["infer_logprobs"])

        input_ids =torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor([1] * len(token_ids), dtype=torch.long).unsqueeze(0)
        response_mask = torch.tensor(response_masks, dtype=torch.bool).unsqueeze(0)

        first_response_idx = response_masks.index(1)
        prompt_masks = [1] * first_response_idx + [0] * (len(token_ids) - first_response_idx)
        prompt_mask =torch.tensor(prompt_masks, dtype=torch.bool).unsqueeze(0)
        score_tensor = torch.tensor([0] * len(token_ids), dtype=torch.float).unsqueeze(0)
        score_tensor[0][-1] = episode_score
        # Huggingface Transformers prefer position_ids to be 0-based.
        # Attn Mask: [1, 1, 1, ..., 1, 0, 0, ..., 0]
        # cumsum: [1, 2, 3, ..., n, n+1, n+1, ..., n+1]
        # cumsum - 1: [0, 1, 2, ..., n-1, n, n, ..., n]
        position_ids = attention_mask.cumsum(dim=-1) - 1

        lm_input = DataProto()
        lm_input.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0])

        response_length = response_mask.sum(dim=-1).float().mean().item()

        # TODO: move pad to pipeline
        input_ids = pad_to_length(input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id)
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        position_ids = pad_to_length(position_ids, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        prompt_mask = pad_to_length(prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        lm_input.batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            "prompt_mask": prompt_mask,
            "scores": score_tensor,
        })
        if len(infer_logprobs):
            infer_logprobs = torch.tensor(infer_logprobs, dtype=torch.float).unsqueeze(0)
            infer_logprobs = pad_to_length(infer_logprobs, length=self.pipeline_config.sequence_length, pad_value=0)
            lm_input.batch["infer_logprobs"] = infer_logprobs[:, 1:]

        lm_input.non_tensor_batch.update({
            "env_ids": np.array([self.rollout_cache.env_id], dtype=object),
            "group_ids": np.array([self.rollout_cache.group_id], dtype=object),
            "tags": np.array([self.rollout_cache.tag], dtype=object),
            "step_scores": np.array([scores], dtype=object),
            "episode_scores": np.array([episode_score], dtype=object),
        })

        metrics_agg_mode = self.rollout_cache.history[-1].get('metrics_agg_mode', {})
        history_metrics = [item.get("metrics", {}) for item in self.rollout_cache.history]
        env_metric = aggregate_metrics(history_metrics=history_metrics, metrics_agg_mode=metrics_agg_mode)
        env_metric["num_actions"] = rollout_cache.step
        if self._resume_e2e_latency_samples:
            env_metric["resume_latency_e2e_mean_s"] = float(np.mean(self._resume_e2e_latency_samples))
            env_metric["resume_latency_e2e_p50_s"] = float(np.percentile(self._resume_e2e_latency_samples, 50))
            env_metric["resume_latency_e2e_p95_s"] = float(np.percentile(self._resume_e2e_latency_samples, 95))
        if self._resume_infer_latency_samples:
            env_metric["resume_infer_latency_mean_s"] = float(np.mean(self._resume_infer_latency_samples))
            env_metric["resume_infer_latency_p50_s"] = float(np.percentile(self._resume_infer_latency_samples, 50))
            env_metric["resume_infer_latency_p95_s"] = float(np.percentile(self._resume_infer_latency_samples, 95))
        if self._resume_prefill_tokens_samples:
            env_metric["resume_prefill_tokens_mean"] = float(np.mean(self._resume_prefill_tokens_samples))
            env_metric["resume_prefill_tokens_p50"] = float(np.percentile(self._resume_prefill_tokens_samples, 50))
            env_metric["resume_prefill_tokens_p95"] = float(np.percentile(self._resume_prefill_tokens_samples, 95))
        if self._external_wait_samples:
            env_metric["external_wait_mean_s"] = float(np.mean(self._external_wait_samples))
            env_metric["external_wait_p50_s"] = float(np.percentile(self._external_wait_samples, 50))
            env_metric["external_wait_p95_s"] = float(np.percentile(self._external_wait_samples, 95))
        if self._resume_queue_wait_samples:
            env_metric["resume_queue_wait_mean_s"] = float(np.mean(self._resume_queue_wait_samples))
            env_metric["resume_queue_wait_p50_s"] = float(np.percentile(self._resume_queue_wait_samples, 50))
            env_metric["resume_queue_wait_p95_s"] = float(np.percentile(self._resume_queue_wait_samples, 95))
        if self._resume_client_submit_before_samples:
            env_metric["resume_client_submit_before_mean_s"] = float(np.mean(self._resume_client_submit_before_samples))
            env_metric["resume_client_submit_before_p50_s"] = float(np.percentile(self._resume_client_submit_before_samples, 50))
            env_metric["resume_client_submit_before_p95_s"] = float(np.percentile(self._resume_client_submit_before_samples, 95))
        if self._resume_pre_router_samples:
            env_metric["resume_pre_router_mean_s"] = float(np.mean(self._resume_pre_router_samples))
            env_metric["resume_pre_router_p50_s"] = float(np.percentile(self._resume_pre_router_samples, 50))
            env_metric["resume_pre_router_p95_s"] = float(np.percentile(self._resume_pre_router_samples, 95))
        if self._resume_router_lookup_samples:
            env_metric["resume_router_lookup_mean_s"] = float(np.mean(self._resume_router_lookup_samples))
            env_metric["resume_router_lookup_p50_s"] = float(np.percentile(self._resume_router_lookup_samples, 50))
            env_metric["resume_router_lookup_p95_s"] = float(np.percentile(self._resume_router_lookup_samples, 95))
        if self._resume_router_priority_samples:
            env_metric["resume_router_priority_mean_s"] = float(np.mean(self._resume_router_priority_samples))
            env_metric["resume_router_priority_p50_s"] = float(np.percentile(self._resume_router_priority_samples, 50))
            env_metric["resume_router_priority_p95_s"] = float(np.percentile(self._resume_router_priority_samples, 95))
        if self._resume_router_schedule_samples:
            env_metric["resume_router_schedule_mean_s"] = float(np.mean(self._resume_router_schedule_samples))
            env_metric["resume_router_schedule_p50_s"] = float(np.percentile(self._resume_router_schedule_samples, 50))
            env_metric["resume_router_schedule_p95_s"] = float(np.percentile(self._resume_router_schedule_samples, 95))
        if self._resume_dispatch_to_engine_start_samples:
            env_metric["resume_dispatch_to_engine_start_mean_s"] = float(np.mean(self._resume_dispatch_to_engine_start_samples))
            env_metric["resume_dispatch_to_engine_start_p50_s"] = float(np.percentile(self._resume_dispatch_to_engine_start_samples, 50))
            env_metric["resume_dispatch_to_engine_start_p95_s"] = float(np.percentile(self._resume_dispatch_to_engine_start_samples, 95))
        if self._resume_engine_ttft_samples:
            env_metric["resume_engine_ttft_mean_s"] = float(np.mean(self._resume_engine_ttft_samples))
            env_metric["resume_engine_ttft_p50_s"] = float(np.percentile(self._resume_engine_ttft_samples, 50))
            env_metric["resume_engine_ttft_p95_s"] = float(np.percentile(self._resume_engine_ttft_samples, 95))
        if self._resume_decode_tail_samples:
            env_metric["resume_decode_tail_mean_s"] = float(np.mean(self._resume_decode_tail_samples))
            env_metric["resume_decode_tail_p50_s"] = float(np.percentile(self._resume_decode_tail_samples, 50))
            env_metric["resume_decode_tail_p95_s"] = float(np.percentile(self._resume_decode_tail_samples, 95))
        if self._resume_router_return_overhead_samples:
            env_metric["resume_router_return_overhead_mean_s"] = float(np.mean(self._resume_router_return_overhead_samples))
            env_metric["resume_router_return_overhead_p50_s"] = float(np.percentile(self._resume_router_return_overhead_samples, 50))
            env_metric["resume_router_return_overhead_p95_s"] = float(np.percentile(self._resume_router_return_overhead_samples, 95))
        if self._resume_post_router_overhead_samples:
            env_metric["resume_post_router_overhead_mean_s"] = float(np.mean(self._resume_post_router_overhead_samples))
            env_metric["resume_post_router_overhead_p50_s"] = float(np.percentile(self._resume_post_router_overhead_samples, 50))
            env_metric["resume_post_router_overhead_p95_s"] = float(np.percentile(self._resume_post_router_overhead_samples, 95))

        env_metric = {f"env/{rollout_cache.tag}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length

        traj_group_id = (
            f"{rollout_cache.tag}_{rollout_cache.group_id}_{self.episode_id}_{self.group_seed}"
        )
        traj_id = f"{traj_group_id}_{rollout_cache.env_id}"
        serializable_history = []
        for item in history:
            entry = {
                k: v
                for k, v in item.items()
                if k not in ("prompt_ids", "response_ids", "infer_logprobs")
            }
            if "metrics" in entry and isinstance(entry["metrics"], dict):
                entry["metrics"] = dict(entry["metrics"])
            serializable_history.append(entry)
        trajectory_payload = {
            "trajectory_id": traj_id,
            "tag": rollout_cache.tag,
            "group_id": rollout_cache.group_id,
            "env_id": rollout_cache.env_id,
            "episode_id": self.episode_id,
            "num_actions": rollout_cache.step,
            "response_length": response_length,
            "episode_score": episode_score,
            "resume_count": len(self._resume_prefill_tokens_samples),
            "tool_use_count": sum(1 for h in history if h.get("use_tool")),
            "resume_prefill_tokens_samples": list(self._resume_prefill_tokens_samples),
            "resume_actual_hit_samples": list(self._resume_actual_hit_samples),
            "resume_matched_prefix_tokens_samples": list(self._resume_matched_prefix_tokens_samples),
            "resume_pinned_kv_gb_seconds_samples": list(self._resume_pinned_kv_gb_seconds_samples),
            "resume_prefill_ratio_samples": list(self._resume_prefill_ratio_samples),
            "resume_saved_prefill_ms_samples": list(self._resume_saved_prefill_ms_samples),
            "resume_latency_e2e_samples": list(self._resume_e2e_latency_samples),
            "resume_infer_latency_samples": list(self._resume_infer_latency_samples),
            "resume_queue_wait_samples": list(self._resume_queue_wait_samples),
            "resume_client_submit_before_samples": list(self._resume_client_submit_before_samples),
            "resume_pre_router_samples": list(self._resume_pre_router_samples),
            "resume_dispatch_to_engine_start_samples": list(self._resume_dispatch_to_engine_start_samples),
            "resume_engine_ttft_samples": list(self._resume_engine_ttft_samples),
            "resume_decode_tail_samples": list(self._resume_decode_tail_samples),
            "resume_router_return_overhead_samples": list(self._resume_router_return_overhead_samples),
            "resume_post_router_overhead_samples": list(self._resume_post_router_overhead_samples),
            "external_wait_samples": list(self._external_wait_samples),
            "history": serializable_history,
        }
        lm_input.non_tensor_batch["trajectory_json"] = np.array(
            [json.dumps(trajectory_payload, ensure_ascii=False)], dtype=object
        )
        # traj_id is set again in run_rollout_loop; do not list it in COLUMMNS_CONFIG
        # because dump_rollout_trajectories pops listed keys before train/log grouping.
        lm_input.meta_info = {
            "metrics": env_metric,
            # Only dump observation fields; dump_rollout_trajectories pops listed keys.
            "COLUMMNS_CONFIG": [
                ["trajectory_json", "string"],
            ],
        }
        return lm_input
