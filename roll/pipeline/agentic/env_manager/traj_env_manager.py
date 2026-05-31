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
        self._external_wait_samples: list[float] = []
        self._resume_queue_wait_samples: list[float] = []
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
        self._external_wait_samples = []
        self._resume_queue_wait_samples = []
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
        return "<tool_call>" in text

    def _maybe_set_pending_tool_suspend_lease(self) -> None:
        """Register tool-wait KV lease before external tool blocks in env.step (L1 suspend)."""
        snapshot = get_scheduling_weight_snapshot()
        if snapshot is None or not self.trajectory_id or self._last_backend_id is None:
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
        route_meta = {
            "trajectory_id": self.trajectory_id,
            "request_type": "resume",
            "last_backend_id": self._last_backend_id,
            "history_len_tokens": float(history_len_tokens),
            **traj_signals,
        }
        state = get_trajectory_scheduling_state()
        t_tool = state.get_t_tool_s(self.trajectory_id)
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
            self.rollout_cache.history[-1].update(info)
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
            self._last_backend_id = selected_backend_id

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
            resume_infer_latency_s = max(0.0, infer_end_ts - infer_start_ts)
            self._resume_infer_latency_samples.append(resume_infer_latency_s)
            self._resume_prefill_tokens_samples.append(float(input_ids.shape[1]))
            if self._pause_ts is not None:
                resume_latency_e2e_s = max(0.0, infer_end_ts - self._pause_ts)
                self._resume_e2e_latency_samples.append(resume_latency_e2e_s)

            if "metrics" not in content or not isinstance(content["metrics"], dict):
                content["metrics"] = {}
            if "metrics_agg_mode" not in content or not isinstance(content["metrics_agg_mode"], dict):
                content["metrics_agg_mode"] = {}
            if self._resume_e2e_latency_samples:
                content["metrics"]["resume_latency_e2e_s"] = self._resume_e2e_latency_samples[-1]
            content["metrics"]["resume_infer_start_ts"] = infer_start_ts
            content["metrics"]["resume_first_token_ts"] = infer_end_ts
            content["metrics"]["resume_infer_end_ts"] = infer_end_ts
            content["metrics"]["resume_infer_latency_s"] = resume_infer_latency_s
            content["metrics"]["resume_prefill_tokens"] = float(input_ids.shape[1])
            content["metrics"]["resume_history_len_tokens"] = float(input_ids.shape[1])
            for key in (
                "resume_enqueue_ts",
                "resume_dispatch_ts",
                "resume_queue_wait_s",
                "context_class_gpu_hit",
                "context_class_cpu_reload",
                "context_class_full_prefill",
                "selected_backend_affinity_hit",
                "selected_backend_migration",
                "worker_load_skew_at_dispatch",
                "selected_worker_load_at_dispatch",
                "routing_policy",
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
                "lookup_resume_found",
                "lookup_hit_tokens",
                "lookup_cache_confidence",
                "lookup_estimated_prefill_tokens",
                "lookup_lease_remaining_s",
                "ttl_remaining_s",
                "actual_hit",
                "matched_prefix_tokens",
                "resume_prefill_tokens",
                "estimated_prefill_tokens",
                "prefill_time_ms",
                "cache_confidence",
            ):
                if key in lm_output.meta_info:
                    value = lm_output.meta_info[key]
                    try:
                        content["metrics"][key] = float(value)
                    except (TypeError, ValueError):
                        content["metrics"][key] = value
            if "resume_queue_wait_s" in lm_output.meta_info:
                self._resume_queue_wait_samples.append(float(lm_output.meta_info["resume_queue_wait_s"]))
            content["metrics_agg_mode"].update({
                "resume_latency_e2e_s": "mean",
                "resume_infer_start_ts": "last",
                "resume_first_token_ts": "last",
                "resume_infer_end_ts": "last",
                "resume_infer_latency_s": "mean",
                "resume_prefill_tokens": "mean",
                "resume_history_len_tokens": "mean",
                "resume_enqueue_ts": "last",
                "resume_dispatch_ts": "last",
                "resume_queue_wait_s": "mean",
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
        if 'observation' in rollout_cache.history[-1]:
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
            "resume_latency_e2e_samples": list(self._resume_e2e_latency_samples),
            "resume_infer_latency_samples": list(self._resume_infer_latency_samples),
            "resume_queue_wait_samples": list(self._resume_queue_wait_samples),
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