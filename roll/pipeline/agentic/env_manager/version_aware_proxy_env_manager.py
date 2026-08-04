import copy
import json
import time
from threading import Lock
from typing import Dict, Optional

import numpy as np
import ray

from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.env_manager.proxy_env_manager import ProxyEnvManager


class VersionAwareProxyEnvManager(ProxyEnvManager):
    """Proxy environment manager with version-aware scheduling and accounting."""

    def __init__(self, *args, **kwargs):
        self._progress_snapshot_lock = Lock()
        self._latest_progress_snapshot: Optional[Dict] = None
        self.trajectory_version = -1
        self.trajectory_id: Optional[str] = None
        self.runtime_phase = "idle"
        self._episode_started = False
        self._episode_completed = False
        self._episode_submitted = False
        self._episode_truncated = False
        self._episode_result: Dict = {}
        self._episode_completed_at: Optional[float] = None
        self._tail_env_seconds = 0.0
        super().__init__(*args, **kwargs)

    def run_rollout_loop(self, data: DataProto):
        assert "seed" in data.meta_info
        self.running = True
        self.group_seed = data.meta_info["seed"] + self.env_config["group_seed"]
        start_step = self.current_step

        while True:
            self.episode_id = ray.get(
                self.output_queue.get_episode_id.remote(
                    self.env_config["group_id"], self.env_config["env_id"]
                )
            )
            start_step = self.current_step
            if self.episode_id is None:
                break

            self.history = []
            self.reset_stats()
            self._start_episode(start_step)
            seed = self.group_seed + self.episode_id

            episode_result = self.agent_runner.run_job(seed)
            if episode_result.status == "NoData":
                self.runtime_phase = "no_data"
                self._publish_progress_snapshot()
                break

            result = episode_result.to_dict()
            self._episode_result = result
            self._episode_completed = episode_result.status == "Finished"
            self._episode_truncated = bool(result.get("truncated", False))
            self._episode_completed_at = time.time()
            self._capture_tail_environment_time()
            self.runtime_phase = "completed_not_submitted"
            self._publish_progress_snapshot()

            self.running = False
            rollout = self.formulate_rollouts(result)
            ray.get(
                self.output_queue.put.remote(
                    self.env_config["group_id"],
                    self.episode_id,
                    start_step,
                    rollout,
                    self.env_config["env_id"],
                )
            )
            self._episode_submitted = True
            self.runtime_phase = "submitted"
            self._publish_progress_snapshot()
            self.running = True

        self.running = False
        ray.get(
            self.output_queue.put.remote(
                self.env_config["group_id"],
                self.episode_id,
                start_step,
                None,
                self.env_config["env_id"],
            )
        )

    def _start_episode(self, start_step: int) -> None:
        tag = self.env_config.get("tag", "proxy")
        traj_group_id = (
            f"{tag}_{self.env_config['group_id']}_{self.episode_id}_{self.group_seed}"
        )
        self.trajectory_id = f"{traj_group_id}_{self.env_config['env_id']}"
        self.trajectory_version = int(start_step)
        self.runtime_phase = "environment_reset"
        self._episode_started = True
        self._episode_completed = False
        self._episode_submitted = False
        self._episode_truncated = False
        self._episode_result = {}
        self._episode_completed_at = None
        self._tail_env_seconds = 0.0
        self._publish_progress_snapshot()

    def _capture_tail_environment_time(self) -> None:
        if self.last_response_finish_time is not None:
            self._tail_env_seconds = max(
                0.0, time.time() - self.last_response_finish_time
            )

    def _tool_calls(self) -> int:
        return sum(
            len(item.get("response_message", {}).get("tool_calls") or [])
            for item in self.history
        )

    def _request_metric_total(self, name: str) -> float:
        total = 0.0
        for item in self.history:
            value = item.get("request_metrics", {}).get(name, 0.0)
            try:
                total += float(value)
            except (TypeError, ValueError):
                continue
        return total

    def _tool_wall_seconds(self) -> float:
        runner_metrics = getattr(self.agent_runner, "runtime_metrics", {}) or {}
        values = (
            self._episode_result.get("tool_wall_seconds", 0.0),
            runner_metrics.get("tool_wall_seconds", 0.0),
        )
        return max(float(value or 0.0) for value in values)

    def _trajectory_runtime_state(self) -> Dict:
        actions_completed = len(self.history)
        max_actions = int(self.env_config.get("max_steps", actions_completed))
        current_version = int(self.current_step)
        return {
            "trajectory_id": self.trajectory_id or "unknown",
            "group_id": int(self.env_config["group_id"]),
            "episode_id": int(self.episode_id),
            "env_id": int(self.env_config["env_id"]),
            "policy_version": int(self.trajectory_version),
            "current_version": current_version,
            "version_age": max(0, current_version - int(self.trajectory_version)),
            "actions_completed": actions_completed,
            "inference_calls": actions_completed,
            "tool_calls": self._tool_calls(),
            "max_actions": max_actions,
            "remaining_actions": max(0, max_actions - actions_completed),
        }

    def _augment_lm_input_meta(
        self,
        lm_input: DataProto,
        request_role: str,
        track_trajectory: bool,
    ) -> None:
        scheduling_enabled = (
            getattr(self.pipeline_config, "trajectory_scheduling_policy", "fifo")
            == "version_priority"
        )
        if scheduling_enabled or getattr(
            self.pipeline_config, "version_boundary_profiler_enabled", False
        ):
            runtime_state = self._trajectory_runtime_state()
            runtime_state["scheduling_enabled"] = scheduling_enabled
            runtime_state["request_role"] = request_role
            runtime_state["track_trajectory"] = track_trajectory
            lm_input.meta_info["trajectory_priority"] = runtime_state

    def _on_proxy_request_start(
        self, request_role: str, track_trajectory: bool
    ) -> None:
        self.runtime_phase = (
            "policy_inference" if track_trajectory else "environment_model"
        )
        self._publish_progress_snapshot()

    def _on_proxy_request_finish(
        self, request_role: str, track_trajectory: bool
    ) -> None:
        self.runtime_phase = "tool_or_environment"
        self._publish_progress_snapshot()

    def _build_progress_snapshot(self) -> Optional[Dict]:
        if not self._episode_started or self.episode_id is None:
            return None

        prompt_tokens = sum(len(item.get("prompt_ids", [])) for item in self.history)
        response_tokens = sum(
            len(item.get("response_ids", [])) for item in self.history
        )
        latest = self.history[-1] if self.history else {}
        latest_prompt_tokens = len(latest.get("prompt_ids", []))
        latest_response_tokens = len(latest.get("response_ids", []))
        max_actions = int(self.env_config.get("max_steps", len(self.history)))
        wall_seconds = max(0.0, time.time() - self.traj_start_time)
        env_seconds = float(sum(self.log_stats.get("env_exec_time", [])))
        env_seconds += float(self._tail_env_seconds)
        remaining_actions = max(0, max_actions - len(self.history))
        action_budget_progress = (
            min(1.0, len(self.history) / max_actions) if max_actions else 0.0
        )

        return {
            "trajectory_id": self.trajectory_id or "unknown",
            "category": (
                "completed_not_submitted"
                if self._episode_completed and not self._episode_submitted
                else "inflight_at_shutdown"
            ),
            "discard_reason": "pipeline_shutdown",
            "group_id": int(self.env_config["group_id"]),
            "episode_id": int(self.episode_id),
            "env_id": int(self.env_config["env_id"]),
            "version_start": int(self.trajectory_version),
            "version_end": int(self.current_step),
            "version_age": max(
                0, int(self.current_step) - int(self.trajectory_version)
            ),
            "reset_completed": True,
            "completed": bool(self._episode_completed),
            "truncated": bool(self._episode_truncated),
            "actions_completed": len(self.history),
            "inference_calls": len(self.history),
            "tool_calls": self._tool_calls(),
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "inference_tokens": prompt_tokens + response_tokens,
            "latest_prompt_tokens": latest_prompt_tokens,
            "latest_response_tokens": latest_response_tokens,
            "current_context_tokens": latest_prompt_tokens + latest_response_tokens,
            "max_actions": max_actions,
            "remaining_actions": remaining_actions,
            "action_budget_progress": action_budget_progress,
            "runtime_phase": self.runtime_phase,
            "generate_seconds": float(
                sum(self.log_stats.get("pure_infer_time", []))
            ),
            "environment_model_seconds": float(
                sum(self.log_stats.get("environment_model_infer_time", []))
            ),
            "env_seconds": env_seconds,
            "tool_wall_seconds": self._tool_wall_seconds(),
            "request_queue_seconds": self._request_metric_total(
                "vllm/request_queue_seconds"
            ),
            "router_scheduling_wait_seconds": self._request_metric_total(
                "router/scheduling_wait_seconds"
            ),
            "router_control_path_seconds": self._request_metric_total(
                "router/control_path_seconds"
            ),
            "router_control_cpu_seconds": self._request_metric_total(
                "router/control_cpu_seconds"
            ),
            "request_ttft_seconds": self._request_metric_total(
                "vllm/request_ttft_seconds"
            ),
            "request_prefill_seconds": self._request_metric_total(
                "vllm/request_prefill_seconds"
            ),
            "request_decode_seconds": self._request_metric_total(
                "vllm/request_decode_seconds"
            ),
            "request_inference_seconds": self._request_metric_total(
                "vllm/request_inference_seconds"
            ),
            "request_latency_seconds": self._request_metric_total(
                "vllm/request_latency_seconds"
            ),
            "model_forward_seconds": self._request_metric_total(
                "vllm/request_model_forward_seconds"
            ),
            "model_execute_seconds": self._request_metric_total(
                "vllm/request_model_execute_seconds"
            ),
            "engine_step_seconds_attributed": self._request_metric_total(
                "vllm/request_engine_step_seconds_attributed"
            ),
            "prefill_engine_step_seconds_attributed": (
                self._request_metric_total(
                    "vllm/request_prefill_engine_step_seconds_attributed"
                )
            ),
            "decode_engine_step_seconds_attributed": (
                self._request_metric_total(
                    "vllm/request_decode_engine_step_seconds_attributed"
                )
            ),
            "engine_prefill_tokens": int(
                self._request_metric_total("vllm/request_prefill_tokens")
            ),
            "engine_cached_prompt_tokens": int(
                self._request_metric_total(
                    "vllm/request_cached_prompt_tokens"
                )
            ),
            "trajectory_started_at_unix": float(self.traj_start_time),
            "trajectory_completed_at_unix": self._episode_completed_at,
            "trajectory_wall_seconds": wall_seconds,
        }

    def _publish_progress_snapshot(self) -> None:
        snapshot = self._build_progress_snapshot()
        with self._progress_snapshot_lock:
            self._latest_progress_snapshot = copy.deepcopy(snapshot)

    def get_progress_snapshot(self) -> Optional[Dict]:
        with self._progress_snapshot_lock:
            snapshot = copy.deepcopy(self._latest_progress_snapshot)
        if snapshot is None:
            return None
        # Once a rollout has entered the scheduler queue it is accounted for by
        # GroupQueueManager. Returning the worker's last submitted snapshot at
        # shutdown would count consumed or buffered work a second time.
        if snapshot.get("runtime_phase") in {"submitted", "idle", "no_data"}:
            return None
        return snapshot

    def formulate_rollouts(self, runner_result: Dict) -> DataProto:
        batch = super().formulate_rollouts(runner_result)
        snapshot = self._build_progress_snapshot() or {}
        batch_size = batch.batch.batch_size[0]
        tag = self.env_config.get("tag", "proxy")
        stale_tolerance = int(
            getattr(self.pipeline_config, "trajectory_staleness_tolerance", -1)
        )

        metric_values = {
            "traj_observed": 1.0,
            "traj_completed": float(self._episode_completed),
            "traj_truncated": float(self._episode_truncated),
            "traj_actions_completed": float(snapshot.get("actions_completed", 0)),
            "traj_inference_calls": float(snapshot.get("inference_calls", 0)),
            "traj_tool_calls": float(snapshot.get("tool_calls", 0)),
            "traj_prompt_tokens_total": float(snapshot.get("prompt_tokens", 0)),
            "traj_response_tokens_total": float(snapshot.get("response_tokens", 0)),
            "traj_inference_tokens_total": float(snapshot.get("inference_tokens", 0)),
            "traj_generate_seconds_total": float(snapshot.get("generate_seconds", 0.0)),
            "traj_env_seconds_total": float(snapshot.get("env_seconds", 0.0)),
            "traj_tool_wall_seconds_total": float(
                snapshot.get("tool_wall_seconds", 0.0)
            ),
            "traj_request_queue_seconds_total": float(
                snapshot.get("request_queue_seconds", 0.0)
            ),
            "traj_router_scheduling_wait_seconds_total": float(
                snapshot.get("router_scheduling_wait_seconds", 0.0)
            ),
            "traj_router_control_path_seconds_total": float(
                snapshot.get("router_control_path_seconds", 0.0)
            ),
            "traj_router_control_cpu_seconds_total": float(
                snapshot.get("router_control_cpu_seconds", 0.0)
            ),
            "traj_request_ttft_seconds_total": float(
                snapshot.get("request_ttft_seconds", 0.0)
            ),
            "traj_request_prefill_seconds_total": float(
                snapshot.get("request_prefill_seconds", 0.0)
            ),
            "traj_request_decode_seconds_total": float(
                snapshot.get("request_decode_seconds", 0.0)
            ),
            "traj_request_inference_seconds_total": float(
                snapshot.get("request_inference_seconds", 0.0)
            ),
            "traj_request_latency_seconds_total": float(
                snapshot.get("request_latency_seconds", 0.0)
            ),
            "traj_model_forward_seconds_total": float(
                snapshot.get("model_forward_seconds", 0.0)
            ),
            "traj_model_execute_seconds_total": float(
                snapshot.get("model_execute_seconds", 0.0)
            ),
            "traj_engine_step_seconds_attributed_total": float(
                snapshot.get("engine_step_seconds_attributed", 0.0)
            ),
            "traj_prefill_engine_step_seconds_attributed_total": float(
                snapshot.get(
                    "prefill_engine_step_seconds_attributed", 0.0
                )
            ),
            "traj_decode_engine_step_seconds_attributed_total": float(
                snapshot.get("decode_engine_step_seconds_attributed", 0.0)
            ),
            "traj_engine_prefill_tokens_total": float(
                snapshot.get("engine_prefill_tokens", 0)
            ),
            "traj_engine_cached_prompt_tokens_total": float(
                snapshot.get("engine_cached_prompt_tokens", 0)
            ),
            "traj_max_actions": float(snapshot.get("max_actions", 0)),
            "traj_remaining_actions": float(
                snapshot.get("remaining_actions", 0)
            ),
            "traj_action_budget_progress": float(
                snapshot.get("action_budget_progress", 0.0)
            ),
            "traj_wall_seconds_total": float(
                snapshot.get("trajectory_wall_seconds", 0.0)
            ),
            "traj_version_start": float(self.trajectory_version),
            "traj_version_end": float(self.current_step),
            "traj_version_age": float(
                max(0, int(self.current_step) - int(self.trajectory_version))
            ),
            "traj_stale_tolerance": float(stale_tolerance),
            "traj_environment_model_calls": float(
                len(self.log_stats.get("environment_model_infer_time", []))
            ),
            "traj_environment_model_seconds_total": float(
                sum(self.log_stats.get("environment_model_infer_time", []))
            ),
            "traj_environment_model_prompt_tokens_total": float(
                sum(self.log_stats.get("environment_model_prompt_tokens", []))
            ),
            "traj_environment_model_response_tokens_total": float(
                sum(self.log_stats.get("environment_model_response_tokens", []))
            ),
        }
        batch.meta_info.setdefault("metrics", {}).update(
            {f"env/{tag}/{key}": value for key, value in metric_values.items()}
        )

        direct_values = {
            "traj_actions_completed": int(snapshot.get("actions_completed", 0)),
            "traj_inference_calls": int(snapshot.get("inference_calls", 0)),
            "traj_tool_calls": int(snapshot.get("tool_calls", 0)),
            "traj_prompt_tokens_total": int(snapshot.get("prompt_tokens", 0)),
            "traj_response_tokens_total": int(snapshot.get("response_tokens", 0)),
            "traj_inference_tokens_total": int(snapshot.get("inference_tokens", 0)),
            "traj_generate_seconds_total": float(snapshot.get("generate_seconds", 0.0)),
            "traj_env_seconds_total": float(snapshot.get("env_seconds", 0.0)),
            "traj_tool_wall_seconds_total": float(
                snapshot.get("tool_wall_seconds", 0.0)
            ),
            "traj_request_queue_seconds_total": float(
                snapshot.get("request_queue_seconds", 0.0)
            ),
            "traj_router_scheduling_wait_seconds_total": float(
                snapshot.get("router_scheduling_wait_seconds", 0.0)
            ),
            "traj_router_control_path_seconds_total": float(
                snapshot.get("router_control_path_seconds", 0.0)
            ),
            "traj_router_control_cpu_seconds_total": float(
                snapshot.get("router_control_cpu_seconds", 0.0)
            ),
            "traj_request_ttft_seconds_total": float(
                snapshot.get("request_ttft_seconds", 0.0)
            ),
            "traj_request_prefill_seconds_total": float(
                snapshot.get("request_prefill_seconds", 0.0)
            ),
            "traj_request_decode_seconds_total": float(
                snapshot.get("request_decode_seconds", 0.0)
            ),
            "traj_request_inference_seconds_total": float(
                snapshot.get("request_inference_seconds", 0.0)
            ),
            "traj_request_latency_seconds_total": float(
                snapshot.get("request_latency_seconds", 0.0)
            ),
            "traj_model_forward_seconds_total": float(
                snapshot.get("model_forward_seconds", 0.0)
            ),
            "traj_model_execute_seconds_total": float(
                snapshot.get("model_execute_seconds", 0.0)
            ),
            "traj_engine_step_seconds_attributed_total": float(
                snapshot.get("engine_step_seconds_attributed", 0.0)
            ),
            "traj_prefill_engine_step_seconds_attributed_total": float(
                snapshot.get(
                    "prefill_engine_step_seconds_attributed", 0.0
                )
            ),
            "traj_decode_engine_step_seconds_attributed_total": float(
                snapshot.get("decode_engine_step_seconds_attributed", 0.0)
            ),
            "traj_engine_prefill_tokens_total": int(
                snapshot.get("engine_prefill_tokens", 0)
            ),
            "traj_engine_cached_prompt_tokens_total": int(
                snapshot.get("engine_cached_prompt_tokens", 0)
            ),
            "traj_max_actions": int(snapshot.get("max_actions", 0)),
            "traj_remaining_actions": int(
                snapshot.get("remaining_actions", 0)
            ),
            "traj_action_budget_progress": float(
                snapshot.get("action_budget_progress", 0.0)
            ),
            "traj_started_at_unix": float(
                snapshot.get("trajectory_started_at_unix", 0.0)
            ),
            "traj_completed_at_unix": float(
                snapshot.get("trajectory_completed_at_unix") or 0.0
            ),
            "traj_wall_seconds_total": float(
                snapshot.get("trajectory_wall_seconds", 0.0)
            ),
        }
        for key, value in direct_values.items():
            batch.non_tensor_batch[key] = np.array(
                [value] * batch_size, dtype=object
            )

        traj_group_id = (
            f"{tag}_{self.env_config['group_id']}_{self.episode_id}_{self.group_seed}"
        )
        stable_ids = [
            self.trajectory_id
            if batch_size == 1
            else f"{self.trajectory_id}_b{idx}"
            for idx in range(batch_size)
        ]
        batch.non_tensor_batch["traj_group_id"] = np.array(
            [traj_group_id] * batch_size, dtype=object
        )
        batch.non_tensor_batch["traj_id"] = np.array(stable_ids, dtype=object)

        trajectory_data = batch.non_tensor_batch.get("trajectory_data")
        if trajectory_data is not None:
            for idx, raw_value in enumerate(trajectory_data):
                try:
                    value = json.loads(raw_value)
                except (TypeError, json.JSONDecodeError):
                    continue
                value["trajectory_id"] = stable_ids[idx]
                value["version_info"] = {
                    "version_start": int(self.trajectory_version),
                    "version_end": int(self.current_step),
                    "version_age": max(
                        0, int(self.current_step) - int(self.trajectory_version)
                    ),
                    "stale_tolerance": stale_tolerance,
                }
                value["progress_info"] = direct_values
                trajectory_data[idx] = json.dumps(value)

        return batch
