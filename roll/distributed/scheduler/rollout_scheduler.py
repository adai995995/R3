import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray._private import profiling
from tqdm import tqdm

from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.router import RouterManager
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_config import EnvManagerConfig, EnvMonitorConfig
from roll.distributed.scheduler.rollout_mock_mixin import RolloutMockMixin
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import append_to_dict
from roll.utils.import_utils import safe_import_class
from roll.utils.logging import get_logger
from roll.utils.telemetry import (
    attach_trace_context,
    extract_trace_context,
    inject_trace_context,
    get_tracer,
)

logger = get_logger()


class EnvActivityMonitor:
    """Environment activity monitor for tracking and detecting hung envs."""

    def __init__(self, config: EnvMonitorConfig, group_queue_dict: Dict[int, 'GroupQueue']):
        """
        Args:
            config: EnvMonitorConfig object
            group_queue_dict: Reference to GroupQueue dict for checking episode status
        """
        self.group_queue_dict = group_queue_dict
        self.enable = config.enable

        # Configuration parameters
        self.monitor_interval = config.monitor_interval  # seconds
        self.hung_timeout = config.hung_timeout  # seconds (default: 1 hour)

        # Tracking data structures - Dual-timestamp approach
        # Track when env starts processing an episode
        # Key: ((group_id, env_id), episode_id) -> Value: timestamp
        self.env_episode_start: Dict[Tuple[Tuple[int, int], int], float] = {}

        # Track when env submits episode rollout
        # Key: ((group_id, env_id), episode_id) -> Value: timestamp
        self.env_episode_submit: Dict[Tuple[Tuple[int, int], int], float] = {}

        # Track each env's current episode (for cleanup)
        # Key: (group_id, env_id) -> Value: episode_id
        self.env_current_episode: Dict[Tuple[int, int], int] = {}

        # Monitor task
        self.monitor_task: Optional[asyncio.Task] = None

    def record_episode_start(self, group_id: int, env_id: int, episode_id: int):
        """
        Record when env starts processing a new episode.
        Called from GroupQueue.get_episode_id() when an episode is assigned to an env.

        Args:
            group_id: Group ID
            env_id: Environment ID
            episode_id: Episode ID assigned to this env
        """
        if not self.enable:
            return

        env_key = (group_id, env_id)
        episode_key = ((group_id, env_id), episode_id)

        # Automatic cleanup: Remove old episode records for this env
        old_episode_id = self.env_current_episode.get(env_key)
        if old_episode_id is not None and old_episode_id != episode_id:
            old_episode_key = ((group_id, env_id), old_episode_id)
            self.env_episode_start.pop(old_episode_key, None)
            self.env_episode_submit.pop(old_episode_key, None)

        # Record new episode start time
        self.env_episode_start[episode_key] = time.time()
        self.env_current_episode[env_key] = episode_id

    def record_activity(self, group_id: int, env_id: int, episode_id: int, rollout: Optional[DataProto]):
        """
        Record env activity when submitting a rollout.
        Called from GroupQueueManager.put() when env submits rollout.

        Args:
            group_id: Group ID
            env_id: Environment ID
            episode_id: Episode ID
            rollout: Rollout data (None means env is exiting)
        """
        if not self.enable:
            return

        env_key = (group_id, env_id)
        episode_key = ((group_id, env_id), episode_id)

        if rollout is None:
            # Env calls put(..., None) to signal exit, remove all tracking
            self.env_episode_start.pop(episode_key, None)
            self.env_episode_submit.pop(episode_key, None)
            self.env_current_episode.pop(env_key, None)
            return

        # Normal rollout submission, record submit time
        self.env_episode_submit[episode_key] = time.time()

    def start_monitoring(self):
        """Start background monitoring task."""
        if not self.enable or self.monitor_task is not None:
            return

        self.monitor_task = asyncio.create_task(self._monitor_loop())

    def stop_monitoring(self):
        """Stop background monitoring task."""
        if self.monitor_task:
            self.monitor_task.cancel()
            self.monitor_task = None

    def cleanup_episode(self, group_id: int, episode_id: int):
        """
        Clean up monitoring data for completed episode.
        Note: With dual-timestamp tracking, cleanup is mostly automatic in record_episode_start().
        This method is kept for compatibility but has minimal work to do.
        """
        if not self.enable:
            return

        # No cleanup needed - dual-timestamp approach handles cleanup automatically
        # when new episodes start via record_episode_start()
        pass

    async def _monitor_loop(self):
        """Background monitoring task that periodically detects hung envs and logs."""
        while True:
            try:
                await asyncio.sleep(self.monitor_interval)
                self.check_and_log_hung_envs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[EnvMonitor] Monitor loop error: {e}")

    def check_and_log_hung_envs(self):
        """
        Detect and log hung envs using dual-timestamp tracking.

        Detection Logic:
        - For each env with a start time recorded:
          - Check if current episode has a submit time
          - If no submit time and (now - start_time) > hung_timeout:
            → Report as hung
          - If submit time exists:
            → Env has completed, don't report (even if timestamp is old)
        """
        now = time.time()
        hung_envs_by_group = {}  # group_id -> list of hung env info

        # Iterate over all episode start records
        for episode_key, start_time in self.env_episode_start.items():
            (group_id, env_id), episode_id = episode_key

            # Check if this episode has been submitted
            submit_time = self.env_episode_submit.get(episode_key)

            if submit_time is None:
                # Env started but hasn't submitted (still processing)
                inactive_time = now - start_time

                if inactive_time > self.hung_timeout:
                    # Report as hung
                    if group_id not in hung_envs_by_group:
                        hung_envs_by_group[group_id] = []

                    hung_envs_by_group[group_id].append({
                        "env_id": env_id,
                        "episode_id": episode_id,
                        "inactive_seconds": int(inactive_time),
                    })
            # else: Episode submitted, env is waiting for next episode (normal)

        # Output logs
        if hung_envs_by_group:
            for group_id, hung_envs in hung_envs_by_group.items():
                hung_env_ids = [e["env_id"] for e in hung_envs]
                logger.warning(
                    f"[EnvMonitor] Group {group_id}: Detected {len(hung_envs)} hung envs: {hung_env_ids}"
                )
                for env_info in hung_envs[:5]:  # Only log details for first 5
                    logger.warning(
                        f"[EnvMonitor]   - env_id={env_info['env_id']}, "
                        f"episode_id={env_info['episode_id']}, "
                        f"inactive_for={env_info['inactive_seconds']}s"
                    )
                if len(hung_envs) > 5:
                    logger.warning(f"[EnvMonitor]   ... and {len(hung_envs) - 5} more")


@dataclass
class GroupData:
    group_id: int
    episode_id: int
    create_step: int
    rollouts: List[DataProto] = field(default_factory=list)
    running_rollouts: int = 0 

class GroupQueue:
    def __init__(
        self,
        group_id,
        progress_bar: tqdm,
        group_size,
        group_size_redundancy,
        max_traj_per_env,
        async_generation_ratio,
        staleness_tolerance,
        group_filter,
        env_monitor: Optional['EnvActivityMonitor'] = None,
        scheduling_policy: str = "fifo",
        fixed_step_admission: bool = True,
    ):
        self.group_id = group_id
        self.progress_bar = progress_bar

        self.group_size = group_size
        self.group_size_redundancy = group_size_redundancy
        self.max_traj_per_env = max_traj_per_env
        self.async_generation_ratio = async_generation_ratio
        self.staleness_tolerance = staleness_tolerance
        self.group_filter = group_filter
        if scheduling_policy not in ("fifo", "version_priority"):
            raise ValueError(f"Unsupported trajectory_scheduling_policy: {scheduling_policy}")
        self.scheduling_policy = scheduling_policy
        self.fixed_step_admission = fixed_step_admission
        self.admission_width = self.group_size + self.group_size_redundancy
        self.group_filter_count = 0
        self.group_filter_rollout_count = 0
        self.group_filter_actions = 0.0
        self.group_filter_actions_ge_1 = 0.0
        self.group_filter_actions_ge_2 = 0.0
        self.group_filter_actions_ge_3 = 0.0
        self.group_filter_actions_ge_4 = 0.0
        self.group_filter_inference_calls = 0.0
        self.group_filter_tool_calls = 0.0
        self.group_filter_prompt_tokens = 0.0
        self.group_filter_response_tokens = 0.0
        self.group_filter_inference_tokens = 0.0
        self.group_filter_env_seconds = 0.0
        self.env_monitor = env_monitor

        self.current_step = None
        self.next_episode_id = 0
        self.groups: Dict[int, GroupData] = {}
        self.retired_groups: Dict[int, GroupData] = {}
        self.discard_records: List[Dict[str, Any]] = []
        self.discard_metrics_cursor = 0

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

        self.quit = False

    def clear(self):
        self.current_step = None
        self.next_episode_id = 0
        self.groups.clear()
        self.retired_groups.clear()
        self.discard_records.clear()
        self.discard_metrics_cursor = 0

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

    def shutdown(self):
        self.quit = True
        self.groups.clear()
        self.progress.set()

    @staticmethod
    def _metric_by_suffix(rollout: DataProto, suffix: str, default: float = 0.0) -> float:
        if rollout is None:
            return default
        metrics = rollout.meta_info.get("metrics", {}) if rollout.meta_info else {}
        for key, value in metrics.items():
            if key.endswith(suffix):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
        return default

    def record_filtered_group(self, group: GroupData):
        for rollout in group.rollouts:
            if rollout is None:
                continue
            self.group_filter_rollout_count += 1
            self.group_filter_actions += self._metric_by_suffix(rollout, "/traj_actions_completed")
            self.group_filter_actions_ge_1 += self._metric_by_suffix(rollout, "/traj_actions_ge_1")
            self.group_filter_actions_ge_2 += self._metric_by_suffix(rollout, "/traj_actions_ge_2")
            self.group_filter_actions_ge_3 += self._metric_by_suffix(rollout, "/traj_actions_ge_3")
            self.group_filter_actions_ge_4 += self._metric_by_suffix(rollout, "/traj_actions_ge_4")
            self.group_filter_inference_calls += self._metric_by_suffix(rollout, "/traj_inference_calls")
            self.group_filter_tool_calls += self._metric_by_suffix(rollout, "/traj_tool_calls")
            self.group_filter_prompt_tokens += self._metric_by_suffix(rollout, "/traj_prompt_tokens_total")
            self.group_filter_response_tokens += self._metric_by_suffix(rollout, "/traj_response_tokens_total")
            self.group_filter_inference_tokens += self._metric_by_suffix(rollout, "/traj_inference_tokens_total")
            self.group_filter_env_seconds += self._metric_by_suffix(rollout, "/traj_env_seconds_total")

    @staticmethod
    def _first_non_tensor_value(rollout: DataProto, key: str, default=None):
        values = rollout.non_tensor_batch.get(key) if rollout.non_tensor_batch else None
        if values is None or len(values) == 0:
            return default
        value = values[0]
        return value.item() if hasattr(value, "item") else value

    def record_discarded_rollout(
        self,
        rollout: Optional[DataProto],
        group: GroupData,
        reason: str,
        observed_step: Optional[int] = None,
    ):
        if rollout is None:
            return
        metric = self._metric_by_suffix
        step = self.current_step if observed_step is None else observed_step
        step = group.create_step if step is None else step
        env_id = self._first_non_tensor_value(rollout, "env_ids", -1)
        self.discard_records.append({
            "trajectory_id": str(self._first_non_tensor_value(rollout, "traj_id", "unknown")),
            "category": "async_discard",
            "discard_reason": reason,
            "group_id": int(group.group_id),
            "episode_id": int(group.episode_id),
            "env_id": int(env_id),
            "version_start": int(metric(rollout, "/traj_version_start", group.create_step)),
            "version_end": int(metric(rollout, "/traj_version_end", group.create_step)),
            "version_age": max(0, int(step) - int(group.create_step)),
            "reset_completed": True,
            "completed": bool(metric(rollout, "/traj_completed", 0)),
            "truncated": bool(metric(rollout, "/traj_truncated", 0)),
            "actions_completed": int(metric(rollout, "/traj_actions_completed", 0)),
            "inference_calls": int(metric(rollout, "/traj_inference_calls", 0)),
            "tool_calls": int(metric(rollout, "/traj_tool_calls", 0)),
            "prompt_tokens": int(metric(rollout, "/traj_prompt_tokens_total", 0)),
            "response_tokens": int(metric(rollout, "/traj_response_tokens_total", 0)),
            "inference_tokens": int(metric(rollout, "/traj_inference_tokens_total", 0)),
            "generate_seconds": float(metric(rollout, "/traj_generate_seconds_total", 0)),
            "env_seconds": float(metric(rollout, "/traj_env_seconds_total", 0)),
        })

    def record_discarded_group(self, group: GroupData, reason: str, observed_step: Optional[int] = None):
        for rollout in group.rollouts:
            self.record_discarded_rollout(rollout, group, reason, observed_step)

    def collect_new_discard_records(self) -> List[Dict[str, Any]]:
        records = self.discard_records[self.discard_metrics_cursor:]
        self.discard_metrics_cursor = len(self.discard_records)
        return records

    def reset_filter_metrics(self):
        self.group_filter_count = 0
        self.group_filter_rollout_count = 0
        self.group_filter_actions = 0.0
        self.group_filter_actions_ge_1 = 0.0
        self.group_filter_actions_ge_2 = 0.0
        self.group_filter_actions_ge_3 = 0.0
        self.group_filter_actions_ge_4 = 0.0
        self.group_filter_inference_calls = 0.0
        self.group_filter_tool_calls = 0.0
        self.group_filter_prompt_tokens = 0.0
        self.group_filter_response_tokens = 0.0
        self.group_filter_inference_tokens = 0.0
        self.group_filter_env_seconds = 0.0

    def advance_group(self, create_step):
        assert not self.quit
        self.groups[self.next_episode_id] = GroupData(
            group_id=self.group_id, episode_id=self.next_episode_id, create_step=create_step)
        self.next_episode_id += 1

    def _ordered_groups(self):
        if self.scheduling_policy == "version_priority":
            return sorted(
                self.groups.items(),
                key=lambda item: (item[1].create_step, item[1].episode_id),
            )
        return self.groups.items()

    def outstanding_snapshot(self, observed_step: Optional[int] = None) -> Dict[str, Any]:
        step = self.current_step if observed_step is None else observed_step
        snapshot = {
            "active_groups": len(self.groups),
            "ready_trajectories": 0,
            "running_trajectories": 0,
            "reserved_trajectories": 0,
            "retired_running_trajectories": 0,
            "outstanding_trajectories": 0,
            "oldest_version_age": 0,
            "age_counts": {},
        }
        for group in self.groups.values():
            ready = min(len(group.rollouts), self.admission_width)
            running = max(0, min(group.running_rollouts, self.admission_width) - ready)
            reserved = max(0, self.admission_width - max(group.running_rollouts, ready))
            age = max(0, int(step) - group.create_step) if step is not None else 0
            snapshot["ready_trajectories"] += ready
            snapshot["running_trajectories"] += running
            snapshot["reserved_trajectories"] += reserved
            snapshot["oldest_version_age"] = max(snapshot["oldest_version_age"], age)
            snapshot["age_counts"][age] = snapshot["age_counts"].get(age, 0) + ready + running + reserved
        for group in self.retired_groups.values():
            running = max(0, group.running_rollouts - len(group.rollouts))
            age = max(0, int(step) - group.create_step) if step is not None else 0
            snapshot["retired_running_trajectories"] += running
            snapshot["oldest_version_age"] = max(snapshot["oldest_version_age"], age)
            snapshot["age_counts"][age] = snapshot["age_counts"].get(age, 0) + running
        snapshot["outstanding_trajectories"] = (
            snapshot["ready_trajectories"]
            + snapshot["running_trajectories"]
            + snapshot["reserved_trajectories"]
            + snapshot["retired_running_trajectories"]
        )
        return snapshot

    def _advance_step(self, create_step):
        if self.max_traj_per_env is None:
            return
        for _ in range(self.max_traj_per_env):
            self.advance_group(create_step)

    def advance_step(self, step, admit_step_groups: bool = True):
        if self.current_step is None and admit_step_groups:
            # first time into advance_step, generate extra groups for async training
            for _ in range(self.async_generation_ratio):
                self._advance_step(step)
        else:
            # remove outdated groups for async training
            expired_episodes = []
            for episode_id, group in self.groups.items():
                if step - group.create_step > self.staleness_tolerance:
                    expired_episodes.append(episode_id)
            for episode_id in expired_episodes:
                group = self.groups.pop(episode_id)
                self.record_discarded_group(group, "version_expired_buffered", step)
                if len(group.rollouts) < group.running_rollouts:
                    self.retired_groups[episode_id] = group
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)

        self.current_step = step
        if admit_step_groups:
            self._advance_step(step)
        self.progress.set()

    async def get_episode_id(self, env_id: Optional[int] = None) -> Optional[int]:
        """
        Get the next episode_id for an env to process.

        Args:
            env_id: Environment ID requesting work (None for backward compatibility)

        Returns:
            episode_id to process, or None if shutting down
        """
        while not self.quit:
            # Version priority is only based on policy age; no reward or training value enters here.
            for episode_id, group in self._ordered_groups():
                if group.running_rollouts < self.group_size + self.group_size_redundancy:
                    group.running_rollouts += 1

                    # Record episode start for hang detection
                    if self.env_monitor and env_id is not None:
                        self.env_monitor.record_episode_start(self.group_id, env_id, episode_id)

                    return episode_id
            if self.max_traj_per_env is None:
                while self.current_step is None:
                    self.progress.clear()
                    await self.progress.wait()
                self.advance_group(self.current_step)
                continue
            else:
                self.progress.clear()
                await self.progress.wait()
        return None

    def put(self, episode_id, start_step, rollout):
        if episode_id not in self.groups:
            group = self.retired_groups.get(episode_id)
            if group is not None:
                group.rollouts.append(rollout)
                is_version_stale = (
                    self.current_step is not None
                    and self.current_step - group.create_step > self.staleness_tolerance
                )
                reason = "version_expired_late_return" if is_version_stale else "redundancy_late_return"
                self.record_discarded_rollout(rollout, group, reason, self.current_step)
                if len(group.rollouts) >= group.running_rollouts:
                    self.retired_groups.pop(episode_id, None)
            # A retired episode has already been consumed or expired.
            return
        group = self.groups[episode_id]
        assert start_step >= group.create_step, f"{start_step=} {group.create_step=}"
        group.rollouts.append(rollout)
        if len(group.rollouts) == self.group_size:
            if all(rollout is None for rollout in group.rollouts):
                logger.info(f"GroupQueue: group {self.group_id} exit")
                self.complete.set()
            elif self.group_filter.filter(group_id=self.group_id, episode_id=episode_id, group=group.rollouts):
                logger.info(f"filter rollout group {group.group_id} episode {group.episode_id}")
                self.group_filter_count += 1
                self.record_filtered_group(group)
                self.groups.pop(episode_id)
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)
                if self.fixed_step_admission:
                    self.advance_group(create_step=self.current_step)
            else:
                self.complete.set()
                self.progress_bar.update(self.group_size)

    async def get(self) -> GroupData:
        while True:
            while not self.groups:
                self.complete.clear()
                await self.complete.wait()
            if self.scheduling_policy == "version_priority":
                episode_id = min(
                    self.groups,
                    key=lambda key: (self.groups[key].create_step, self.groups[key].episode_id),
                )
            else:
                episode_id = next(iter(self.groups)) # preserve original FIFO behavior
            group = self.groups[episode_id]
            if len(group.rollouts) >= self.group_size:
                self.groups.pop(episode_id)
                if len(group.rollouts) < group.running_rollouts:
                    self.retired_groups[episode_id] = group
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)
                return group
            self.complete.clear()
            await self.complete.wait()

@ray.remote
class GroupQueueManager:
    def __init__(self, config, env_manager_config: EnvManagerConfig, mode):
        self.mode = mode
        self.env_manager_config = env_manager_config
        self.group_size = self.env_manager_config.group_size
        self.progress_bar = tqdm(desc=f"{self.mode} rollout progress(total trajectory)", mininterval=self.env_manager_config.max_traj_per_env)
        self.pending_gets = set()
        self.rollout_complete = {}

        group_filter_cls = safe_import_class(env_manager_config.group_filter_cls)
        assert group_filter_cls
        self.group_filter = group_filter_cls(config, env_manager_config, mode)

        if self.mode == "train":
            self.async_generation_ratio = config.async_generation_ratio
            configured_tolerance = getattr(config, "trajectory_staleness_tolerance", None)
            self.staleness_tolerance = (
                int(configured_tolerance)
                if configured_tolerance is not None
                else int(self.async_generation_ratio)
            )
            self.max_traj_per_env = env_manager_config.max_traj_per_env if config.rollout_batch_size > 0 else None
        else:
            self.async_generation_ratio = 0
            self.staleness_tolerance = 0
            self.max_traj_per_env = env_manager_config.max_traj_per_env if config.val_batch_size > 0 else None

        self.scheduling_policy = (
            getattr(config, "trajectory_scheduling_policy", "fifo") if self.mode == "train" else "fifo"
        )
        self.admission_policy = (
            getattr(config, "trajectory_admission_policy", "step") if self.mode == "train" else "step"
        )
        if self.admission_policy not in ("step", "outstanding_watermark"):
            raise ValueError(f"Unsupported trajectory_admission_policy: {self.admission_policy}")
        configured_watermark = getattr(config, "max_outstanding_trajectories", None)
        if self.admission_policy == "outstanding_watermark":
            default_watermark = math.ceil(
                (1 + float(self.async_generation_ratio)) * int(config.rollout_batch_size)
            )
            self.max_outstanding_trajectories = int(configured_watermark or default_watermark)
        else:
            self.max_outstanding_trajectories = None
        self.admission_cursor = 0
        self.admitted_trajectories_total = 0
        self.admission_throttled_total = 0

        # Initialize env activity monitor first (before creating GroupQueues)
        self.group_queue: Dict[int, GroupQueue] = {}
        self.env_monitor = EnvActivityMonitor(
            config=config.env_monitor,
            group_queue_dict=self.group_queue
        )

        # Create GroupQueues with env_monitor reference
        for rank, rank_env_configs in env_manager_config.env_configs.items():
            for env_id, env_config in rank_env_configs.items():
                group_id = env_config["group_id"]
                if group_id not in self.group_queue:
                    self.group_queue[group_id] = GroupQueue(
                        group_id=group_id,
                        progress_bar=self.progress_bar,
                        group_size=env_manager_config.group_size,
                        group_size_redundancy=env_manager_config.group_size_redundancy,
                        max_traj_per_env=self.max_traj_per_env,
                        async_generation_ratio=self.async_generation_ratio,
                        staleness_tolerance=self.staleness_tolerance,
                        group_filter=self.group_filter,
                        env_monitor=self.env_monitor,
                        scheduling_policy=self.scheduling_policy,
                        fixed_step_admission=self.admission_policy == "step",
                    )

        # Start monitoring after all GroupQueues are created
        if config.env_monitor.enable:
            self.env_monitor.start_monitoring()

        # for debug
        self.total = 0
        self.waiting = 0

    def _pending_ready_snapshot(self, observed_step: Optional[int]) -> Dict[str, Any]:
        ready = 0
        oldest_age = 0
        age_counts: Dict[int, int] = {}
        for task in self.pending_gets:
            if task.cancelled() or not task.done():
                continue
            try:
                group = task.result()
            except Exception:
                continue
            count = len(group.rollouts)
            age = max(0, int(observed_step) - group.create_step) if observed_step is not None else 0
            ready += count
            oldest_age = max(oldest_age, age)
            age_counts[age] = age_counts.get(age, 0) + count
        return {
            "ready_trajectories": ready,
            "oldest_version_age": oldest_age,
            "age_counts": age_counts,
        }

    def _outstanding_snapshot(self, observed_step: Optional[int] = None) -> Dict[str, Any]:
        if observed_step is None:
            steps = [queue.current_step for queue in self.group_queue.values() if queue.current_step is not None]
            observed_step = max(steps) if steps else None
        snapshot = {
            "active_groups": 0,
            "ready_trajectories": 0,
            "running_trajectories": 0,
            "reserved_trajectories": 0,
            "retired_running_trajectories": 0,
            "outstanding_trajectories": 0,
            "oldest_version_age": 0,
            "age_counts": {},
        }
        for queue in self.group_queue.values():
            queue_snapshot = queue.outstanding_snapshot(observed_step)
            for key in (
                "active_groups",
                "ready_trajectories",
                "running_trajectories",
                "reserved_trajectories",
                "retired_running_trajectories",
                "outstanding_trajectories",
            ):
                snapshot[key] += queue_snapshot[key]
            snapshot["oldest_version_age"] = max(
                snapshot["oldest_version_age"], queue_snapshot["oldest_version_age"]
            )
            for age, count in queue_snapshot["age_counts"].items():
                snapshot["age_counts"][age] = snapshot["age_counts"].get(age, 0) + count
        pending = self._pending_ready_snapshot(observed_step)
        snapshot["ready_trajectories"] += pending["ready_trajectories"]
        snapshot["outstanding_trajectories"] += pending["ready_trajectories"]
        snapshot["oldest_version_age"] = max(
            snapshot["oldest_version_age"], pending["oldest_version_age"]
        )
        for age, count in pending["age_counts"].items():
            snapshot["age_counts"][age] = snapshot["age_counts"].get(age, 0) + count
        return snapshot

    def _refill_to_watermark(self, create_step: int):
        if self.admission_policy != "outstanding_watermark" or not self.group_queue:
            return
        queues = [queue for queue in self.group_queue.values() if not queue.quit]
        if not queues:
            return
        width = queues[0].admission_width
        if self.max_outstanding_trajectories < width:
            raise ValueError(
                "max_outstanding_trajectories must be at least one rollout group "
                f"({width})"
            )
        outstanding = self._outstanding_snapshot(create_step)["outstanding_trajectories"]
        admitted = 0
        touched = set()
        while outstanding + width <= self.max_outstanding_trajectories:
            queue = queues[self.admission_cursor % len(queues)]
            self.admission_cursor += 1
            queue.advance_group(create_step)
            touched.add(queue.group_id)
            outstanding += width
            admitted += width
        for group_id in touched:
            self.group_queue[group_id].progress.set()
        self.admitted_trajectories_total += admitted
        if admitted == 0 and outstanding + width > self.max_outstanding_trajectories:
            self.admission_throttled_total += 1

    def collect_metrics(self):
        outstanding = self._outstanding_snapshot()
        filter_metrics = {
            "scheduler/async_generation_ratio": self.async_generation_ratio,
            "scheduler/trajectory_staleness_tolerance": self.staleness_tolerance,
            "scheduler/version_priority_enabled": int(self.scheduling_policy == "version_priority"),
            "scheduler/watermark_admission_enabled": int(self.admission_policy == "outstanding_watermark"),
            "scheduler/max_outstanding_trajectories": self.max_outstanding_trajectories or 0,
            "scheduler/outstanding_trajectories": outstanding["outstanding_trajectories"],
            "scheduler/outstanding_active_groups": outstanding["active_groups"],
            "scheduler/outstanding_ready_trajectories": outstanding["ready_trajectories"],
            "scheduler/outstanding_running_trajectories": outstanding["running_trajectories"],
            "scheduler/outstanding_reserved_trajectories": outstanding["reserved_trajectories"],
            "scheduler/outstanding_retired_running_trajectories": outstanding["retired_running_trajectories"],
            "scheduler/outstanding_oldest_version_age": outstanding["oldest_version_age"],
            "scheduler/admitted_trajectories_total": self.admitted_trajectories_total,
            "scheduler/admission_throttled_total": self.admission_throttled_total,
            "scheduler/group_filter_count": 0,
            "scheduler/group_filter_rollouts": 0,
            "scheduler/group_filter_actions": 0.0,
            "scheduler/group_filter_actions_ge_1": 0.0,
            "scheduler/group_filter_actions_ge_2": 0.0,
            "scheduler/group_filter_actions_ge_3": 0.0,
            "scheduler/group_filter_actions_ge_4": 0.0,
            "scheduler/group_filter_inference_calls": 0.0,
            "scheduler/group_filter_tool_calls": 0.0,
            "scheduler/group_filter_prompt_tokens": 0.0,
            "scheduler/group_filter_response_tokens": 0.0,
            "scheduler/group_filter_inference_tokens": 0.0,
            "scheduler/group_filter_env_seconds": 0.0,
        }
        new_discard_records = []
        for group_queue in self.group_queue.values():
            new_discard_records.extend(group_queue.collect_new_discard_records())
            filter_metrics["scheduler/group_filter_count"] += group_queue.group_filter_count
            filter_metrics["scheduler/group_filter_rollouts"] += group_queue.group_filter_rollout_count
            filter_metrics["scheduler/group_filter_actions"] += group_queue.group_filter_actions
            filter_metrics["scheduler/group_filter_actions_ge_1"] += group_queue.group_filter_actions_ge_1
            filter_metrics["scheduler/group_filter_actions_ge_2"] += group_queue.group_filter_actions_ge_2
            filter_metrics["scheduler/group_filter_actions_ge_3"] += group_queue.group_filter_actions_ge_3
            filter_metrics["scheduler/group_filter_actions_ge_4"] += group_queue.group_filter_actions_ge_4
            filter_metrics["scheduler/group_filter_inference_calls"] += group_queue.group_filter_inference_calls
            filter_metrics["scheduler/group_filter_tool_calls"] += group_queue.group_filter_tool_calls
            filter_metrics["scheduler/group_filter_prompt_tokens"] += group_queue.group_filter_prompt_tokens
            filter_metrics["scheduler/group_filter_response_tokens"] += group_queue.group_filter_response_tokens
            filter_metrics["scheduler/group_filter_inference_tokens"] += group_queue.group_filter_inference_tokens
            filter_metrics["scheduler/group_filter_env_seconds"] += group_queue.group_filter_env_seconds
            group_queue.reset_filter_metrics()
        for age in range(4):
            filter_metrics[f"scheduler/outstanding_version_age_{age}"] = outstanding["age_counts"].get(age, 0)
        filter_metrics["scheduler/outstanding_version_age_ge_4"] = sum(
            count for age, count in outstanding["age_counts"].items() if age >= 4
        )
        near_expiry_age = max(0, self.staleness_tolerance - 1)
        filter_metrics["scheduler/outstanding_near_expiry_trajectories"] = sum(
            count for age, count in outstanding["age_counts"].items() if age >= near_expiry_age
        )
        discard_metrics, _ = self._aggregate_discard_records(new_discard_records, "scheduler/async_discard")
        filter_metrics.update(discard_metrics)
        return filter_metrics

    @staticmethod
    def _aggregate_discard_records(records: List[Dict[str, Any]], prefix: str):
        actions_histogram: Dict[str, int] = {}
        inference_histogram: Dict[str, int] = {}
        tool_histogram: Dict[str, int] = {}
        for record in records:
            for histogram, field in (
                (actions_histogram, "actions_completed"),
                (inference_histogram, "inference_calls"),
                (tool_histogram, "tool_calls"),
            ):
                bucket = str(int(record.get(field, 0)))
                histogram[bucket] = histogram.get(bucket, 0) + 1

        metrics = {
            f"{prefix}/trajectories": len(records),
            f"{prefix}/version_stale_trajectories": sum(
                str(record.get("discard_reason", "")).startswith("version_") for record in records
            ),
            f"{prefix}/redundancy_trajectories": sum(
                str(record.get("discard_reason", "")).startswith("redundancy_") for record in records
            ),
            f"{prefix}/actions": sum(int(record.get("actions_completed", 0)) for record in records),
            f"{prefix}/inference_calls": sum(int(record.get("inference_calls", 0)) for record in records),
            f"{prefix}/tool_calls": sum(int(record.get("tool_calls", 0)) for record in records),
            f"{prefix}/prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records),
            f"{prefix}/response_tokens": sum(int(record.get("response_tokens", 0)) for record in records),
            f"{prefix}/inference_tokens": sum(int(record.get("inference_tokens", 0)) for record in records),
            f"{prefix}/env_seconds": sum(float(record.get("env_seconds", 0.0)) for record in records),
        }
        for threshold in (1, 2, 3, 4, 8):
            metrics[f"{prefix}/trajectories_actions_ge_{threshold}"] = sum(
                int(record.get("actions_completed", 0)) >= threshold for record in records
            )
            metrics[f"{prefix}/trajectories_inference_ge_{threshold}"] = sum(
                int(record.get("inference_calls", 0)) >= threshold for record in records
            )
        for threshold in (1, 2, 4):
            metrics[f"{prefix}/trajectories_tool_calls_ge_{threshold}"] = sum(
                int(record.get("tool_calls", 0)) >= threshold for record in records
            )
        return metrics, {
            "actions_completed": actions_histogram,
            "inference_calls": inference_histogram,
            "tool_calls": tool_histogram,
        }

    @staticmethod
    def _first_non_tensor_value(rollout: DataProto, key: str, default=None):
        values = rollout.non_tensor_batch.get(key) if rollout.non_tensor_batch else None
        if values is None or len(values) == 0:
            return default
        value = values[0]
        return value.item() if hasattr(value, "item") else value

    def _completed_rollout_record(self, rollout: DataProto, group: GroupData) -> Dict[str, Any]:
        metric = GroupQueue._metric_by_suffix
        env_id = self._first_non_tensor_value(rollout, "env_ids", -1)
        return {
            "trajectory_id": str(self._first_non_tensor_value(rollout, "traj_id", "unknown")),
            "category": "completed_unconsumed",
            "discard_reason": "pipeline_shutdown",
            "group_id": int(group.group_id),
            "episode_id": int(group.episode_id),
            "env_id": int(env_id),
            "version_start": int(metric(rollout, "/traj_version_start", group.create_step)),
            "version_end": int(metric(rollout, "/traj_version_end", group.create_step)),
            "version_age": int(metric(rollout, "/traj_version_age", 0)),
            "reset_completed": True,
            "completed": True,
            "truncated": bool(metric(rollout, "/traj_truncated", 0)),
            "actions_completed": int(metric(rollout, "/traj_actions_completed", 0)),
            "inference_calls": int(metric(rollout, "/traj_inference_calls", 0)),
            "tool_calls": int(metric(rollout, "/traj_tool_calls", 0)),
            "prompt_tokens": int(metric(rollout, "/traj_prompt_tokens_total", 0)),
            "response_tokens": int(metric(rollout, "/traj_response_tokens_total", 0)),
            "inference_tokens": int(metric(rollout, "/traj_inference_tokens_total", 0)),
            "generate_seconds": float(metric(rollout, "/traj_generate_seconds_total", 0)),
            "env_seconds": float(metric(rollout, "/traj_env_seconds_total", 0)),
        }

    def collect_shutdown_waste(self, inflight_records: List[Dict[str, Any]]):
        records = []
        for group_queue in self.group_queue.values():
            for group in group_queue.groups.values():
                for rollout in group.rollouts:
                    if rollout is not None:
                        records.append(self._completed_rollout_record(rollout, group))

        records.extend(record for record in inflight_records if record is not None)
        records.sort(
            key=lambda item: (
                -int(item.get("actions_completed", 0)),
                -int(item.get("inference_calls", 0)),
                str(item.get("trajectory_id", "")),
            )
        )

        actions_histogram: Dict[str, int] = {}
        inference_histogram: Dict[str, int] = {}
        tool_histogram: Dict[str, int] = {}
        for record in records:
            for histogram, field in (
                (actions_histogram, "actions_completed"),
                (inference_histogram, "inference_calls"),
                (tool_histogram, "tool_calls"),
            ):
                bucket = str(int(record.get(field, 0)))
                histogram[bucket] = histogram.get(bucket, 0) + 1

        metrics = {
            "terminal_waste/trajectories": len(records),
            "terminal_waste/completed_unconsumed": sum(
                record.get("category") == "completed_unconsumed" for record in records
            ),
            "terminal_waste/completed_not_submitted": sum(
                record.get("category") == "completed_not_submitted" for record in records
            ),
            "terminal_waste/inflight": sum(
                record.get("category") == "inflight_at_shutdown" for record in records
            ),
            "terminal_waste/reset_only": sum(int(record.get("inference_calls", 0)) == 0 for record in records),
            "terminal_waste/actions": sum(int(record.get("actions_completed", 0)) for record in records),
            "terminal_waste/inference_calls": sum(int(record.get("inference_calls", 0)) for record in records),
            "terminal_waste/tool_calls": sum(int(record.get("tool_calls", 0)) for record in records),
            "terminal_waste/prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records),
            "terminal_waste/response_tokens": sum(int(record.get("response_tokens", 0)) for record in records),
            "terminal_waste/inference_tokens": sum(int(record.get("inference_tokens", 0)) for record in records),
            "terminal_waste/env_seconds": sum(float(record.get("env_seconds", 0.0)) for record in records),
        }
        for threshold in (1, 2, 3, 4, 8):
            metrics[f"terminal_waste/trajectories_actions_ge_{threshold}"] = sum(
                int(record.get("actions_completed", 0)) >= threshold for record in records
            )
            metrics[f"terminal_waste/trajectories_inference_ge_{threshold}"] = sum(
                int(record.get("inference_calls", 0)) >= threshold for record in records
            )
        for threshold in (1, 2, 4):
            metrics[f"terminal_waste/trajectories_tool_calls_ge_{threshold}"] = sum(
                int(record.get("tool_calls", 0)) >= threshold for record in records
            )
        async_discard_records = [
            record
            for group_queue in self.group_queue.values()
            for record in group_queue.discard_records
        ]
        async_discard_records.sort(
            key=lambda item: (-int(item.get("actions_completed", 0)), str(item.get("discard_reason", "")))
        )
        async_metrics, async_histograms = self._aggregate_discard_records(
            async_discard_records, "async_waste"
        )
        metrics.update(async_metrics)
        return {
            "metrics": metrics,
            "histograms": {
                "actions_completed": actions_histogram,
                "inference_calls": inference_histogram,
                "tool_calls": tool_histogram,
            },
            "records": records,
            "async_discard": {
                "metrics": async_metrics,
                "histograms": async_histograms,
                "records": async_discard_records,
            },
        }

    def clear(self):
        self.rollout_complete = {}
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.clear()

    def advance_step(self, step):
        fixed_step_admission = self.admission_policy == "step"
        for group_queue in self.group_queue.values():
            group_queue.advance_step(step, admit_step_groups=fixed_step_admission)
        if not fixed_step_admission:
            self._refill_to_watermark(step)

    async def get_episode_id(self, group_id, env_id=None):
        """
        Get the next episode ID for an environment.

        Args:
            group_id: Group ID
            env_id: Environment ID (for hang detection tracking)

        Returns:
            episode_id to process
        """
        assert group_id in self.group_queue
        return await self.group_queue[group_id].get_episode_id(env_id)

    def shutdown(self):
        # Stop monitoring task
        self.env_monitor.stop_monitoring()

        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.shutdown()

    def put(self, group_id, episode_id, start_step, rollout: DataProto, env_id=None):
        """
        Put rollout data to queue.

        Args:
            group_id: Group ID
            episode_id: Episode ID
            start_step: Starting step
            rollout: Rollout data (can be None for final submission)
            env_id: Environment ID (optional, for monitoring)

        Backward compatibility:
        - Old calls: put(group_id, episode_id, start_step, rollout) - env_id defaults to None
        - New calls: put(group_id, episode_id, start_step, rollout, env_id) - enables monitoring
        """
        assert group_id in self.group_queue

        # Record env activity only if env_id is provided
        if env_id is not None:
            self.env_monitor.record_activity(group_id, env_id, episode_id, rollout)

        self.waiting += 1
        self.group_queue[group_id].put(episode_id, start_step, rollout)
        self.waiting -= 1
        self.total += 1
        if self.admission_policy == "outstanding_watermark":
            current_step = self.group_queue[group_id].current_step
            if current_step is not None:
                self._refill_to_watermark(current_step)

    async def get_batch(self, batch_size, current_step) -> List[DataProto]:
        """
        return completed rollouts group by group_id with least start_step
        """
        # TODO: No need to get from every group queue, instead we can reuse 
        # a group queue as long as there are enough rollouts to avoid tail-latency?
        # But this will cause im-balance in episode_id.

        # When batch_size < 0, iterate until exit run_rollout_loop immediately.
        ret: List[DataProto] = []
        progress_bar = tqdm(desc=f"{self.mode} rollout get_batch progress(trajectory)", mininterval=self.group_size)
        while batch_size < 0 or len(ret) < batch_size:

            if len(self.rollout_complete) == len(self.group_queue):
                break

            async def wait_a_episode():
                # Only wait for new episode when there are no pending GroupQueue.get,
                # this way we can avoid starvation of some env.
                if not self.pending_gets:
                    pending = set(
                        [
                            asyncio.create_task(self.group_queue[group_id].get(), name=str(group_id))
                            for group_id in self.group_queue if str(group_id) not in self.rollout_complete
                        ]
                    )
                else:
                    pending = self.pending_gets
                    self.pending_gets = set()

                while pending and (batch_size < 0 or len(ret) < batch_size):

                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    while done and (batch_size < 0 or len(ret) < batch_size):
                        if self.scheduling_policy == "version_priority":
                            d = min(
                                done,
                                key=lambda task: (
                                    task.result().create_step,
                                    task.result().episode_id,
                                    task.result().group_id,
                                ),
                            )
                            done.remove(d)
                        else:
                            d = done.pop()
                        group = await d
                        group_rollout = group.rollouts
                        self.total -= len(group_rollout)

                        group_rollout = [rollout for rollout in group_rollout if rollout is not None]
                        if len(group_rollout) == 0:
                            self.rollout_complete[d.get_name()] = True
                            continue

                        if current_step - group.create_step > self.staleness_tolerance:
                            self.group_queue[group.group_id].record_discarded_group(
                                group, "version_stale_at_consume", current_step
                            )
                            logger.info(f"ignore rollout, current_step({current_step}) - create_step({group.create_step}) "
                                        f"exceed trajectory_staleness_tolerance({self.staleness_tolerance}) "
                                        f"{group.group_id=} {group.episode_id=}")
                            continue

                        for rollout in group_rollout[self.group_size:]:
                            self.group_queue[group.group_id].record_discarded_rollout(
                                rollout, group, "redundancy_trim", current_step
                            )
                        group_rollout = group_rollout[:self.group_size]
                        ret.extend(group_rollout)
                        progress_bar.update(len(group_rollout))

                    assert batch_size < 0 or (done and len(ret) >= batch_size) or (not done and len(ret) <= batch_size), f"{batch_size=}, {len(ret)=}, {done=}"
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)
                self._refill_to_watermark(current_step)

            await wait_a_episode()
        get_batch_return_start_time = time.time()
        for d in ret:
            d.meta_info["get_batch_return_start_time"] = get_batch_return_start_time
        return ret

class RolloutScheduler(RolloutMockMixin):
    """
    Usage:
        # User should control load_states/offload_states in pipeline by themselves.
        actor_infer
        train_rollout_scheduler = RolloutScheduler(actor_infer)
        val_rollout_scheduler = RolloutScheduler(actor_infer)
        while True:
            ray.get(train_rollout_scheduler.suspend.remote())
            model_update()
            if val:
                ray.get(val_rollout_scheduler.get_batch.remote())
            ray.get(train_rollout_scheduler.get_batch.remote())
            rollout()
        ray.get(train_rollout_scheduler.shutdown.remote())
    """
    shutdown_timeout_seconds = 30.0

    def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, collator=None):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode
        self.collator = collator

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = GroupQueueManager.options(
            name=f"GroupQueueManager-{mode}",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False),
            max_concurrency = env_num + 1 # reserve extra one for get_batch
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )

        self.router_manager = ray.remote(RouterManager).options(
                name=f"RouterManager-{self.env_manager_config.name}-{mode}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency = env_num + 1 # reserve extra one for suspend/resume
            ).remote(actor_cluster=self.infer_cluster, router_args=config.router_args, num_gpus_per_node=config.num_gpus_per_node)

        self.es_manager: Any = Cluster(
            name=self.env_manager_config.name,
            worker_cls=self.env_manager_config.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.env_manager_config,
        )

        self.rollout_task = None

        # Initialize rollout mock mechanism from mixin
        self._init_rollout_mock()

    async def initialize(self):
        await self.router_manager.initialize.remote()
        await asyncio.gather(*self.es_manager.initialize(
            pipeline_config=self.config,
            generate_scheduler=self.router_manager,
            output_queue=self.env_output_queue,
            collator=self.collator,
            mode=self.mode,
            blocking=False,
        ))

    async def shutdown(self):
        if self.rollout_task is None:
            return None

        timeout_seconds = self.shutdown_timeout_seconds
        timeout_stages = []
        worker_snapshots = []

        # Snapshot first: an environment can be blocked in an external sandbox call
        # long after the learner has stopped consuming trajectories.
        try:
            worker_snapshots = await asyncio.wait_for(
                asyncio.gather(*self.es_manager.collect_trajectory_progress(blocking=False)),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_stages.append("snapshot")
            logger.warning("timed out collecting terminal trajectory snapshots")

        try:
            await asyncio.wait_for(
                asyncio.gather(*self.es_manager.stop(blocking=False)),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_stages.append("worker_stop")
            logger.warning("timed out stopping environment workers")

        try:
            await asyncio.wait_for(self.rollout_task, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            timeout_stages.append("rollout_loop")
            logger.warning("timed out waiting for rollout loop shutdown")

        inflight_records = [
            record
            for worker_records in worker_snapshots
            for record in worker_records
        ]
        shutdown_report = await self.env_output_queue.collect_shutdown_waste.remote(inflight_records)
        await self.env_output_queue.shutdown.remote()
        try:
            await asyncio.wait_for(
                self.router_manager.shutdown.remote(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_stages.append("router")
            logger.warning("timed out stopping rollout router")

        shutdown_report["shutdown"] = {
            "timeout_seconds": timeout_seconds,
            "timeout_stages": timeout_stages,
        }
        shutdown_report["metrics"]["terminal_waste/shutdown_timeouts"] = len(timeout_stages)
        self.rollout_task = None
        return shutdown_report

    async def suspend(self):
        await self.router_manager.suspend.remote()
        await self.router_manager.abort_all.remote()
        await self.router_manager.wait_complete.remote()

    async def _run_rollout_loop(self, seed):
        await asyncio.gather(*self.es_manager.run_rollout_loop(seed, blocking=False))

    async def _get_batch(self, batch_size, global_step):
        return await self.env_output_queue.get_batch.remote(batch_size, global_step)

    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        # MOCK MODE: Load pre-recorded data, skip rollout (from mixin)
        if self._should_load_mock(global_step):
            return await self._load_mock_batch(global_step)

        with (
            attach_trace_context(data.meta_info),
            get_tracer("scheduler").start_as_current_span(
                "get_batch",
                attributes={
                    "global_step": global_step,
                    "batch_size": batch_size,
                },
            ),
        ):
            return await self._get_batch_impl(data, batch_size)

    async def _get_batch_impl(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        # start env manager
        if self.rollout_task is None:
            if self.mode == "train":
                seed = (
                    self.config.rollout_seed
                    if self.config.rollout_seed is not None
                    else random.randint(0, 1000000)
                )
            else:
                seed = self.config.seed
            self.rollout_task = asyncio.create_task(self._run_rollout_loop(seed))

        await asyncio.gather(*self.es_manager.update_step(global_step, inject_trace_context({}), blocking=False))
        await self.env_output_queue.advance_step.remote(global_step)
        await self.router_manager.resume.remote()

        get_task = asyncio.create_task(self._get_batch(batch_size, global_step))
        await asyncio.wait({get_task, self.rollout_task}, return_when=asyncio.FIRST_COMPLETED)
        if self.rollout_task.done() and self.rollout_task.exception() is not None:
            await self.rollout_task
        data_batch = await get_task
        if batch_size <= 0:
            await self.rollout_task
            self.rollout_task = None
            await self.env_output_queue.clear.remote()

        if len(data_batch) == 0:
            return None

        metrics = {}
        get_batch_return_start_time = None
        for d_item in data_batch:
            get_batch_return_start_time = d_item.meta_info.pop("get_batch_return_start_time", None)
            append_to_dict(metrics, d_item.meta_info["metrics"])
        if get_batch_return_start_time is not None:
            metrics["time/get_batch_cost_gqm"] = time.time() - get_batch_return_start_time
        metrics.update(await self.env_output_queue.collect_metrics.remote())
        batch = DataProto.concat(data_batch)
        batch.meta_info["metrics"] = metrics
        batch.meta_info["get_batch_return_start_time"] = time.time()

        # DUMP MODE: Save merged batch (from mixin)
        await self._maybe_dump_batch(batch, global_step)

        with get_tracer("scheduler").start_as_current_span("to_remote"):
            loop = asyncio.get_running_loop()
            batch = await loop.run_in_executor(None, DataProto.to_remote, batch)
        return batch

    async def shrink_sampler(self, target_gpus: List[int]) -> Dict[str, Any]:
        """Thin wrapper: Delegate shrink operation to RequestScheduler.

        v4.6 ARCHITECTURAL CHANGE: RolloutScheduler no longer performs validation,
        calculation, or state management. All worker lifecycle operations are now
        owned by RequestScheduler for atomic execution under routing_lock.

        Args:
            target_gpus: GPU IDs to free (e.g., [4,5] for actor_train or [6,7] for critic)

        Returns:
            Dict with metrics from RequestScheduler.shrink_workers():
                - "shrink_duration_ms": Total shrink operation time
                - "offload_ranks": DP ranks offloaded
                - "aborted": Number of requests aborted
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "rollout_scheduler_duration_ms": Timing from RolloutScheduler perspective

        Raises:
            RuntimeError: If shrink_workers() fails (propagated from RequestScheduler)

        Side Effects:
            - Calls RequestScheduler.shrink_workers() which performs:
              * Validation, calculation, rebalancing, state offload atomically
              * All operations protected by routing_lock

        Example:
            # Shrink before training to free actor_train GPUs
            metrics = await rollout_scheduler.shrink_sampler.remote([4, 5, 6, 7])
            # RequestScheduler handles: validation → calculation → rebalance → offload
        """
        start_time = time.time()

        # Delegate complete shrink operation to RequestScheduler (atomic under routing_lock)
        result = await self.router_manager.shrink_workers.remote(target_gpus)

        # Add timing from RolloutScheduler perspective
        result["rollout_scheduler_duration_ms"] = (time.time() - start_time) * 1000

        return result

    async def expand_sampler(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        """Thin wrapper: Delegate expand operation to RequestScheduler.

        v4.6 ARCHITECTURAL CHANGE: RolloutScheduler no longer performs validation,
        calculation, or state management. All worker lifecycle operations are now
        owned by RequestScheduler for atomic execution under routing_lock.

        Args:
            target_gpus: GPU IDs to restore (e.g., [4,5] for actor_train or [6,7] for critic)
            skip_load: If True, skip model loading (use when model_update already loaded states).
                      This only updates active_dp_ranks to restore routing state.

        Returns:
            Dict with metrics from RequestScheduler.expand_workers():
                - "expand_duration_ms": Total expand operation time
                - "load_ranks": DP ranks reloaded
                - "aborted": Number of requests aborted (proportional rebalancing)
                - "remapped": Number of src_ranks remapped (same as aborted)
                - "rollout_scheduler_duration_ms": Timing from RolloutScheduler perspective

        Raises:
            RuntimeError: If expand_workers() fails (propagated from RequestScheduler)

        Side Effects:
            - Calls RequestScheduler.expand_workers() which performs:
              * Validation, calculation, state loading (unless skip_load=True), routing updates atomically
              * All operations protected by routing_lock

        Example:
            # Expand after training to restore actor_train GPUs
            metrics = await rollout_scheduler.expand_sampler.remote([4, 5, 6, 7])
            # RequestScheduler handles: validation → calculation → load → rebalance

            # After model_update already loaded states, just restore routing:
            metrics = await rollout_scheduler.expand_sampler.remote([4, 5, 6, 7], skip_load=True)
        """
        start_time = time.time()

        # Delegate complete expand operation to RequestScheduler (atomic under routing_lock)
        result = await self.router_manager.expand_workers.remote(target_gpus, skip_load)

        # Add timing from RolloutScheduler perspective
        result["rollout_scheduler_duration_ms"] = (time.time() - start_time) * 1000

        return result
