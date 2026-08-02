import asyncio
import hashlib
import heapq
import itertools
import math
import time
import uuid
import httpx
import weakref
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote

import ray

from roll.distributed.executor.cluster import Cluster
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.configs.base_config import RouterArguments
from roll.models.model_providers import default_tokenizer_provider
from roll.utils.functionals import gather_unpadded_input_ids
from roll.utils.checkpoint_manager import download_model
from roll.utils.logging import get_logger


logger = get_logger()


def update_request_metric_totals(
    totals: MutableMapping[str, float], request_metrics: Dict[str, Any]
) -> None:
    """Accumulate one completed request, including rebuild-only token totals."""
    for name, value in request_metrics.items():
        # Per-request ratios are observations, not additive counters. The
        # aggregate ratio is derived from summed token/block counters below.
        if name in (
            "vllm/request_kv_hit_ratio",
            "vllm/request_scheduler_batch_id",
            "vllm/request_decode_tokens_per_second",
        ):
            continue
        if isinstance(value, (int, float)):
            totals[name] += float(value)
    if not request_metrics.get("router/post_update_rebuild_request", 0):
        return
    totals["router/rebuild_prompt_tokens"] += float(
        request_metrics.get("vllm/request_prompt_tokens", 0)
    )
    if "vllm/request_cached_prompt_tokens" in request_metrics:
        totals["router/rebuild_cached_prompt_tokens"] += float(
            request_metrics["vllm/request_cached_prompt_tokens"]
        )
    if "vllm/request_prefill_tokens" in request_metrics:
        totals["router/rebuild_prefill_tokens"] += float(
            request_metrics["vllm/request_prefill_tokens"]
        )


def summarize_request_metric_totals(
    totals: Dict[str, float], *, scope: str
) -> Dict[str, float]:
    """Derive comparable KV and scheduling rates from additive counters."""
    metrics = dict(totals)
    prompt_tokens = metrics.get("vllm/request_prompt_tokens", 0.0)
    cached_tokens = metrics.get("vllm/request_cached_prompt_tokens")
    if cached_tokens is not None:
        metrics[f"vllm/{scope}_kv_hit_ratio"] = (
            cached_tokens / prompt_tokens if prompt_tokens else 0.0
        )
    rebuild_prompt = metrics.get("router/rebuild_prompt_tokens", 0.0)
    rebuild_cached = metrics.get("router/rebuild_cached_prompt_tokens")
    if rebuild_cached is not None:
        metrics["router/rebuild_kv_hit_ratio"] = (
            rebuild_cached / rebuild_prompt if rebuild_prompt else 0.0
        )
    query_blocks = metrics.get(
        "vllm/engine_prefix_cache_query_blocks_delta", 0.0
    )
    hit_blocks = metrics.get(
        "vllm/engine_prefix_cache_hit_blocks_delta", 0.0
    )
    engine_cached_tokens = metrics.get(
        "vllm/engine_prefix_cache_cached_tokens_delta", 0.0
    )
    metrics.update(
        {
            "router/kv_cache_requests": metrics.get(
                "vllm/engine_prefix_cache_requests_delta", 0.0
            ),
            "router/kv_query_blocks": query_blocks,
            "router/kv_hit_blocks": hit_blocks,
            "router/kv_query_tokens": metrics.get(
                "vllm/engine_prefix_cache_query_tokens_delta", 0.0
            ),
            "router/kv_cached_tokens": engine_cached_tokens,
            "router/kv_saved_prefill_tokens": engine_cached_tokens,
            "router/kv_cacheable_reprefill_tokens": max(
                0.0,
                metrics.get("vllm/engine_prefix_cache_query_tokens_delta", 0.0)
                - engine_cached_tokens,
            ),
            "router/kv_block_hit_ratio": (
                hit_blocks / query_blocks if query_blocks else 0.0
            ),
            "router/kv_cache_resets": metrics.get(
                "vllm/engine_prefix_cache_resets_delta", 0.0
            ),
            "router/engine_priority_requests": metrics.get(
                "vllm/request_priority_enabled", 0.0
            ),
            "router/engine_priority_queued_requests": metrics.get(
                "vllm/request_priority_queued", 0.0
            ),
            "router/engine_priority_queue_seconds": metrics.get(
                "vllm/request_priority_queue_seconds", 0.0
            ),
        }
    )
    engine_priority_requests = metrics["router/engine_priority_requests"]
    metrics["router/engine_priority_queued_ratio"] = (
        metrics["router/engine_priority_queued_requests"]
        / engine_priority_requests
        if engine_priority_requests
        else 0.0
    )
    scheduling_decisions = metrics.get("router/scheduling_decisions", 0.0)
    if scheduling_decisions:
        for total_name, derived_name in (
            ("router/scheduling_wait_seconds", "router/scheduling_wait_seconds_mean"),
            (
                "router/scheduling_decision_seconds",
                "router/scheduling_decision_seconds_mean",
            ),
            ("router/selected_worker_pressure", "router/selected_worker_pressure_mean"),
            ("router/affinity_selected", "router/affinity_selected_ratio"),
            ("router/load_override", "router/load_override_ratio"),
            ("router/least_loaded_selected", "router/least_loaded_selected_ratio"),
            (
                "router/working_set_prefix_selected",
                "router/working_set_prefix_selected_ratio",
            ),
            (
                "router/rebuild_candidate_request",
                "router/rebuild_candidate_request_ratio",
            ),
            (
                "router/planned_priority_candidate_request",
                "router/planned_priority_candidate_request_ratio",
            ),
            ("router/priority_queued_requests", "router/priority_queued_ratio"),
            (
                "router/priority_coalesced_requests",
                "router/priority_coalesced_ratio",
            ),
            (
                "router/priority_reordered_requests",
                "router/priority_reordered_ratio",
            ),
            ("router/priority_queue_depth", "router/priority_queue_depth_mean"),
            (
                "router/engine_kv_feedback_requests",
                "router/engine_kv_feedback_ratio",
            ),
            (
                "router/completion_eta_selected",
                "router/completion_eta_selected_ratio",
            ),
            (
                "router/predicted_completion_eta_seconds",
                "router/predicted_completion_eta_seconds_mean",
            ),
            (
                "router/predicted_queue_eta_seconds",
                "router/predicted_queue_eta_seconds_mean",
            ),
            (
                "router/predicted_prefill_tokens",
                "router/predicted_prefill_tokens_mean",
            ),
            (
                "router/request_service_seconds",
                "router/request_service_seconds_mean",
            ),
            (
                "router/actual_completion_seconds",
                "router/actual_completion_seconds_mean",
            ),
            (
                "router/completion_eta_absolute_error_seconds",
                "router/completion_eta_absolute_error_seconds_mean",
            ),
            (
                "router/planned_completion_probability",
                "router/planned_completion_probability_mean",
            ),
            (
                "router/planned_completion_eta_seconds",
                "router/planned_completion_eta_seconds_mean",
            ),
            (
                "router/actual_prefill_tokens",
                "router/actual_prefill_tokens_mean",
            ),
            (
                "router/prefill_prediction_absolute_error_tokens",
                "router/prefill_prediction_absolute_error_tokens_mean",
            ),
        ):
            metrics[derived_name] = (
                metrics.get(total_name, 0.0) / scheduling_decisions
            )
    rebuild_requests = metrics.get("router/post_update_rebuild_request", 0.0)
    if rebuild_requests:
        metrics["router/post_update_rebuild_wave_size_mean"] = (
            metrics.get("router/post_update_rebuild_wave_size", 0.0)
            / rebuild_requests
        )
        metrics["router/post_update_rebuild_coalesced_ratio"] = (
            metrics.get("router/post_update_rebuild_coalesced", 0.0)
            / rebuild_requests
        )
        metrics["router/rebuild_seed_ratio"] = (
            metrics.get("router/rebuild_seed_request", 0.0)
            / rebuild_requests
        )
        metrics["router/rebuild_follower_ratio"] = (
            metrics.get("router/rebuild_follower_request", 0.0)
            / rebuild_requests
        )
    rebuild_followers = metrics.get("router/rebuild_follower_request", 0.0)
    if rebuild_followers:
        metrics["router/rebuild_follower_wait_seconds_mean"] = (
            metrics.get("router/rebuild_follower_wait_seconds", 0.0)
            / rebuild_followers
        )
    batch_reported = metrics.get("vllm/request_scheduler_batch_reported", 0.0)
    if batch_reported:
        metrics["vllm/request_scheduler_batch_size_mean"] = (
            metrics.get("vllm/request_scheduler_batch_size", 0.0)
            / batch_reported
        )
    return metrics


@dataclass(frozen=True)
class TrajectoryRuntimeState:
    trajectory_id: str
    policy_version: int
    current_version: int
    version_age: int
    actions_completed: int
    max_actions: int
    group_id: int = -1
    episode_id: int = -1
    env_id: int = -1

    @property
    def remaining_actions(self) -> int:
        return max(0, self.max_actions - self.actions_completed)

    @property
    def priority_key(self):
        # Lexicographic, system-only ordering: deadline, invested work, then
        # distance to the hard action limit. FIFO is appended by the router.
        return (
            self.policy_version,
            int(self.actions_completed == 0),
            self.remaining_actions,
        )

    @property
    def group_key(self) -> str:
        return f"{self.group_id}:{self.episode_id}"

    @classmethod
    def from_priority(cls, priority, fallback_id) -> "TrajectoryRuntimeState":
        if isinstance(priority, dict):
            policy_version = int(priority.get("policy_version", 0))
            current_version = int(priority.get("current_version", policy_version))
            actions_completed = int(priority.get("actions_completed", 0))
            max_actions = int(priority.get("max_actions", actions_completed))
            return cls(
                trajectory_id=str(priority.get("trajectory_id", fallback_id)),
                policy_version=policy_version,
                current_version=current_version,
                version_age=max(
                    0,
                    int(priority.get("version_age", current_version - policy_version)),
                ),
                actions_completed=max(0, actions_completed),
                max_actions=max(actions_completed, max_actions),
                group_id=int(priority.get("group_id", -1)),
                episode_id=int(priority.get("episode_id", -1)),
                env_id=int(priority.get("env_id", -1)),
            )
        policy_version = int(priority or 0)
        return cls(
            trajectory_id=str(fallback_id),
            policy_version=policy_version,
            current_version=policy_version,
            version_age=0,
            actions_completed=0,
            max_actions=0,
        )


def build_runtime_priority_key(
    runtime_state: TrajectoryRuntimeState,
    planned_candidate_ranks: Dict[str, int],
):
    """Combine the boundary plan with request-local version/progress state."""
    if not planned_candidate_ranks:
        return runtime_state.priority_key
    candidate_rank = planned_candidate_ranks.get(runtime_state.group_key)
    return (
        int(candidate_rank is None),
        (
            len(planned_candidate_ranks)
            if candidate_rank is None
            else int(candidate_rank)
        ),
        *runtime_state.priority_key,
    )


def build_engine_request_priority(
    runtime_state: TrajectoryRuntimeState,
    planned_candidate_ranks: Dict[str, int],
) -> int:
    """Encode the router's lexicographic order as a vLLM integer priority."""
    rank_bits = 20
    version_bits = 24
    remaining_bits = 16
    rank_limit = (1 << rank_bits) - 1
    version_limit = (1 << version_bits) - 1
    remaining_limit = (1 << remaining_bits) - 1

    if planned_candidate_ranks:
        candidate_rank = planned_candidate_ranks.get(runtime_state.group_key)
        outside_plan = int(candidate_rank is None)
        candidate_rank = rank_limit if candidate_rank is None else int(candidate_rank)
    else:
        outside_plan = 0
        candidate_rank = 0

    candidate_rank = min(rank_limit, max(0, candidate_rank))
    policy_version = min(version_limit, max(0, int(runtime_state.policy_version)))
    remaining_actions = min(
        remaining_limit, max(0, int(runtime_state.remaining_actions))
    )
    unstarted = int(runtime_state.actions_completed == 0)

    priority = outside_plan
    priority = (priority << rank_bits) | candidate_rank
    priority = (priority << version_bits) | policy_version
    priority = (priority << 1) | unstarted
    priority = (priority << remaining_bits) | remaining_actions
    return priority


def build_router_progress_snapshot(
    runtime_state: TrajectoryRuntimeState, prompt_tokens: int
) -> Dict[str, Any]:
    """Expose scheduling progress already carried by an inference request."""
    return {
        "trajectory_id": runtime_state.trajectory_id,
        "group_id": runtime_state.group_id,
        "episode_id": runtime_state.episode_id,
        "env_id": runtime_state.env_id,
        "version_start": runtime_state.policy_version,
        "version_end": runtime_state.current_version,
        "version_age": runtime_state.version_age,
        "reset_completed": True,
        "completed": False,
        "truncated": False,
        "actions_completed": runtime_state.actions_completed,
        "inference_calls": runtime_state.actions_completed,
        "tool_calls": 0,
        "prompt_tokens": 0,
        "response_tokens": 0,
        "inference_tokens": 0,
        "latest_prompt_tokens": max(0, int(prompt_tokens)),
        "latest_response_tokens": 0,
        "current_context_tokens": max(0, int(prompt_tokens)),
        "max_actions": runtime_state.max_actions,
        "remaining_actions": runtime_state.remaining_actions,
        "runtime_phase": "router_last_request",
        "generate_seconds": 0.0,
        "env_seconds": 0.0,
        "trajectory_wall_seconds": 0.0,
        "progress_source": "router",
    }


def build_boundary_recovery_record(
    runtime_state: TrajectoryRuntimeState,
    *,
    cache_epoch: int,
    boundary_version: Any,
    worker_rank: int,
    route_reason: str,
    prompt_tokens: int,
    response_metrics: Dict[str, Any],
    boundary_resumed_at: Optional[float] = None,
    request_dispatched_at: Optional[float] = None,
    request_completed_at: Optional[float] = None,
    prefix_fingerprints: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Describe a survivor's first request in a new cache epoch."""
    cached_tokens = response_metrics.get("vllm/request_cached_prompt_tokens")
    prefill_tokens = response_metrics.get("vllm/request_prefill_tokens")
    resolved_boundary_version = int(
        runtime_state.current_version
        if boundary_version is None else boundary_version
    )
    record = {
        "trajectory_id": runtime_state.trajectory_id,
        "group_id": runtime_state.group_id,
        "episode_id": runtime_state.episode_id,
        "policy_version": runtime_state.policy_version,
        "boundary_version": resolved_boundary_version,
        "version_age": max(
            0,
            resolved_boundary_version - runtime_state.policy_version,
        ),
        "actions_completed": runtime_state.actions_completed,
        "remaining_actions": runtime_state.remaining_actions,
        "cache_epoch": int(cache_epoch),
        "worker_rank": int(worker_rank),
        "route_reason": str(route_reason),
        "logical_prompt_tokens": max(0, int(prompt_tokens)),
        "reported_cached_prompt_tokens": (
            None if cached_tokens is None else max(0, int(cached_tokens))
        ),
        "reported_prefill_tokens": (
            None if prefill_tokens is None else max(0, int(prefill_tokens))
        ),
        "engine_scheduler_batch_id": response_metrics.get(
            "vllm/request_scheduler_batch_id"
        ),
        "engine_scheduler_batch_size": response_metrics.get(
            "vllm/request_scheduler_batch_size"
        ),
        "request_queue_seconds": float(
            response_metrics.get("vllm/request_queue_seconds", 0.0)
        ),
        "request_ttft_seconds": float(
            response_metrics.get("vllm/request_ttft_seconds", 0.0)
        ),
        "request_prefill_seconds": float(
            response_metrics.get("vllm/request_prefill_seconds", 0.0)
        ),
        "request_decode_seconds": float(
            response_metrics.get("vllm/request_decode_seconds", 0.0)
        ),
        "request_inference_seconds": float(
            response_metrics.get("vllm/request_inference_seconds", 0.0)
        ),
        "request_latency_seconds": float(
            response_metrics.get("vllm/request_latency_seconds", 0.0)
        ),
        "request_model_forward_seconds": float(
            response_metrics.get("vllm/request_model_forward_seconds", 0.0)
        ),
        "request_model_execute_seconds": float(
            response_metrics.get("vllm/request_model_execute_seconds", 0.0)
        ),
        "request_engine_step_seconds_attributed": float(
            response_metrics.get(
                "vllm/request_engine_step_seconds_attributed", 0.0
            )
        ),
        "request_prefill_engine_step_seconds_attributed": float(
            response_metrics.get(
                "vllm/request_prefill_engine_step_seconds_attributed", 0.0
            )
        ),
        "request_decode_engine_step_seconds_attributed": float(
            response_metrics.get(
                "vllm/request_decode_engine_step_seconds_attributed", 0.0
            )
        ),
        "request_output_tokens": int(
            response_metrics.get("vllm/request_output_tokens", 0)
        ),
        "request_decode_tokens": int(
            response_metrics.get("vllm/request_decode_tokens", 0)
        ),
        "prefix_fingerprints": list(prefix_fingerprints or []),
    }
    if boundary_resumed_at is not None:
        if request_dispatched_at is not None:
            record["dispatch_after_boundary_seconds"] = max(
                0.0, float(request_dispatched_at) - float(boundary_resumed_at)
            )
            record["first_token_after_boundary_seconds"] = (
                record["dispatch_after_boundary_seconds"]
                + record["request_ttft_seconds"]
            )
        if request_completed_at is not None:
            record["finish_after_boundary_seconds"] = max(
                0.0, float(request_completed_at) - float(boundary_resumed_at)
            )
            if record["request_latency_seconds"] > 0:
                record["first_token_after_boundary_seconds"] = max(
                    record.get("dispatch_after_boundary_seconds", 0.0),
                    record["finish_after_boundary_seconds"]
                    - max(
                        0.0,
                        record["request_latency_seconds"]
                        - record["request_ttft_seconds"],
                    ),
                )
    record["logical_reprefill_exposure_tokens"] = (
        record["reported_prefill_tokens"]
        if record["reported_prefill_tokens"] is not None
        else max(
            0,
            record["logical_prompt_tokens"]
            - (record["reported_cached_prompt_tokens"] or 0),
        )
    )
    record["reprefill_measurement"] = (
        "engine_reported_prefill"
        if record["reported_prefill_tokens"] is not None
        else (
            "request_cached_tokens"
            if record["reported_cached_prompt_tokens"] is not None
            else "logical_prompt_upper_bound"
        )
    )
    return record


def build_prefix_fingerprints(
    prompt_tokens: Sequence[int],
    depths: Sequence[int] = (128, 256, 512, 1024, 2048, 4096, 8192),
    block_size: int = 16,
) -> List[Dict[str, Any]]:
    """Build compact cumulative hashes at block-aligned prompt depths."""
    if block_size <= 0 or not prompt_tokens:
        return []
    prompt_length = len(prompt_tokens)
    targets = {
        min(prompt_length, max(0, int(depth))) // block_size * block_size
        for depth in depths
    }
    targets.add(prompt_length // block_size * block_size)
    targets.discard(0)
    ordered_targets = sorted(targets)
    if not ordered_targets:
        return []

    digest = hashlib.blake2b(digest_size=12)
    fingerprints = []
    target_index = 0
    for index, token in enumerate(prompt_tokens[: ordered_targets[-1]], start=1):
        digest.update(int(token).to_bytes(8, "little", signed=False))
        if index == ordered_targets[target_index]:
            fingerprints.append(
                {
                    "prefix_tokens": index,
                    "fingerprint": digest.hexdigest(),
                }
            )
            target_index += 1
            if target_index == len(ordered_targets):
                break
    return fingerprints


PrefixDirectoryKey = Tuple[int, str]


def build_prefix_directory_keys(
    prompt_tokens: Sequence[int],
    prefix_limit: int,
    block_size: int = 16,
) -> Tuple[PrefixDirectoryKey, ...]:
    """Return block-aligned prefix keys from shallowest to deepest."""
    limit = max(0, min(len(prompt_tokens), int(prefix_limit)))
    if limit < block_size:
        return ()
    depths = tuple(
        depth
        for depth in (128, 256, 512, 1024, 2048, 4096, 8192)
        if depth <= limit
    )
    if not depths or depths[-1] != limit:
        depths = (*depths, limit)
    return tuple(
        (int(item["prefix_tokens"]), str(item["fingerprint"]))
        for item in build_prefix_fingerprints(
            prompt_tokens[:limit], depths=depths, block_size=block_size
        )
    )


def build_refresh_request_record(
    runtime_state: TrajectoryRuntimeState,
    *,
    cache_epoch: int,
    boundary_version: Any,
    worker_rank: int,
    route_reason: str,
    prompt_tokens: Sequence[int],
    response_metrics: Dict[str, Any],
    request_dispatched_at: float,
    request_completed_at: float,
    boundary_resumed_at: Optional[float],
    first_epoch_request: bool,
) -> Dict[str, Any]:
    """Capture one policy request for refresh-aligned throughput analysis."""
    boundary = (
        runtime_state.current_version
        if boundary_version is None
        else int(boundary_version)
    )
    record = {
        "trajectory_id": runtime_state.trajectory_id,
        "policy_version": runtime_state.policy_version,
        "boundary_version": boundary,
        "version_age": max(0, boundary - runtime_state.policy_version),
        "cache_epoch": int(cache_epoch),
        "worker_rank": int(worker_rank),
        "route_reason": str(route_reason),
        "first_epoch_request": bool(first_epoch_request),
        "survivor_request": runtime_state.policy_version < boundary,
        "prompt_tokens": len(prompt_tokens),
        "cached_prompt_tokens": int(
            response_metrics.get("vllm/request_cached_prompt_tokens", 0)
        ),
        "prefill_tokens": int(
            response_metrics.get(
                "vllm/request_prefill_tokens", len(prompt_tokens)
            )
        ),
        "decode_tokens": int(
            response_metrics.get("vllm/request_decode_tokens", 0)
        ),
        "output_tokens": int(
            response_metrics.get("vllm/request_output_tokens", 0)
        ),
        "request_queue_seconds": float(
            response_metrics.get("vllm/request_queue_seconds", 0.0)
        ),
        "request_ttft_seconds": float(
            response_metrics.get("vllm/request_ttft_seconds", 0.0)
        ),
        "request_prefill_seconds": float(
            response_metrics.get("vllm/request_prefill_seconds", 0.0)
        ),
        "request_decode_seconds": float(
            response_metrics.get("vllm/request_decode_seconds", 0.0)
        ),
        "request_engine_step_seconds_attributed": float(
            response_metrics.get(
                "vllm/request_engine_step_seconds_attributed", 0.0
            )
        ),
        "request_prefill_engine_step_seconds_attributed": float(
            response_metrics.get(
                "vllm/request_prefill_engine_step_seconds_attributed", 0.0
            )
        ),
        "request_decode_engine_step_seconds_attributed": float(
            response_metrics.get(
                "vllm/request_decode_engine_step_seconds_attributed", 0.0
            )
        ),
        "engine_scheduler_batch_id": response_metrics.get(
            "vllm/request_scheduler_batch_id"
        ),
        "engine_scheduler_batch_size": response_metrics.get(
            "vllm/request_scheduler_batch_size"
        ),
        "request_dispatched_at_monotonic": float(request_dispatched_at),
        "request_completed_at_monotonic": float(request_completed_at),
        "prefix_fingerprints": (
            build_prefix_fingerprints(prompt_tokens)
            if first_epoch_request
            else []
        ),
    }
    if boundary_resumed_at is not None:
        record["dispatch_after_boundary_seconds"] = max(
            0.0, request_dispatched_at - boundary_resumed_at
        )
        record["finish_after_boundary_seconds"] = max(
            0.0, request_completed_at - boundary_resumed_at
        )
        record["first_token_after_boundary_seconds"] = (
            record["dispatch_after_boundary_seconds"]
            + record["request_ttft_seconds"]
        )
    return record


def is_post_boundary_request(
    runtime_state: TrajectoryRuntimeState,
    boundary_version: Any,
) -> bool:
    """Return whether this trajectory started before the active cache epoch."""
    resolved_boundary_version = int(
        runtime_state.current_version
        if boundary_version is None else boundary_version
    )
    return resolved_boundary_version > runtime_state.policy_version


def select_soft_locality_worker(
    affinity_rank,
    active_dp_ranks,
    worker_pressure,
    cache_valid: bool,
    load_slack: int,
):
    """Choose affinity unless its queue pressure exceeds the least-loaded worker."""
    candidates = list(active_dp_ranks)
    if not candidates:
        raise RuntimeError("No active DP ranks")
    least_loaded = min(candidates, key=lambda rank: (worker_pressure.get(rank, 0), rank))
    if (
        cache_valid
        and affinity_rank in active_dp_ranks
        and worker_pressure.get(affinity_rank, 0)
        <= worker_pressure.get(least_loaded, 0) + max(0, load_slack)
    ):
        return affinity_rank, "affinity"
    if cache_valid and affinity_rank in active_dp_ranks:
        return least_loaded, "load_override"
    return least_loaded, "least_loaded"


def common_prefix_tokens(left, right, limit: int) -> int:
    """Count equal leading tokens, bounded to keep rebuild routing inexpensive."""
    count = 0
    for left_token, right_token in zip(left, right):
        if count >= limit or int(left_token) != int(right_token):
            break
        count += 1
    return count


def select_rebuild_worker(
    prompt_tokens,
    worker_prompts,
    active_dp_ranks,
    assigned_counts,
    prefix_limit: int,
):
    """Choose the worker whose current rebuild wave is least prefix-similar."""
    candidates = list(active_dp_ranks)
    if not candidates:
        raise RuntimeError("No active DP ranks")

    def score(dp_rank):
        similarities = [
            common_prefix_tokens(prompt_tokens, previous, prefix_limit)
            for previous in worker_prompts.get(dp_rank, [])
        ]
        max_similarity = max(similarities, default=0)
        return max_similarity, assigned_counts.get(dp_rank, 0), dp_rank

    selected = min(candidates, key=score)
    similarities = [
        common_prefix_tokens(prompt_tokens, previous, prefix_limit)
        for previous in worker_prompts.get(selected, [])
    ]
    return selected, max(similarities, default=0)


def select_prefix_locality_worker(
    prompt_tokens,
    worker_prompts,
    active_dp_ranks,
    worker_pressure,
    load_slack: int,
    prefix_limit: int,
):
    """Prefer the current-version working set unless queue pressure is excessive."""
    candidates = list(active_dp_ranks)
    if not candidates:
        raise RuntimeError("No active DP ranks")
    least_loaded = min(candidates, key=lambda rank: (worker_pressure.get(rank, 0), rank))

    def cached_tokens(dp_rank):
        return max(
            (
                common_prefix_tokens(prompt_tokens, previous, prefix_limit)
                for previous in worker_prompts.get(dp_rank, [])
            ),
            default=0,
        )

    best_rank = max(
        candidates,
        key=lambda rank: (cached_tokens(rank), -worker_pressure.get(rank, 0), -rank),
    )
    best_cached = cached_tokens(best_rank)
    if best_cached <= 0:
        return least_loaded, "least_loaded", 0
    if (
        worker_pressure.get(best_rank, 0)
        <= worker_pressure.get(least_loaded, 0) + max(0, load_slack)
    ):
        return best_rank, "prefix_locality", best_cached
    return least_loaded, "prefix_load_override", 0


def select_prefix_directory_worker(
    prefix_keys: Sequence[PrefixDirectoryKey],
    ready_workers: Dict[PrefixDirectoryKey, Set[int]],
    active_dp_ranks,
    worker_pressure,
    load_slack: int,
):
    """Use the deepest ready prefix owner without scanning stored prompts."""
    candidates = list(active_dp_ranks)
    if not candidates:
        raise RuntimeError("No active DP ranks")
    least_loaded = min(
        candidates, key=lambda rank: (worker_pressure.get(rank, 0), rank)
    )
    for prefix_tokens, fingerprint in reversed(tuple(prefix_keys)):
        owners = ready_workers.get((prefix_tokens, fingerprint), set())
        owners = [rank for rank in owners if rank in active_dp_ranks]
        if not owners:
            continue
        owner = min(
            owners, key=lambda rank: (worker_pressure.get(rank, 0), rank)
        )
        if (
            worker_pressure.get(owner, 0)
            <= worker_pressure.get(least_loaded, 0) + max(0, load_slack)
        ):
            return owner, "prefix_directory", int(prefix_tokens)
        return least_loaded, "prefix_load_override", 0
    return least_loaded, "least_loaded", 0


def select_completion_eta_worker(
    prompt_tokens,
    worker_prompts,
    active_dp_ranks,
    worker_pressure,
    worker_service_seconds,
    token_seconds: float,
    prefix_limit: int,
    *,
    affinity_rank=None,
    affinity_cached_tokens: int = 0,
    affinity_cache_valid: bool = False,
    locality_slack_seconds: float = 0.0,
):
    """Choose the worker with the lowest estimated request completion time."""
    candidates = list(active_dp_ranks)
    if not candidates:
        raise RuntimeError("No active DP ranks")
    observed_services = [
        float(worker_service_seconds.get(rank, 0.0))
        for rank in candidates
        if float(worker_service_seconds.get(rank, 0.0)) > 0
    ]
    fallback_service = (
        sum(observed_services) / len(observed_services)
        if observed_services
        else 1.0
    )

    def cached_tokens(dp_rank):
        cached = max(
            (
                common_prefix_tokens(prompt_tokens, previous, prefix_limit)
                for previous in worker_prompts.get(dp_rank, [])
            ),
            default=0,
        )
        if affinity_cache_valid and dp_rank == affinity_rank:
            cached = max(cached, max(0, int(affinity_cached_tokens)))
        return min(len(prompt_tokens), cached)

    estimates = {}
    for dp_rank in candidates:
        service_seconds = max(
            1e-6,
            float(worker_service_seconds.get(dp_rank, 0.0))
            or fallback_service,
        )
        pressure = max(0, int(worker_pressure.get(dp_rank, 0)))
        cached = cached_tokens(dp_rank)
        prefill_tokens = max(0, len(prompt_tokens) - cached)
        queue_eta = pressure * service_seconds
        request_eta = max(
            service_seconds,
            prefill_tokens * max(0.0, float(token_seconds)),
        )
        estimates[dp_rank] = {
            "eta_seconds": queue_eta + request_eta,
            "queue_eta_seconds": queue_eta,
            "prefill_tokens": prefill_tokens,
            "cached_tokens": cached,
        }

    selected = min(
        candidates,
        key=lambda rank: (
            estimates[rank]["eta_seconds"],
            -estimates[rank]["cached_tokens"],
            rank,
        ),
    )
    if affinity_cache_valid and affinity_rank in estimates:
        if (
            estimates[affinity_rank]["eta_seconds"]
            <= estimates[selected]["eta_seconds"]
            + max(0.0, float(locality_slack_seconds))
        ):
            selected = affinity_rank

    selected_estimate = estimates[selected]
    if affinity_cache_valid and selected == affinity_rank:
        reason = "completion_eta_affinity"
    elif selected_estimate["cached_tokens"] > 0:
        reason = "completion_eta_prefix"
    else:
        reason = "completion_eta_load"
    return selected, reason, selected_estimate


def completion_eta_observation(
    predicted_seconds: float,
    scheduling_wait_seconds: float,
    request_service_seconds: float,
):
    """Return observed end-to-end request latency and its ETA error."""
    actual_seconds = max(0.0, float(scheduling_wait_seconds)) + max(
        0.0, float(request_service_seconds)
    )
    predicted = max(0.0, float(predicted_seconds))
    absolute_error = (
        abs(actual_seconds - predicted) if predicted > 0 else 0.0
    )
    return actual_seconds, absolute_error


def _create_sampling_params_for_sglang(gen_kwargs: dict):
    return dict(
        max_new_tokens=gen_kwargs["max_new_tokens"],
        min_new_tokens=gen_kwargs.get("min_new_tokens", 0),
        ignore_eos=gen_kwargs.get("ignore_eos", False),
        temperature=gen_kwargs["temperature"],
        top_p=gen_kwargs["top_p"],
        top_k=gen_kwargs["top_k"],
        stop_token_ids=gen_kwargs["eos_token_id"],
        repetition_penalty=gen_kwargs["repetition_penalty"],
        n=gen_kwargs["num_return_sequences"],
        stop=gen_kwargs["stop_strings"],
        no_stop_trim=gen_kwargs.get("include_stop_str_in_output", True),
    )


def is_report_data_finished(data: DataProto) -> bool:
    finish_reasons = data.meta_info.get("finish_reasons", [])
    assert isinstance(finish_reasons, list), f"{finish_reasons}"
    assert all(isinstance(finish_reason, str) for finish_reason in finish_reasons), f"{finish_reasons}"
    return not any(finish_reason == "abort" for finish_reason in finish_reasons)

def raise_for_status(response: httpx.Response):
    if not response.is_success:
        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(str(e))

async def wait_sglang_router_ready(router_process, url):
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        for attempt in range(60):
            await asyncio.sleep(1)
            try:
                response = await client.get(url)
                if response.status_code in [200, 404]:
                    break
                else:
                    logger.info(f"Waiting for sglang router {url} to ready ({attempt=}) (status={response.status_code})...")
                    raise_for_status(response)
                assert router_process.is_alive()
            except httpx.ConnectError:
                logger.info(f"Waiting for sglang router {url} to start ({attempt=})...")

async def wait_sglang_router_workflow(router_url, expected):
    expected = set(expected)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        while True:
            await asyncio.sleep(3)
            response = await client.get(f"{router_url}/workers")
            raise_for_status(response)
            response = response.json()
            if {worker["url"] for worker in response["workers"]} == expected:
                break
            logger.info(f"Waiting for sglang router worker workflow {router_url} ready, "
                        f"{expected=}, current count={response['total']}, workers={response['workers']} ...")

class RouterManager:
    def __init__(self, actor_cluster: Cluster, router_args: RouterArguments, num_gpus_per_node: int):
        self.actor_cluster = actor_cluster
        self.workers = actor_cluster.workers

        self.strategy_name = actor_cluster.worker_config.strategy_args.strategy_name
        self.model_path = download_model(actor_cluster.worker_config.model_args.model_name_or_path)
        self.tokenizer = default_tokenizer_provider(model_args=actor_cluster.worker_config.model_args)

        router_name = router_args.router_name
        if router_name == "PromptAffinityRouter":
            self.router_cls = PromptAffinityRouter
        elif router_name == "EnvAffinityRouter":
            self.router_cls = EnvAffinityRouter
        else:
            self.router_cls = SglangRouter
        assert self.router_cls is not SglangRouter or self.strategy_name == "sglang"
        assert (self.router_cls is SglangRouter) == (actor_cluster.worker_config.strategy_args.strategy_config.get("grpc_mode", None) is not None) # xnor
        logger.info(f"RouterManager use router {self.router_cls.__name__}")
        self.router: Router = self.router_cls(router_manager=self, workers=self.workers, model_path=self.model_path, router_args=router_args)

        self.inflight_requests = set()
        self.need_suspend = False
        self.need_shutdown = False
        self.suspend_notifier = asyncio.Event()
        self.empty_notifier = asyncio.Event()

        self.partial_gpu_manager = PartialGPUManager(actor_cluster=actor_cluster, router=self.router, num_gpus_per_node=num_gpus_per_node)
        self.request_metric_totals = defaultdict(float)
        self.request_metric_lifetime_totals = defaultdict(float)

    async def initialize(self):
        await self.router.initialize()

    def router_meta(self):
        return {
            "strategy_name": self.strategy_name,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "sglang_router": self.router_cls is SglangRouter,
            "router_ip": self.router.router_ip if self.router_cls is SglangRouter else None,
            "router_port": self.router.router_port if self.router_cls is SglangRouter else None,
            "worker_urls": self.router.worker_urls if self.router_cls is SglangRouter else None,
        }

    @classmethod
    def create_client_sync(cls, self) -> "RouterClient":
        if isinstance(self, ray.actor.ActorHandle):
            meta = ray.get(self.router_meta.remote())
            proxy_cls = RayProxy
        elif isinstance(self, cls):
            meta = self.router_meta()
            proxy_cls = InprocProxy
        else:
            raise ValueError(f"self {self} is not a ray actor or RouterManager")

        proxy = proxy_cls(self)
        if meta["sglang_router"]:
            proxy = SglangProxy(proxy, meta)
        return RouterClient(proxy, meta)

    @classmethod
    async def create_client(cls, self) -> "RouterClient":
        """
        self may be a ray actor or normal class.
        """
        if isinstance(self, ray.actor.ActorHandle):
            meta = await self.router_meta.remote()
            proxy_cls = RayProxy
        elif isinstance(self, cls):
            meta = self.router_meta()
            proxy_cls = InprocProxy
        else:
            raise ValueError(f"self {self} is not a ray actor or RouterManager")

        proxy = proxy_cls(self)
        if meta["sglang_router"]:
            proxy = SglangProxy(proxy, meta)
        return RouterClient(proxy, meta)

    async def generate_request(self, payload, request_id, uid, priority=None):
        response = await self.router.generate_request(
            payload=payload, request_id=request_id, uid=uid, priority=priority
        )
        request_metrics = response.get("metrics", {})
        update_request_metric_totals(self.request_metric_totals, request_metrics)
        update_request_metric_totals(
            self.request_metric_lifetime_totals, request_metrics
        )
        return response

    def collect_request_metrics(self):
        metrics = summarize_request_metric_totals(
            dict(self.request_metric_totals), scope="interval"
        )
        self.request_metric_totals.clear()
        return metrics

    def collect_lifetime_request_metrics(self):
        return summarize_request_metric_totals(
            dict(self.request_metric_lifetime_totals), scope="lifetime"
        )

    def collect_version_boundary_profile(self):
        return self.router.collect_version_boundary_profile()

    def collect_trajectory_progress(self):
        return self.router.collect_trajectory_progress()

    def collect_runtime_feedback(self):
        collector = getattr(self.router, "collect_runtime_feedback", None)
        feedback = collector() if collector is not None else {}
        feedback = dict(feedback or {})
        feedback["request_metrics"] = dict(self.request_metric_lifetime_totals)
        return feedback

    async def abort_requests(self, request_ids, uid):
        return await self.router.abort_requests(request_ids, uid)

    async def abort_all(self):
        logger.info(f"abort all requests, remaining requests: {len(self.inflight_requests)}")
        return await self.router.abort_all(list(self.inflight_requests))

    async def on_send_request(self, request_id) -> bool:
        while self.need_suspend:
            await self.suspend_notifier.wait()
        if self.need_shutdown:
            return False
        self.inflight_requests.add(request_id)
        return True

    async def on_request_routed(self, request_id):
        self.inflight_requests.remove(request_id)
        self.empty_notifier.set()

    def suspend(self):
        """
        Suspend all running requests.

        All following call of generate will be blocked until resume.
        """
        if self.need_suspend:
            return
        self.suspend_notifier.clear()
        self.need_suspend = True

    def resume(self, version=None):
        if not self.need_suspend:
            return
        self.router.on_version_resume(version)
        self.need_suspend = False
        self.suspend_notifier.set()

    def update_runtime_plan(self, plan=None):
        self.router.on_runtime_plan_update(plan)

    async def shutdown(self):
        self.need_shutdown = True
        await self.abort_all()
        self.resume()
        await self.wait_complete()

    async def wait_complete(self):
        """
        Wait until all running requests are finished (no matter whether suspended or not).
        """
        logger.info(f"RouterManager: wait all requests complete {self.inflight_requests=}")
        while len(self.inflight_requests) > 0:
            self.empty_notifier.clear()
            await self.empty_notifier.wait()
        logger.info(f"RouterManager: all requests completed")

    def size(self):
        return len(self.inflight_requests)

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        logger.info(f"RouterManager shrink_workers {target_gpus=}")
        return await self.partial_gpu_manager.shrink_workers(target_gpus)

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        logger.info(f"RouterManager expand_workers {target_gpus=}")
        return await self.partial_gpu_manager.expand_workers(target_gpus, skip_load)

class PartialGPUManager:
    def __init__(self, actor_cluster, router, num_gpus_per_node: int):
        self.infer_cluster = actor_cluster
        self.router = router
        self.num_gpus_per_node = num_gpus_per_node

    def _get_gpus_for_dp_rank(self, dp_rank: int) -> List[int]:
        """Map DP rank to GPU IDs using cluster's device info.

        Args:
            dp_rank: Data parallel rank index (0 to dp_size-1)

        Returns:
            List of GPU IDs used by this DP rank's workers

        Example:
            # Pure DP: rank == dp_rank
            # DP rank 0 uses GPUs [0], DP rank 1 uses GPUs [1], etc.
            gpus = self._get_gpus_for_dp_rank(dp_rank=0)
            # Returns: [0]
        """
        # In agentic pipeline (pure DP): rank == dp_rank, so directly access rank2devices
        devices_info = self.infer_cluster.rank2devices[dp_rank]

        # Extract GPU IDs: gpu_id = node_rank * num_gpus_per_node + gpu_rank
        gpu_ids = []
        for device in devices_info:
            gpu_id = device["node_rank"] * self.num_gpus_per_node + device["gpu_rank"]
            gpu_ids.append(gpu_id)

        return sorted(set(gpu_ids))  # Remove duplicates and sort

    def _validate_target_gpus(self, target_gpus: List[int], mode: str) -> None:
        """Validate target_gpus input for shrink/expand operations.

        Args:
            target_gpus: List of GPU IDs to free (shrink) or restore (expand)
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If target_gpus is empty, has duplicates, or mode is invalid

        Example:
            self._validate_target_gpus([4, 5, 6, 7], mode="shrink")
            # Validates successfully

            self._validate_target_gpus([], mode="shrink")
            # Raises: ValueError("[shrink] target_gpus cannot be empty")

            self._validate_target_gpus([4, 4, 5], mode="expand")
            # Raises: ValueError("[expand] target_gpus has duplicates: [4, 4, 5]")
        """
        # VAL: VAL_NON_EMPTY
        if not target_gpus:
            raise ValueError(f"[{mode}] target_gpus cannot be empty")

        # VAL: VAL_NO_DUPLICATES
        if len(target_gpus) != len(set(target_gpus)):
            raise ValueError(f"[{mode}] target_gpus has duplicates: {target_gpus}")

        if mode not in ("shrink", "expand"):
            raise ValueError(f"Invalid mode: {mode}")

    def _validate_calculated_ranks(self, ranks: List[int], mode: str) -> None:
        """Validate calculated DP ranks against current active_dp_ranks state.

        Args:
            ranks: List of DP ranks calculated from target_gpus
            mode: Operation mode ("shrink" or "expand")

        Raises:
            ValueError: If ranks is empty, contains out-of-range values,
                       or violates state consistency (shrink: must be active,
                       expand: must be inactive)

        Example:
            # Shrink validation
            self.active_dp_ranks = {0, 1, 2, 3}
            self._validate_calculated_ranks([2, 3], mode="shrink")
            # Validates successfully (ranks 2, 3 are active)

            self._validate_calculated_ranks([4], mode="shrink")
            # Raises: ValueError("[shrink] DP rank 4 not active")

            # Expand validation
            self.active_dp_ranks = {0, 1}
            self._validate_calculated_ranks([2, 3], mode="expand")
            # Validates successfully (ranks 2, 3 are inactive)

            self._validate_calculated_ranks([0], mode="expand")
            # Raises: ValueError("[expand] DP rank 0 already active")
        """
        # VAL: VAL_NON_EMPTY
        if not ranks:
            raise ValueError(f"[{mode}] Calculated ranks list is empty")

        # VAL: VAL_INT_RANGE
        for dp_rank in ranks:
            if not (0 <= dp_rank < self.infer_cluster.world_size):
                raise ValueError(f"[{mode}] DP rank {dp_rank} out of range [0, {self.infer_cluster.world_size})")

        # AST: State consistency

        # TODO: fix this validation and move to EnvAffinityRouter
        # for dp_rank in ranks:
        #     if dp_rank not in self.active_dp_ranks:
        #         raise ValueError(f"DP rank {dp_rank} not active {mode=}")

    async def shrink_workers(self, target_gpus: List[int]) -> Dict[str, Any]:
        """Complete atomic shrink operation: validate → rebalance → offload → update routing.

        Orchestrates the full worker shrink process:
        1. Validates target_gpus input
        2. Calculates DP ranks to offload based on GPU overlap
        3. Validates calculated ranks against active state
        4. Do shrink:
           - Rebalances routing (aborts requests on shrinking workers)
           - Offloads model states from shrinking workers
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to free (e.g., [4, 5, 6, 7] to free second half of 8 GPUs)

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "shrink_duration_ms": Total operation time in milliseconds
                - "offload_ranks": List of DP ranks that were offloaded

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (not active, out of range)
            RuntimeError: If rebalance or offload operations fail

        Example:
            # Shrink to free GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.shrink_workers([4, 5, 6, 7])
            # Returns: {"aborted": 10, "remapped": 5, "shrink_duration_ms": 2340.5, "offload_ranks": [2, 3]}

        Side Effects:
            - Updates active_dp_ranks (removes offload_ranks)
            - Aborts in-flight requests on shrinking workers
            - Clears src_rank mappings for remapped environments
            - Offloads model states from shrinking workers to CPU
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="shrink")
        # Calculate DP ranks to offload
        target_gpus = set(target_gpus)
        offload_ranks = [dp for dp in range(self.infer_cluster.world_size)
                         if set(self._get_gpus_for_dp_rank(dp)).intersection(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        self._validate_calculated_ranks(offload_ranks, mode="shrink")

        result = await self.router.rebalance_on_shrink(offload_ranks)

        # release the lock before blocking offload so that active dp rank can work immediately
        # Offload states from target workers
        offload_refs = self.infer_cluster.offload_states_partial(offload_ranks, blocking=False)
        await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in offload_refs])

        return {**result, "shrink_duration_ms": (time.time() - start_time) * 1000,
                "offload_ranks": offload_ranks}

    async def expand_workers(self, target_gpus: List[int], skip_load: bool = False) -> Dict[str, Any]:
        """Complete atomic expand operation: validate → load → rebalance → update routing.

        Orchestrates the full worker expand process:
        1. Validates target_gpus input
        2. Calculates DP ranks to restore based on GPU overlap
        3. Validates calculated ranks against active state (skip if skip_load=True)
        4. Do expand:
           - Loads model states on expanding workers (skip if skip_load=True)
           - Rebalances routing (proportionally redistributes requests)
        5. Returns metrics for monitoring

        Args:
            target_gpus: GPU IDs to restore (e.g., [4, 5, 6, 7] to restore second half of 8 GPUs)
            skip_load: If True, skip model loading and validation (use when model_update already loaded states).
                      This only updates active_dp_ranks to restore routing state without re-loading models.

        Returns:
            Metrics dict containing:
                - "aborted": Number of requests aborted during rebalancing (proportional redistribution)
                - "remapped": Number of src_ranks remapped (cleared from routing)
                - "expand_duration_ms": Total operation time in milliseconds
                - "load_ranks": List of DP ranks that were restored

        Raises:
            ValueError: If target_gpus invalid (empty, duplicates) or
                       calculated ranks invalid (already active, out of range)
            RuntimeError: If load or rebalance operations fail

        Example:
            # Expand to restore GPUs [4, 5, 6, 7] (second half of 8-GPU setup)
            result = await scheduler.expand_workers([4, 5, 6, 7])
            # Returns: {"aborted": 3, "remapped": 3, "expand_duration_ms": 1850.2, "load_ranks": [2, 3]}

            # After model_update already loaded states to all GPUs, just restore routing:
            result = await scheduler.expand_workers([4, 5, 6, 7], skip_load=True)

        Side Effects:
            - Updates active_dp_ranks (adds load_ranks)
            - Loads model states from CPU to expanding workers (unless skip_load=True)
            - Aborts some requests from old workers for proportional rebalancing
            - Clears src_rank mappings for rebalanced environments (will route to new workers)
        """
        start_time = time.time()

        # VAL: VAL_NON_EMPTY, VAL_NO_DUPLICATES
        self._validate_target_gpus(target_gpus, mode="expand")

        # Calculate DP ranks to restore
        target_gpus = set(target_gpus)
        load_ranks = [dp for dp in range(self.infer_cluster.world_size)
                      if set(self._get_gpus_for_dp_rank(dp)).issubset(target_gpus)]

        # VAL: VAL_NON_EMPTY, state consistency check
        # Skip validation when skip_load=True because ranks may already be "active" in cluster
        # (model states loaded by model_update) but not tracked in active_dp_ranks yet
        if not skip_load:
            self._validate_calculated_ranks(load_ranks, mode="expand")
            load_refs = self.infer_cluster.load_states_partial(load_ranks, blocking=False)
            await asyncio.gather(*[asyncio.wrap_future(ref.future()) for ref in load_refs])

        result = await self.router.rebalance_on_expand(load_ranks)

        return {**result, "expand_duration_ms": (time.time() - start_time) * 1000,
                "load_ranks": load_ranks}

class RouterProxy:
    """
    Proxy to RouterManager
    """
    @abstractmethod
    async def generate_request(self, payload, request_id, uid, priority=None):
        pass

    @abstractmethod
    async def on_send_request(self, request_id):
        pass

    @abstractmethod
    async def on_request_routed(self, request_id):
        pass

    def generate_request_sync(self, payload, request_id, uid, priority=None):
        raise NotImplementedError

    def on_send_request_sync(self, request_id):
        raise NotImplementedError

    def on_request_routed_sync(self, request_id):
        raise NotImplementedError

class InprocProxy(RouterProxy):
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager

    async def generate_request(self, payload, request_id, uid, priority=None):
        return await self.router_manager.generate_request(payload=payload, request_id=request_id, uid=uid, priority=priority)

    async def on_send_request(self, request_id):
        return await self.router_manager.on_send_request(request_id)

    async def on_request_routed(self, request_id):
        return await self.router_manager.on_request_routed(request_id)

class RayProxy(RouterProxy):
    def __init__(self, router_manager: RouterManager):
        self.router_manager = router_manager

    async def generate_request(self, payload, request_id, uid, priority=None):
        return await self.router_manager.generate_request.remote(payload=payload, request_id=request_id, uid=uid, priority=priority)

    async def on_send_request(self, request_id):
        return await self.router_manager.on_send_request.remote(request_id)

    async def on_request_routed(self, request_id):
        return await self.router_manager.on_request_routed.remote(request_id)

    def generate_request_sync(self, payload, request_id, uid, priority=None):
        return ray.get(self.router_manager.generate_request.remote(payload=payload, request_id=request_id, uid=uid, priority=priority))

    def on_send_request_sync(self, request_id):
        return ray.get(self.router_manager.on_send_request.remote(request_id))

    def on_request_routed_sync(self, request_id):
        return ray.get(self.router_manager.on_request_routed.remote(request_id))

class SglangProxy(RouterProxy):
    def __init__(self, proxy: RouterProxy, router_meta):
        self.proxy = proxy
        self.router_ip = router_meta["router_ip"]
        self.router_port = router_meta["router_port"]
        self.worker_urls = router_meta["worker_urls"]
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        self.client_sync = httpx.Client(timeout=httpx.Timeout(None))

    async def generate_request(self, payload, request_id, uid, priority=None):
        from roll.distributed.strategy.sglang_strategy import postprocess_generate
        assert "multi_modal_data" not in payload
        url = f"http://{self.router_ip}:{self.router_port}/generate"
        response = await self.client.post(url, json=payload)
        raise_for_status(response)
        response = response.json()
        response = response if isinstance(response, list) else [response]
        return postprocess_generate(response)

    async def on_send_request(self, request_id):
        return await self.proxy.on_send_request(request_id)

    async def on_request_routed(self, request_id):
        return await self.proxy.on_request_routed(request_id)

    def generate_request_sync(self, payload, request_id, uid, priority=None):
        from roll.distributed.strategy.sglang_strategy import postprocess_generate
        assert "multi_modal_data" not in payload
        url = f"http://{self.router_ip}:{self.router_port}/generate"
        response = self.client_sync.post(url, json=payload)
        raise_for_status(response)
        response = response.json()
        response = response if isinstance(response, list) else [response]
        return postprocess_generate(response)

    def on_send_request_sync(self, request_id):
        return self.proxy.on_send_request_sync(request_id)

    def on_request_routed_sync(self, request_id):
        return self.proxy.on_request_routed_sync(request_id)

class RouterClient:
    def __init__(self, proxy, meta):
        self.proxy = proxy
        self.strategy_name = meta["strategy_name"]
        self.eos_token_id = meta["eos_token_id"]
        self.pad_token_id = meta["pad_token_id"]

    def _preprocess_generate(self, req: DataProto, request_id):
        if request_id is None:
            request_id = str(uuid.uuid4())
        payload = {"rid": str(request_id)}

        generation_config = req.meta_info.get("generation_config")
        collect_unfinished = req.meta_info.get("collect_unfinished", False)
        num_return_sequences = generation_config["num_return_sequences"]
        assert num_return_sequences == 1 or not collect_unfinished, "collect_unfinished is not supported in parallel sampling"

        max_new_tokens = req.meta_info.get("max_new_tokens", generation_config["max_new_tokens"])
        max_new_tokens = min(max_new_tokens, generation_config["max_new_tokens"])
        generation_config["max_new_tokens"] = max_new_tokens

        generation_config["eos_token_id"] = [self.eos_token_id, self.pad_token_id]
        generation_config["pad_token_id"] = self.pad_token_id

        if "multi_modal_data" in req.non_tensor_batch:
            multi_modal_data = req.non_tensor_batch["multi_modal_data"]
            assert len(multi_modal_data) == 1
            if 'multi_modal_data' in multi_modal_data[0] and 'video' in multi_modal_data[0]['multi_modal_data'] and self.strategy_name == 'sglang':
                multi_modal_data = req.non_tensor_batch["multi_modal_inputs"]
                assert len(multi_modal_data) == 1
                payload["multi_modal_data"] = {'multi_modal_data': {'video': multi_modal_data[0]}}
                input_ids = req.batch["input_ids"]
                attention_mask = req.batch["attention_mask"]
                input_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
                payload["multi_modal_data"]["prompt_token_ids"] = input_ids[0]
            else:
                payload["multi_modal_data"] = multi_modal_data[0]

        else:
            input_ids = req.batch["input_ids"]
            assert not collect_unfinished or input_ids.size(0) == 1
            attention_mask = req.batch["attention_mask"]
            input_ids = gather_unpadded_input_ids(input_ids=input_ids, attention_mask=attention_mask)
            payload["input_ids"] = input_ids[0]

        match self.strategy_name:
            case "sglang":
                sampling_params = _create_sampling_params_for_sglang(gen_kwargs=generation_config)
                payload["sampling_params"] = sampling_params
                payload["return_logprob"] = generation_config.get("logprobs", 0) is not None
            case "vllm":
                from roll.distributed.strategy.vllm_strategy import create_sampling_params_for_vllm
                # vllm is hard coded to return logprob
                sampling_params = create_sampling_params_for_vllm(generation_config, collect_unfinished)
                payload["sampling_params"] = sampling_params
            case _:
                raise NotImplementedError(f"strategy {self.strategy_name} is not supported")
        return payload, request_id

    def _postprocess_generate(self, req, response):
        output_data = DataProto(meta_info=req.meta_info)
        output_data.meta_info["finish_reasons"] = response["finish_reasons"]
        output_data.meta_info["output_token_ids"] = response.get("output_token_ids", None)
        output_data.meta_info["output_logprobs"] = response.get("output_logprobs", None)
        # TODO: The size of routed_experts is [b * s * layer * topk].
        # For the 30A3 model, this data block is tens of MB in size.
        # The serialization overhead of Ray transmission needs to be profiled again.
        output_data.meta_info["routed_experts"] = response.get("routed_experts", None)
        output_data.meta_info["eos_token_id"] = [self.eos_token_id, self.pad_token_id]
        output_data.meta_info["pad_token_id"] = self.pad_token_id

        # Merge metrics from response (e.g., speculative decoding metrics)
        if "metrics" in response:
            output_data.meta_info.setdefault("metrics", {}).update(response["metrics"])
        if "runtime_attribution" in response:
            output_data.meta_info["runtime_attribution"] = dict(
                response["runtime_attribution"]
            )

        return output_data

    async def generate_request(self, req: DataProto, request_id, uid):
        """
        Request format is adapted for sglang generate (specificly, use rid rather than request_id),
        which can be directly used by SglangRouter.
        Request is expected to be scalar (single request).

        Response format is adapted for ROLL DataProto.
        Response is expected to be vector (expanded for parallel sample).
        """
        payload, request_id = self._preprocess_generate(req, request_id)

        if not await self.proxy.on_send_request(request_id):
            return None # shutdown
        try:
            priority = req.meta_info.get("trajectory_priority")
            response = await self.proxy.generate_request(payload=payload, request_id=request_id, uid=uid, priority=priority)
        finally:
            await self.proxy.on_request_routed(request_id)

        return self._postprocess_generate(req, response)

    def generate_request_sync(self, req: DataProto, request_id, uid):
        payload, request_id = self._preprocess_generate(req, request_id)

        if not self.proxy.on_send_request_sync(request_id):
            return None # shutdown
        try:
            priority = req.meta_info.get("trajectory_priority")
            response = self.proxy.generate_request_sync(payload=payload, request_id=request_id, uid=uid, priority=priority)
        finally:
            self.proxy.on_request_routed_sync(request_id)

        return self._postprocess_generate(req, response)

class Router:
    def __init__(self, router_manager, workers, model_path, router_args: RouterArguments):
        self.router_manager_ref = weakref.ref(router_manager)
        self.workers = workers
        self.model_path = model_path
        self.router_args = router_args

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def generate_request(self, payload, request_id, uid, priority=None):
        pass

    @abstractmethod
    async def abort_requests(self, request_ids, uid):
        pass

    @abstractmethod
    async def abort_all(self, request_ids):
        pass

    def on_version_resume(self, version=None):
        """Notify routers that model weights and prefix-cache generation changed."""
        return None

    def on_runtime_plan_update(self, plan=None):
        """Apply an online decision revision without changing model/cache epoch."""
        return None

    def collect_version_boundary_profile(self):
        return {"metrics": {}, "records": []}

    def collect_trajectory_progress(self):
        return []

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        raise NotImplementedError

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        raise NotImplementedError

class SglangRouter(Router):
    """
    Wrap of https://docs.sglang.io/advanced_features/router.html#api-surface

    This is act as a client to sglang-router, can instantiate one SglangRouterClient for every env,
    """
    async def initialize(self):
        self.router_ip = Worker.get_node_ip()
        self.router_port = Worker.get_free_port()

        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None))

        self.worker_urls = await asyncio.gather(
            *[
                worker.get_url.remote()
                for worker in self.workers
            ]
        )
        self.http_mode = False if self.worker_urls[0].startswith("grpc") else True
        assert self.http_mode

        import multiprocessing
        from sglang_router.launch_router import RouterArgs, launch_router

        multiprocessing.set_start_method("spawn")

        router_config = {
            "host": self.router_ip,
            "port": self.router_port,
            "prometheus_port": Worker.get_free_port(),
            "log_level": "warn",
            "policy": "cache_aware",
            "request_timeout_secs": 1800,
            "max_concurrent_requests": -1,
            "dp_aware": False,
            "worker_urls": self.worker_urls,
        }
        extra_router_config = self.router_args.router_config
        if router_config:
            router_config.update(extra_router_config)
        router_args = RouterArgs(**router_config)
        self.router_process = multiprocessing.Process(
            target=launch_router,
            args=(router_args,),
            daemon=True
        )
        self.router_process.start()
        logger.info(f"Launch sglang-router {router_args=}")
        await wait_sglang_router_ready(self.router_process, f"http://{self.router_ip}:{self.router_port}")
        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", self.worker_urls)

    async def generate_request(self, payload, request_id, uid, priority=None):
        raise RuntimeError("SglangRouter.generate_request is not expected to be called directly, use RouterClient.")

    async def abort_requests(self, request_ids, uid):
        async def abort_request(self, url, request_id):
            response = await self.client.post(f"{url}/abort_request", json={"rid": request_id})
            raise_for_status(response)
        await asyncio.gather(
            *[
                abort_request(self, url=url, request_id=request_id)
                for request_id in request_ids for url in self.worker_urls
            ]
        )

    async def abort_all(self, request_ids):
        # Cannot use abort_all of sglang, because actor_cluster may be shared between different Routers.
        await self.abort_requests(request_ids, uid=None)

    async def abort_all_worker(self, url):
        # Can only be used when router is not shared between two scheudlers.
        response = await self.client.post(f"{url}/abort_request", json={"abort_all": True})
        raise_for_status(response)

    async def post_workers(self, urls):
        responses = await asyncio.gather(
            *[
                self.client.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json={"url": url},
                )
                for url in urls
            ]
        )
        for response in responses:
            raise_for_status(response)

    async def delete_workers(self, urls):
        encoded_urls = [quote(url, safe="") for url in urls]
        responses = await asyncio.gather(
            *[self.client.delete(f"http://{self.router_ip}:{self.router_port}/workers/{url}") for url in encoded_urls]
        )
        for response in responses:
            raise_for_status(response)

    async def get_worker_loads(self, url):
        response = await self.client.get(f"{url}/get_load")
        raise_for_status(response)
        return response.json()

    async def wait_worker_complete(self, url):
        while True:
            loads = await self.get_worker_loads(url)
            if all(load["num_reqs"] == 0 and load["num_waiting_reqs"] == 0 for load in loads):
                break
            await asyncio.sleep(1)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        shrink_urls = [self.worker_urls[dp_rank] for dp_rank in shrink_dp_ranks]

        router_manager: RouterManager = self.router_manager_ref()
        router_manager.suspend()

        await self.delete_workers(shrink_urls)
        logger.info(f"SglangRouter: delete workers on shrink {shrink_dp_ranks=} {shrink_urls=}")

        # FIXME: Do not abort and wait for all workers.
        # Because call wait_worker_complete of shrink workers may not be accurate. There may be
        # a client called on_request_routed but has not calling generate_request yet.
        # Instead, we use RouterManager.wait_complete to make sure no more requests to shrink workers.
        await asyncio.gather(*[self.abort_all_worker(url) for url in self.worker_urls])
        logger.info(f"SglangRouter: abort all requests on shrink {shrink_dp_ranks=} {shrink_urls=}")

        logger.info(f"SglangRouter: wait for running requests on shrink ")
        await router_manager.wait_complete()

        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", {url for url in self.worker_urls if url not in shrink_urls})

        router_manager.resume()

        logger.info(f"SglangRouter: rebalance on shrink finish")

        return {"aborted": 0, "remapped": 0} # for compatibility

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        expand_urls = [self.worker_urls[dp_rank] for dp_rank in expand_dp_ranks]

        await self.post_workers(expand_urls)
        logger.info(f"SglangRouter: post workers on expand {expand_dp_ranks=}")

        # simply abort all requests to let sglang-router to re-schedule
        await asyncio.gather(*[self.abort_all_worker(url) for url in self.worker_urls])
        logger.info(f"SglangRouter: aborted all requests on expand {expand_dp_ranks=}")

        # FIXME: assume expand all workers currently
        await wait_sglang_router_workflow(f"http://{self.router_ip}:{self.router_port}", self.worker_urls)

        return {"aborted": 0, "remapped": 0} # for compatibility

class PromptAffinityRouter(Router):
    """
    Schedule requests of the same prompt to the same worker. Choose worker using best fit
    strategy (using linear search for simplicity), blocking generate request if no worker available.

    Limit the number of running requests of each dp rank below max_running_requests.
    """
    async def initialize(self):
        self.max_running_requests = self.router_args.max_running_requests

        # key: dp_rank, value: num_inflight_requests
        self.worker_loads = {dp_rank: 0 for dp_rank in range(len(self.workers))}
        # cache-aware scheduling by uid
        self.id_to_dp_rank: Dict[int, int] = {}
        # dp_rank -> request_ids, used by abort_all
        self.dp_inflight_requests: List[int, Set[str]] = [set() for _ in self.workers]

        self.lock = asyncio.Lock()
        # used by acquire
        self.event = asyncio.Event()
        # used by reacquire
        self.worker_event = {dp_rank: asyncio.Event() for dp_rank in range(len(self.workers))}

    def __repr__(self):
        return f"worker loads: {self.worker_loads}"

    async def generate_request(self, payload, request_id, uid, priority=None):
        credit = payload["sampling_params"]["n"]
        dp_rank = None
        if uid not in self.id_to_dp_rank:
            # To prevent multiple generate requests for the same prompt.
            # It is safe and no performance issue to acquire lock here.
            # Because acquire is guaranteed to return as long as there has
            # one worker whose running_requests < max_running_requests no matter
            # how large credit is.
            async with self.lock:
                if uid not in self.id_to_dp_rank:
                    dp_rank = await self.acquire(credit=credit)
                    self.id_to_dp_rank[uid] = dp_rank
        if dp_rank is None:
            assert uid in self.id_to_dp_rank
            dp_rank = self.id_to_dp_rank[uid]
            assert dp_rank is not None
            await self.reacquire(dp_rank=dp_rank, credit=credit)
        try:
            self.dp_inflight_requests[dp_rank].add(request_id)
            # InferWorker.generate_request only return data with finish_reason=="abort" on abort
            # but not raise asyncio.CancelledError. This try finally block may be not necessary.
            return await self.workers[dp_rank].generate_request.remote(payload)
            # TODO ray.cancel(ref) on asyncio.CancelledError
        finally:
            self.dp_inflight_requests[dp_rank].remove(request_id)
            self.release(dp_rank=dp_rank, credit=credit)

    async def abort_requests(self, request_ids, uid):
        assert uid is not None
        dp_rank = self.id_to_dp_rank[uid]
        await self.workers[dp_rank].abort_requests.remote(request_ids=request_ids)

    async def abort_all(self, request_ids):
        await asyncio.gather(
            *[
                self.workers[dp_rank].abort_requests.remote(list(request_ids))
                for dp_rank, request_ids in enumerate(self.dp_inflight_requests)
            ]
        )
        self.id_to_dp_rank.clear() # gc uid cache here

    async def acquire(self, credit: int) -> int:
        while True:
            # TODO add check of suspend here to stop early
            target = -1
            for dp_rank, running_requests in self.worker_loads.items():
                if running_requests >= self.max_running_requests:
                    continue
                if target == -1 or running_requests < self.worker_loads[target]:
                    target = dp_rank
            if target != -1:
                # may send more requests than max_running_requests,
                # i.e. worker_loads[target] + credit > max_running_requests
                self.worker_loads[target] += credit
                return target
            self.event.clear()
            await self.event.wait()

    async def reacquire(self, dp_rank: int, credit: int):
        assert dp_rank in self.worker_loads
        while True:
            # TODO add check of suspend here to stop early
            if self.worker_loads[dp_rank] < self.max_running_requests:
                self.worker_loads[dp_rank] += credit
                return
            self.worker_event[dp_rank].clear()
            await self.worker_event[dp_rank].wait()

    def release(self, dp_rank: int, credit: int):
        assert credit >= 0
        self.worker_loads[dp_rank] -= credit
        assert self.worker_loads[dp_rank] >= 0
        self.event.set()
        self.worker_event[dp_rank].set()

    def size(self):
        return sum(self.worker_loads.values())

    def full(self) -> bool:
        return all(running_requests >= self.max_running_requests for running_requests in self.worker_loads.values())

class EnvAffinityRouter(Router):
    """
    Schedule requests of the same (env) uid, to the same dp_rank.

    Choose dp_rank by RR for the first time.

    No rate limit now.

    Do not support partial rollout now.
    """
    async def initialize(self):
        self.src_rank2_dp_rank = {}
        self.src_rank_cache_epoch = {}
        self.src_rank_last_prompt_tokens = {}
        self.cache_epoch = 0
        self.request_id_2_src_rank: Dict[str, int] = {}  # Reverse lookup for abort
        self.running_requests: List[set[str]] = [set() for _ in range(len(self.workers))]
        self.worker_iter = itertools.cycle(range(len(self.workers)))

        # Active DP ranks for request routing
        self.active_dp_ranks: Set[int] = set(range(len(self.workers)))  # All ranks initially active
        self.routing_lock = asyncio.Lock()  # Protect routing updates
        self.max_running_requests = self.router_args.max_running_requests
        self.priority_sequence = itertools.count()
        self.priority_waiters = [[] for _ in self.workers]
        self.priority_inflight = [0 for _ in self.workers]
        self.priority_conditions = [asyncio.Condition() for _ in self.workers]
        self.priority_candidate_ranks: Dict[str, int] = {}
        self.priority_candidate_estimates: Dict[str, Dict[str, Any]] = {}
        config = getattr(self.router_args, "router_config", None) or {}
        self.engine_priority_scheduling_enabled = bool(
            config.get("engine_priority_scheduling_enabled", False)
        )
        self.priority_max_running_requests = int(
            config.get("priority_max_running_requests", self.max_running_requests)
        )
        self.priority_rebuild_max_running_requests = max(
            self.priority_max_running_requests,
            min(
                self.max_running_requests,
                int(
                    config.get(
                        "priority_rebuild_max_running_requests",
                        self.priority_max_running_requests,
                    )
                ),
            ),
        )
        self.priority_coalesce_seconds = max(
            0.0, float(config.get("priority_coalesce_seconds", 0.0))
        )
        self.priority_batch_deadline = [0.0 for _ in self.workers]
        self.post_update_rebuild_enabled = bool(
            config.get("post_update_rebuild_enabled", False)
        )
        self.post_update_rebuild_requests = int(
            config.get("post_update_rebuild_requests", 0)
        )
        self.post_update_rebuild_observe_requests = int(
            config.get(
                "post_update_rebuild_observe_requests",
                self.post_update_rebuild_requests,
            )
        )
        self.post_update_rebuild_prefix_tokens = int(
            config.get("post_update_rebuild_prefix_tokens", 2048)
        )
        self.post_update_rebuild_coalesce_seconds = max(
            0.0,
            float(config.get("post_update_rebuild_coalesce_seconds", 0.0)),
        )
        self.post_update_rebuild_seed_slots_per_worker = max(
            1,
            int(config.get("post_update_rebuild_seed_slots_per_worker", 2)),
        )
        self.post_update_rebuild_min_reuse_requests = max(
            1,
            int(config.get("post_update_rebuild_min_reuse_requests", 1)),
        )
        self.post_update_rebuild_min_outstanding_per_worker = max(
            0,
            int(
                config.get(
                    "post_update_rebuild_min_outstanding_per_worker",
                    self.priority_max_running_requests,
                )
            ),
        )
        self.soft_locality_enabled = bool(
            config.get("soft_locality_enabled", False)
        )
        self.soft_locality_load_slack = int(
            config.get("soft_locality_load_slack", 1)
        )
        self.working_set_routing_enabled = bool(
            config.get("working_set_routing_enabled", self.post_update_rebuild_enabled)
        )
        self.working_set_max_prompts_per_worker = int(
            config.get("working_set_max_prompts_per_worker", 64)
        )
        self.completion_eta_routing_enabled = bool(
            config.get("completion_eta_routing_enabled", False)
        )
        self.completion_eta_ewma_alpha = min(
            1.0,
            max(0.0, float(config.get("completion_eta_ewma_alpha", 0.2))),
        )
        self.completion_eta_locality_slack_seconds = max(
            0.0,
            float(
                config.get(
                    "completion_eta_locality_slack_seconds", 0.05
                )
            ),
        )
        self.worker_service_seconds_ewma: List[Optional[float]] = [
            None for _ in self.workers
        ]
        self.model_token_seconds_ewma: Optional[float] = None
        self.refresh_profile_enabled = bool(
            config.get("refresh_profile_enabled", False)
        )
        self.refresh_profile_max_records = max(
            1, int(config.get("refresh_profile_max_records", 20000))
        )
        self.rebuild_epoch = None
        self.rebuild_load_eligible = True
        self.boundary_resumed_at: Dict[int, float] = {}
        self.rebuild_remaining = 0
        self.rebuild_target = 0
        self.rebuild_observe_remaining = 0
        self.rebuild_candidate_groups = set()
        self.rebuild_candidate_trajectories = set()
        self.rebuild_cohort_exact = False
        self.rebuild_seen_trajectories = set()
        self.rebuild_worker_prompts = defaultdict(list)
        self.rebuild_assigned_counts = defaultdict(int)
        self.rebuild_pending = []
        self.rebuild_coalesce_lock = asyncio.Lock()
        self.rebuild_flush_task = None
        self.rebuild_prefix_ready_workers = defaultdict(set)
        self.rebuild_prefix_warming_workers = defaultdict(set)
        self.rebuild_prefix_waiters = defaultdict(list)
        self.prefix_ready_workers = defaultdict(set)
        self.prefix_keys_by_worker = defaultdict(list)
        self.working_set_worker_prompts = defaultdict(list)
        self.runtime_plan = {}
        self.boundary_recovery_records: List[Dict[str, Any]] = []
        self.refresh_request_records: List[Dict[str, Any]] = []
        self.latest_trajectory_progress: Dict[str, Dict[str, Any]] = {}
        self.worker_engine_kv_feedback = [
            {
                "requests": 0,
                "query_blocks": 0,
                "hit_blocks": 0,
                "cached_tokens": 0,
                "resets": 0,
            }
            for _ in self.workers
        ]

    def _is_rebuild_load_eligible(self, plan: Dict[str, Any]) -> bool:
        minimum = self.post_update_rebuild_min_outstanding_per_worker
        if minimum <= 0 or "outstanding_trajectories" not in plan:
            return True
        worker_count = max(
            1, int(plan.get("worker_count", len(self.workers)))
        )
        outstanding = max(0, int(plan.get("outstanding_trajectories", 0)))
        return outstanding >= worker_count * minimum

    def on_version_resume(self, version=None):
        plan = dict(version) if isinstance(version, dict) else {"version": version}
        epoch = plan.get("version")
        if epoch == self.rebuild_epoch:
            return
        self._cancel_pending_rebuild_wave()
        self.cache_epoch += 1
        self.rebuild_epoch = epoch
        self.boundary_resumed_at[self.cache_epoch] = time.monotonic()
        self.runtime_plan = plan
        priority_candidates = (
            plan.get("priority_candidate_groups", [])
            if bool(plan.get("priority_enabled", False))
            else []
        )
        self.priority_candidate_ranks = {
            str(group_key): rank
            for rank, group_key in enumerate(priority_candidates)
        }
        self.priority_candidate_estimates = {
            str(estimate.get("group_key")): dict(estimate)
            for estimate in plan.get("priority_candidate_estimates", [])
            if estimate.get("group_key") is not None
        }
        self.rebuild_candidate_groups = set(plan.get("rebuild_candidate_groups", []))
        self.rebuild_candidate_trajectories = set(
            str(trajectory_id)
            for trajectory_id in plan.get("rebuild_candidate_trajectories", [])
        )
        self.rebuild_cohort_exact = bool(
            plan.get("rebuild_cohort_exact", False)
        )
        self.rebuild_load_eligible = self._is_rebuild_load_eligible(plan)
        planned_target = int(
            plan.get("rebuild_target_trajectories", self.post_update_rebuild_requests)
        )
        if not self.rebuild_load_eligible:
            planned_target = 0
        elif (
            not self.rebuild_cohort_exact
            and not self.rebuild_candidate_groups
            and planned_target <= 0
        ):
            planned_target = self.post_update_rebuild_requests
        self.rebuild_target = min(
            max(0, self.post_update_rebuild_requests),
            max(0, planned_target),
        )
        self.rebuild_remaining = self.rebuild_target
        self.rebuild_observe_remaining = max(
            self.rebuild_target,
            max(0, self.post_update_rebuild_observe_requests),
        )
        self.rebuild_seen_trajectories.clear()
        self.rebuild_worker_prompts.clear()
        self.rebuild_assigned_counts.clear()
        self.rebuild_prefix_ready_workers.clear()
        self.rebuild_prefix_warming_workers.clear()
        self.rebuild_prefix_waiters.clear()
        self.prefix_ready_workers.clear()
        self.prefix_keys_by_worker.clear()
        self.working_set_worker_prompts.clear()
        # A refresh invalidates KV, but the previous trajectory-to-worker map is
        # still a safe, balanced fallback. Keep it unless a rebuild cluster has
        # enough followers to justify changing placement.

    def _cancel_pending_rebuild_wave(self):
        task = self.rebuild_flush_task
        if task is not None and not task.done():
            task.cancel()
        pending = self.rebuild_pending
        self.rebuild_pending = []
        self.rebuild_flush_task = None
        for item in pending:
            future = item.get("future")
            if future is not None and not future.done():
                future.set_result(None)
        waiters = self.rebuild_prefix_waiters
        self.rebuild_prefix_waiters = defaultdict(list)
        for prefix_waiters in waiters.values():
            for item in prefix_waiters:
                future = item.get("future")
                if future is not None and not future.done():
                    future.set_result(None)

    def _observe_first_epoch_request(self, trajectory_id: str) -> bool:
        if trajectory_id in self.rebuild_seen_trajectories:
            return False
        self.rebuild_seen_trajectories.add(trajectory_id)
        self.rebuild_observe_remaining = max(0, self.rebuild_observe_remaining - 1)
        return True

    async def _flush_rebuild_wave(self):
        if self.post_update_rebuild_coalesce_seconds > 0:
            await asyncio.sleep(self.post_update_rebuild_coalesce_seconds)
        async with self.rebuild_coalesce_lock:
            pending = self.rebuild_pending
            self.rebuild_pending = []
            self.rebuild_flush_task = None
            if not pending:
                return
            pending.sort(
                key=lambda item: build_runtime_priority_key(
                    item["runtime_state"], self.priority_candidate_ranks
                )
            )
            wave_size = len(pending)
            groups = {}
            for item in pending:
                prefix_keys = build_prefix_directory_keys(
                    item["prompt_tokens"],
                    self.post_update_rebuild_prefix_tokens,
                )
                item["prefix_keys"] = prefix_keys
                cluster_key = (
                    prefix_keys[-1]
                    if prefix_keys
                    else (0, f'trajectory:{item["trajectory_id"]}')
                )
                item["cluster_key"] = cluster_key
                groups.setdefault(cluster_key, []).append(item)

            active_ranks = sorted(self.active_dp_ranks)
            for cluster_key, cluster in groups.items():
                # One seed per worker preserves rollout parallelism. Additional
                # same-prefix requests wait until a seed publishes current-epoch KV.
                seed_count = min(
                    len(cluster),
                    max(
                        1,
                        len(active_ranks)
                        * self.post_update_rebuild_seed_slots_per_worker,
                    ),
                )
                reuse_requests = len(cluster) - seed_count
                if reuse_requests < self.post_update_rebuild_min_reuse_requests:
                    # No realizable reuse in this wave. Preserve the normal
                    # affinity path instead of perturbing placement for a
                    # reconstruction plan that cannot save any prefill.
                    for item in cluster:
                        if not item["future"].done():
                            item["future"].set_result(None)
                    continue
                cluster_seed_workers = set()
                for index, item in enumerate(cluster):
                    if index >= seed_count:
                        item["wave_size"] = wave_size
                        item["wait_started"] = time.perf_counter()
                        self.rebuild_prefix_waiters[cluster_key].append(item)
                        continue
                    dp_rank = min(
                        active_ranks,
                        key=lambda rank: (
                            int(rank in cluster_seed_workers),
                            self.rebuild_assigned_counts.get(rank, 0),
                            self._worker_pressure().get(rank, 0),
                            rank,
                        ),
                    )
                    cluster_seed_workers.add(dp_rank)
                    self.rebuild_assigned_counts[dp_rank] += 1
                    self.rebuild_prefix_warming_workers[cluster_key].add(dp_rank)
                    if not item["future"].done():
                        item["future"].set_result(
                            (
                                dp_rank,
                                0,
                                wave_size,
                                cluster_key,
                                "seed",
                                self.cache_epoch,
                                0.0,
                            )
                        )

    async def _prepare_first_epoch_request(
        self,
        runtime_state: TrajectoryRuntimeState,
        trajectory_id: str,
        prompt_tokens,
    ):
        """Observe an epoch request and optionally join its bounded rebuild wave."""
        future = None
        async with self.rebuild_coalesce_lock:
            first_epoch_request = self._observe_first_epoch_request(trajectory_id)
            rebuild_candidate = (
                trajectory_id in self.rebuild_candidate_trajectories
                if self.rebuild_cohort_exact
                else runtime_state.group_key in self.rebuild_candidate_groups
            )
            rebuild_fallback = (
                not self.rebuild_cohort_exact
                and (
                    not self.rebuild_candidate_groups
                    or self.rebuild_observe_remaining <= self.rebuild_remaining
                )
            )
            if (
                self.post_update_rebuild_enabled
                and self.rebuild_remaining > 0
                and first_epoch_request
                and (rebuild_candidate or rebuild_fallback)
                and len(prompt_tokens) > 0
            ):
                future = asyncio.get_running_loop().create_future()
                self.rebuild_pending.append(
                    {
                        "runtime_state": runtime_state,
                        "trajectory_id": trajectory_id,
                        "prompt_tokens": prompt_tokens,
                        "future": future,
                    }
                )
                self.rebuild_remaining -= 1
                if self.rebuild_flush_task is None:
                    self.rebuild_flush_task = asyncio.create_task(
                        self._flush_rebuild_wave()
                    )
        assignment = await future if future is not None else None
        return first_epoch_request, assignment

    def _register_prefix_keys_ready(self, dp_rank: int, prompt_tokens) -> None:
        keys = build_prefix_directory_keys(
            prompt_tokens, self.post_update_rebuild_prefix_tokens
        )
        if not keys:
            return
        worker_keys = self.prefix_keys_by_worker[dp_rank]
        max_keys = max(1, self.working_set_max_prompts_per_worker) * 8
        for key in keys:
            owners = self.prefix_ready_workers[key]
            if dp_rank in owners:
                continue
            owners.add(dp_rank)
            worker_keys.append(key)
        while len(worker_keys) > max_keys:
            evicted = worker_keys.pop(0)
            owners = self.prefix_ready_workers.get(evicted)
            if owners is None:
                continue
            owners.discard(dp_rank)
            if not owners:
                self.prefix_ready_workers.pop(evicted, None)

    def _invalidate_prefix_worker(self, dp_rank: int) -> None:
        for key in self.prefix_keys_by_worker.pop(dp_rank, []):
            owners = self.prefix_ready_workers.get(key)
            if owners is not None:
                owners.discard(dp_rank)
                if not owners:
                    self.prefix_ready_workers.pop(key, None)
        for mapping in (
            self.rebuild_prefix_ready_workers,
            self.rebuild_prefix_warming_workers,
        ):
            for key in list(mapping):
                mapping[key].discard(dp_rank)
                if not mapping[key]:
                    mapping.pop(key, None)

    async def _complete_rebuild_seed(
        self,
        *,
        cache_epoch: int,
        cluster_key: PrefixDirectoryKey,
        dp_rank: int,
        prompt_tokens,
        success: bool,
    ) -> None:
        """Publish a seed's KV and release same-prefix followers conservatively."""
        async with self.rebuild_coalesce_lock:
            if cache_epoch != self.cache_epoch:
                return
            warming = self.rebuild_prefix_warming_workers.get(cluster_key, set())
            warming.discard(dp_rank)
            if not warming:
                self.rebuild_prefix_warming_workers.pop(cluster_key, None)
            if success:
                self.rebuild_prefix_ready_workers[cluster_key].add(dp_rank)
                self._register_prefix_keys_ready(dp_rank, prompt_tokens)

            waiters = self.rebuild_prefix_waiters.get(cluster_key, [])
            if not waiters:
                return
            ready = sorted(self.rebuild_prefix_ready_workers.get(cluster_key, set()))
            warming_count = len(
                self.rebuild_prefix_warming_workers.get(cluster_key, set())
            )
            if not ready:
                if warming_count:
                    return
                replacement = waiters.pop(0)
                replacement_rank = min(
                    self.active_dp_ranks,
                    key=lambda rank: (self._worker_pressure().get(rank, 0), rank),
                )
                self.rebuild_prefix_warming_workers[cluster_key].add(
                    replacement_rank
                )
                future = replacement["future"]
                if not future.done():
                    future.set_result(
                        (
                            replacement_rank,
                            0,
                            replacement["wave_size"],
                            cluster_key,
                            "seed",
                            self.cache_epoch,
                            max(
                                0.0,
                                time.perf_counter()
                                - replacement["wait_started"],
                            ),
                        )
                    )
                return

            release_count = (
                len(waiters)
                if warming_count == 0
                else max(1, math.ceil(len(waiters) / (warming_count + 1)))
            )
            assigned = defaultdict(int)
            for _ in range(min(release_count, len(waiters))):
                follower = waiters.pop(0)
                if warming_count:
                    follower_rank = dp_rank
                else:
                    follower_rank = min(
                        ready,
                        key=lambda rank: (
                            self._worker_pressure().get(rank, 0)
                            + assigned[rank],
                            rank,
                        ),
                    )
                assigned[follower_rank] += 1
                future = follower["future"]
                if not future.done():
                    future.set_result(
                        (
                            follower_rank,
                            int(cluster_key[0]),
                            follower["wave_size"],
                            cluster_key,
                            "follower",
                            self.cache_epoch,
                            max(
                                0.0,
                                time.perf_counter() - follower["wait_started"],
                            ),
                        )
                    )
            if not waiters:
                self.rebuild_prefix_waiters.pop(cluster_key, None)

    def on_runtime_plan_update(self, plan=None):
        revised = dict(plan) if isinstance(plan, dict) else {}
        if revised.get("version") != self.rebuild_epoch:
            return
        if int(revised.get("revision", 0)) <= int(
            self.runtime_plan.get("revision", 0)
        ):
            return
        self._cancel_pending_rebuild_wave()
        self.runtime_plan = revised
        priority_candidates = (
            revised.get("priority_candidate_groups", [])
            if bool(revised.get("priority_enabled", False))
            else []
        )
        self.priority_candidate_ranks = {
            str(group_key): rank
            for rank, group_key in enumerate(priority_candidates)
        }
        self.priority_candidate_estimates = {
            str(estimate.get("group_key")): dict(estimate)
            for estimate in revised.get("priority_candidate_estimates", [])
            if estimate.get("group_key") is not None
        }
        self.rebuild_candidate_groups = set(
            str(group_key)
            for group_key in revised.get("rebuild_candidate_groups", [])
        )
        self.rebuild_candidate_trajectories = set(
            str(trajectory_id)
            for trajectory_id in revised.get(
                "rebuild_candidate_trajectories", []
            )
        )
        self.rebuild_cohort_exact = bool(
            revised.get("rebuild_cohort_exact", False)
        )
        self.rebuild_load_eligible = self._is_rebuild_load_eligible(revised)
        planned_target = int(
            revised.get("rebuild_target_trajectories", self.rebuild_target)
        )
        if not self.rebuild_load_eligible:
            planned_target = 0
        self.rebuild_target = min(
            max(0, self.post_update_rebuild_requests),
            max(0, planned_target),
        )
        assigned = sum(self.rebuild_assigned_counts.values())
        self.rebuild_remaining = max(0, self.rebuild_target - assigned)
        self.rebuild_observe_remaining = max(
            0,
            max(
                self.rebuild_target,
                max(0, self.post_update_rebuild_observe_requests),
            )
            - len(self.rebuild_seen_trajectories),
        )

    def _apply_engine_kv_feedback(self, dp_rank: int, response_metrics):
        """Feed exact engine counters back into Router's worker-local cache state."""
        metric_names = {
            "requests": "vllm/engine_prefix_cache_requests_delta",
            "query_blocks": "vllm/engine_prefix_cache_query_blocks_delta",
            "hit_blocks": "vllm/engine_prefix_cache_hit_blocks_delta",
            "cached_tokens": "vllm/engine_prefix_cache_cached_tokens_delta",
            "resets": "vllm/engine_prefix_cache_resets_delta",
        }
        observed = any(name in response_metrics for name in metric_names.values())
        if not observed:
            return False, 0, 0

        feedback = self.worker_engine_kv_feedback[dp_rank]
        for field, name in metric_names.items():
            feedback[field] += max(0, int(response_metrics.get(name, 0)))
        resets = max(
            0,
            int(response_metrics.get("vllm/engine_prefix_cache_resets_delta", 0)),
        )
        invalidated = 0
        if resets:
            self._invalidate_prefix_worker(dp_rank)
            self.working_set_worker_prompts.pop(dp_rank, None)
            self.rebuild_worker_prompts.pop(dp_rank, None)
            self.rebuild_assigned_counts.pop(dp_rank, None)
            affected = [
                key
                for key, assigned_rank in self.src_rank2_dp_rank.items()
                if assigned_rank == dp_rank
            ]
            invalidated = len(affected)
            for key in affected:
                self.src_rank2_dp_rank.pop(key, None)
                self.src_rank_cache_epoch.pop(key, None)
                self.src_rank_last_prompt_tokens.pop(key, None)
        return True, resets, invalidated

    def _completion_eta_model_ready(self) -> bool:
        return any(
            value is not None and value > 0
            for value in self.worker_service_seconds_ewma
        )

    def _update_completion_eta_model(
        self,
        dp_rank: int,
        service_seconds: float,
        prompt_tokens: int,
        response,
    ) -> None:
        if service_seconds <= 0:
            return
        alpha = self.completion_eta_ewma_alpha
        previous = self.worker_service_seconds_ewma[dp_rank]
        self.worker_service_seconds_ewma[dp_rank] = (
            service_seconds
            if previous is None
            else alpha * service_seconds + (1 - alpha) * previous
        )
        response_metrics = response.get("metrics", {})
        prefill_tokens = response_metrics.get("vllm/request_prefill_tokens")
        if prefill_tokens is None:
            cached = response_metrics.get("vllm/request_cached_prompt_tokens")
            prefill_tokens = max(
                0, int(prompt_tokens) - max(0, int(cached or 0))
            )
        output_tokens = sum(
            len(tokens) for tokens in response.get("output_token_ids", [])
        )
        model_tokens = max(0, int(prefill_tokens)) + max(0, output_tokens)
        if model_tokens <= 0:
            return
        sample = service_seconds / model_tokens
        self.model_token_seconds_ewma = (
            sample
            if self.model_token_seconds_ewma is None
            else alpha * sample
            + (1 - alpha) * self.model_token_seconds_ewma
        )

    async def generate_request(self, payload, request_id, uid, priority=None):
        src_rank = uid
        runtime_state = TrajectoryRuntimeState.from_priority(priority, src_rank)
        track_trajectory = not isinstance(priority, dict) or bool(
            priority.get("track_trajectory", True)
        )
        scheduling_enabled = not isinstance(priority, dict) or bool(
            priority.get("scheduling_enabled", True)
        )
        routing_key = runtime_state.trajectory_id if scheduling_enabled else src_rank
        epoch_trajectory_id = (
            runtime_state.trajectory_id if isinstance(priority, dict) else str(routing_key)
        )
        prompt_tokens = payload.get("input_ids", [])
        # Most requests resume an existing trajectory and use O(1) sticky
        # affinity. Build cross-trajectory prefix keys only when that affinity
        # is unavailable and the prefix directory can affect placement.
        prefix_keys = ()
        if runtime_state.group_id >= 0 and runtime_state.episode_id >= 0:
            self.latest_trajectory_progress[runtime_state.trajectory_id] = (
                build_router_progress_snapshot(runtime_state, len(prompt_tokens))
            )
            while len(self.latest_trajectory_progress) > 4096:
                self.latest_trajectory_progress.pop(next(iter(self.latest_trajectory_progress)))
        decision_started = time.perf_counter()
        rebuild_request = False
        rebuild_lcp_tokens = 0
        rebuild_candidate = (
            runtime_state.trajectory_id in self.rebuild_candidate_trajectories
            if self.rebuild_cohort_exact
            else runtime_state.group_key in self.rebuild_candidate_groups
        )
        priority_candidate = (
            runtime_state.group_key in self.priority_candidate_ranks
        )
        priority_estimate = self.priority_candidate_estimates.get(
            runtime_state.group_key, {}
        )
        route_reason = "least_loaded"
        affinity_candidate = False
        affinity_cache_valid = False
        estimated_cached_tokens = 0
        selected_pressure = 0
        predicted_completion_eta_seconds = 0.0
        predicted_queue_eta_seconds = 0.0
        predicted_prefill_tokens = 0
        completion_eta_model_ready = False
        if track_trajectory:
            first_epoch_request, rebuild_assignment = (
                await self._prepare_first_epoch_request(
                    runtime_state,
                    epoch_trajectory_id,
                    prompt_tokens,
                )
            )
        else:
            first_epoch_request, rebuild_assignment = False, None
        rebuild_wave_size = 0
        rebuild_cluster_key = None
        rebuild_role = "none"
        rebuild_assignment_epoch = self.cache_epoch
        rebuild_follower_wait_seconds = 0.0
        rebuild_seed_resolved = False
        # Atomic routing assignment under lock to prevent TOCTOU race with shrink/expand
        async with self.routing_lock:
            affinity_rank = self.src_rank2_dp_rank.get(routing_key)
            affinity_candidate = affinity_rank is not None
            affinity_cache_valid = (
                affinity_candidate
                and self.src_rank_cache_epoch.get(routing_key) == self.cache_epoch
            )
            worker_pressure = self._worker_pressure()
            completion_eta_model_ready = (
                self.completion_eta_routing_enabled
                and self._completion_eta_model_ready()
            )
            if affinity_cache_valid:
                estimated_cached_tokens = min(
                    len(prompt_tokens),
                    int(self.src_rank_last_prompt_tokens.get(routing_key, 0)),
                )

            if rebuild_assignment is not None:
                (
                    dp_rank,
                    rebuild_lcp_tokens,
                    rebuild_wave_size,
                    rebuild_cluster_key,
                    rebuild_role,
                    rebuild_assignment_epoch,
                    rebuild_follower_wait_seconds,
                ) = rebuild_assignment
                rebuild_request = True
                route_reason = f"rebuild_{rebuild_role}"
                if dp_rank != affinity_rank:
                    self.src_rank2_dp_rank[routing_key] = dp_rank
                    self.src_rank_cache_epoch.pop(routing_key, None)
                    self.src_rank_last_prompt_tokens.pop(routing_key, None)
            elif affinity_rank is None:
                if completion_eta_model_ready:
                    dp_rank, route_reason, eta_estimate = (
                        select_completion_eta_worker(
                            prompt_tokens,
                            self.working_set_worker_prompts,
                            self.active_dp_ranks,
                            worker_pressure,
                            {
                                rank: (
                                    self.worker_service_seconds_ewma[rank]
                                    or 0.0
                                )
                                for rank in self.active_dp_ranks
                            },
                            self.model_token_seconds_ewma or 0.0,
                            self.post_update_rebuild_prefix_tokens,
                            locality_slack_seconds=(
                                self.completion_eta_locality_slack_seconds
                            ),
                        )
                    )
                    estimated_cached_tokens = int(
                        eta_estimate["cached_tokens"]
                    )
                    predicted_completion_eta_seconds = float(
                        eta_estimate["eta_seconds"]
                    )
                    predicted_queue_eta_seconds = float(
                        eta_estimate["queue_eta_seconds"]
                    )
                    predicted_prefill_tokens = int(
                        eta_estimate["prefill_tokens"]
                    )
                elif self.working_set_routing_enabled and len(prompt_tokens) > 0:
                    prefix_keys = build_prefix_directory_keys(
                        prompt_tokens, self.post_update_rebuild_prefix_tokens
                    )
                    dp_rank, route_reason, estimated_cached_tokens = (
                        select_prefix_directory_worker(
                            prefix_keys,
                            self.prefix_ready_workers,
                            self.active_dp_ranks,
                            worker_pressure,
                            self.soft_locality_load_slack,
                        )
                    )
                else:
                    dp_rank, route_reason = select_soft_locality_worker(
                        None,
                        self.active_dp_ranks,
                        worker_pressure,
                        False,
                        self.soft_locality_load_slack,
                    )
                self.src_rank2_dp_rank[routing_key] = dp_rank
            elif completion_eta_model_ready:
                dp_rank, route_reason, eta_estimate = (
                    select_completion_eta_worker(
                        prompt_tokens,
                        self.working_set_worker_prompts,
                        self.active_dp_ranks,
                        worker_pressure,
                        {
                            rank: (
                                self.worker_service_seconds_ewma[rank] or 0.0
                            )
                            for rank in self.active_dp_ranks
                        },
                        self.model_token_seconds_ewma or 0.0,
                        self.post_update_rebuild_prefix_tokens,
                        affinity_rank=affinity_rank,
                        affinity_cached_tokens=estimated_cached_tokens,
                        affinity_cache_valid=affinity_cache_valid,
                        locality_slack_seconds=(
                            self.completion_eta_locality_slack_seconds
                        ),
                    )
                )
                estimated_cached_tokens = int(
                    eta_estimate["cached_tokens"]
                )
                predicted_completion_eta_seconds = float(
                    eta_estimate["eta_seconds"]
                )
                predicted_queue_eta_seconds = float(
                    eta_estimate["queue_eta_seconds"]
                )
                predicted_prefill_tokens = int(
                    eta_estimate["prefill_tokens"]
                )
                if dp_rank != affinity_rank:
                    self.src_rank2_dp_rank[routing_key] = dp_rank
                    self.src_rank_cache_epoch.pop(routing_key, None)
                    self.src_rank_last_prompt_tokens.pop(routing_key, None)
            elif self.soft_locality_enabled:
                dp_rank, route_reason = select_soft_locality_worker(
                    affinity_rank,
                    self.active_dp_ranks,
                    worker_pressure,
                    affinity_cache_valid,
                    self.soft_locality_load_slack,
                )
                if dp_rank != affinity_rank:
                    self.src_rank2_dp_rank[routing_key] = dp_rank
                    self.src_rank_cache_epoch.pop(routing_key, None)
                    self.src_rank_last_prompt_tokens.pop(routing_key, None)
                    estimated_cached_tokens = 0
            else:
                dp_rank = affinity_rank
                route_reason = "affinity"
            selected_pressure = worker_pressure.get(dp_rank, 0)

        routing_decision_seconds = time.perf_counter() - decision_started
        wait_started = time.perf_counter()
        (
            has_priority_slot,
            priority_queue_depth,
            priority_was_queued,
            priority_was_coalesced,
            priority_was_reordered,
        ) = (
            await self._acquire_priority_slot(
                dp_rank,
                priority,
                request_id,
                max_running_requests=(
                    self.priority_rebuild_max_running_requests
                    if rebuild_request
                    else None
                ),
            )
        )
        scheduling_wait_seconds = time.perf_counter() - wait_started

        self.request_id_2_src_rank[request_id] = src_rank
        self.running_requests[dp_rank].add(request_id)

        try:
            engine_request_priority = None
            worker_payload = payload
            if rebuild_request or (
                self.engine_priority_scheduling_enabled and scheduling_enabled
            ):
                worker_payload = dict(payload)
            if self.engine_priority_scheduling_enabled and scheduling_enabled:
                engine_request_priority = build_engine_request_priority(
                    runtime_state, self.priority_candidate_ranks
                )
                worker_payload["_roll_request_priority"] = engine_request_priority
            if rebuild_request:
                worker_payload["_roll_prefill_rebuild"] = True
            request_cache_epoch = self.cache_epoch
            request_boundary_version = self.rebuild_epoch
            request_boundary_resumed_at = self.boundary_resumed_at.get(
                request_cache_epoch
            )
            request_dispatched_at = time.monotonic()
            response = await self.workers[dp_rank].generate_request.remote(
                worker_payload
            )
            request_completed_at = time.monotonic()
            request_service_seconds = max(
                0.0, request_completed_at - request_dispatched_at
            )
            (
                actual_completion_seconds,
                completion_eta_absolute_error_seconds,
            ) = completion_eta_observation(
                predicted_completion_eta_seconds,
                scheduling_wait_seconds,
                request_service_seconds,
            )
            self._update_completion_eta_model(
                dp_rank,
                request_service_seconds,
                len(prompt_tokens),
                response,
            )
            response_metrics = response.setdefault("metrics", {})
            actual_prefill_tokens = response_metrics.get(
                "vllm/request_prefill_tokens"
            )
            if actual_prefill_tokens is None:
                actual_prefill_tokens = max(
                    0,
                    len(prompt_tokens)
                    - max(
                        0,
                        int(
                            response_metrics.get(
                                "vllm/request_cached_prompt_tokens", 0
                            )
                        ),
                    ),
                )
            refresh_request_record = None
            if self.refresh_profile_enabled and track_trajectory:
                refresh_request_record = build_refresh_request_record(
                    runtime_state,
                    cache_epoch=request_cache_epoch,
                    boundary_version=request_boundary_version,
                    worker_rank=dp_rank,
                    route_reason=route_reason,
                    prompt_tokens=prompt_tokens,
                    response_metrics=response_metrics,
                    request_dispatched_at=request_dispatched_at,
                    request_completed_at=request_completed_at,
                    boundary_resumed_at=request_boundary_resumed_at,
                    first_epoch_request=first_epoch_request,
                )
                self.refresh_request_records.append(refresh_request_record)
                if (
                    len(self.refresh_request_records)
                    > self.refresh_profile_max_records
                ):
                    del self.refresh_request_records[
                        : len(self.refresh_request_records)
                        - self.refresh_profile_max_records
                    ]
            (
                engine_feedback_observed,
                engine_resets_observed,
                engine_shadow_invalidations,
            ) = self._apply_engine_kv_feedback(dp_rank, response_metrics)
            if first_epoch_request and is_post_boundary_request(
                runtime_state,
                request_boundary_version,
            ):
                self.boundary_recovery_records.append(
                    build_boundary_recovery_record(
                        runtime_state,
                        cache_epoch=request_cache_epoch,
                        boundary_version=request_boundary_version,
                        worker_rank=dp_rank,
                        route_reason=route_reason,
                        prompt_tokens=len(prompt_tokens),
                        response_metrics=response_metrics,
                        boundary_resumed_at=request_boundary_resumed_at,
                        request_dispatched_at=request_dispatched_at,
                        request_completed_at=request_completed_at,
                        prefix_fingerprints=(
                            refresh_request_record.get(
                                "prefix_fingerprints", []
                            )
                            if refresh_request_record is not None
                            else (
                                build_prefix_fingerprints(prompt_tokens)
                                if self.refresh_profile_enabled
                                else []
                            )
                        ),
                    )
                )
            finish_reasons = response.get("finish_reasons", [])
            request_succeeded = not any(
                reason == "abort" for reason in finish_reasons
            )
            if request_succeeded and request_cache_epoch == self.cache_epoch:
                self.src_rank_cache_epoch[routing_key] = request_cache_epoch
                self.src_rank_last_prompt_tokens[routing_key] = len(prompt_tokens)
                if rebuild_role != "seed" and (
                    first_epoch_request
                    or not affinity_candidate
                    or route_reason == "prefix_directory"
                    or rebuild_role == "follower"
                ):
                    self._register_prefix_keys_ready(dp_rank, prompt_tokens)
                if self.completion_eta_routing_enabled:
                    cached_prompts = self.working_set_worker_prompts[dp_rank]
                    cached_prompts.append(
                        tuple(
                            int(token)
                            for token in prompt_tokens[
                                : self.post_update_rebuild_prefix_tokens
                            ]
                        )
                    )
                    max_prompts = max(1, self.working_set_max_prompts_per_worker)
                    if len(cached_prompts) > max_prompts:
                        del cached_prompts[:-max_prompts]
            if rebuild_role == "seed" and rebuild_cluster_key is not None:
                await self._complete_rebuild_seed(
                    cache_epoch=rebuild_assignment_epoch,
                    cluster_key=rebuild_cluster_key,
                    dp_rank=dp_rank,
                    prompt_tokens=prompt_tokens,
                    success=request_succeeded,
                )
                rebuild_seed_resolved = True
            response_metrics.update({
                "router/scheduling_decisions": 1,
                "router/scheduling_decision_seconds": routing_decision_seconds,
                "router/scheduling_wait_seconds": scheduling_wait_seconds,
                "router/scheduling_version_age": runtime_state.version_age,
                "router/scheduling_actions_completed": runtime_state.actions_completed,
                "router/scheduling_remaining_actions": runtime_state.remaining_actions,
                "router/affinity_candidate": int(affinity_candidate),
                "router/affinity_cache_valid": int(affinity_cache_valid),
                "router/affinity_selected": int(route_reason == "affinity"),
                "router/load_override": int(
                    route_reason in ("load_override", "prefix_load_override")
                ),
                "router/least_loaded_selected": int(route_reason == "least_loaded"),
                "router/working_set_prefix_selected": int(
                    route_reason in ("prefix_locality", "prefix_directory")
                ),
                "router/rebuild_selected": int(
                    route_reason.startswith("rebuild_")
                ),
                "router/rebuild_seed_request": int(rebuild_role == "seed"),
                "router/rebuild_follower_request": int(
                    rebuild_role == "follower"
                ),
                "router/rebuild_follower_wait_seconds": (
                    rebuild_follower_wait_seconds
                ),
                "router/rebuild_candidate_request": int(rebuild_candidate),
                "router/rebuild_load_eligible": int(
                    self.rebuild_load_eligible
                ),
                "router/planned_priority_candidate_request": int(
                    priority_candidate
                ),
                "router/planned_completion_probability": float(
                    priority_estimate.get("completion_probability", 0.0)
                ),
                "router/planned_completion_eta_seconds": float(
                    priority_estimate.get("eta_seconds", 0.0)
                ),
                "router/planned_completion_laxity_seconds": float(
                    priority_estimate.get("laxity_seconds", 0.0)
                ),
                "router/planned_completion_feasible": int(
                    bool(priority_estimate.get("feasible", False))
                ),
                "router/version_runtime_plan_request": int(bool(self.runtime_plan)),
                "router/priority_queued_requests": int(priority_was_queued),
                "router/priority_coalesced_requests": int(
                    priority_was_coalesced
                ),
                "router/priority_reordered_requests": int(
                    priority_was_reordered
                ),
                "router/priority_queue_depth": priority_queue_depth,
                "router/priority_slot_capacity": (
                    0
                    if self.engine_priority_scheduling_enabled
                    else (
                        self.priority_rebuild_max_running_requests
                        if rebuild_request
                        else self.priority_max_running_requests
                    )
                ),
                "router/engine_priority_scheduling_enabled": int(
                    self.engine_priority_scheduling_enabled
                ),
                "router/engine_priority_request": int(
                    engine_request_priority is not None
                ),
                "router/engine_request_priority": float(
                    engine_request_priority or 0
                ),
                "router/router_priority_gate_used": int(has_priority_slot),
                "router/rebuild_burst_request": int(
                    rebuild_request
                    and not self.engine_priority_scheduling_enabled
                    and self.priority_rebuild_max_running_requests
                    > self.priority_max_running_requests
                ),
                "router/selected_worker_pressure": selected_pressure,
                "router/soft_locality_estimated_cached_tokens": estimated_cached_tokens,
                "router/completion_eta_model_ready": int(
                    completion_eta_model_ready
                ),
                "router/predicted_completion_eta_seconds": (
                    predicted_completion_eta_seconds
                ),
                "router/predicted_queue_eta_seconds": (
                    predicted_queue_eta_seconds
                ),
                "router/predicted_prefill_tokens": predicted_prefill_tokens,
                "router/actual_prefill_tokens": actual_prefill_tokens,
                "router/prefill_prediction_absolute_error_tokens": (
                    abs(actual_prefill_tokens - predicted_prefill_tokens)
                    if completion_eta_model_ready
                    else 0
                ),
                "router/request_service_seconds": request_service_seconds,
                "router/actual_completion_seconds": actual_completion_seconds,
                "router/completion_eta_absolute_error_seconds": (
                    completion_eta_absolute_error_seconds
                ),
                "router/completion_eta_selected": int(
                    route_reason.startswith("completion_eta_")
                ),
                "router/engine_kv_feedback_requests": int(
                    engine_feedback_observed
                ),
                "router/engine_kv_resets_observed": engine_resets_observed,
                "router/engine_kv_shadow_invalidations": (
                    engine_shadow_invalidations
                ),
            })
            response["runtime_attribution"] = {
                "plan_id": str(self.runtime_plan.get("plan_id", "")),
                "forecast_id": str(
                    self.runtime_plan.get("forecast_id", "")
                ),
                "version": self.runtime_plan.get("version"),
                "revision": int(self.runtime_plan.get("revision", 0)),
                "estimator_revision": int(
                    self.runtime_plan.get("estimator_revision", 0)
                ),
                "group_key": runtime_state.group_key,
                "trajectory_id": runtime_state.trajectory_id,
            }
            if rebuild_request:
                response_metrics.update({
                    "router/post_update_rebuild_request": 1,
                    "router/post_update_rebuild_lcp_tokens": rebuild_lcp_tokens,
                    "router/post_update_rebuild_dp_rank": dp_rank,
                    "router/post_update_rebuild_wave_size": rebuild_wave_size,
                    "router/post_update_rebuild_coalesced": int(
                        rebuild_wave_size > 1
                    ),
                })
            return response
        finally:
            self.running_requests[dp_rank].remove(request_id)
            if (
                rebuild_role == "seed"
                and rebuild_cluster_key is not None
                and not rebuild_seed_resolved
            ):
                await self._complete_rebuild_seed(
                    cache_epoch=rebuild_assignment_epoch,
                    cluster_key=rebuild_cluster_key,
                    dp_rank=dp_rank,
                    prompt_tokens=prompt_tokens,
                    success=False,
                )
            # Cleanup tracking (on both success and abort paths)
            self.request_id_2_src_rank.pop(request_id, None)
            if has_priority_slot:
                await self._release_priority_slot(dp_rank)

    def collect_version_boundary_profile(self):
        records = list(self.boundary_recovery_records)
        request_records = list(self.refresh_request_records)
        engine_batches = defaultdict(int)
        for record in records:
            batch_id = record.get("engine_scheduler_batch_id")
            if batch_id is None:
                continue
            engine_batches[
                (
                    int(record["cache_epoch"]),
                    int(record["worker_rank"]),
                    int(batch_id),
                )
            ] += 1
        return {
            "metrics": {
                "survivor_first_requests": len(records),
                "logical_prompt_tokens": sum(
                    int(record["logical_prompt_tokens"]) for record in records
                ),
                "logical_reprefill_exposure_tokens": sum(
                    int(record["logical_reprefill_exposure_tokens"])
                    for record in records
                ),
                "engine_reported_records": sum(
                    record["reported_prefill_tokens"] is not None for record in records
                ),
                "engine_reported_prefill_tokens": sum(
                    int(record["reported_prefill_tokens"] or 0) for record in records
                ),
                "engine_scheduler_batch_records": sum(engine_batches.values()),
                "engine_scheduler_batches": len(engine_batches),
                "engine_batches_with_multiple_survivors": sum(
                    count > 1 for count in engine_batches.values()
                ),
                "engine_cobatched_survivor_requests": sum(
                    count for count in engine_batches.values() if count > 1
                ),
                "engine_scheduler_batch_size_max": max(
                    (
                        int(record.get("engine_scheduler_batch_size") or 0)
                        for record in records
                    ),
                    default=0,
                ),
                "request_queue_seconds": sum(
                    float(record.get("request_queue_seconds", 0.0))
                    for record in records
                ),
                "request_ttft_seconds": sum(
                    float(record.get("request_ttft_seconds", 0.0))
                    for record in records
                ),
                "request_prefill_seconds": sum(
                    float(record.get("request_prefill_seconds", 0.0))
                    for record in records
                ),
                "request_decode_seconds": sum(
                    float(record.get("request_decode_seconds", 0.0))
                    for record in records
                ),
                "request_inference_seconds": sum(
                    float(record.get("request_inference_seconds", 0.0))
                    for record in records
                ),
                "request_model_forward_seconds": sum(
                    float(record.get("request_model_forward_seconds", 0.0))
                    for record in records
                ),
                "request_model_execute_seconds": sum(
                    float(record.get("request_model_execute_seconds", 0.0))
                    for record in records
                ),
                "request_engine_step_seconds_attributed": sum(
                    float(
                        record.get(
                            "request_engine_step_seconds_attributed", 0.0
                        )
                    )
                    for record in records
                ),
                "request_prefill_engine_step_seconds_attributed": sum(
                    float(
                        record.get(
                            "request_prefill_engine_step_seconds_attributed",
                            0.0,
                        )
                    )
                    for record in records
                ),
                "request_decode_engine_step_seconds_attributed": sum(
                    float(
                        record.get(
                            "request_decode_engine_step_seconds_attributed",
                            0.0,
                        )
                    )
                    for record in records
                ),
                "recovery_finish_seconds_max": max(
                    (
                        float(record.get("finish_after_boundary_seconds", 0.0))
                        for record in records
                    ),
                    default=0.0,
                ),
                "refresh_profile_requests": len(request_records),
                "refresh_profile_decode_tokens": sum(
                    int(record.get("decode_tokens", 0))
                    for record in request_records
                ),
                "refresh_profile_prefill_tokens": sum(
                    int(record.get("prefill_tokens", 0))
                    for record in request_records
                ),
            },
            "records": records,
            "request_records": request_records,
            "refresh_profile_enabled": self.refresh_profile_enabled,
        }

    def collect_trajectory_progress(self):
        return list(self.latest_trajectory_progress.values())

    def collect_runtime_feedback(self):
        totals = {
            field: sum(int(worker.get(field, 0)) for worker in self.worker_engine_kv_feedback)
            for field in ("requests", "query_blocks", "hit_blocks", "cached_tokens", "resets")
        }
        totals["hit_ratio"] = (
            totals["hit_blocks"] / totals["query_blocks"]
            if totals["query_blocks"]
            else 0.0
        )
        totals["workers"] = [dict(worker) for worker in self.worker_engine_kv_feedback]
        running_by_worker = [
            len(requests) for requests in self.running_requests
        ]
        queued_by_worker = [
            len(waiters) for waiters in self.priority_waiters
        ]
        totals.update(
            worker_count=len(self.workers),
            busy_workers=sum(
                int(running > 0 or queued > 0)
                for running, queued in zip(
                    running_by_worker, queued_by_worker
                )
            ),
            running_requests=sum(running_by_worker),
            queued_requests=sum(queued_by_worker),
            running_requests_by_worker=running_by_worker,
            queued_requests_by_worker=queued_by_worker,
        )
        return totals

    async def _acquire_priority_slot(
        self,
        dp_rank: int,
        priority,
        request_id: str,
        max_running_requests: Optional[int] = None,
    ):
        if (
            self.engine_priority_scheduling_enabled
            or priority is None
            or self.priority_max_running_requests <= 0
            or (
                isinstance(priority, dict)
                and not bool(priority.get("scheduling_enabled", True))
            )
        ):
            return False, 0, False, False, False
        condition = self.priority_conditions[dp_rank]
        capacity = (
            self.priority_max_running_requests
            if max_running_requests is None
            else max(
                self.priority_max_running_requests,
                min(self.max_running_requests, int(max_running_requests)),
            )
        )
        runtime_state = TrajectoryRuntimeState.from_priority(priority, request_id)
        runtime_priority_key = build_runtime_priority_key(
            runtime_state, self.priority_candidate_ranks
        )
        entry = (runtime_priority_key, next(self.priority_sequence), request_id)
        async with condition:
            queue_depth = len(self.priority_waiters[dp_rank])
            capacity_full = self.priority_inflight[dp_rank] >= capacity
            should_coalesce = self.priority_coalesce_seconds > 0 and (
                queue_depth > 0 or capacity_full
            )
            if queue_depth == 0:
                self.priority_batch_deadline[dp_rank] = (
                    asyncio.get_running_loop().time()
                    + (
                        self.priority_coalesce_seconds
                        if should_coalesce
                        else 0.0
                    )
                )
            was_queued = (
                queue_depth > 0
                or capacity_full
            )
            was_coalesced = should_coalesce
            heapq.heappush(self.priority_waiters[dp_rank], entry)
            try:
                while True:
                    is_head = self.priority_waiters[dp_rank][0] == entry
                    has_capacity = (
                        self.priority_inflight[dp_rank] < capacity
                    )
                    delay = (
                        self.priority_batch_deadline[dp_rank]
                        - asyncio.get_running_loop().time()
                    )
                    if is_head and has_capacity and delay <= 0:
                        break
                    if is_head and has_capacity:
                        try:
                            await asyncio.wait_for(condition.wait(), timeout=delay)
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await condition.wait()
                was_reordered = any(
                    other[1] < entry[1]
                    for other in self.priority_waiters[dp_rank]
                    if other != entry
                )
                heapq.heappop(self.priority_waiters[dp_rank])
                self.priority_inflight[dp_rank] += 1
                condition.notify_all()
                return (
                    True,
                    queue_depth,
                    was_queued,
                    was_coalesced,
                    was_reordered,
                )
            except BaseException:
                try:
                    self.priority_waiters[dp_rank].remove(entry)
                    heapq.heapify(self.priority_waiters[dp_rank])
                except ValueError:
                    pass
                condition.notify_all()
                raise

    async def _release_priority_slot(self, dp_rank: int):
        condition = self.priority_conditions[dp_rank]
        async with condition:
            self.priority_inflight[dp_rank] -= 1
            assert self.priority_inflight[dp_rank] >= 0
            condition.notify_all()

    async def abort_requests(self, request_ids, uid):
        raise NotImplementedError

    async def abort_all(self, request_ids):
        await asyncio.gather(*(
            self.workers[dp_rank].abort_requests.remote(list(self.running_requests[dp_rank]))
            for dp_rank in range(len(self.workers))
            if self.running_requests[dp_rank]
        ))

    def _get_least_active_dp_rank(self) -> int:
        """Find DP rank with fewest assigned src_ranks (environments).

        Returns:
            DP rank with minimum src_rank count from src_rank2_dp_rank

        Raises:
            RuntimeError: If no active ranks

        Note:
            Counts unique src_ranks (environments) per worker, not in-flight requests.
            With sticky mapping, one src_rank generates multiple sequential requests.
        """
        candidate_ranks = list(self.active_dp_ranks)
        if not candidate_ranks:
            raise RuntimeError("No active DP ranks")
        # todo optimization: (yangpeng) not efficient, better to use counter for this
        # Count src_ranks per dp_rank
        src_rank_count = defaultdict(int)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in self.active_dp_ranks:
                src_rank_count[dp_rank] += 1

        # Return dp_rank with minimum src_rank count
        return min(candidate_ranks, key=lambda r: src_rank_count[r])

    def _worker_pressure(self) -> Dict[int, int]:
        return {
            dp_rank: len(self.running_requests[dp_rank]) + len(self.priority_waiters[dp_rank])
            for dp_rank in self.active_dp_ranks
        }

    def _clear_src_rank_mappings(self, src_ranks: Set[int]) -> None:
        """Clear sticky mappings to allow re-routing on retry."""
        for src_rank in src_ranks:
            self.src_rank2_dp_rank.pop(src_rank, None)

    async def rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (abort + update active_dp_ranks)
            return await self.rebalance_on_shrink_impl(shrink_dp_ranks)

    async def rebalance_on_shrink_impl(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Abort requests on shrinking workers, clear mappings for natural re-dispatch.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If shrink_dp_ranks empty/invalid/duplicates
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not shrink_dp_ranks:
            raise ValueError("shrink_dp_ranks cannot be empty")

        for rank in shrink_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < len(self.workers)):
                raise ValueError(f"rank {rank} out of range")

        if len(shrink_dp_ranks) != len(set(shrink_dp_ranks)):
            raise ValueError(f"Duplicates in shrink_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_shrink(shrink_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_shrink timed out after 30s")

    async def _rebalance_on_shrink(self, shrink_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of shrink rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (shrink_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Args:
            shrink_dp_ranks: DP ranks to remove from active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            RuntimeError: If shrink operation fails
        """
        keep_ranks = list(self.active_dp_ranks - set(shrink_dp_ranks))
        if not keep_ranks:
            raise ValueError("Cannot shrink to zero active ranks")

        old_active_ranks = self.active_dp_ranks.copy()
        self.active_dp_ranks = set(keep_ranks)

        try:
            total_aborted = 0
            abort_futures = []

            for dp_rank in shrink_dp_ranks:
                request_ids = list(self.running_requests[dp_rank])
                if not request_ids:
                    continue

                total_aborted += len(request_ids)

                abort_futures.append(
                    self.workers[dp_rank].abort_requests.remote(request_ids)
                )

            await asyncio.gather(*abort_futures)

            while True:
                remain = sum(len(self.running_requests[dp_rank]) for dp_rank in shrink_dp_ranks)
                if remain == 0:
                    break
                logger.info(f"Shrink: waiting for {len(shrink_dp_ranks)} workers {remain=} to finish abort")
                await asyncio.sleep(3)

            # Clear ALL mappings pointing to shrinking workers (not just in-flight)
            shrink_dp_ranks_set = set(shrink_dp_ranks)
            src_ranks_to_remap = set([
                src_rank for src_rank, dp_rank in self.src_rank2_dp_rank.items()
                if dp_rank in shrink_dp_ranks_set
            ])
            self._clear_src_rank_mappings(src_ranks_to_remap)

            logger.info(
                f"Shrink: aborted {total_aborted} requests, "
                f"cleared {len(src_ranks_to_remap)} mappings"
            )

            return {"aborted": total_aborted, "remapped": len(src_ranks_to_remap)}

        except Exception as e:
            self.active_dp_ranks = old_active_ranks
            raise RuntimeError(f"Shrink failed: {e}") from e

    async def rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        # Atomic operation under routing_lock
        async with self.routing_lock:
            # Rebalance (update active_dp_ranks + conditional abort)
            return await self.rebalance_on_expand_impl(expand_dp_ranks)

    async def rebalance_on_expand_impl(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Add workers and rebalance via src_rank-level abort.

        Args:
            expand_dp_ranks: DP ranks to add to active set

        Returns:
            {"aborted": count, "remapped": count}

        Raises:
            ValueError: If expand_dp_ranks invalid
            RuntimeError: If timeout or operation fails
        """
        # VAL: VAL_NON_EMPTY, VAL_TYPE_CHECK, VAL_INT_RANGE, VAL_NO_DUPLICATES
        if not expand_dp_ranks:
            raise ValueError("expand_dp_ranks cannot be empty")
        for rank in expand_dp_ranks:
            if not isinstance(rank, int):
                raise TypeError(f"Expected int, got {type(rank)}")
            if not (0 <= rank < len(self.workers)):
                raise ValueError(f"rank {rank} out of range")
        if len(expand_dp_ranks) != len(set(expand_dp_ranks)):
            raise ValueError(f"Duplicates in expand_dp_ranks")

        # P0: LOCK_TIMEOUT
        try:
            return await asyncio.wait_for(
                self._rebalance_on_expand(expand_dp_ranks),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("rebalance_on_expand timed out after 30s")

    async def _rebalance_on_expand(self, expand_dp_ranks: List[int]) -> Dict[str, int]:
        """Internal implementation of expand rebalancing.

        PRE-CONDITION: routing_lock MUST be held by caller (expand_workers).
        This method does NOT acquire the lock internally to avoid double-lock deadlock.

        Algorithm: Round-robin selection across old workers
        1. Calculate proportional src_ranks to abort: src_ranks_to_keep = ceil(total * old_count / new_count)
        2. Group existing src_ranks by dp_rank (only old workers)
        3. Round-robin iterate over old workers using cycle()
        4. Select one src_rank at a time until remaining_to_abort reaches 0
        5. Abort ALL requests from selected src_ranks
        6. Clear src_rank mappings for reallocation to new workers

        Implementation Notes:
        - Uses cycle() for infinite round-robin iteration over old workers
        - Check at line 1146 (if not dp_rank in old_active_dp_ranks) is redundant
          since dp_rank_to_src_ranks already contains only old workers, but kept as defensive guard
        - Loop terminates when remaining_to_abort <= 0 or all worker lists are exhausted
        - If all workers exhausted before reaching target, loop may cycle indefinitely
          (no explicit check for empty state, but pop(0) will eventually empty all lists)

        Args:
            expand_dp_ranks: DP ranks to add to active set (already validated)

        Returns:
            {"aborted": count, "remapped": count} - count of src_ranks aborted/remapped

        Preconditions:
            - routing_lock MUST be held by caller
            - expand_dp_ranks validated (non-empty, int, in range, no duplicates)

        Postconditions:
            - active_dp_ranks updated with expand_dp_ranks
            - Selected src_ranks aborted and removed from mappings
            - Requests from aborted src_ranks reported as is_abort=True
        """
        # Calculate counts before updating active_dp_ranks
        old_dp_count = len(self.active_dp_ranks)
        old_active_dp_ranks = self.active_dp_ranks.copy()

        self.active_dp_ranks.update(expand_dp_ranks)
        new_dp_count = len(self.active_dp_ranks)

        total_src_ranks = len(self.src_rank2_dp_rank)
        if total_src_ranks == 0:
            return {"aborted": 0, "remapped": 0}

        # Proportional calculation
        src_ranks_to_keep = math.ceil(int(total_src_ranks * old_dp_count / new_dp_count))
        src_ranks_to_abort = total_src_ranks - src_ranks_to_keep

        if src_ranks_to_abort <= 0:
            logger.info("Expand: no rebalancing needed (src_ranks_to_abort <= 0)")
            return {"aborted": 0, "remapped": 0}

        # Group src_ranks by dp_rank (old workers only)
        dp_rank_to_src_ranks = defaultdict(list)
        for src_rank, dp_rank in self.src_rank2_dp_rank.items():
            if dp_rank in old_active_dp_ranks:
                dp_rank_to_src_ranks[dp_rank].append(src_rank)

        # Round-robin selection: iterate over old workers and select one src_rank at a time
        # todo optimization:(yangpeng) take uneven dp load into consideration and do dynamic load balancing, not just RR
        selected_src_ranks = []
        remaining_to_abort = src_ranks_to_abort
        for dp_rank in itertools.cycle(dp_rank_to_src_ranks.keys()):
            if not dp_rank in old_active_dp_ranks:
                continue

            if remaining_to_abort <= 0:
                break

            src_ranks_on_worker = dp_rank_to_src_ranks.get(dp_rank, [])
            if not src_ranks_on_worker:
                continue
            selected_src_ranks.append(src_ranks_on_worker.pop(0))

            remaining_to_abort -= 1

        # Remove from mapping and group by dp_rank for abort
        abort_by_dp_rank = defaultdict(list)
        for src_rank in selected_src_ranks:
            dp_rank = self.src_rank2_dp_rank.pop(src_rank)

            # Find request_id(s) for this src_rank
            for request_id, sr in self.request_id_2_src_rank.items():
                if sr == src_rank:
                    abort_by_dp_rank[dp_rank].append(request_id)

        # Send batched ABORT commands
        abort_futures = []
        total_aborted = 0
        for dp_rank, request_ids in abort_by_dp_rank.items():
            if not request_ids:
                continue

            total_aborted += len(request_ids)
            abort_futures.append(
                self.workers[dp_rank].abort_requests.remote(request_ids)
            )


        await asyncio.gather(*abort_futures)

        logger.info(
            f"Expand: aborted {len(selected_src_ranks)} src_ranks, "
            f"cleared {len(selected_src_ranks)} mappings "
            f"(proportional: {old_dp_count}/{new_dp_count})"
        )

        return {"aborted": len(selected_src_ranks), "remapped": len(selected_src_ranks)}
