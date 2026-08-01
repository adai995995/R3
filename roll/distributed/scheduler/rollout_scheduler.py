import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field, replace
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


def compute_dynamic_reserve(
    reserve: int,
    version: int,
    learner_wait_ewma: float,
    stale_ewma: float,
    prediction_error_ewma: float,
    *,
    reserve_min: int,
    reserve_max: int,
    additive_step: int,
    multiplicative_decay: float,
    warmup_versions: int,
    wait_high: float,
    stale_high: float,
    prediction_error_margin: float,
) -> Tuple[int, int]:
    """Return the next reserve and a numeric reason code for metrics."""
    if version < warmup_versions:
        return reserve, 4  # warmup

    step = max(1, additive_step)
    if stale_ewma > stale_high:
        decayed = math.floor((reserve * multiplicative_decay) / step) * step
        return max(reserve_min, decayed), 2  # stale pressure
    if prediction_error_ewma < -prediction_error_margin:
        decayed = math.floor((reserve * multiplicative_decay) / step) * step
        return max(reserve_min, decayed), 3  # supply over-prediction
    if learner_wait_ewma > wait_high:
        return min(reserve_max, reserve + step), 1  # learner starvation
    return reserve, 0  # deadband hold


def compute_closed_loop_reserve(
    reserve: int,
    version: int,
    learner_wait_ewma: float,
    overload_ewma: float,
    *,
    reserve_min: int,
    reserve_max: int,
    additive_step: int,
    multiplicative_decay: float,
    warmup_versions: int,
    wait_high: float,
    overload_high: float,
) -> Tuple[int, int]:
    """Adjust reserve from actuator errors, not supply-model residuals.

    Supply prediction error is estimator supervision.  It is intentionally
    absent here because the same negative residual can mean either transient
    under-supply or completion degradation caused by an overloaded pool.
    """
    if version < warmup_versions:
        return reserve, 4  # warmup

    step = max(1, additive_step)
    if overload_ewma > overload_high:
        decayed = math.floor((reserve * multiplicative_decay) / step) * step
        return max(reserve_min, decayed), 2  # overload / expired work
    if learner_wait_ewma > wait_high:
        return min(reserve_max, reserve + step), 1  # measured under-supply
    return reserve, 0  # deadband hold


DYNAMIC_RESERVE_REASON_NAMES = {
    0: "hold",
    1: "learner_starvation",
    2: "expired_waste",
    3: "supply_overprediction",
    4: "warmup",
    5: "disabled",
    6: "hysteresis_or_cooldown",
    7: "gpu_queue_conflict",
    8: "starvation_waste_conflict",
    9: "tool_wait_diversity",
    10: "gpu_queue_pressure",
}


def compute_state_feedback_reserve(
    reserve: int,
    version: int,
    starvation_fraction: float,
    waste_fraction: float,
    queue_pressure: float,
    inference_ready: int,
    tool_wait_fraction: float,
    learner_demand: int,
    *,
    reserve_min: int,
    reserve_max: int,
    additive_step: int,
    warmup_versions: int,
    starvation_high: float,
    waste_high: float,
    queue_high: float,
    tool_wait_high: float,
) -> Tuple[int, int]:
    """Choose a reserve direction from starvation and overload together.

    The controller changes only admission supply.  When learner starvation is
    accompanied by a busy inference queue, adding work cannot resolve the
    bottleneck, so the decision is held for the scheduler instead.
    """
    if version < warmup_versions:
        return reserve, 4

    step = max(1, int(additive_step))
    starvation = max(0.0, float(starvation_fraction))
    waste = max(0.0, float(waste_fraction))
    queue = max(0.0, float(queue_pressure))
    tool_wait = max(0.0, float(tool_wait_fraction))
    wait_high = starvation > max(0.0, float(starvation_high))
    wait_low = starvation <= max(0.0, float(starvation_high)) * 0.5
    waste_is_high = waste > max(0.0, float(waste_high))
    queue_is_high = queue > max(0.0, float(queue_high))
    ready_is_low = max(0, int(inference_ready)) < max(1, int(learner_demand))

    if wait_high and waste_is_high:
        return reserve, 8
    if wait_high and queue_is_high:
        return reserve, 7
    if wait_high and ready_is_low:
        reason = 9 if tool_wait > max(0.0, float(tool_wait_high)) else 1
        return min(reserve_max, reserve + step), reason
    if wait_low and waste_is_high:
        return max(reserve_min, reserve - step), 2
    if wait_low and queue_is_high:
        return max(reserve_min, reserve - step), 10
    return reserve, 0


def apply_dynamic_reserve_hysteresis(
    reserve: int,
    candidate_reserve: int,
    candidate_reason: int,
    pending_direction: int,
    pending_count: int,
    cooldown_remaining: int,
    *,
    signal_patience: int,
    cooldown_versions: int,
) -> Tuple[int, int, int, int, int]:
    """Confirm persistent pressure and prevent consecutive reserve changes."""
    if candidate_reason in (4, 5):
        return reserve, candidate_reason, 0, 0, cooldown_remaining
    if cooldown_remaining > 0:
        return reserve, 6, 0, 0, cooldown_remaining - 1

    direction = (candidate_reserve > reserve) - (candidate_reserve < reserve)
    if direction == 0:
        return reserve, candidate_reason, 0, 0, 0

    confirmed = pending_count + 1 if direction == pending_direction else 1
    if confirmed < max(1, signal_patience):
        return reserve, 6, direction, confirmed, 0
    return candidate_reserve, candidate_reason, 0, 0, max(0, cooldown_versions)


def compute_effective_rollout_utility(
    consumed_response_tokens: int,
    consumed_inference_tokens: int,
    stale_inference_tokens: int,
    elapsed_seconds: float,
    waste_weight: float,
) -> Tuple[float, float, float, float]:
    """Return useful output rate discounted by the fraction of wasted inference work."""
    elapsed = max(float(elapsed_seconds), 1e-6)
    response_rate = max(0, int(consumed_response_tokens)) / elapsed
    consumed_work = max(0, int(consumed_inference_tokens))
    stale_work = max(0, int(stale_inference_tokens))
    weighted_work = consumed_work + max(0.0, waste_weight) * stale_work
    compute_efficiency = consumed_work / weighted_work if weighted_work > 0 else 1.0
    stale_rate = stale_work / elapsed
    return response_rate * compute_efficiency, response_rate, stale_rate, compute_efficiency


def compute_stale_control_signal(
    stale_tokens: int,
    consumed_tokens: int,
    stale_trajectories: int,
    consumed_trajectories: int,
) -> Tuple[Optional[float], float]:
    """Keep compute-waste control separate from trajectory-count diagnostics."""
    token_denominator = max(0, int(stale_tokens)) + max(
        0, int(consumed_tokens)
    )
    token_fraction = (
        max(0, int(stale_tokens)) / token_denominator
        if token_denominator > 0
        else None
    )
    trajectory_denominator = max(0, int(stale_trajectories)) + max(
        0, int(consumed_trajectories)
    )
    trajectory_fraction = (
        max(0, int(stale_trajectories)) / trajectory_denominator
        if trajectory_denominator > 0
        else 0.0
    )
    return token_fraction, trajectory_fraction


def update_utility_hill_climb(
    reserve: int,
    direction: int,
    utility: float,
    previous_utility: Optional[float],
    *,
    reserve_min: int,
    reserve_max: int,
    additive_step: int,
    improvement_margin: float,
) -> Tuple[int, int, int]:
    """Move one reserve step using windowed perturb-and-observe feedback."""
    direction = 1 if direction >= 0 else -1
    if previous_utility is None:
        reason = 7  # establish baseline, then probe
    else:
        relative_delta = (utility - previous_utility) / max(abs(previous_utility), 1e-6)
        if relative_delta < -max(0.0, improvement_margin):
            direction *= -1
            reason = 9  # utility regressed, reverse
        elif relative_delta > max(0.0, improvement_margin):
            reason = 8  # utility improved, continue
        else:
            reason = 10  # utility deadband, continue bounded probe

    step = max(1, additive_step)
    candidate = reserve + direction * step
    if candidate < reserve_min or candidate > reserve_max:
        direction *= -1
        candidate = reserve + direction * step
    return min(reserve_max, max(reserve_min, candidate)), direction, reason


def update_constrained_utility_hill_climb(
    reserve: int,
    direction: int,
    utility: float,
    previous_utility: Optional[float],
    compute_efficiency: float,
    *,
    min_compute_efficiency: float,
    reserve_min: int,
    reserve_max: int,
    additive_step: int,
    improvement_margin: float,
) -> Tuple[int, int, int]:
    """Optimize useful throughput while treating compute efficiency as a hard constraint."""
    if compute_efficiency < min_compute_efficiency:
        step = max(1, additive_step)
        return max(reserve_min, reserve - step), -1, 11
    return update_utility_hill_climb(
        reserve,
        direction,
        utility,
        previous_utility,
        reserve_min=reserve_min,
        reserve_max=reserve_max,
        additive_step=additive_step,
        improvement_margin=improvement_margin,
    )


def compute_progress_topup_groups(
    missing_trajectories: int,
    valid_potential: int,
    outstanding_trajectories: int,
    max_outstanding_trajectories: int,
    group_size: int,
    admission_width: int,
) -> int:
    """Return the minimum bounded group count needed to preserve learner progress."""
    deficit = max(0, missing_trajectories - valid_potential)
    needed = math.ceil(deficit / max(1, group_size))
    available = max(
        0,
        (max_outstanding_trajectories - outstanding_trajectories)
        // max(1, admission_width),
    )
    return min(needed, available)


FINISH_RATE_AGE_BUCKETS = ("age_0", "age_1", "age_2", "age_3", "age_ge_4")
FINISH_RATE_PROGRESS_BUCKETS = (
    "actions_0",
    "actions_1",
    "actions_2_3",
    "actions_4_7",
    "actions_ge_8",
)
FINISH_RATE_BUCKET_KEYS = tuple(
    f"{age}__{progress}"
    for age in FINISH_RATE_AGE_BUCKETS
    for progress in FINISH_RATE_PROGRESS_BUCKETS
)

VERSION_RUNTIME_ADMISSION_REASON_CODES = {
    "disabled": 0,
    "existing_supply_sufficient": 1,
    "supply_deficit": 2,
    "partial_capacity": 3,
    "outstanding_cap": 4,
    "progress_reconcile": 5,
}


def finish_rate_bucket(version_age: int, actions_completed: int) -> str:
    """Map a carry-over group to a stable version-age and progress bucket."""
    age = max(0, int(version_age))
    if age >= 4:
        age_bucket = "age_ge_4"
    else:
        age_bucket = f"age_{age}"

    actions = max(0, int(actions_completed))
    if actions >= 8:
        progress_bucket = "actions_ge_8"
    elif actions >= 4:
        progress_bucket = "actions_4_7"
    elif actions >= 2:
        progress_bucket = "actions_2_3"
    else:
        progress_bucket = f"actions_{actions}"
    return f"{age_bucket}__{progress_bucket}"


def runtime_finish_rate_bucket(
    version_age: int,
    actions_completed: int,
    runtime_summary: Dict[str, Any],
) -> str:
    """Add readiness to the stable age/progress completion bucket."""
    base = finish_rate_bucket(version_age, actions_completed)
    inference_ready = max(0, int(runtime_summary.get("inference_ready", 0)))
    tool_waiting = max(0, int(runtime_summary.get("tool_waiting", 0)))
    if inference_ready and tool_waiting:
        readiness = "mixed"
    elif inference_ready:
        readiness = "ready"
    elif tool_waiting:
        readiness = "tool_wait"
    else:
        readiness = "other"
    return f"{base}__readiness_{readiness}"


def coarse_finish_rate_bucket(bucket: str) -> str:
    """Return the age/progress parent used when a readiness bucket is cold."""
    return str(bucket).split("__readiness_", 1)[0]


def update_bucketed_finish_ratios(
    ratios: Dict[str, float],
    sample_counts: Dict[str, int],
    cohort_counts: Dict[str, int],
    completed_counts: Dict[str, int],
    ewma_alpha: float,
) -> None:
    """Update per-cohort completion EWMAs from one policy-version window."""
    alpha = min(1.0, max(0.0, float(ewma_alpha)))
    for bucket, cohort_count in cohort_counts.items():
        cohort_count = max(0, int(cohort_count))
        if cohort_count == 0:
            continue
        completed = min(cohort_count, max(0, int(completed_counts.get(bucket, 0))))
        observed_ratio = completed / cohort_count
        previous = ratios.get(bucket)
        ratios[bucket] = (
            observed_ratio
            if previous is None
            else alpha * observed_ratio + (1 - alpha) * previous
        )
        sample_counts[bucket] = sample_counts.get(bucket, 0) + cohort_count


def predict_bucketed_finish_supply(
    cohort_counts: Dict[str, int],
    ratios: Dict[str, float],
    sample_counts: Dict[str, int],
    fallback_ratio: float,
    min_bucket_samples: int,
) -> Tuple[float, int, int]:
    """Predict completions and report learned-bucket versus fallback population."""
    expected = 0.0
    learned_population = 0
    fallback_population = 0
    threshold = max(1, int(min_bucket_samples))
    for bucket, count in cohort_counts.items():
        count = max(0, int(count))
        if sample_counts.get(bucket, 0) >= threshold and bucket in ratios:
            ratio = ratios[bucket]
            learned_population += count
        else:
            ratio = fallback_ratio
            fallback_population += count
        expected += count * min(1.0, max(0.0, float(ratio)))
    return expected, learned_population, fallback_population


def version_runtime_plan_id(version: int, revision: int) -> str:
    return f"version-{int(version)}-revision-{max(0, int(revision))}"


def version_runtime_forecast_id(version: int, estimator_revision: int) -> str:
    return (
        f"version-{int(version)}-estimator-"
        f"{max(0, int(estimator_revision))}"
    )


@dataclass(frozen=True)
class VersionRuntimeForecast:
    """Supply forecast captured before one policy-version decision."""

    version: int
    forecast_id: str
    estimator_revision: int
    ready_valid_slots: int
    predicted_inflight_slots: float
    predicted_supply_by_bucket: Tuple[Tuple[str, float], ...] = ()
    learned_population: int = 0
    fallback_population: int = 0

    @property
    def expected_existing_supply(self) -> float:
        return max(0, int(self.ready_valid_slots)) + max(
            0.0, float(self.predicted_inflight_slots)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "forecast_id": str(self.forecast_id),
            "estimator_revision": max(0, int(self.estimator_revision)),
            "ready_valid_slots": max(0, int(self.ready_valid_slots)),
            "predicted_inflight_slots": max(
                0.0, float(self.predicted_inflight_slots)
            ),
            "expected_existing_supply": self.expected_existing_supply,
            "predicted_supply_by_bucket": {
                str(bucket): max(0.0, float(value))
                for bucket, value in self.predicted_supply_by_bucket
            },
            "learned_population": max(0, int(self.learned_population)),
            "fallback_population": max(0, int(self.fallback_population)),
        }


@dataclass(frozen=True)
class VersionRuntimeOutcome:
    """Observed result attributable to one finalized runtime plan."""

    plan_id: str
    forecast_id: str
    version: int
    final_revision: int
    predicted_existing_supply: float
    actual_existing_valid_slots: int
    admitted_trajectories: int
    completed_valid_slots: int
    consumed_valid_slots: int
    learner_wait_seconds: float
    next_batch_latency_seconds: float
    policy_update_interval_seconds: float = 0.0
    expired_trajectories: int = 0
    expired_actions: int = 0
    expired_tokens: int = 0
    expired_tool_calls: int = 0
    expired_tool_seconds: float = 0.0
    reprefill_tokens: int = 0
    prefill_tokens: int = 0
    prefill_seconds: float = 0.0
    scheduling_wait_seconds: float = 0.0
    scheduling_requests: int = 0
    scheduling_wait_mean_seconds: float = 0.0
    inference_queue_seconds: float = 0.0
    inference_service_seconds: float = 0.0
    queue_pressure: float = 0.0
    starvation_fraction: float = 0.0
    waste_fraction: float = 0.0
    inference_ready_trajectories: int = 0
    tool_waiting_trajectories: int = 0
    tool_wait_fraction: float = 0.0
    completion_eta_absolute_error_seconds: float = 0.0

    @property
    def supply_prediction_error(self) -> float:
        return float(self.actual_existing_valid_slots) - float(
            self.predicted_existing_supply
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": str(self.plan_id),
            "forecast_id": str(self.forecast_id),
            "version": int(self.version),
            "final_revision": max(0, int(self.final_revision)),
            "predicted_existing_supply": max(
                0.0, float(self.predicted_existing_supply)
            ),
            "actual_existing_valid_slots": max(
                0, int(self.actual_existing_valid_slots)
            ),
            "supply_prediction_error": self.supply_prediction_error,
            "admitted_trajectories": max(0, int(self.admitted_trajectories)),
            "completed_valid_slots": max(0, int(self.completed_valid_slots)),
            "consumed_valid_slots": max(0, int(self.consumed_valid_slots)),
            "learner_wait_seconds": max(0.0, float(self.learner_wait_seconds)),
            "next_batch_latency_seconds": max(
                0.0, float(self.next_batch_latency_seconds)
            ),
            "policy_update_interval_seconds": max(
                0.0, float(self.policy_update_interval_seconds)
            ),
            "expired_trajectories": max(0, int(self.expired_trajectories)),
            "expired_actions": max(0, int(self.expired_actions)),
            "expired_tokens": max(0, int(self.expired_tokens)),
            "expired_tool_calls": max(0, int(self.expired_tool_calls)),
            "expired_tool_seconds": max(0.0, float(self.expired_tool_seconds)),
            "reprefill_tokens": max(0, int(self.reprefill_tokens)),
            "prefill_tokens": max(0, int(self.prefill_tokens)),
            "prefill_seconds": max(0.0, float(self.prefill_seconds)),
            "scheduling_wait_seconds": max(
                0.0, float(self.scheduling_wait_seconds)
            ),
            "scheduling_requests": max(0, int(self.scheduling_requests)),
            "scheduling_wait_mean_seconds": max(
                0.0, float(self.scheduling_wait_mean_seconds)
            ),
            "inference_queue_seconds": max(
                0.0, float(self.inference_queue_seconds)
            ),
            "inference_service_seconds": max(
                0.0, float(self.inference_service_seconds)
            ),
            "queue_pressure": min(1.0, max(0.0, float(self.queue_pressure))),
            "starvation_fraction": min(
                1.0, max(0.0, float(self.starvation_fraction))
            ),
            "waste_fraction": min(1.0, max(0.0, float(self.waste_fraction))),
            "inference_ready_trajectories": max(
                0, int(self.inference_ready_trajectories)
            ),
            "tool_waiting_trajectories": max(
                0, int(self.tool_waiting_trajectories)
            ),
            "tool_wait_fraction": min(
                1.0, max(0.0, float(self.tool_wait_fraction))
            ),
            "completion_eta_absolute_error_seconds": max(
                0.0, float(self.completion_eta_absolute_error_seconds)
            ),
        }


@dataclass(frozen=True)
class PolicyUpdateTrace:
    """Critical-path and resource diagnostics for one policy version."""

    version: int
    batch_wait_seconds: float = 0.0
    learner_compute_seconds: float = 0.0
    publish_activate_seconds: float = 0.0
    update_interval_seconds: float = 0.0
    other_seconds: float = 0.0
    finalized: bool = False

    consumed_queue_seconds: float = 0.0
    consumed_queue_mean_seconds: float = 0.0
    consumed_queue_p95_seconds: float = 0.0
    consumed_queue_max_seconds: float = 0.0
    consumed_tool_seconds: float = 0.0
    consumed_tool_mean_seconds: float = 0.0
    consumed_tool_p95_seconds: float = 0.0
    consumed_tool_max_seconds: float = 0.0
    consumed_generate_seconds: float = 0.0
    consumed_generate_mean_seconds: float = 0.0
    consumed_generate_p95_seconds: float = 0.0
    consumed_generate_max_seconds: float = 0.0
    batch_closing_trajectory_id: str = ""
    batch_closing_completion_unix: float = 0.0
    batch_closing_queue_seconds: float = 0.0
    batch_closing_tool_seconds: float = 0.0
    batch_closing_generate_seconds: float = 0.0

    expired_tokens: int = 0
    expired_actions: int = 0
    expired_tool_calls: int = 0
    starvation_fraction: float = 0.0
    waste_fraction: float = 0.0
    queue_pressure: float = 0.0
    inference_ready_trajectories: int = 0
    tool_waiting_trajectories: int = 0
    tool_wait_fraction: float = 0.0

    reserve_before: int = 0
    reserve_after: int = 0
    decision_reason: str = "unobserved"
    shadow_mode: bool = False
    shadow_reserve_after: int = 0
    shadow_decision_reason: str = "unobserved"

    @property
    def decomposition_error_seconds(self) -> float:
        if not self.finalized or self.update_interval_seconds <= 0:
            return 0.0
        accounted = (
            self.batch_wait_seconds
            + self.learner_compute_seconds
            + self.publish_activate_seconds
            + self.other_seconds
        )
        return float(self.update_interval_seconds) - accounted

    def to_dict(self) -> Dict[str, Any]:
        result = dict(vars(self))
        result["decomposition_error_seconds"] = self.decomposition_error_seconds
        return result


@dataclass(frozen=True)
class RuntimeCandidateEstimate:
    """System-only completion estimate for one learner group."""

    group_key: str
    completion_probability: float
    eta_seconds: float
    laxity_seconds: float
    feasible: bool
    version_age: int
    progress_actions: int
    invested_trajectories: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_key": str(self.group_key),
            "completion_probability": min(
                1.0, max(0.0, float(self.completion_probability))
            ),
            "eta_seconds": max(0.0, float(self.eta_seconds)),
            "laxity_seconds": float(self.laxity_seconds),
            "feasible": bool(self.feasible),
            "version_age": max(0, int(self.version_age)),
            "progress_actions": max(0, int(self.progress_actions)),
            "invested_trajectories": max(
                0, int(self.invested_trajectories)
            ),
        }


class RuntimeEstimator:
    """Online, system-only estimator shared by runtime decisions.

    The first implementation deliberately keeps the existing bounded
    version-age/progress buckets.  It centralizes their prediction and
    supervision so later service-time and KV estimates can be added without
    giving each actuator an independent feedback model.
    """

    def __init__(
        self,
        *,
        initial_finish_ratio: float,
        ewma_alpha: float,
        bucketed_finish_enabled: bool,
        bucket_min_samples: int,
    ):
        self.finish_ratio = min(1.0, max(0.0, float(initial_finish_ratio)))
        self.ewma_alpha = min(1.0, max(0.0, float(ewma_alpha)))
        self.bucketed_finish_enabled = bool(bucketed_finish_enabled)
        self.bucket_min_samples = max(1, int(bucket_min_samples))
        self.bucket_finish_ratios: Dict[str, float] = {}
        self.bucket_sample_counts: Dict[str, int] = {}
        self.coarse_bucket_finish_ratios: Dict[str, float] = {}
        self.coarse_bucket_sample_counts: Dict[str, int] = {}
        self.revision = 0
        self.supply_error_ewma: Optional[float] = None
        self.supply_abs_error_ewma: Optional[float] = None
        self.policy_interval_seconds_ewma: Optional[float] = None
        self.generation_seconds_per_action_ewma: Optional[float] = None
        self.tool_seconds_per_action_ewma: Optional[float] = None
        self.prefill_seconds_per_token_ewma: Optional[float] = None
        self._observed_completed_records = set()

    def _ewma(self, current: Optional[float], sample: float) -> float:
        if current is None:
            return float(sample)
        return (
            self.ewma_alpha * float(sample)
            + (1 - self.ewma_alpha) * current
        )

    def bucket_finish_ratio(self, bucket: str) -> float:
        if (
            self.bucketed_finish_enabled
            and self.bucket_sample_counts.get(bucket, 0)
            >= self.bucket_min_samples
            and bucket in self.bucket_finish_ratios
        ):
            return min(
                1.0, max(0.0, self.bucket_finish_ratios[bucket])
            )
        coarse = coarse_finish_rate_bucket(bucket)
        if (
            self.bucketed_finish_enabled
            and self.coarse_bucket_sample_counts.get(coarse, 0)
            >= self.bucket_min_samples
            and coarse in self.coarse_bucket_finish_ratios
        ):
            return min(
                1.0, max(0.0, self.coarse_bucket_finish_ratios[coarse])
            )
        return min(1.0, max(0.0, self.finish_ratio))

    def has_learned_finish_ratio(self, bucket: str) -> bool:
        coarse = coarse_finish_rate_bucket(bucket)
        return bool(
            self.bucketed_finish_enabled
            and (
                self.bucket_sample_counts.get(bucket, 0)
                >= self.bucket_min_samples
                or self.coarse_bucket_sample_counts.get(coarse, 0)
                >= self.bucket_min_samples
            )
        )

    def observe_policy_interval(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self.policy_interval_seconds_ewma = self._ewma(
            self.policy_interval_seconds_ewma, seconds
        )

    def observe_completed_records(
        self, records: List[Dict[str, Any]]
    ) -> None:
        """Learn service rates once from each completed trajectory."""
        for record in records:
            if not bool(record.get("completed", False)):
                continue
            key = (
                int(record.get("group_id", -1)),
                int(record.get("episode_id", -1)),
                int(record.get("env_id", -1)),
                int(record.get("version_start", 0)),
            )
            if key in self._observed_completed_records:
                continue
            self._observed_completed_records.add(key)
            inference_calls = max(
                0, int(record.get("inference_calls", 0))
            )
            actions = max(0, int(record.get("actions_completed", 0)))
            tool_calls = max(0, int(record.get("tool_calls", 0)))
            generate_seconds = max(
                0.0, float(record.get("generate_seconds", 0.0))
            )
            tool_seconds = max(
                0.0, float(record.get("env_seconds", 0.0))
            )
            if inference_calls > 0 and generate_seconds > 0:
                self.generation_seconds_per_action_ewma = self._ewma(
                    self.generation_seconds_per_action_ewma,
                    generate_seconds / inference_calls,
                )
            service_actions = max(actions, tool_calls)
            if service_actions > 0 and tool_seconds > 0:
                self.tool_seconds_per_action_ewma = self._ewma(
                    self.tool_seconds_per_action_ewma,
                    tool_seconds / service_actions,
                )
            prefill_tokens = max(
                0, int(record.get("engine_prefill_tokens", 0))
            )
            prefill_seconds = max(
                0.0, float(record.get("request_prefill_seconds", 0.0))
            )
            if prefill_tokens > 0 and prefill_seconds > 0:
                self.prefill_seconds_per_token_ewma = self._ewma(
                    self.prefill_seconds_per_token_ewma,
                    prefill_seconds / prefill_tokens,
                )

        # Keep attribution state bounded during long runs.
        if len(self._observed_completed_records) > 16384:
            self._observed_completed_records = set(
                list(self._observed_completed_records)[-8192:]
            )

    def estimate_candidate(
        self,
        *,
        group_key: str,
        version_age: int,
        staleness_tolerance: int,
        progress_actions: int,
        invested_trajectories: int,
        finish_bucket: str,
        runtime_summary: Dict[str, Any],
    ) -> RuntimeCandidateEstimate:
        remaining_actions = max(
            0, int(runtime_summary.get("frontier_remaining_actions", 0))
        )
        local_generation = max(
            0.0,
            float(runtime_summary.get("generation_seconds_per_action", 0.0)),
        )
        local_tool = max(
            0.0, float(runtime_summary.get("tool_seconds_per_action", 0.0))
        )
        generation_per_action = (
            local_generation
            or self.generation_seconds_per_action_ewma
            or 1.0
        )
        tool_per_action = (
            local_tool or self.tool_seconds_per_action_ewma or 0.0
        )
        eta_seconds = remaining_actions * (
            generation_per_action + tool_per_action
        )
        if runtime_summary.get("tool_waiting", 0):
            eta_seconds += tool_per_action
        if version_age > 0 and self.prefill_seconds_per_token_ewma:
            eta_seconds += max(
                0.0, float(runtime_summary.get("mean_context_tokens", 0.0))
            ) * self.prefill_seconds_per_token_ewma

        slack_versions = max(
            0, int(staleness_tolerance) - max(0, int(version_age))
        )
        if self.policy_interval_seconds_ewma is None:
            feasible = True
            # Preserve version slack as the dominant cold-start ordering.
            laxity_seconds = float(slack_versions) * 1e9 - eta_seconds
        else:
            deadline_seconds = (
                slack_versions + 1
            ) * self.policy_interval_seconds_ewma
            laxity_seconds = deadline_seconds - eta_seconds
            feasible = laxity_seconds >= 0

        return RuntimeCandidateEstimate(
            group_key=str(group_key),
            completion_probability=self.bucket_finish_ratio(finish_bucket),
            eta_seconds=max(0.0, eta_seconds),
            laxity_seconds=float(laxity_seconds),
            feasible=feasible,
            version_age=max(0, int(version_age)),
            progress_actions=max(0, int(progress_actions)),
            invested_trajectories=max(0, int(invested_trajectories)),
        )

    def predict_unfinished_supply(
        self,
        *,
        salvageable_inflight: int,
        unfinished_bucket_counts: Dict[str, int],
    ) -> Tuple[float, int, int, Dict[str, float]]:
        salvageable = max(0, int(salvageable_inflight))
        if not self.bucketed_finish_enabled:
            expected = self.finish_ratio * salvageable
            return expected, 0, salvageable, {"global": expected}

        expected = 0.0
        learned = 0
        fallback = 0
        by_bucket: Dict[str, float] = {}
        for bucket, count in unfinished_bucket_counts.items():
            count = max(0, int(count))
            ratio = self.bucket_finish_ratio(bucket)
            if self.has_learned_finish_ratio(bucket):
                learned += count
            else:
                fallback += count
            predicted = count * min(1.0, max(0.0, float(ratio)))
            by_bucket[str(bucket)] = predicted
            expected += predicted
        return expected, learned, fallback, by_bucket

    def build_forecast(
        self,
        version: int,
        *,
        ready_valid_slots: int,
        salvageable_inflight: int,
        unfinished_bucket_counts: Dict[str, int],
    ) -> VersionRuntimeForecast:
        predicted, learned, fallback, by_bucket = (
            self.predict_unfinished_supply(
                salvageable_inflight=salvageable_inflight,
                unfinished_bucket_counts=unfinished_bucket_counts,
            )
        )
        return VersionRuntimeForecast(
            version=int(version),
            forecast_id=version_runtime_forecast_id(version, self.revision),
            estimator_revision=self.revision,
            ready_valid_slots=max(0, int(ready_valid_slots)),
            predicted_inflight_slots=max(0.0, float(predicted)),
            predicted_supply_by_bucket=tuple(sorted(by_bucket.items())),
            learned_population=learned,
            fallback_population=fallback,
        )

    def observe_supply(
        self,
        *,
        salvageable_inflight: int,
        completed_inflight: int,
        cohort_counts: Dict[str, int],
        completed_counts: Dict[str, int],
        prediction_error: Optional[float] = None,
    ) -> None:
        salvageable = max(0, int(salvageable_inflight))
        completed = min(salvageable, max(0, int(completed_inflight)))
        if salvageable > 0:
            observed_ratio = completed / salvageable
            self.finish_ratio = (
                self.ewma_alpha * observed_ratio
                + (1 - self.ewma_alpha) * self.finish_ratio
            )
        if self.bucketed_finish_enabled:
            update_bucketed_finish_ratios(
                self.bucket_finish_ratios,
                self.bucket_sample_counts,
                cohort_counts,
                completed_counts,
                self.ewma_alpha,
            )
            coarse_cohorts: Dict[str, int] = {}
            coarse_completed: Dict[str, int] = {}
            for bucket, count in cohort_counts.items():
                coarse = coarse_finish_rate_bucket(bucket)
                coarse_cohorts[coarse] = (
                    coarse_cohorts.get(coarse, 0) + max(0, int(count))
                )
                coarse_completed[coarse] = (
                    coarse_completed.get(coarse, 0)
                    + max(0, int(completed_counts.get(bucket, 0)))
                )
            update_bucketed_finish_ratios(
                self.coarse_bucket_finish_ratios,
                self.coarse_bucket_sample_counts,
                coarse_cohorts,
                coarse_completed,
                self.ewma_alpha,
            )
        if prediction_error is not None:
            error = float(prediction_error)
            if self.supply_error_ewma is None:
                self.supply_error_ewma = error
                self.supply_abs_error_ewma = abs(error)
            else:
                self.supply_error_ewma = (
                    self.ewma_alpha * error
                    + (1 - self.ewma_alpha) * self.supply_error_ewma
                )
                self.supply_abs_error_ewma = (
                    self.ewma_alpha * abs(error)
                    + (1 - self.ewma_alpha)
                    * float(self.supply_abs_error_ewma or 0.0)
                )
        self.revision += 1


@dataclass(frozen=True)
class VersionRuntimeState:
    """System-only state observed at one policy-version boundary."""

    version: int
    learner_demand: int
    safety_reserve: int
    expected_existing_supply: float
    outstanding_trajectories: int
    max_outstanding_trajectories: int
    admission_width: int
    group_size: int
    staleness_tolerance: int
    invested_candidate_groups: Tuple[Tuple[int, int, int, int, int], ...]
    admission_enabled: bool = True
    priority_enabled: bool = True
    revision: int = 0
    gpu_invested_candidate_groups: Optional[
        Tuple[Tuple[int, int, int, int, int], ...]
    ] = None
    gpu_invested_candidate_trajectories: Tuple[
        Tuple[str, int, int, int, int, int], ...
    ] = ()
    exact_rebuild_cohort: bool = False
    kv_feedback_requests: int = 0
    kv_feedback_hit_ratio: float = 0.0
    kv_feedback_resets: int = 0
    forecast_id: str = ""
    estimator_revision: int = 0
    ready_valid_slots: int = 0
    predicted_inflight_slots: float = 0.0
    reserve_before: int = 0
    undersupply_signal: float = 0.0
    overload_signal: float = 0.0
    candidate_estimates: Tuple[RuntimeCandidateEstimate, ...] = ()


@dataclass(frozen=True)
class VersionRuntimePlan:
    """One policy-version decision shared by admission and inference routing."""

    version: int
    learner_demand: int
    safety_reserve: int
    expected_existing_supply: float
    outstanding_trajectories: int
    admission_enabled: bool
    admission_budget: int
    admission_budget_trainable: int
    admission_deficit: float
    admission_capacity: int
    admission_reason: str
    priority_enabled: bool
    priority_deadline_version: int
    priority_candidate_groups: Tuple[str, ...]
    rebuild_candidate_groups: Tuple[str, ...]
    rebuild_candidate_trajectories: Tuple[str, ...]
    rebuild_cohort_exact: bool
    rebuild_target_trajectories: int
    kv_feedback_requests: int = 0
    kv_feedback_hit_ratio: float = 0.0
    kv_feedback_resets: int = 0
    revision: int = 0
    admission_delta_trajectories: int = 0
    plan_id: str = ""
    forecast_id: str = ""
    estimator_revision: int = 0
    ready_valid_slots: int = 0
    predicted_inflight_slots: float = 0.0
    reserve_before: int = 0
    reserve_after: int = 0
    undersupply_signal: float = 0.0
    overload_signal: float = 0.0
    priority_candidate_estimates: Tuple[RuntimeCandidateEstimate, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "learner_demand": self.learner_demand,
            "safety_reserve": self.safety_reserve,
            "expected_existing_supply": self.expected_existing_supply,
            "outstanding_trajectories": self.outstanding_trajectories,
            "admission_enabled": self.admission_enabled,
            "admission_budget": self.admission_budget,
            "admission_budget_trainable": self.admission_budget_trainable,
            "admission_deficit": self.admission_deficit,
            "admission_capacity": self.admission_capacity,
            "admission_reason": self.admission_reason,
            "priority_enabled": self.priority_enabled,
            "priority_deadline_version": self.priority_deadline_version,
            "priority_candidate_groups": list(self.priority_candidate_groups),
            "rebuild_candidate_groups": list(self.rebuild_candidate_groups),
            "rebuild_candidate_trajectories": list(
                self.rebuild_candidate_trajectories
            ),
            "rebuild_cohort_exact": self.rebuild_cohort_exact,
            "rebuild_target_trajectories": self.rebuild_target_trajectories,
            "kv_feedback_requests": self.kv_feedback_requests,
            "kv_feedback_hit_ratio": self.kv_feedback_hit_ratio,
            "kv_feedback_resets": self.kv_feedback_resets,
            "revision": self.revision,
            "admission_delta_trajectories": self.admission_delta_trajectories,
            "plan_id": self.plan_id,
            "forecast_id": self.forecast_id,
            "estimator_revision": self.estimator_revision,
            "ready_valid_slots": self.ready_valid_slots,
            "predicted_inflight_slots": self.predicted_inflight_slots,
            "reserve_before": self.reserve_before,
            "reserve_after": self.reserve_after,
            "undersupply_signal": self.undersupply_signal,
            "overload_signal": self.overload_signal,
            "priority_candidate_estimates": [
                estimate.to_dict()
                for estimate in self.priority_candidate_estimates
            ],
        }


class VersionAwareRuntimeController:
    """Convert one boundary snapshot into a deterministic runtime decision."""

    def decide(
        self,
        state: VersionRuntimeState,
        *,
        active_plan: Optional[VersionRuntimePlan] = None,
        missing_trajectories: int = 0,
        current_batch_missing: int = 0,
        learner_wait_seconds: float = 0.0,
        reconcile_wait_seconds: float = 0.0,
        max_revisions_per_version: int = 0,
        max_admission_groups: int = 1,
    ) -> Optional[VersionRuntimePlan]:
        """Return the complete boundary plan or an online revision of it."""
        if active_plan is None:
            return self._decide_boundary(state)
        if (
            active_plan.version != state.version
            or learner_wait_seconds < reconcile_wait_seconds
            or active_plan.revision >= max(0, int(max_revisions_per_version))
        ):
            return None

        missing = min(
            max(0, int(missing_trajectories)),
            max(0, int(current_batch_missing)),
        )
        if missing <= 0:
            return None
        admitted_groups = compute_progress_topup_groups(
            missing,
            int(round(max(0.0, state.expected_existing_supply))),
            state.outstanding_trajectories,
            state.max_outstanding_trajectories,
            state.group_size,
            state.admission_width,
        )
        admitted_groups = min(
            admitted_groups, max(0, int(max_admission_groups))
        )
        if admitted_groups <= 0:
            return None

        candidate_plan = self._decide_boundary(
            replace(state, admission_enabled=False, safety_reserve=0)
        )
        admission_delta = admitted_groups * max(1, int(state.admission_width))
        revised_revision = active_plan.revision + 1
        return replace(
            active_plan,
            learner_demand=candidate_plan.learner_demand,
            safety_reserve=candidate_plan.safety_reserve,
            expected_existing_supply=candidate_plan.expected_existing_supply,
            outstanding_trajectories=candidate_plan.outstanding_trajectories,
            revision=revised_revision,
            admission_enabled=True,
            admission_budget=active_plan.admission_budget + admission_delta,
            admission_budget_trainable=(
                active_plan.admission_budget_trainable
                + admitted_groups * max(1, int(state.group_size))
            ),
            admission_reason="progress_reconcile",
            admission_deficit=candidate_plan.admission_deficit,
            admission_capacity=candidate_plan.admission_capacity,
            priority_candidate_groups=candidate_plan.priority_candidate_groups,
            rebuild_candidate_groups=candidate_plan.rebuild_candidate_groups,
            rebuild_candidate_trajectories=(
                candidate_plan.rebuild_candidate_trajectories
            ),
            rebuild_cohort_exact=candidate_plan.rebuild_cohort_exact,
            rebuild_target_trajectories=(
                candidate_plan.rebuild_target_trajectories
            ),
            kv_feedback_requests=candidate_plan.kv_feedback_requests,
            kv_feedback_hit_ratio=candidate_plan.kv_feedback_hit_ratio,
            kv_feedback_resets=candidate_plan.kv_feedback_resets,
            admission_delta_trajectories=admission_delta,
            plan_id=version_runtime_plan_id(state.version, revised_revision),
            forecast_id=(
                state.forecast_id
                or version_runtime_forecast_id(
                    state.version, state.estimator_revision
                )
            ),
            estimator_revision=max(0, int(state.estimator_revision)),
            ready_valid_slots=max(0, int(state.ready_valid_slots)),
            predicted_inflight_slots=max(
                0.0, float(state.predicted_inflight_slots)
            ),
            reserve_before=max(0, int(state.reserve_before)),
            reserve_after=max(0, int(state.safety_reserve)),
            undersupply_signal=max(0.0, float(state.undersupply_signal)),
            overload_signal=max(0.0, float(state.overload_signal)),
            priority_candidate_estimates=(
                candidate_plan.priority_candidate_estimates
            ),
        )

    def _decide_boundary(self, state: VersionRuntimeState) -> VersionRuntimePlan:
        width = max(1, int(state.admission_width))
        trainable_width = max(1, int(state.group_size))
        demand = max(0, int(state.learner_demand)) + max(
            0, int(state.safety_reserve)
        )
        deficit = max(
            0.0,
            demand - max(0.0, float(state.expected_existing_supply)),
        )
        desired_groups = math.ceil(deficit / trainable_width)
        available_trajectories = max(
            0,
            max(0, int(state.max_outstanding_trajectories))
            - max(0, int(state.outstanding_trajectories)),
        )
        available_groups = available_trajectories // width
        admitted_groups = (
            min(desired_groups, available_groups)
            if state.admission_enabled
            else 0
        )

        if not state.admission_enabled:
            admission_reason = "disabled"
        elif desired_groups == 0:
            admission_reason = "existing_supply_sufficient"
        elif available_groups == 0:
            admission_reason = "outstanding_cap"
        elif admitted_groups < desired_groups:
            admission_reason = "partial_capacity"
        else:
            admission_reason = "supply_deficit"

        estimate_by_group = {
            estimate.group_key: estimate
            for estimate in state.candidate_estimates
        }

        def candidate_order(item):
            group_key = f"{int(item[0])}:{int(item[1])}"
            estimate = estimate_by_group.get(group_key)
            if estimate is None:
                return (
                    0,
                    float(-int(item[2])),
                    float(-int(item[3])),
                    -float(int(item[4])),
                    int(item[0]),
                    int(item[1]),
                )
            if estimate.feasible:
                primary = float(estimate.laxity_seconds)
                secondary = float(estimate.eta_seconds)
            else:
                primary = float(estimate.eta_seconds)
                secondary = -float(estimate.completion_probability)
            return (
                int(not estimate.feasible),
                primary,
                secondary,
                -float(estimate.invested_trajectories),
                int(item[0]),
                int(item[1]),
            )

        ordered_candidates = sorted(
            state.invested_candidate_groups,
            key=candidate_order,
        )
        candidate_keys = tuple(
            f"{int(group_id)}:{int(episode_id)}"
            for group_id, episode_id, _, _, _ in ordered_candidates
        )
        rebuild_ordered_candidates = sorted(
            (
                state.invested_candidate_groups
                if state.gpu_invested_candidate_groups is None
                else state.gpu_invested_candidate_groups
            ),
            key=candidate_order,
        )
        rebuild_candidate_keys = tuple(
            f"{int(group_id)}:{int(episode_id)}"
            for group_id, episode_id, _, _, _ in rebuild_ordered_candidates
        )
        candidate_rank = {
            group_key: rank for rank, group_key in enumerate(candidate_keys)
        }
        rebuild_ordered_trajectories = sorted(
            state.gpu_invested_candidate_trajectories,
            key=lambda item: (
                candidate_rank.get(
                    f"{int(item[1])}:{int(item[2])}", len(candidate_rank)
                ),
                -int(item[4]),
                -int(item[5]),
                int(item[1]),
                int(item[2]),
                int(item[3]),
                str(item[0]),
            ),
        )
        rebuild_candidate_trajectories = tuple(
            str(trajectory_id)
            for trajectory_id, _, _, _, _, _ in rebuild_ordered_trajectories
        )
        if rebuild_candidate_trajectories:
            rebuild_candidate_keys = tuple(
                dict.fromkeys(
                    f"{int(group_id)}:{int(episode_id)}"
                    for _, group_id, episode_id, _, _, _ in rebuild_ordered_trajectories
                )
            )
        priority_candidates = candidate_keys if state.priority_enabled else ()
        return VersionRuntimePlan(
            version=int(state.version),
            learner_demand=max(0, int(state.learner_demand)),
            safety_reserve=max(0, int(state.safety_reserve)),
            expected_existing_supply=max(
                0.0, float(state.expected_existing_supply)
            ),
            outstanding_trajectories=max(
                0, int(state.outstanding_trajectories)
            ),
            admission_enabled=bool(state.admission_enabled),
            admission_budget=admitted_groups * width,
            admission_budget_trainable=admitted_groups * trainable_width,
            admission_deficit=deficit,
            admission_capacity=available_groups * width,
            admission_reason=admission_reason,
            priority_enabled=bool(state.priority_enabled),
            priority_deadline_version=int(state.version)
            + max(0, int(state.staleness_tolerance)),
            priority_candidate_groups=priority_candidates,
            rebuild_candidate_groups=rebuild_candidate_keys,
            rebuild_candidate_trajectories=rebuild_candidate_trajectories,
            rebuild_cohort_exact=bool(state.exact_rebuild_cohort),
            rebuild_target_trajectories=(
                len(rebuild_candidate_trajectories)
                if state.exact_rebuild_cohort
                else sum(
                    max(0, int(invested))
                    for _, _, _, _, invested in rebuild_ordered_candidates
                )
            ),
            kv_feedback_requests=max(0, int(state.kv_feedback_requests)),
            kv_feedback_hit_ratio=min(
                1.0, max(0.0, float(state.kv_feedback_hit_ratio))
            ),
            kv_feedback_resets=max(0, int(state.kv_feedback_resets)),
            revision=max(0, int(state.revision)),
            admission_delta_trajectories=admitted_groups * width,
            plan_id=version_runtime_plan_id(state.version, state.revision),
            forecast_id=(
                state.forecast_id
                or version_runtime_forecast_id(
                    state.version, state.estimator_revision
                )
            ),
            estimator_revision=max(0, int(state.estimator_revision)),
            ready_valid_slots=max(0, int(state.ready_valid_slots)),
            predicted_inflight_slots=max(
                0.0, float(state.predicted_inflight_slots)
            ),
            reserve_before=max(0, int(state.reserve_before)),
            reserve_after=max(0, int(state.safety_reserve)),
            undersupply_signal=max(0.0, float(state.undersupply_signal)),
            overload_signal=max(0.0, float(state.overload_signal)),
            priority_candidate_estimates=tuple(
                estimate_by_group[group_key]
                for group_key in candidate_keys
                if group_key in estimate_by_group
            ),
        )


def build_version_runtime_plan(
    *,
    version: int,
    learner_demand: int,
    safety_reserve: int,
    expected_existing_supply: float,
    outstanding_trajectories: int,
    max_outstanding_trajectories: int,
    admission_width: int,
    group_size: int,
    staleness_tolerance: int,
    invested_candidate_groups: List[Tuple[int, int, int, int, int]],
    admission_enabled: bool = True,
    priority_enabled: bool = True,
    revision: int = 0,
    gpu_invested_candidate_groups: Optional[
        List[Tuple[int, int, int, int, int]]
    ] = None,
    gpu_invested_candidate_trajectories: Optional[
        List[Tuple[str, int, int, int, int, int]]
    ] = None,
    kv_feedback_requests: int = 0,
    kv_feedback_hit_ratio: float = 0.0,
    kv_feedback_resets: int = 0,
    forecast_id: str = "",
    estimator_revision: int = 0,
    ready_valid_slots: int = 0,
    predicted_inflight_slots: float = 0.0,
    reserve_before: int = 0,
    undersupply_signal: float = 0.0,
    overload_signal: float = 0.0,
    candidate_estimates: Optional[List[RuntimeCandidateEstimate]] = None,
) -> VersionRuntimePlan:
    """Build the boundary plan consumed by both the queue manager and Router.

    Candidate tuples are ``(group_id, episode_id, version_age, progress, invested)``.
    They are ordered by deadline first and invested progress second. This only
    changes runtime timing; it does not inspect rewards or learner-side values.
    """
    state = VersionRuntimeState(
        version=version,
        learner_demand=learner_demand,
        safety_reserve=safety_reserve,
        expected_existing_supply=expected_existing_supply,
        outstanding_trajectories=outstanding_trajectories,
        max_outstanding_trajectories=max_outstanding_trajectories,
        admission_width=admission_width,
        group_size=group_size,
        staleness_tolerance=staleness_tolerance,
        invested_candidate_groups=tuple(invested_candidate_groups),
        admission_enabled=admission_enabled,
        priority_enabled=priority_enabled,
        revision=revision,
        gpu_invested_candidate_groups=(
            None
            if gpu_invested_candidate_groups is None
            else tuple(gpu_invested_candidate_groups)
        ),
        gpu_invested_candidate_trajectories=tuple(
            gpu_invested_candidate_trajectories or ()
        ),
        exact_rebuild_cohort=gpu_invested_candidate_trajectories is not None,
        kv_feedback_requests=kv_feedback_requests,
        kv_feedback_hit_ratio=kv_feedback_hit_ratio,
        kv_feedback_resets=kv_feedback_resets,
        forecast_id=forecast_id,
        estimator_revision=estimator_revision,
        ready_valid_slots=ready_valid_slots,
        predicted_inflight_slots=predicted_inflight_slots,
        reserve_before=reserve_before,
        undersupply_signal=undersupply_signal,
        overload_signal=overload_signal,
        candidate_estimates=tuple(candidate_estimates or ()),
    )
    plan = VersionAwareRuntimeController().decide(state)
    assert plan is not None
    return plan


def summarize_version_boundary_records(
    records: List[Dict[str, Any]],
    *,
    from_version: int,
    to_version: int,
    staleness_tolerance: int,
    reserved_unstarted: int = 0,
    unobserved_started: int = 0,
) -> Dict[str, Any]:
    """Aggregate a policy boundary without using reward or learner-side value."""
    expired_records = [record for record in records if record.get("will_expire", False)]
    unfinished_records = [record for record in records if not record.get("completed", False)]
    completed_records = [record for record in records if record.get("completed", False)]
    cross_version_records = [
        record
        for record in unfinished_records
        if int(record.get("version_age_at_boundary", 0)) > 0
    ]
    unfinished_survivor_records = [
        record for record in unfinished_records if not record.get("will_expire", False)
    ]
    unfinished_expired_records = [
        record for record in unfinished_records if record.get("will_expire", False)
    ]
    invested_records = [
        record
        for record in records
        if int(record.get("actions_completed", 0)) > 0
        or int(record.get("inference_calls", 0)) > 0
    ]
    return {
        "from_version": int(from_version),
        "to_version": int(to_version),
        "staleness_tolerance": max(0, int(staleness_tolerance)),
        "observed_started_trajectories": len(records),
        "unobserved_started_trajectories": max(0, int(unobserved_started)),
        "reserved_unstarted_trajectories": max(0, int(reserved_unstarted)),
        "unfinished_started_trajectories": len(unfinished_records),
        "completed_ready_trajectories": len(completed_records),
        "completed_carryover_trajectories": sum(
            int(record.get("version_age_at_boundary", 0)) > 0
            for record in completed_records
        ),
        "cross_version_trajectories": len(cross_version_records),
        "cross_version_invested_trajectories": sum(
            int(record.get("actions_completed", 0)) > 0
            or int(record.get("inference_calls", 0)) > 0
            for record in cross_version_records
        ),
        "survivor_trajectories": len(unfinished_survivor_records),
        "completed_survivor_trajectories": sum(
            not record.get("will_expire", False) for record in completed_records
        ),
        "expired_trajectories": len(expired_records),
        "unfinished_expired_trajectories": len(unfinished_expired_records),
        "invested_trajectories": len(invested_records),
        "actions_completed": sum(int(record.get("actions_completed", 0)) for record in records),
        "inference_calls": sum(int(record.get("inference_calls", 0)) for record in records),
        "tool_calls": sum(int(record.get("tool_calls", 0)) for record in records),
        "prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records),
        "response_tokens": sum(int(record.get("response_tokens", 0)) for record in records),
        "logical_inference_tokens": sum(
            int(record.get("inference_tokens", 0)) for record in records
        ),
        "current_context_tokens": sum(
            int(record.get("current_context_tokens", 0)) for record in records
        ),
        "generate_seconds": sum(float(record.get("generate_seconds", 0.0)) for record in records),
        "env_seconds": sum(float(record.get("env_seconds", 0.0)) for record in records),
        "tool_wall_seconds": sum(
            float(record.get("tool_wall_seconds", 0.0)) for record in records
        ),
        "model_execute_seconds": sum(
            float(record.get("model_execute_seconds", 0.0))
            for record in records
        ),
        "engine_step_seconds_attributed": sum(
            float(record.get("engine_step_seconds_attributed", 0.0))
            for record in records
        ),
        "trajectory_wall_seconds": sum(
            float(record.get("trajectory_wall_seconds", 0.0)) for record in records
        ),
        "unfinished_actions": sum(
            int(record.get("actions_completed", 0)) for record in unfinished_records
        ),
        "unfinished_logical_inference_tokens": sum(
            int(record.get("inference_tokens", 0)) for record in unfinished_records
        ),
        "unfinished_current_context_tokens": sum(
            int(record.get("current_context_tokens", 0)) for record in unfinished_records
        ),
        "expired_actions": sum(
            int(record.get("actions_completed", 0)) for record in expired_records
        ),
        "expired_logical_inference_tokens": sum(
            int(record.get("inference_tokens", 0)) for record in expired_records
        ),
        "expired_tool_wall_seconds": sum(
            float(record.get("tool_wall_seconds", 0.0))
            for record in expired_records
        ),
        "expired_model_execute_seconds": sum(
            float(record.get("model_execute_seconds", 0.0))
            for record in expired_records
        ),
        "expired_engine_step_seconds_attributed": sum(
            float(record.get("engine_step_seconds_attributed", 0.0))
            for record in expired_records
        ),
    }


def summarize_rollout_goodput(
    consumed_records: List[Dict[str, Any]],
    discard_records: List[Dict[str, Any]],
    terminal_records: List[Dict[str, Any]],
    *,
    elapsed_seconds: float,
    learner_wait_seconds: float,
    admitted_trajectories: Optional[int] = None,
) -> Dict[str, Any]:
    """Separate raw rollout production from learner-consumable goodput."""
    elapsed = max(1e-6, float(elapsed_seconds))
    all_records = consumed_records + discard_records + terminal_records
    stale_records = [
        record
        for record in discard_records
        if str(record.get("discard_reason", "")).startswith("version_")
    ]

    def total(records, field):
        return sum(int(record.get(field, 0)) for record in records)

    def total_seconds(records, field):
        return sum(float(record.get(field, 0.0)) for record in records)

    valid_consumed_records = [
        record
        for record in consumed_records
        if bool(record.get("trainable_valid", True))
    ]
    raw_response_tokens = total(all_records, "response_tokens")
    raw_inference_tokens = total(all_records, "inference_tokens")
    trainable_response_tokens = total(valid_consumed_records, "response_tokens")
    trainable_inference_tokens = total(valid_consumed_records, "inference_tokens")
    stale_inference_tokens = total(stale_records, "inference_tokens")
    raw_model_execute_seconds = total_seconds(all_records, "model_execute_seconds")
    stale_model_execute_seconds = total_seconds(
        stale_records, "model_execute_seconds"
    )
    raw_engine_step_seconds = total_seconds(
        all_records, "engine_step_seconds_attributed"
    )
    stale_engine_step_seconds = total_seconds(
        stale_records, "engine_step_seconds_attributed"
    )
    admitted = (
        len(all_records)
        if admitted_trajectories is None
        else max(0, int(admitted_trajectories))
    )
    return {
        "rollout/elapsed_seconds": elapsed,
        "rollout/admitted_trajectories": admitted,
        "rollout/raw_trajectories": len(all_records),
        "rollout/raw_completed_trajectories": sum(
            bool(record.get("completed", False)) for record in all_records
        ),
        "rollout/raw_response_tokens": raw_response_tokens,
        "rollout/raw_logical_inference_tokens": raw_inference_tokens,
        "rollout/raw_response_tokens_per_second": raw_response_tokens / elapsed,
        "rollout/raw_logical_inference_tokens_per_second": raw_inference_tokens / elapsed,
        "rollout/learner_consumed_trajectories": len(consumed_records),
        "rollout/trainable_trajectories": len(valid_consumed_records),
        "rollout/placeholder_trajectories": (
            len(consumed_records) - len(valid_consumed_records)
        ),
        "rollout/trainable_response_tokens": trainable_response_tokens,
        "rollout/trainable_logical_inference_tokens": trainable_inference_tokens,
        "rollout/trainable_trajectories_per_second": (
            len(valid_consumed_records) / elapsed
        ),
        "rollout/trainable_response_tokens_per_second": trainable_response_tokens / elapsed,
        "rollout/trainable_logical_inference_tokens_per_second": (
            trainable_inference_tokens / elapsed
        ),
        "rollout/stale_trajectories": len(stale_records),
        "rollout/stale_logical_inference_tokens": stale_inference_tokens,
        "rollout/raw_model_execute_seconds": raw_model_execute_seconds,
        "rollout/stale_model_execute_seconds": stale_model_execute_seconds,
        "rollout/raw_engine_step_seconds_attributed": raw_engine_step_seconds,
        "rollout/stale_engine_step_seconds_attributed": (
            stale_engine_step_seconds
        ),
        "rollout/stale_trajectory_fraction": (
            len(stale_records) / admitted if admitted else 0.0
        ),
        "rollout/stale_trajectory_fraction_of_recorded": (
            len(stale_records) / len(all_records) if all_records else 0.0
        ),
        "rollout/stale_logical_token_fraction": (
            stale_inference_tokens / raw_inference_tokens if raw_inference_tokens else 0.0
        ),
        "rollout/stale_model_execute_fraction": (
            stale_model_execute_seconds / raw_model_execute_seconds
            if raw_model_execute_seconds
            else 0.0
        ),
        "rollout/stale_engine_step_fraction": (
            stale_engine_step_seconds / raw_engine_step_seconds
            if raw_engine_step_seconds
            else 0.0
        ),
        "learner/wait_seconds": max(0.0, float(learner_wait_seconds)),
        "learner/wait_fraction": min(
            1.0, max(0.0, float(learner_wait_seconds)) / elapsed
        ),
    }


def build_learner_wait_record(
    *,
    step: int,
    wait_seconds: float,
    batch_size: int,
    consumed_records: List[Dict[str, Any]],
    discard_records: List[Dict[str, Any]],
    recorded_at_unix: Optional[float] = None,
) -> Dict[str, Any]:
    """Describe the usable-batch formation work performed for one update."""
    consumed = [
        record
        for record in consumed_records
        if int(record.get("consumed_at_step", -1)) == int(step)
    ]
    stale = [
        record
        for record in discard_records
        if int(record.get("discarded_at_step", -1)) == int(step)
        and str(record.get("discard_reason", "")).startswith("version_")
    ]
    valid = [
        record
        for record in consumed
        if bool(record.get("trainable_valid", True))
    ]
    version_ages = [max(0, int(record.get("version_age", 0))) for record in valid]

    def total(records, field):
        return sum(float(record.get(field, 0.0) or 0.0) for record in records)

    def distribution(records, field):
        values = sorted(
            max(0.0, float(record.get(field, 0.0) or 0.0))
            for record in records
        )
        if not values:
            return 0.0, 0.0, 0.0
        rank = min(len(values) - 1, max(0, math.ceil(0.95 * len(values)) - 1))
        return sum(values) / len(values), values[rank], values[-1]

    queue_mean, queue_p95, queue_max = distribution(valid, "request_queue_seconds")
    tool_mean, tool_p95, tool_max = distribution(valid, "tool_wall_seconds")
    generate_mean, generate_p95, generate_max = distribution(
        valid, "generate_seconds"
    )
    batch_closer = max(
        valid,
        key=lambda record: float(record.get("trajectory_completed_at_unix", 0.0)),
        default={},
    )

    return {
        "step": int(step),
        "batch_size": max(0, int(batch_size)),
        "wait_seconds": max(0.0, float(wait_seconds)),
        "consumed_trajectories": len(consumed),
        "valid_trajectories": len(valid),
        "valid_version_age_mean": (
            sum(version_ages) / len(version_ages) if version_ages else 0.0
        ),
        "valid_version_age_max": max(version_ages, default=0),
        "stale_discarded_trajectories": len(stale),
        "stale_discarded_actions": int(total(stale, "actions_completed")),
        "stale_discarded_output_tokens": int(total(stale, "response_tokens")),
        "stale_discarded_tool_calls": int(total(stale, "tool_calls")),
        "stale_discarded_tool_wall_seconds": total(stale, "tool_wall_seconds"),
        "stale_discarded_engine_step_seconds_attributed": total(
            stale, "engine_step_seconds_attributed"
        ),
        "consumed_queue_seconds": total(valid, "request_queue_seconds"),
        "consumed_queue_mean_seconds": queue_mean,
        "consumed_queue_p95_seconds": queue_p95,
        "consumed_queue_max_seconds": queue_max,
        "consumed_tool_seconds": total(valid, "tool_wall_seconds"),
        "consumed_tool_mean_seconds": tool_mean,
        "consumed_tool_p95_seconds": tool_p95,
        "consumed_tool_max_seconds": tool_max,
        "consumed_generate_seconds": total(valid, "generate_seconds"),
        "consumed_generate_mean_seconds": generate_mean,
        "consumed_generate_p95_seconds": generate_p95,
        "consumed_generate_max_seconds": generate_max,
        "batch_closing_trajectory_id": str(
            batch_closer.get("trajectory_id", "")
        ),
        "batch_closing_completion_unix": float(
            batch_closer.get("trajectory_completed_at_unix", 0.0)
        ),
        "batch_closing_queue_seconds": float(
            batch_closer.get("request_queue_seconds", 0.0)
        ),
        "batch_closing_tool_seconds": float(
            batch_closer.get("tool_wall_seconds", 0.0)
        ),
        "batch_closing_generate_seconds": float(
            batch_closer.get("generate_seconds", 0.0)
        ),
        "recorded_at_unix": (
            time.time() if recorded_at_unix is None else float(recorded_at_unix)
        ),
    }


def consume_utility_settle(settle_remaining: int) -> Tuple[bool, int]:
    """Return whether to record this observation and the next settle counter."""
    if settle_remaining > 0:
        return False, settle_remaining - 1
    return True, 0


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
        self.version_priority_ready_bypass_total = 0
        self.env_monitor = env_monitor

        self.current_step = None
        self.next_episode_id = 0
        self.groups: Dict[int, GroupData] = {}
        self.retired_groups: Dict[int, GroupData] = {}
        self.discard_records: List[Dict[str, Any]] = []
        self.discard_record_indices: Dict[Tuple[int, int, int], int] = {}
        self.dirty_discard_indices = set()
        self.progress_snapshots: Dict[Tuple[int, int], Dict[str, Any]] = {}

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

        self.quit = False

    def clear(self):
        self.current_step = None
        self.next_episode_id = 0
        self.groups.clear()
        self.retired_groups.clear()
        self.discard_records.clear()
        self.discard_record_indices.clear()
        self.dirty_discard_indices.clear()
        self.progress_snapshots.clear()

        self.progress = asyncio.Event()
        self.complete = asyncio.Event()

    def shutdown(self):
        self.stop_admission()
        self.groups.clear()

    def stop_admission(self):
        self.quit = True
        self.progress.set()

    @staticmethod
    def _metric_by_suffix(rollout: DataProto, suffix: str, default: float = 0.0) -> float:
        if rollout is None:
            return default
        metrics = rollout.meta_info.get("metrics", {}) if rollout.meta_info else {}
        for key, value in metrics.items():
            if key.endswith(suffix):
                if hasattr(value, "tolist"):
                    value = value.tolist()
                if isinstance(value, (list, tuple)):
                    numeric_values = []
                    for item in value:
                        if hasattr(item, "item"):
                            item = item.item()
                        try:
                            numeric_values.append(float(item))
                        except (TypeError, ValueError):
                            continue
                    if numeric_values:
                        # Trajectory totals can be duplicated by DataProto.concat.
                        return max(numeric_values)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    break

        # DataProto metadata is not guaranteed to survive every concat/Ray boundary.
        # The final sample carries the same counters in structured trajectory data.
        values = rollout.non_tensor_batch.get("trajectory_data") if rollout.non_tensor_batch else None
        trajectory_data = None
        if values is not None:
            for value in reversed(values):
                value = value.item() if hasattr(value, "item") else value
                if value is None or value == "":
                    continue
                try:
                    trajectory_data = json.loads(value) if isinstance(value, str) else value
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(trajectory_data, dict):
                    break
                trajectory_data = None

        if trajectory_data is None:
            return default

        waste_info = trajectory_data.get("waste_info", {})
        progress_info = trajectory_data.get("progress_info", {})
        version_info = trajectory_data.get("version_info", {})
        waste_fields = {
            "/traj_completed": "completed",
            "/traj_truncated": "truncated",
            "/traj_reset_failed": "reset_failed",
            "/traj_env_timeout": "env_timeout",
            "/traj_actions_completed": "actions_completed",
            "/traj_inference_calls": "inference_calls",
            "/traj_tool_calls": "tool_calls",
            "/traj_prompt_tokens_total": "prompt_tokens_total",
            "/traj_response_tokens_total": "response_tokens_total",
            "/traj_inference_tokens_total": "inference_tokens_total",
            "/traj_env_seconds_total": "env_seconds_total",
            "/traj_generate_seconds_total": "generate_seconds_total",
        }
        progress_fields = {
            "/traj_tool_wall_seconds_total": "traj_tool_wall_seconds_total",
            "/traj_request_queue_seconds_total": "traj_request_queue_seconds_total",
            "/traj_router_scheduling_wait_seconds_total": (
                "traj_router_scheduling_wait_seconds_total"
            ),
            "/traj_request_ttft_seconds_total": "traj_request_ttft_seconds_total",
            "/traj_request_prefill_seconds_total": (
                "traj_request_prefill_seconds_total"
            ),
            "/traj_request_decode_seconds_total": "traj_request_decode_seconds_total",
            "/traj_request_inference_seconds_total": (
                "traj_request_inference_seconds_total"
            ),
            "/traj_request_latency_seconds_total": "traj_request_latency_seconds_total",
            "/traj_model_forward_seconds_total": "traj_model_forward_seconds_total",
            "/traj_model_execute_seconds_total": "traj_model_execute_seconds_total",
            "/traj_engine_step_seconds_attributed_total": (
                "traj_engine_step_seconds_attributed_total"
            ),
            "/traj_prefill_engine_step_seconds_attributed_total": (
                "traj_prefill_engine_step_seconds_attributed_total"
            ),
            "/traj_decode_engine_step_seconds_attributed_total": (
                "traj_decode_engine_step_seconds_attributed_total"
            ),
            "/traj_engine_prefill_tokens_total": "traj_engine_prefill_tokens_total",
            "/traj_engine_cached_prompt_tokens_total": (
                "traj_engine_cached_prompt_tokens_total"
            ),
            "/traj_max_actions": "traj_max_actions",
            "/traj_remaining_actions": "traj_remaining_actions",
            "/traj_action_budget_progress": "traj_action_budget_progress",
            "/traj_started_at_unix": "traj_started_at_unix",
            "/traj_completed_at_unix": "traj_completed_at_unix",
        }
        version_fields = {
            "/traj_version_start": "version_start",
            "/traj_version_end": "version_end",
            "/traj_version_age": "version_age",
            "/traj_stale_tolerance": "stale_tolerance",
        }
        if suffix in waste_fields:
            value = waste_info.get(waste_fields[suffix], default)
        elif suffix in progress_fields:
            value = progress_info.get(progress_fields[suffix], default)
        elif suffix in version_fields:
            value = version_info.get(version_fields[suffix], default)
        elif suffix.startswith("/traj_actions_ge_"):
            try:
                threshold = int(suffix.rsplit("_", 1)[-1])
                value = int(waste_info.get("actions_completed", 0)) >= threshold
            except ValueError:
                value = default
        else:
            value = default
        try:
            return float(value)
        except (TypeError, ValueError):
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

    @staticmethod
    def _tensor_progress(rollout: DataProto) -> Dict[str, int]:
        try:
            response_mask = rollout.batch["response_mask"] > 0
            response_tokens = int(response_mask.sum().item())
        except (KeyError, AttributeError, TypeError):
            return {}
        if response_tokens <= 0:
            return {}
        segment_starts = response_mask.clone()
        if response_mask.shape[-1] > 1:
            segment_starts[..., 1:] &= ~response_mask[..., :-1]
        inference_calls = int(segment_starts.sum().item())
        try:
            attention_mask = rollout.batch["attention_mask"] > 0
            sequence_length = min(attention_mask.shape[-1], segment_starts.shape[-1])
            prefix_lengths = attention_mask[..., :sequence_length].cumsum(dim=-1) - 1
            prompt_tokens = int(
                prefix_lengths[segment_starts[..., :sequence_length]].sum().item()
            )
        except (KeyError, AttributeError, TypeError):
            try:
                prompt_tokens = int(rollout.batch["prompt_mask"].sum().item())
            except (KeyError, AttributeError, TypeError):
                prompt_tokens = 0
        try:
            current_context_tokens = int(
                (rollout.batch["attention_mask"] > 0).sum(dim=-1).max().item()
            )
        except (KeyError, AttributeError, TypeError):
            current_context_tokens = 0
        return {
            "actions_completed": inference_calls,
            "inference_calls": inference_calls,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "inference_tokens": prompt_tokens + response_tokens,
            "current_context_tokens": current_context_tokens,
        }

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
        record = {
            "trajectory_id": str(self._first_non_tensor_value(rollout, "traj_id", "unknown")),
            "category": "async_discard",
            "discard_reason": reason,
            "group_id": int(group.group_id),
            "episode_id": int(group.episode_id),
            "env_id": int(env_id),
            "version_start": int(metric(rollout, "/traj_version_start", group.create_step)),
            "version_end": int(metric(rollout, "/traj_version_end", group.create_step)),
            "version_age": max(0, int(step) - int(group.create_step)),
            "discarded_at_step": int(step),
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
            "tool_wall_seconds": float(
                metric(rollout, "/traj_tool_wall_seconds_total", 0)
            ),
            "request_queue_seconds": float(
                metric(rollout, "/traj_request_queue_seconds_total", 0)
            ),
            "router_scheduling_wait_seconds": float(
                metric(
                    rollout,
                    "/traj_router_scheduling_wait_seconds_total",
                    0,
                )
            ),
            "request_ttft_seconds": float(
                metric(rollout, "/traj_request_ttft_seconds_total", 0)
            ),
            "request_prefill_seconds": float(
                metric(rollout, "/traj_request_prefill_seconds_total", 0)
            ),
            "request_decode_seconds": float(
                metric(rollout, "/traj_request_decode_seconds_total", 0)
            ),
            "request_inference_seconds": float(
                metric(rollout, "/traj_request_inference_seconds_total", 0)
            ),
            "request_latency_seconds": float(
                metric(rollout, "/traj_request_latency_seconds_total", 0)
            ),
            "model_forward_seconds": float(
                metric(rollout, "/traj_model_forward_seconds_total", 0)
            ),
            "model_execute_seconds": float(
                metric(rollout, "/traj_model_execute_seconds_total", 0)
            ),
            "engine_step_seconds_attributed": float(
                metric(
                    rollout,
                    "/traj_engine_step_seconds_attributed_total",
                    0,
                )
            ),
            "prefill_engine_step_seconds_attributed": float(
                metric(
                    rollout,
                    "/traj_prefill_engine_step_seconds_attributed_total",
                    0,
                )
            ),
            "decode_engine_step_seconds_attributed": float(
                metric(
                    rollout,
                    "/traj_decode_engine_step_seconds_attributed_total",
                    0,
                )
            ),
            "engine_prefill_tokens": int(
                metric(rollout, "/traj_engine_prefill_tokens_total", 0)
            ),
            "engine_cached_prompt_tokens": int(
                metric(rollout, "/traj_engine_cached_prompt_tokens_total", 0)
            ),
            "max_actions": int(metric(rollout, "/traj_max_actions", 0)),
            "remaining_actions": int(
                metric(rollout, "/traj_remaining_actions", 0)
            ),
            "action_budget_progress": float(
                metric(rollout, "/traj_action_budget_progress", 0)
            ),
            "trajectory_started_at_unix": float(
                metric(rollout, "/traj_started_at_unix", 0)
            ),
            "trajectory_completed_at_unix": float(
                metric(rollout, "/traj_completed_at_unix", 0)
            ),
            "trajectory_discarded_at_unix": time.time(),
            "trajectory_wall_seconds": float(
                self._first_non_tensor_value(rollout, "traj_wall_seconds_total", 0)
            ),
        }
        tensor_progress = self._tensor_progress(rollout)
        for field, value in tensor_progress.items():
            record[field] = max(record.get(field, 0), value)
        if tensor_progress:
            record["completed"] = True
        direct_progress_fields = {
            "actions_completed": "traj_actions_completed",
            "inference_calls": "traj_inference_calls",
            "tool_calls": "traj_tool_calls",
            "prompt_tokens": "traj_prompt_tokens_total",
            "response_tokens": "traj_response_tokens_total",
            "inference_tokens": "traj_inference_tokens_total",
            "generate_seconds": "traj_generate_seconds_total",
            "env_seconds": "traj_env_seconds_total",
            "tool_wall_seconds": "traj_tool_wall_seconds_total",
            "request_queue_seconds": "traj_request_queue_seconds_total",
            "router_scheduling_wait_seconds": (
                "traj_router_scheduling_wait_seconds_total"
            ),
            "request_ttft_seconds": "traj_request_ttft_seconds_total",
            "request_prefill_seconds": "traj_request_prefill_seconds_total",
            "request_decode_seconds": "traj_request_decode_seconds_total",
            "request_inference_seconds": (
                "traj_request_inference_seconds_total"
            ),
            "request_latency_seconds": "traj_request_latency_seconds_total",
            "model_forward_seconds": "traj_model_forward_seconds_total",
            "model_execute_seconds": "traj_model_execute_seconds_total",
            "engine_step_seconds_attributed": (
                "traj_engine_step_seconds_attributed_total"
            ),
            "prefill_engine_step_seconds_attributed": (
                "traj_prefill_engine_step_seconds_attributed_total"
            ),
            "decode_engine_step_seconds_attributed": (
                "traj_decode_engine_step_seconds_attributed_total"
            ),
            "engine_prefill_tokens": "traj_engine_prefill_tokens_total",
            "engine_cached_prompt_tokens": (
                "traj_engine_cached_prompt_tokens_total"
            ),
            "max_actions": "traj_max_actions",
            "remaining_actions": "traj_remaining_actions",
            "action_budget_progress": "traj_action_budget_progress",
            "trajectory_started_at_unix": "traj_started_at_unix",
            "trajectory_completed_at_unix": "traj_completed_at_unix",
            "trajectory_wall_seconds": "traj_wall_seconds_total",
        }
        for field, key in direct_progress_fields.items():
            value = self._first_non_tensor_value(rollout, key, 0)
            try:
                record[field] = max(record[field], float(value))
            except (TypeError, ValueError):
                continue
        snapshot = self.progress_snapshots.get((int(group.episode_id), int(env_id)))
        if snapshot is not None:
            for field in (
                "actions_completed",
                "inference_calls",
                "tool_calls",
                "prompt_tokens",
                "response_tokens",
                "inference_tokens",
                "generate_seconds",
                "env_seconds",
                "tool_wall_seconds",
                "request_queue_seconds",
                "router_scheduling_wait_seconds",
                "request_ttft_seconds",
                "request_prefill_seconds",
                "request_decode_seconds",
                "request_inference_seconds",
                "request_latency_seconds",
                "model_forward_seconds",
                "model_execute_seconds",
                "engine_step_seconds_attributed",
                "prefill_engine_step_seconds_attributed",
                "decode_engine_step_seconds_attributed",
                "engine_prefill_tokens",
                "engine_cached_prompt_tokens",
                "max_actions",
                "action_budget_progress",
                "trajectory_started_at_unix",
                "trajectory_completed_at_unix",
                "trajectory_wall_seconds",
            ):
                value = snapshot.get(field)
                if value is None:
                    continue
                try:
                    record[field] = max(
                        float(record.get(field, 0) or 0),
                        float(value),
                    )
                except (TypeError, ValueError):
                    continue
            record["remaining_actions"] = int(
                snapshot.get(
                    "remaining_actions", record.get("remaining_actions", 0)
                )
            )
            for field in ("completed", "truncated", "reset_completed"):
                record[field] = bool(record[field] or snapshot.get(field, False))
            record["version_start"] = int(snapshot.get("version_start", record["version_start"]))
            record["version_end"] = int(snapshot.get("version_end", record["version_end"]))

        record["reset_only"] = bool(
            record.get("reset_completed", False)
            and int(record.get("inference_calls", 0)) == 0
        )
        if record["max_actions"] > 0:
            record["remaining_actions"] = max(
                0, record["max_actions"] - record["actions_completed"]
            )
            record["action_budget_progress"] = min(
                1.0, record["actions_completed"] / record["max_actions"]
            )

        record_key = (int(group.group_id), int(group.episode_id), int(env_id))
        existing_index = self.discard_record_indices.get(record_key)
        if existing_index is None:
            existing_index = len(self.discard_records)
            self.discard_record_indices[record_key] = existing_index
            self.discard_records.append(record)
        else:
            existing = self.discard_records[existing_index]
            for field in (
                "actions_completed",
                "inference_calls",
                "tool_calls",
                "prompt_tokens",
                "response_tokens",
                "inference_tokens",
                "generate_seconds",
                "env_seconds",
                "tool_wall_seconds",
                "request_queue_seconds",
                "router_scheduling_wait_seconds",
                "request_ttft_seconds",
                "request_prefill_seconds",
                "request_decode_seconds",
                "request_inference_seconds",
                "request_latency_seconds",
                "model_forward_seconds",
                "model_execute_seconds",
                "engine_step_seconds_attributed",
                "prefill_engine_step_seconds_attributed",
                "decode_engine_step_seconds_attributed",
                "engine_prefill_tokens",
                "engine_cached_prompt_tokens",
                "max_actions",
                "action_budget_progress",
                "trajectory_started_at_unix",
                "trajectory_completed_at_unix",
                "trajectory_wall_seconds",
            ):
                record[field] = max(record[field], existing.get(field, 0))
            if reason == "version_expired_late_return":
                record["discard_reason"] = reason
            if record["max_actions"] > 0:
                record["remaining_actions"] = max(
                    0, record["max_actions"] - record["actions_completed"]
                )
                record["action_budget_progress"] = min(
                    1.0, record["actions_completed"] / record["max_actions"]
                )
            self.discard_records[existing_index] = record
        self.dirty_discard_indices.add(existing_index)

    def record_discarded_group(self, group: GroupData, reason: str, observed_step: Optional[int] = None):
        for rollout in group.rollouts:
            self.record_discarded_rollout(rollout, group, reason, observed_step)

    def collect_new_discard_records(self) -> List[Dict[str, Any]]:
        records = [self.discard_records[index] for index in sorted(self.dirty_discard_indices)]
        self.dirty_discard_indices.clear()
        return records

    def update_progress_snapshots(self, snapshots: List[Dict[str, Any]]):
        for snapshot in snapshots:
            if int(snapshot.get("group_id", -1)) != self.group_id:
                continue
            key = (int(snapshot.get("episode_id", -1)), int(snapshot.get("env_id", -1)))
            existing = self.progress_snapshots.get(key)
            if existing is None:
                self.progress_snapshots[key] = snapshot
                continue
            merged = dict(existing)
            for field in (
                "actions_completed",
                "inference_calls",
                "tool_calls",
                "prompt_tokens",
                "response_tokens",
                "inference_tokens",
                "latest_prompt_tokens",
                "latest_response_tokens",
                "current_context_tokens",
                "max_actions",
                "generate_seconds",
                "env_seconds",
                "trajectory_wall_seconds",
            ):
                merged[field] = max(existing.get(field, 0), snapshot.get(field, 0))
            for field in ("completed", "truncated", "reset_completed"):
                merged[field] = bool(existing.get(field, False) or snapshot.get(field, False))
            merged["version_end"] = max(
                int(existing.get("version_end", 0)), int(snapshot.get("version_end", 0))
            )
            merged["remaining_actions"] = int(
                snapshot.get("remaining_actions", existing.get("remaining_actions", 0))
            )
            if snapshot.get("progress_source") == "router":
                merged["router_runtime_phase"] = str(
                    snapshot.get("runtime_phase", "router_last_request")
                )
            else:
                merged["runtime_phase"] = str(
                    snapshot.get(
                        "runtime_phase",
                        existing.get("runtime_phase", "unknown"),
                    )
                )
            self.progress_snapshots[key] = merged

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

    def trainable_progress_summary(self, group: GroupData) -> Dict[str, int]:
        """Summarize progress across the best candidates that can make the group trainable."""
        actions_by_env: Dict[Any, int] = {}
        gpu_invested_envs = set()
        for (episode_id, env_id), progress in self.progress_snapshots.items():
            if episode_id != group.episode_id:
                continue
            actions_by_env[env_id] = max(
                actions_by_env.get(env_id, 0),
                int(progress.get("actions_completed", 0)),
            )
            if (
                int(progress.get("inference_calls", 0)) > 0
                or int(progress.get("current_context_tokens", 0)) > 0
                or int(progress.get("inference_tokens", 0)) > 0
            ):
                gpu_invested_envs.add(env_id)

        anonymous_index = 0
        for rollout in group.rollouts:
            if rollout is None:
                continue
            env_id = self._first_non_tensor_value(rollout, "env_ids", None)
            if env_id is None:
                env_id = f"completed_{anonymous_index}"
                anonymous_index += 1
            actions = int(self._metric_by_suffix(rollout, "/traj_actions_completed", 0))
            tensor_progress = self._tensor_progress(rollout)
            actions = max(actions, int(tensor_progress.get("actions_completed", 0)))
            actions_by_env[env_id] = max(actions_by_env.get(env_id, 0), actions)
            if (
                int(self._metric_by_suffix(rollout, "/traj_inference_calls", 0)) > 0
                or int(tensor_progress.get("inference_calls", 0)) > 0
                or int(tensor_progress.get("current_context_tokens", 0)) > 0
            ):
                gpu_invested_envs.add(env_id)

        candidate_actions = sorted(actions_by_env.values(), reverse=True)
        candidate_actions.extend([0] * max(0, self.group_size - len(candidate_actions)))
        trainable_actions = candidate_actions[:self.group_size]
        if not trainable_actions:
            return {
                "mean_actions": 0,
                "frontier_actions": 0,
                "max_actions": 0,
                "observed_candidates": 0,
                "gpu_invested_candidates": 0,
            }
        return {
            "mean_actions": sum(trainable_actions) // len(trainable_actions),
            "frontier_actions": trainable_actions[-1],
            "max_actions": trainable_actions[0],
            "observed_candidates": min(len(actions_by_env), self.group_size),
            "gpu_invested_candidates": min(
                len(gpu_invested_envs), self.group_size
            ),
        }

    def trainable_frontier_actions(self, group: GroupData) -> int:
        return self.trainable_progress_summary(group)["frontier_actions"]

    def trainable_runtime_summary(self, group: GroupData) -> Dict[str, Any]:
        """Summarize remaining system service for the group frontier."""
        candidates: Dict[Any, Dict[str, Any]] = {}
        for (episode_id, env_id), progress in self.progress_snapshots.items():
            if episode_id != group.episode_id:
                continue
            actions = max(0, int(progress.get("actions_completed", 0)))
            max_actions = max(
                actions, int(progress.get("max_actions", actions))
            )
            remaining = max(
                0,
                int(
                    progress.get(
                        "remaining_actions", max_actions - actions
                    )
                ),
            )
            candidates[env_id] = {
                "actions": actions,
                "remaining_actions": remaining,
                "completed": bool(progress.get("completed", False)),
                "runtime_phase": str(
                    progress.get("runtime_phase", "unknown")
                ),
                "context_tokens": max(
                    0, int(progress.get("current_context_tokens", 0))
                ),
                "generate_seconds": max(
                    0.0, float(progress.get("generate_seconds", 0.0))
                ),
                "tool_seconds": max(
                    0.0, float(progress.get("env_seconds", 0.0))
                ),
                "inference_calls": max(
                    0, int(progress.get("inference_calls", 0))
                ),
                "tool_calls": max(0, int(progress.get("tool_calls", 0))),
            }

        anonymous_index = 0
        for rollout in group.rollouts:
            if rollout is None:
                continue
            env_id = self._first_non_tensor_value(rollout, "env_ids", None)
            if env_id is None:
                env_id = f"completed_{anonymous_index}"
                anonymous_index += 1
            actions = max(
                0,
                int(
                    self._metric_by_suffix(
                        rollout, "/traj_actions_completed", 0
                    )
                ),
            )
            candidates[env_id] = {
                "actions": actions,
                "remaining_actions": 0,
                "completed": True,
                "runtime_phase": "completed",
                "context_tokens": max(
                    0,
                    int(
                        self._metric_by_suffix(
                            rollout, "/traj_context_tokens", 0
                        )
                    ),
                ),
                "generate_seconds": max(
                    0.0,
                    float(
                        self._metric_by_suffix(
                            rollout, "/traj_generate_seconds_total", 0
                        )
                    ),
                ),
                "tool_seconds": max(
                    0.0,
                    float(
                        self._metric_by_suffix(
                            rollout, "/traj_env_seconds_total", 0
                        )
                    ),
                ),
                "inference_calls": max(
                    0,
                    int(
                        self._metric_by_suffix(
                            rollout, "/traj_inference_calls", actions
                        )
                    ),
                ),
                "tool_calls": max(
                    0,
                    int(
                        self._metric_by_suffix(
                            rollout, "/traj_tool_calls", 0
                        )
                    ),
                ),
            }

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                int(item["completed"]),
                int(item["actions"]),
                -int(item["remaining_actions"]),
            ),
            reverse=True,
        )
        default_remaining = max(
            [int(item["remaining_actions"]) for item in ordered] or [1]
        )
        while len(ordered) < self.group_size:
            ordered.append(
                {
                    "actions": 0,
                    "remaining_actions": default_remaining,
                    "completed": False,
                    "runtime_phase": "unstarted",
                    "context_tokens": 0,
                    "generate_seconds": 0.0,
                    "tool_seconds": 0.0,
                    "inference_calls": 0,
                    "tool_calls": 0,
                }
            )
        frontier = ordered[: self.group_size]
        inference_calls = sum(
            int(item["inference_calls"]) for item in frontier
        )
        service_actions = sum(
            max(int(item["actions"]), int(item["tool_calls"]))
            for item in frontier
        )
        return {
            "frontier_remaining_actions": max(
                int(item["remaining_actions"]) for item in frontier
            ),
            "mean_remaining_actions": sum(
                int(item["remaining_actions"]) for item in frontier
            )
            / max(1, len(frontier)),
            "mean_context_tokens": sum(
                int(item["context_tokens"]) for item in frontier
            )
            / max(1, len(frontier)),
            "inference_ready": sum(
                item["runtime_phase"]
                in {"ready_for_inference", "router_last_request"}
                for item in frontier
            ),
            "tool_waiting": sum(
                item["runtime_phase"] == "tool_or_environment"
                for item in frontier
            ),
            "generation_seconds_per_action": (
                sum(float(item["generate_seconds"]) for item in frontier)
                / inference_calls
                if inference_calls > 0
                else 0.0
            ),
            "tool_seconds_per_action": (
                sum(float(item["tool_seconds"]) for item in frontier)
                / service_actions
                if service_actions > 0
                else 0.0
            ),
        }

    def gpu_invested_trajectory_candidates(
        self, group: GroupData
    ) -> List[Tuple[str, int, int]]:
        """Return unfinished trajectory identities that already used inference work."""
        completed_env_ids = {
            int(self._first_non_tensor_value(rollout, "env_ids", -1))
            for rollout in group.rollouts
            if rollout is not None
        }
        candidates: Dict[int, Tuple[str, int, int]] = {}
        for (episode_id, env_id), progress in self.progress_snapshots.items():
            if episode_id != group.episode_id or int(env_id) in completed_env_ids:
                continue
            if bool(progress.get("completed", False)):
                continue
            if not (
                int(progress.get("inference_calls", 0)) > 0
                or int(progress.get("current_context_tokens", 0)) > 0
                or int(progress.get("inference_tokens", 0)) > 0
            ):
                continue
            trajectory_id = str(
                progress.get(
                    "trajectory_id",
                    f"{self.group_id}:{group.episode_id}:{int(env_id)}",
                )
            )
            actions = max(0, int(progress.get("actions_completed", 0)))
            existing = candidates.get(int(env_id))
            if existing is None or actions > existing[1]:
                candidates[int(env_id)] = (trajectory_id, actions, int(env_id))
        return list(candidates.values())

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
            return False
        group = self.groups[episode_id]
        assert start_step >= group.create_step, f"{start_step=} {group.create_step=}"
        group.rollouts.append(rollout)
        if len(group.rollouts) == self.group_size:
            if all(rollout is None for rollout in group.rollouts):
                logger.info(f"GroupQueue: group {self.group_id} exit")
                self.complete.set()
                return False
            elif self.group_filter.filter(group_id=self.group_id, episode_id=episode_id, group=group.rollouts):
                logger.info(f"filter rollout group {group.group_id} episode {group.episode_id}")
                self.group_filter_count += 1
                self.record_filtered_group(group)
                self.groups.pop(episode_id)
                if self.env_monitor:
                    self.env_monitor.cleanup_episode(self.group_id, episode_id)
                if self.fixed_step_admission:
                    self.advance_group(create_step=self.current_step)
                return False
            else:
                self.complete.set()
                self.progress_bar.update(self.group_size)
                return True
        return False

    async def get(self) -> GroupData:
        while True:
            while not self.groups:
                self.complete.clear()
                await self.complete.wait()
            if self.scheduling_policy == "version_priority":
                ready_episode_ids = [
                    episode_id
                    for episode_id, group in self.groups.items()
                    if len(group.rollouts) >= self.group_size
                ]
                if not ready_episode_ids:
                    self.complete.clear()
                    await self.complete.wait()
                    continue
                episode_id = min(
                    ready_episode_ids,
                    key=lambda key: (
                        self.groups[key].create_step,
                        self.groups[key].episode_id,
                    ),
                )
                selected_order = (
                    self.groups[episode_id].create_step,
                    self.groups[episode_id].episode_id,
                )
                if any(
                    (group.create_step, group.episode_id) < selected_order
                    and len(group.rollouts) < self.group_size
                    for group in self.groups.values()
                ):
                    self.version_priority_ready_bypass_total += 1
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
        if self.admission_policy not in ("step", "outstanding_watermark", "version_adaptive"):
            raise ValueError(f"Unsupported trajectory_admission_policy: {self.admission_policy}")
        configured_watermark = getattr(config, "max_outstanding_trajectories", None)
        if self.admission_policy in ("outstanding_watermark", "version_adaptive"):
            default_watermark = math.ceil(
                (1 + float(self.async_generation_ratio)) * int(config.rollout_batch_size)
            )
            self.max_outstanding_trajectories = int(configured_watermark or default_watermark)
        else:
            self.max_outstanding_trajectories = None
        self.admission_cursor = 0
        self.admitted_trajectories_total = 0
        self.admission_throttled_total = 0
        self.rollout_batch_size = int(config.rollout_batch_size) if self.mode == "train" else 0
        self.adaptive_reserve = int(getattr(config, "adaptive_admission_reserve_trajectories", 0))
        self.version_adaptive_progress_floor_enabled = bool(
            getattr(config, "version_adaptive_progress_floor_enabled", False)
        )
        self.version_runtime_reconcile_wait_seconds = float(
            getattr(config, "version_runtime_reconcile_wait_seconds", 30.0)
        )
        self.version_runtime_max_revisions_per_version = int(
            getattr(config, "version_runtime_max_revisions_per_version", 4)
        )
        self.adaptive_finish_ratio = float(
            getattr(config, "adaptive_admission_initial_finish_ratio", 0.5)
        )
        self.adaptive_ewma_alpha = float(getattr(config, "adaptive_admission_ewma_alpha", 0.5))
        self.bucketed_finish_enabled = bool(
            getattr(config, "adaptive_admission_bucketed_finish_enabled", False)
        )
        self.bucketed_finish_min_samples = int(
            getattr(config, "adaptive_admission_bucket_min_samples", 4)
        )
        self.runtime_estimator = RuntimeEstimator(
            initial_finish_ratio=self.adaptive_finish_ratio,
            ewma_alpha=self.adaptive_ewma_alpha,
            bucketed_finish_enabled=self.bucketed_finish_enabled,
            bucket_min_samples=self.bucketed_finish_min_samples,
        )
        # Existing fixed bucket metrics expose the coarse age/progress model.
        self.bucketed_finish_ratios = (
            self.runtime_estimator.coarse_bucket_finish_ratios
        )
        self.bucketed_finish_sample_counts = (
            self.runtime_estimator.coarse_bucket_sample_counts
        )
        self.dynamic_reserve_enabled = bool(
            getattr(config, "dynamic_admission_reserve_enabled", False)
        )
        self.dynamic_reserve_min = int(getattr(config, "dynamic_admission_reserve_min", 0))
        self.dynamic_reserve_max = int(getattr(config, "dynamic_admission_reserve_max", 8))
        self.dynamic_reserve_additive_step = int(
            getattr(config, "dynamic_admission_reserve_additive_step", 2)
        )
        self.dynamic_reserve_decay = float(
            getattr(config, "dynamic_admission_reserve_multiplicative_decay", 0.5)
        )
        self.dynamic_reserve_warmup_versions = int(
            getattr(config, "dynamic_admission_reserve_warmup_versions", 2)
        )
        self.dynamic_reserve_wait_high = float(
            getattr(config, "dynamic_admission_reserve_wait_high_seconds", 2.0)
        )
        self.dynamic_reserve_stale_high = float(
            getattr(config, "dynamic_admission_reserve_stale_high", 0.25)
        )
        self.dynamic_starvation_high = float(
            getattr(config, "dynamic_admission_starvation_high_fraction", 0.10)
        )
        self.dynamic_queue_high = float(
            getattr(config, "dynamic_admission_queue_high_fraction", 0.20)
        )
        self.dynamic_tool_wait_high = float(
            getattr(config, "dynamic_admission_tool_wait_high_fraction", 0.50)
        )
        self.dynamic_reserve_shadow_mode = bool(
            getattr(config, "dynamic_admission_reserve_shadow_mode", False)
        )
        self.dynamic_reserve_prediction_error_margin = float(
            getattr(config, "dynamic_admission_reserve_prediction_error_margin", 1.0)
        )
        self.dynamic_reserve_ewma_alpha = float(
            getattr(config, "dynamic_admission_reserve_ewma_alpha", 0.5)
        )
        self.dynamic_reserve_signal_patience = int(
            getattr(config, "dynamic_admission_reserve_signal_patience", 2)
        )
        self.dynamic_reserve_cooldown_versions = int(
            getattr(config, "dynamic_admission_reserve_cooldown_versions", 2)
        )
        self.dynamic_reserve_controller = str(
            getattr(
                config,
                "dynamic_admission_reserve_controller",
                "closed_loop_aimd",
            )
        )
        self.dynamic_utility_window_versions = int(
            getattr(config, "dynamic_admission_utility_window_versions", 4)
        )
        self.dynamic_utility_waste_weight = float(
            getattr(config, "dynamic_admission_utility_waste_weight", 1.0)
        )
        self.dynamic_utility_improvement_margin = float(
            getattr(config, "dynamic_admission_utility_improvement_margin", 0.05)
        )
        self.dynamic_utility_settle_versions = int(
            getattr(config, "dynamic_admission_utility_settle_versions", 2)
        )
        self.dynamic_utility_min_compute_efficiency = float(
            getattr(config, "dynamic_admission_utility_min_compute_efficiency", 0.95)
        )
        self.dynamic_learner_wait_ewma: Optional[float] = None
        self.dynamic_stale_ewma: Optional[float] = None
        self.dynamic_stale_record_tokens_seen: Dict[
            Tuple[int, int, int], int
        ] = {}
        self.dynamic_stale_record_ids_seen = set()
        self.dynamic_prediction_error_ewma: Optional[float] = None
        self.dynamic_reserve_update_reason = 5 if not self.dynamic_reserve_enabled else 4
        self.dynamic_reserve_pending_direction = 0
        self.dynamic_reserve_pending_count = 0
        self.dynamic_reserve_cooldown_remaining = 0
        self.dynamic_shadow_reserve = self.adaptive_reserve
        self.dynamic_shadow_update_reason = 5
        self.dynamic_shadow_pending_direction = 0
        self.dynamic_shadow_pending_count = 0
        self.dynamic_shadow_cooldown_remaining = 0
        self.dynamic_utility_direction = 1
        self.dynamic_utility_window_sum = 0.0
        self.dynamic_utility_window_count = 0
        self.dynamic_utility_window_response_tokens = 0
        self.dynamic_utility_window_consumed_tokens = 0
        self.dynamic_utility_window_stale_tokens = 0
        self.dynamic_utility_window_seconds = 0.0
        self.dynamic_utility_previous_window: Optional[float] = None
        self.dynamic_utility_last_window = 0.0
        self.dynamic_utility_last_window_efficiency = 1.0
        self.dynamic_utility_sample = 0.0
        self.dynamic_useful_token_rate = 0.0
        self.dynamic_stale_token_rate = 0.0
        self.dynamic_compute_efficiency = 1.0
        self.dynamic_last_observation_time = time.monotonic()
        self.dynamic_utility_settle_remaining = 0
        self.version_progress_topup_events = 0
        self.version_progress_topup_trajectories = 0
        self.version_runtime_revision = 0
        self.current_batch_missing = 0
        self.dynamic_reserve_increase_total = 0
        self.dynamic_reserve_decrease_total = 0
        self.dynamic_reserve_hold_total = 0
        self.version_admission_version = -1
        self.version_admission_budget = 0
        self.version_admission_budget_trainable = 0
        self.version_admission_used = 0
        self.version_admission_remaining = 0
        self.version_valid_ready_at_boundary = 0
        self.version_salvageable_inflight_at_boundary = 0
        self.version_invested_inflight_at_boundary = 0
        self.version_gpu_invested_inflight_at_boundary = 0
        self.version_reset_only_inflight_at_boundary = 0
        self.version_reserved_unstarted_at_boundary = 0
        self.version_near_expiry_at_boundary = 0
        self.version_expected_existing_supply = 0.0
        self.version_actual_existing_supply = 0
        self.version_actual_existing_consumed = 0
        self.version_admission_prediction_error = 0.0
        self.version_expected_inflight_supply = 0.0
        self.version_bucket_learned_population = 0
        self.version_bucket_fallback_population = 0
        self.version_unfinished_bucket_counts: Dict[str, int] = {}
        self.version_progress_observed_candidates = 0
        self.version_progress_mean_actions_sum = 0
        self.version_progress_frontier_actions_sum = 0
        self.version_progress_max_actions = 0
        self.version_runtime_controller = VersionAwareRuntimeController()
        self.version_runtime_plan: Optional[VersionRuntimePlan] = None
        self.version_runtime_forecast: Optional[VersionRuntimeForecast] = None
        self.version_runtime_last_outcome: Optional[VersionRuntimeOutcome] = None
        self.version_runtime_outcomes: List[VersionRuntimeOutcome] = []
        self.version_runtime_max_outcomes = max(
            1, int(getattr(config, "version_runtime_max_outcomes", 256))
        )
        self.version_runtime_boundary_started_at: Optional[float] = None
        self.version_runtime_last_learner_wait_seconds = 0.0
        self.version_runtime_undersupply_ewma: Optional[float] = None
        self.version_runtime_overload_ewma: Optional[float] = None
        self.version_runtime_starvation_ewma: Optional[float] = None
        self.version_runtime_waste_ewma: Optional[float] = None
        self.version_runtime_queue_pressure_ewma: Optional[float] = None
        self.version_runtime_tool_wait_fraction_ewma: Optional[float] = None
        self.version_runtime_inference_ready = 0
        self.version_runtime_tool_waiting = 0
        self.version_runtime_reserve_before = self.adaptive_reserve
        self.version_runtime_waste_totals = {
            "expired_trajectories": 0,
            "expired_actions": 0,
            "expired_tokens": 0,
            "expired_tool_calls": 0,
            "expired_tool_seconds": 0.0,
        }
        self.version_runtime_consumed_tokens_total = 0
        self.version_runtime_request_metric_totals: Dict[str, float] = {}
        self.latest_kv_feedback: Dict[str, Any] = {}
        self._tracked_existing_groups = set()
        self._tracked_unfinished_groups = set()
        self._tracked_unfinished_group_buckets: Dict[Tuple[int, int], str] = {}
        self._tracked_unfinished_bucket_counts: Dict[str, int] = {}
        self._tracked_unfinished_bucket_completed: Dict[str, int] = {}
        self._tracked_existing_consumed = 0
        self._tracked_unfinished_consumed = 0
        self._tracked_unfinished_completed = 0
        self.consumed_records: List[Dict[str, Any]] = []
        self.new_consumed_records: List[Dict[str, Any]] = []
        self.version_boundary_profiler_enabled = bool(
            getattr(config, "version_boundary_profiler_enabled", False)
        )
        self.version_boundary_profiler_max_records = max(
            0, int(getattr(config, "version_boundary_profiler_max_records", 4096))
        )
        self.version_boundary_events: List[Dict[str, Any]] = []
        self.latest_version_boundary_summary: Dict[str, Any] = {}
        self.rollout_started_at: Optional[float] = None
        self.rollout_finished_at: Optional[float] = None
        self.learner_wait_seconds_total = 0.0
        self.learner_wait_events = 0
        self.learner_wait_records: List[Dict[str, Any]] = []
        self.policy_update_traces: Dict[int, PolicyUpdateTrace] = {}

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

    def _decorate_boundary_record(
        self,
        record: Dict[str, Any],
        *,
        group: GroupData,
        to_version: int,
        state: str,
    ) -> Dict[str, Any]:
        decorated = dict(record)
        version_start = int(decorated.get("version_start", group.create_step))
        version_age = max(0, int(to_version) - version_start)
        decorated.update(
            group_id=int(group.group_id),
            episode_id=int(group.episode_id),
            version_start=version_start,
            version_age_at_boundary=version_age,
            boundary_version=int(to_version),
            runtime_state=str(state),
            will_expire=version_age > self.staleness_tolerance,
        )
        decorated.pop("discard_reason", None)
        decorated.pop("category", None)
        return decorated

    def _collect_version_boundary_records(self, to_version: int):
        records: List[Dict[str, Any]] = []
        reserved_unstarted = 0
        unobserved_started = 0
        observed_groups = set()

        for group_id, queue in self.group_queue.items():
            for episode_id, group in queue.groups.items():
                observed_groups.add((group_id, episode_id))
                completed_env_ids = set()
                for rollout in group.rollouts:
                    if rollout is None:
                        continue
                    record = self._completed_rollout_record(rollout, group)
                    env_id = int(record.get("env_id", -1))
                    completed_env_ids.add(env_id)
                    state = (
                        "completed_ready"
                        if len(group.rollouts) >= self.group_size
                        else "completed_partial_group"
                    )
                    records.append(
                        self._decorate_boundary_record(
                            record, group=group, to_version=to_version, state=state
                        )
                    )

                observed_active = 0
                for (snapshot_episode_id, env_id), snapshot in queue.progress_snapshots.items():
                    if snapshot_episode_id != episode_id or env_id in completed_env_ids:
                        continue
                    observed_active += 1
                    state = str(snapshot.get("runtime_phase", "running"))
                    records.append(
                        self._decorate_boundary_record(
                            snapshot, group=group, to_version=to_version, state=state
                        )
                    )

                started = max(len(group.rollouts), group.running_rollouts)
                unobserved_started += max(
                    0, started - len(completed_env_ids) - observed_active
                )
                reserved_unstarted += max(0, queue.admission_width - started)

        # Completed GroupQueue.get tasks have left queue.groups but still occupy
        # trainable supply until the learner consumes them.
        for task in self.pending_gets:
            if task.cancelled() or not task.done():
                continue
            try:
                group = task.result()
            except Exception:
                continue
            if (group.group_id, group.episode_id) in observed_groups:
                continue
            for rollout in group.rollouts:
                if rollout is None:
                    continue
                records.append(
                    self._decorate_boundary_record(
                        self._completed_rollout_record(rollout, group),
                        group=group,
                        to_version=to_version,
                        state="completed_pending_consume",
                    )
                )

        return records, reserved_unstarted, unobserved_started

    def _capture_version_boundary(self, from_version: int, to_version: int):
        records, reserved_unstarted, unobserved_started = (
            self._collect_version_boundary_records(to_version)
        )
        summary = summarize_version_boundary_records(
            records,
            from_version=from_version,
            to_version=to_version,
            staleness_tolerance=self.staleness_tolerance,
            reserved_unstarted=reserved_unstarted,
            unobserved_started=unobserved_started,
        )
        retained_records = records[:self.version_boundary_profiler_max_records]
        return {
            "timestamp": time.time(),
            "summary": summary,
            "pre_boundary_outstanding": self._outstanding_snapshot(to_version),
            "records_total": len(records),
            "records_truncated": max(0, len(records) - len(retained_records)),
            "records": retained_records,
        }

    def _version_supply_snapshot(self, observed_step: int) -> Dict[str, Any]:
        ready = 0
        unfinished = 0
        invested_inflight = 0
        gpu_invested_inflight = 0
        reserved_unstarted = 0
        near_expiry = 0
        existing_groups = set()
        unfinished_groups = set()
        unfinished_group_buckets: Dict[Tuple[int, int], str] = {}
        invested_candidate_groups: List[Tuple[int, int, int, int, int]] = []
        gpu_invested_candidate_groups: List[
            Tuple[int, int, int, int, int]
        ] = []
        gpu_invested_candidate_trajectories: List[
            Tuple[str, int, int, int, int, int]
        ] = []
        candidate_estimates: List[RuntimeCandidateEstimate] = []
        unfinished_bucket_counts: Dict[str, int] = {}
        unfinished_progress_observed_candidates = 0
        unfinished_progress_mean_actions_sum = 0
        unfinished_progress_frontier_actions_sum = 0
        unfinished_progress_max_actions = 0
        inference_ready_trajectories = 0
        tool_waiting_trajectories = 0
        near_expiry_age = max(0, self.staleness_tolerance - 1)

        for group_id, queue in self.group_queue.items():
            for episode_id, group in queue.groups.items():
                age = max(0, observed_step - group.create_step)
                if age > self.staleness_tolerance:
                    continue
                key = (group_id, episode_id)
                existing_groups.add(key)
                if len(group.rollouts) >= self.group_size:
                    ready += self.group_size
                else:
                    unfinished += self.group_size
                    unfinished_groups.add(key)
                    progress = queue.trainable_progress_summary(group)
                    runtime_summary = queue.trainable_runtime_summary(group)
                    inference_ready_trajectories += max(
                        0, int(runtime_summary.get("inference_ready", 0))
                    )
                    tool_waiting_trajectories += max(
                        0, int(runtime_summary.get("tool_waiting", 0))
                    )
                    bucket = runtime_finish_rate_bucket(
                        age, progress["mean_actions"], runtime_summary
                    )
                    unfinished_group_buckets[key] = bucket
                    unfinished_bucket_counts[bucket] = (
                        unfinished_bucket_counts.get(bucket, 0) + self.group_size
                    )
                    unfinished_progress_observed_candidates += progress["observed_candidates"]
                    unfinished_progress_mean_actions_sum += progress["mean_actions"]
                    unfinished_progress_frontier_actions_sum += progress["frontier_actions"]
                    unfinished_progress_max_actions = max(
                        unfinished_progress_max_actions, progress["max_actions"]
                    )
                    invested = min(
                        self.group_size,
                        max(len(group.rollouts), group.running_rollouts),
                    )
                    invested_inflight += invested
                    if invested > 0:
                        invested_candidate_groups.append(
                            (
                                group_id,
                                episode_id,
                                age,
                                progress["mean_actions"],
                                invested,
                            )
                        )
                        candidate_estimates.append(
                            self.runtime_estimator.estimate_candidate(
                                group_key=f"{group_id}:{episode_id}",
                                version_age=age,
                                staleness_tolerance=self.staleness_tolerance,
                                progress_actions=progress["mean_actions"],
                                invested_trajectories=invested,
                                finish_bucket=bucket,
                                runtime_summary=runtime_summary,
                            )
                        )
                    gpu_invested = min(
                        self.group_size,
                        int(progress["gpu_invested_candidates"]),
                    )
                    gpu_invested_inflight += gpu_invested
                    if gpu_invested > 0:
                        gpu_invested_candidate_groups.append(
                            (
                                group_id,
                                episode_id,
                                age,
                                progress["mean_actions"],
                                gpu_invested,
                            )
                        )
                    for trajectory_id, actions, env_id in (
                        queue.gpu_invested_trajectory_candidates(group)
                    ):
                        gpu_invested_candidate_trajectories.append(
                            (
                                trajectory_id,
                                group_id,
                                episode_id,
                                env_id,
                                age,
                                actions,
                            )
                        )
                    reserved_unstarted += self.group_size - invested
                if age >= near_expiry_age:
                    near_expiry += self.group_size

        # A completed GroupQueue.get task has already removed its group from queue.groups.
        for task in self.pending_gets:
            if task.cancelled() or not task.done():
                continue
            try:
                group = task.result()
            except Exception:
                continue
            age = max(0, observed_step - group.create_step)
            if age > self.staleness_tolerance:
                continue
            key = (group.group_id, group.episode_id)
            if key in existing_groups:
                continue
            existing_groups.add(key)
            ready += self.group_size
            if age >= near_expiry_age:
                near_expiry += self.group_size

        return {
            "valid_ready": ready,
            "salvageable_inflight": unfinished,
            "invested_inflight": invested_inflight,
            "gpu_invested_inflight": gpu_invested_inflight,
            "reset_only_inflight": max(
                0, invested_inflight - gpu_invested_inflight
            ),
            "reserved_unstarted": reserved_unstarted,
            "near_expiry": near_expiry,
            "existing_groups": existing_groups,
            "unfinished_groups": unfinished_groups,
            "unfinished_group_buckets": unfinished_group_buckets,
            "invested_candidate_groups": invested_candidate_groups,
            "gpu_invested_candidate_groups": gpu_invested_candidate_groups,
            "gpu_invested_candidate_trajectories": (
                gpu_invested_candidate_trajectories
            ),
            "candidate_estimates": candidate_estimates,
            "unfinished_bucket_counts": unfinished_bucket_counts,
            "unfinished_progress_observed_candidates": unfinished_progress_observed_candidates,
            "unfinished_progress_mean_actions_sum": unfinished_progress_mean_actions_sum,
            "unfinished_progress_frontier_actions_sum": unfinished_progress_frontier_actions_sum,
            "unfinished_progress_max_actions": unfinished_progress_max_actions,
            "inference_ready_trajectories": inference_ready_trajectories,
            "tool_waiting_trajectories": tool_waiting_trajectories,
        }

    def _record_version_adaptive_consumption(self, group: GroupData, count: int):
        if self.admission_policy != "version_adaptive":
            return
        key = (group.group_id, group.episode_id)
        if key in self._tracked_existing_groups:
            self._tracked_existing_consumed += count
        if key in self._tracked_unfinished_groups:
            self._tracked_unfinished_consumed += count

    def _record_version_adaptive_completion(self, group_id: int, episode_id: int):
        if self.admission_policy != "version_adaptive":
            return
        key = (group_id, episode_id)
        if key not in self._tracked_unfinished_groups:
            return
        self._tracked_unfinished_completed += self.group_size
        bucket = self._tracked_unfinished_group_buckets.get(key)
        if bucket is not None:
            self._tracked_unfinished_bucket_completed[bucket] = (
                self._tracked_unfinished_bucket_completed.get(bucket, 0) + self.group_size
            )

    def _predict_unfinished_supply(self, supply: Dict[str, Any]) -> Tuple[float, int, int]:
        expected, learned, fallback, _ = (
            self.runtime_estimator.predict_unfinished_supply(
                salvageable_inflight=supply["salvageable_inflight"],
                unfinished_bucket_counts=supply["unfinished_bucket_counts"],
            )
        )
        return expected, learned, fallback

    def _update_ewma(self, current: Optional[float], sample: float) -> float:
        if current is None:
            return float(sample)
        alpha = self.dynamic_reserve_ewma_alpha
        return alpha * float(sample) + (1 - alpha) * current

    def _merge_policy_update_trace(
        self, version: int, **updates: Any
    ) -> PolicyUpdateTrace:
        version = int(version)
        trace = self.policy_update_traces.get(version, PolicyUpdateTrace(version=version))
        allowed = set(PolicyUpdateTrace.__dataclass_fields__) - {"version"}
        normalized = {
            name: value
            for name, value in updates.items()
            if name in allowed and value is not None
        }
        incoming_finalized = bool(normalized.get("finalized", False))
        if trace.finalized and not incoming_finalized:
            for name in (
                "publish_activate_seconds",
                "update_interval_seconds",
                "other_seconds",
                "finalized",
            ):
                normalized.pop(name, None)
        if "finalized" in normalized:
            normalized["finalized"] = trace.finalized or bool(
                normalized["finalized"]
            )
        trace = replace(trace, **normalized)
        self.policy_update_traces[version] = trace
        return trace

    def record_policy_update_timing(
        self, version: int, timing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge driver-side critical-path timestamps into one version trace."""
        timing_fields = dict(timing)
        timing_fields.pop("version", None)
        trace = self._merge_policy_update_trace(int(version), **timing_fields)
        return trace.to_dict()

    def record_learner_wait(
        self,
        wait_seconds: float,
        step: Optional[int] = None,
        batch_size: int = 0,
    ):
        if self.mode != "train":
            return
        observed_wait = max(0.0, float(wait_seconds))
        self.learner_wait_seconds_total += observed_wait
        self.learner_wait_events += 1
        self.version_runtime_last_learner_wait_seconds = observed_wait
        self.dynamic_learner_wait_ewma = self._update_ewma(
            self.dynamic_learner_wait_ewma, observed_wait
        )
        if step is None:
            return
        discard_records = [
            record
            for group_queue in self.group_queue.values()
            for record in group_queue.discard_records
        ]
        wait_record = build_learner_wait_record(
            step=int(step),
            wait_seconds=wait_seconds,
            batch_size=batch_size,
            consumed_records=self.consumed_records,
            discard_records=discard_records,
        )
        self.learner_wait_records.append(wait_record)
        self._merge_policy_update_trace(
            int(step),
            batch_wait_seconds=observed_wait,
            consumed_queue_seconds=wait_record["consumed_queue_seconds"],
            consumed_queue_mean_seconds=wait_record[
                "consumed_queue_mean_seconds"
            ],
            consumed_queue_p95_seconds=wait_record[
                "consumed_queue_p95_seconds"
            ],
            consumed_queue_max_seconds=wait_record[
                "consumed_queue_max_seconds"
            ],
            consumed_tool_seconds=wait_record["consumed_tool_seconds"],
            consumed_tool_mean_seconds=wait_record["consumed_tool_mean_seconds"],
            consumed_tool_p95_seconds=wait_record["consumed_tool_p95_seconds"],
            consumed_tool_max_seconds=wait_record["consumed_tool_max_seconds"],
            consumed_generate_seconds=wait_record["consumed_generate_seconds"],
            consumed_generate_mean_seconds=wait_record[
                "consumed_generate_mean_seconds"
            ],
            consumed_generate_p95_seconds=wait_record[
                "consumed_generate_p95_seconds"
            ],
            consumed_generate_max_seconds=wait_record[
                "consumed_generate_max_seconds"
            ],
            batch_closing_trajectory_id=wait_record[
                "batch_closing_trajectory_id"
            ],
            batch_closing_completion_unix=wait_record[
                "batch_closing_completion_unix"
            ],
            batch_closing_queue_seconds=wait_record[
                "batch_closing_queue_seconds"
            ],
            batch_closing_tool_seconds=wait_record[
                "batch_closing_tool_seconds"
            ],
            batch_closing_generate_seconds=wait_record[
                "batch_closing_generate_seconds"
            ],
        )

    def _update_dynamic_reserve(self, version: int):
        if not self.dynamic_reserve_enabled or self.admission_policy != "version_adaptive":
            self.dynamic_reserve_update_reason = 5
            self.dynamic_shadow_update_reason = 5
            return
        if self.dynamic_reserve_controller == "utility_hill_climb":
            self._update_utility_reserve(version)
            return
        shadow_mode = self.dynamic_reserve_shadow_mode
        previous = (
            self.dynamic_shadow_reserve if shadow_mode else self.adaptive_reserve
        )
        if self.dynamic_reserve_controller == "state_feedback":
            candidate, candidate_reason = compute_state_feedback_reserve(
                previous,
                version,
                self.version_runtime_starvation_ewma or 0.0,
                self.version_runtime_waste_ewma or 0.0,
                self.version_runtime_queue_pressure_ewma or 0.0,
                self.version_runtime_inference_ready,
                self.version_runtime_tool_wait_fraction_ewma or 0.0,
                self.rollout_batch_size,
                reserve_min=self.dynamic_reserve_min,
                reserve_max=self.dynamic_reserve_max,
                additive_step=self.dynamic_reserve_additive_step,
                warmup_versions=self.dynamic_reserve_warmup_versions,
                starvation_high=self.dynamic_starvation_high,
                waste_high=self.dynamic_reserve_stale_high,
                queue_high=self.dynamic_queue_high,
                tool_wait_high=self.dynamic_tool_wait_high,
            )
        elif self.dynamic_reserve_controller == "closed_loop_aimd":
            candidate, candidate_reason = compute_closed_loop_reserve(
                previous,
                version,
                self.version_runtime_undersupply_ewma or 0.0,
                self.version_runtime_overload_ewma or 0.0,
                reserve_min=self.dynamic_reserve_min,
                reserve_max=self.dynamic_reserve_max,
                additive_step=self.dynamic_reserve_additive_step,
                multiplicative_decay=self.dynamic_reserve_decay,
                warmup_versions=self.dynamic_reserve_warmup_versions,
                wait_high=self.dynamic_reserve_wait_high,
                overload_high=self.dynamic_reserve_stale_high,
            )
        else:
            candidate, candidate_reason = compute_dynamic_reserve(
                previous,
                version,
                self.dynamic_learner_wait_ewma or 0.0,
                self.dynamic_stale_ewma or 0.0,
                self.dynamic_prediction_error_ewma or 0.0,
                reserve_min=self.dynamic_reserve_min,
                reserve_max=self.dynamic_reserve_max,
                additive_step=self.dynamic_reserve_additive_step,
                multiplicative_decay=self.dynamic_reserve_decay,
                warmup_versions=self.dynamic_reserve_warmup_versions,
                wait_high=self.dynamic_reserve_wait_high,
                stale_high=self.dynamic_reserve_stale_high,
                prediction_error_margin=(
                    self.dynamic_reserve_prediction_error_margin
                ),
            )
        pending_direction = (
            self.dynamic_shadow_pending_direction
            if shadow_mode
            else self.dynamic_reserve_pending_direction
        )
        pending_count = (
            self.dynamic_shadow_pending_count
            if shadow_mode
            else self.dynamic_reserve_pending_count
        )
        cooldown_remaining = (
            self.dynamic_shadow_cooldown_remaining
            if shadow_mode
            else self.dynamic_reserve_cooldown_remaining
        )
        updated, reason, pending_direction, pending_count, cooldown_remaining = (
            apply_dynamic_reserve_hysteresis(
                previous,
                candidate,
                candidate_reason,
                pending_direction,
                pending_count,
                cooldown_remaining,
                signal_patience=self.dynamic_reserve_signal_patience,
                cooldown_versions=self.dynamic_reserve_cooldown_versions,
            )
        )
        if shadow_mode:
            self.dynamic_shadow_reserve = updated
            self.dynamic_shadow_update_reason = reason
            self.dynamic_shadow_pending_direction = pending_direction
            self.dynamic_shadow_pending_count = pending_count
            self.dynamic_shadow_cooldown_remaining = cooldown_remaining
            self.dynamic_reserve_update_reason = reason
            self.dynamic_reserve_hold_total += 1
            return
        self.adaptive_reserve = updated
        self.dynamic_reserve_update_reason = reason
        self.dynamic_reserve_pending_direction = pending_direction
        self.dynamic_reserve_pending_count = pending_count
        self.dynamic_reserve_cooldown_remaining = cooldown_remaining
        if updated > previous:
            self.dynamic_reserve_increase_total += 1
        elif updated < previous:
            self.dynamic_reserve_decrease_total += 1
        else:
            self.dynamic_reserve_hold_total += 1

    def _update_utility_reserve(self, version: int):
        previous = self.adaptive_reserve
        if version < self.dynamic_reserve_warmup_versions:
            self.dynamic_utility_window_sum = 0.0
            self.dynamic_utility_window_count = 0
            self._reset_utility_window_totals()
            self.dynamic_reserve_update_reason = 4
            self.dynamic_reserve_hold_total += 1
            return
        if self.dynamic_utility_window_count < self.dynamic_utility_window_versions:
            self.dynamic_reserve_update_reason = 6
            self.dynamic_reserve_hold_total += 1
            return

        window_utility, _, _, window_efficiency = compute_effective_rollout_utility(
            self.dynamic_utility_window_response_tokens,
            self.dynamic_utility_window_consumed_tokens,
            self.dynamic_utility_window_stale_tokens,
            self.dynamic_utility_window_seconds,
            self.dynamic_utility_waste_weight,
        )
        updated, direction, reason = update_constrained_utility_hill_climb(
            previous,
            self.dynamic_utility_direction,
            window_utility,
            self.dynamic_utility_previous_window,
            window_efficiency,
            min_compute_efficiency=self.dynamic_utility_min_compute_efficiency,
            reserve_min=self.dynamic_reserve_min,
            reserve_max=self.dynamic_reserve_max,
            additive_step=self.dynamic_reserve_additive_step,
            improvement_margin=self.dynamic_utility_improvement_margin,
        )
        self.dynamic_utility_previous_window = window_utility
        self.dynamic_utility_last_window = window_utility
        self.dynamic_utility_last_window_efficiency = window_efficiency
        self.dynamic_utility_window_sum = 0.0
        self.dynamic_utility_window_count = 0
        self._reset_utility_window_totals()
        self.dynamic_utility_direction = direction
        self.adaptive_reserve = updated
        self.dynamic_reserve_update_reason = reason
        if updated != previous:
            self.dynamic_utility_settle_remaining = self.dynamic_utility_settle_versions
        if updated > previous:
            self.dynamic_reserve_increase_total += 1
        elif updated < previous:
            self.dynamic_reserve_decrease_total += 1
        else:
            self.dynamic_reserve_hold_total += 1

    def _reset_utility_window_totals(self):
        self.dynamic_utility_window_response_tokens = 0
        self.dynamic_utility_window_consumed_tokens = 0
        self.dynamic_utility_window_stale_tokens = 0
        self.dynamic_utility_window_seconds = 0.0

    def _admit_version_budget(self, create_step: int):
        queues = [queue for queue in self.group_queue.values() if not queue.quit]
        if not queues:
            return
        width = queues[0].admission_width
        touched = set()
        while self.version_admission_remaining >= width:
            queue = queues[self.admission_cursor % len(queues)]
            self.admission_cursor += 1
            queue.advance_group(create_step)
            touched.add(queue.group_id)
            self.version_admission_remaining -= width
            self.version_admission_used += width
            self.admitted_trajectories_total += width
        for group_id in touched:
            self.group_queue[group_id].progress.set()

    def reconcile_version_progress(
        self,
        current_step: int,
        missing_trajectories: int,
        learner_wait_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        """Revise the active plan when predicted carry-over fails to materialize."""
        if (
            not self.version_adaptive_progress_floor_enabled
            or self.admission_policy != "version_adaptive"
            or not self.group_queue
            or current_step != self.version_admission_version
            or self.version_runtime_plan is None
        ):
            return None
        supply = self._version_supply_snapshot(current_step)
        forecast = self.runtime_estimator.build_forecast(
            current_step,
            ready_valid_slots=supply["valid_ready"],
            salvageable_inflight=supply["salvageable_inflight"],
            unfinished_bucket_counts=supply["unfinished_bucket_counts"],
        )
        forecast = replace(
            forecast,
            forecast_id=(
                f"{forecast.forecast_id}-reconcile-"
                f"{self.version_runtime_revision + 1}"
            ),
        )
        potential = forecast.expected_existing_supply
        queues = [queue for queue in self.group_queue.values() if not queue.quit]
        if not queues:
            return None
        width = queues[0].admission_width
        outstanding = self._outstanding_snapshot(current_step)["outstanding_trajectories"]
        state = VersionRuntimeState(
            version=current_step,
            learner_demand=self.rollout_batch_size,
            safety_reserve=self.adaptive_reserve,
            expected_existing_supply=potential,
            outstanding_trajectories=outstanding,
            max_outstanding_trajectories=self.max_outstanding_trajectories,
            admission_width=width,
            group_size=self.group_size,
            staleness_tolerance=self.staleness_tolerance,
            invested_candidate_groups=tuple(
                supply["invested_candidate_groups"]
            ),
            gpu_invested_candidate_groups=tuple(
                supply["gpu_invested_candidate_groups"]
            ),
            gpu_invested_candidate_trajectories=tuple(
                supply["gpu_invested_candidate_trajectories"]
            ),
            exact_rebuild_cohort=True,
            admission_enabled=True,
            priority_enabled=self.scheduling_policy == "version_priority",
            revision=self.version_runtime_revision,
            kv_feedback_requests=int(
                self.latest_kv_feedback.get("requests", 0)
            ),
            kv_feedback_hit_ratio=float(
                self.latest_kv_feedback.get("hit_ratio", 0.0)
            ),
            kv_feedback_resets=int(
                self.latest_kv_feedback.get("resets", 0)
            ),
            forecast_id=forecast.forecast_id,
            estimator_revision=forecast.estimator_revision,
            ready_valid_slots=forecast.ready_valid_slots,
            predicted_inflight_slots=forecast.predicted_inflight_slots,
            reserve_before=self.version_runtime_reserve_before,
            undersupply_signal=(
                self.version_runtime_undersupply_ewma or 0.0
            ),
            overload_signal=self.version_runtime_overload_ewma or 0.0,
            candidate_estimates=tuple(supply["candidate_estimates"]),
        )
        revised_plan = self.version_runtime_controller.decide(
            state,
            active_plan=self.version_runtime_plan,
            missing_trajectories=missing_trajectories,
            current_batch_missing=self.current_batch_missing,
            learner_wait_seconds=learner_wait_seconds,
            reconcile_wait_seconds=self.version_runtime_reconcile_wait_seconds,
            max_revisions_per_version=(
                self.version_runtime_max_revisions_per_version
            ),
            max_admission_groups=1,
        )
        if revised_plan is None:
            return None

        admitted = revised_plan.admission_delta_trajectories
        admitted_groups = admitted // width
        touched = set()
        for _ in range(admitted_groups):
            queue = queues[self.admission_cursor % len(queues)]
            self.admission_cursor += 1
            queue.advance_group(current_step)
            touched.add(queue.group_id)
        for group_id in touched:
            self.group_queue[group_id].progress.set()

        self.admitted_trajectories_total += admitted
        self.version_progress_topup_events += 1
        self.version_progress_topup_trajectories += admitted
        self.version_runtime_revision = revised_plan.revision
        self.version_admission_budget = revised_plan.admission_budget
        self.version_admission_budget_trainable = (
            revised_plan.admission_budget_trainable
        )
        self.version_admission_used += admitted
        self.version_runtime_plan = revised_plan
        self.version_runtime_forecast = forecast
        # Supervise the revised forecast against the exact pre-top-up pool.
        # The groups admitted above are new supply and must not be credited to
        # the existing-pool forecast that caused this revision.
        self.version_valid_ready_at_boundary = forecast.ready_valid_slots
        self.version_salvageable_inflight_at_boundary = supply[
            "salvageable_inflight"
        ]
        self.version_expected_inflight_supply = (
            forecast.predicted_inflight_slots
        )
        self.version_expected_existing_supply = (
            forecast.expected_existing_supply
        )
        self.version_unfinished_bucket_counts = dict(
            supply["unfinished_bucket_counts"]
        )
        self._tracked_existing_groups = set(supply["existing_groups"])
        self._tracked_unfinished_groups = set(supply["unfinished_groups"])
        self._tracked_unfinished_group_buckets = dict(
            supply["unfinished_group_buckets"]
        )
        self._tracked_unfinished_bucket_counts = dict(
            supply["unfinished_bucket_counts"]
        )
        self._tracked_unfinished_bucket_completed = {}
        self._tracked_existing_consumed = 0
        self._tracked_unfinished_consumed = 0
        self._tracked_unfinished_completed = 0
        return self.version_runtime_plan.to_dict()

    def _build_version_runtime_plan(
        self,
        create_step: int,
        supply: Dict[str, Any],
        expected_existing_supply: float,
        *,
        admission_enabled: bool,
    ) -> VersionRuntimePlan:
        width = next(iter(self.group_queue.values())).admission_width
        forecast = self.version_runtime_forecast
        if forecast is None or forecast.version != int(create_step):
            forecast = self.runtime_estimator.build_forecast(
                create_step,
                ready_valid_slots=supply["valid_ready"],
                salvageable_inflight=supply["salvageable_inflight"],
                unfinished_bucket_counts=supply["unfinished_bucket_counts"],
            )
            self.version_runtime_forecast = forecast
        outstanding = self._outstanding_snapshot(create_step)[
            "outstanding_trajectories"
        ]
        max_outstanding = (
            self.max_outstanding_trajectories
            if self.max_outstanding_trajectories is not None
            else outstanding
        )
        state = VersionRuntimeState(
            version=create_step,
            learner_demand=self.rollout_batch_size,
            safety_reserve=self.adaptive_reserve if admission_enabled else 0,
            expected_existing_supply=expected_existing_supply,
            outstanding_trajectories=outstanding,
            max_outstanding_trajectories=max_outstanding,
            admission_width=width,
            group_size=self.group_size,
            staleness_tolerance=self.staleness_tolerance,
            invested_candidate_groups=tuple(
                supply["invested_candidate_groups"]
            ),
            gpu_invested_candidate_groups=tuple(
                supply["gpu_invested_candidate_groups"]
            ),
            gpu_invested_candidate_trajectories=tuple(
                supply["gpu_invested_candidate_trajectories"]
            ),
            exact_rebuild_cohort=True,
            admission_enabled=admission_enabled,
            priority_enabled=self.scheduling_policy == "version_priority",
            revision=self.version_runtime_revision,
            kv_feedback_requests=int(
                self.latest_kv_feedback.get("requests", 0)
            ),
            kv_feedback_hit_ratio=float(
                self.latest_kv_feedback.get("hit_ratio", 0.0)
            ),
            kv_feedback_resets=int(
                self.latest_kv_feedback.get("resets", 0)
            ),
            forecast_id=forecast.forecast_id,
            estimator_revision=forecast.estimator_revision,
            ready_valid_slots=forecast.ready_valid_slots,
            predicted_inflight_slots=forecast.predicted_inflight_slots,
            reserve_before=self.version_runtime_reserve_before,
            undersupply_signal=(
                self.version_runtime_undersupply_ewma or 0.0
            ),
            overload_signal=self.version_runtime_overload_ewma or 0.0,
            candidate_estimates=tuple(supply["candidate_estimates"]),
        )
        plan = self.version_runtime_controller.decide(state)
        assert plan is not None
        return plan

    def _refresh_nonadaptive_runtime_plan(self, create_step: int):
        """Publish scheduling/KV decisions even when admission is an ablation."""
        if self.mode != "train" or not self.group_queue:
            return
        supply = self._version_supply_snapshot(create_step)
        expected_inflight, _, _ = self._predict_unfinished_supply(supply)
        self.version_runtime_plan = self._build_version_runtime_plan(
            create_step,
            supply,
            supply["valid_ready"] + expected_inflight,
            admission_enabled=False,
        )

    def _runtime_expired_waste_totals(self) -> Dict[str, Any]:
        """Return cumulative version-expiration cost with late updates folded in."""
        records: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for queue in self.group_queue.values():
            for record in queue.discard_records:
                if not str(record.get("discard_reason", "")).startswith(
                    "version_"
                ):
                    continue
                key = (
                    int(record.get("group_id", queue.group_id)),
                    int(record.get("episode_id", -1)),
                    int(record.get("env_id", -1)),
                )
                existing = records.get(key)
                if existing is None or int(record.get("inference_tokens", 0)) >= int(
                    existing.get("inference_tokens", 0)
                ):
                    records[key] = record
        return {
            "expired_trajectories": len(records),
            "expired_actions": sum(
                max(0, int(record.get("actions_completed", 0)))
                for record in records.values()
            ),
            "expired_tokens": sum(
                max(0, int(record.get("inference_tokens", 0)))
                for record in records.values()
            ),
            "expired_tool_calls": sum(
                max(0, int(record.get("tool_calls", 0)))
                for record in records.values()
            ),
            "expired_tool_seconds": sum(
                max(
                    0.0,
                    float(
                        record.get(
                            "tool_wall_seconds",
                            record.get("env_seconds", 0.0),
                        )
                    ),
                )
                for record in records.values()
            ),
        }

    @staticmethod
    def _runtime_counter_delta(
        current: Dict[str, Any], previous: Dict[str, Any]
    ) -> Dict[str, Any]:
        delta: Dict[str, Any] = {}
        for field, value in current.items():
            if isinstance(value, float):
                delta[field] = max(
                    0.0, float(value) - float(previous.get(field, 0.0))
                )
            else:
                delta[field] = max(
                    0, int(value) - int(previous.get(field, 0))
                )
        return delta

    def _finalize_version_runtime_outcome(
        self,
        next_version: int,
        boundary_supply: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Close the previous plan before constructing the next forecast."""
        if (
            self.version_runtime_plan is None
            or self.version_admission_version < 0
        ):
            self.version_runtime_boundary_started_at = time.monotonic()
            return

        now = time.monotonic()
        if boundary_supply is None:
            boundary_supply = self._version_supply_snapshot(next_version)
        interval_seconds = (
            max(0.0, now - self.version_runtime_boundary_started_at)
            if self.version_runtime_boundary_started_at is not None
            else self.version_runtime_last_learner_wait_seconds
        )
        policy_trace = self.policy_update_traces.get(
            self.version_runtime_plan.version
        )
        if (
            policy_trace is not None
            and policy_trace.finalized
            and policy_trace.update_interval_seconds > 0
        ):
            interval_seconds = policy_trace.update_interval_seconds
            observed_batch_wait = policy_trace.batch_wait_seconds
        else:
            observed_batch_wait = self.version_runtime_last_learner_wait_seconds
        current_waste = self._runtime_expired_waste_totals()
        waste_delta = self._runtime_counter_delta(
            current_waste, self.version_runtime_waste_totals
        )
        self.version_runtime_waste_totals = current_waste

        consumed_tokens_total = sum(
            max(0, int(record.get("inference_tokens", 0)))
            for record in self.consumed_records
        )
        consumed_tokens_delta = max(
            0,
            consumed_tokens_total
            - self.version_runtime_consumed_tokens_total,
        )
        self.version_runtime_consumed_tokens_total = consumed_tokens_total
        request_metric_delta: Dict[str, Any] = {}
        request_metrics = self.latest_kv_feedback.get("request_metrics")
        if isinstance(request_metrics, dict):
            current_request_metrics = {
                str(name): float(value)
                for name, value in request_metrics.items()
                if isinstance(value, (int, float))
            }
            request_metric_delta = self._runtime_counter_delta(
                current_request_metrics,
                self.version_runtime_request_metric_totals,
            )
            self.version_runtime_request_metric_totals = current_request_metrics
        expired_tokens = int(waste_delta["expired_tokens"])
        overload_denominator = expired_tokens + consumed_tokens_delta
        expired_overload_signal = (
            expired_tokens / overload_denominator
            if overload_denominator > 0
            else 0.0
        )
        scheduling_requests = int(
            request_metric_delta.get("router/scheduling_decisions", 0)
        )
        scheduling_wait_seconds = float(
            request_metric_delta.get("router/scheduling_wait_seconds", 0.0)
        )
        scheduling_wait_mean_seconds = (
            scheduling_wait_seconds / scheduling_requests
            if scheduling_requests > 0
            else 0.0
        )
        inference_queue_seconds = float(
            request_metric_delta.get("vllm/request_queue_seconds", 0.0)
        )
        inference_service_seconds = float(
            request_metric_delta.get("vllm/request_inference_seconds", 0.0)
        )
        if inference_service_seconds <= 0:
            inference_service_seconds = float(
                request_metric_delta.get("vllm/request_prefill_seconds", 0.0)
            ) + float(request_metric_delta.get("vllm/request_decode_seconds", 0.0))
        queue_denominator = inference_queue_seconds + inference_service_seconds
        queue_overload_signal = (
            inference_queue_seconds / queue_denominator
            if queue_denominator > 0
            else 0.0
        )
        overload_signal = max(
            expired_overload_signal, queue_overload_signal
        )
        undersupply_signal = max(
            0.0, observed_batch_wait
        )
        starvation_signal = (
            undersupply_signal / interval_seconds if interval_seconds > 0 else 0.0
        )
        inference_ready = max(
            0, int(boundary_supply.get("inference_ready_trajectories", 0))
        )
        tool_waiting = max(
            0, int(boundary_supply.get("tool_waiting_trajectories", 0))
        )
        readiness_population = inference_ready + tool_waiting
        tool_wait_fraction = (
            tool_waiting / readiness_population if readiness_population > 0 else 0.0
        )

        outcome = VersionRuntimeOutcome(
            plan_id=self.version_runtime_plan.plan_id,
            forecast_id=self.version_runtime_plan.forecast_id,
            version=self.version_admission_version,
            final_revision=self.version_runtime_plan.revision,
            predicted_existing_supply=(
                self.version_runtime_plan.expected_existing_supply
            ),
            actual_existing_valid_slots=self.version_actual_existing_supply,
            admitted_trajectories=self.version_admission_used,
            completed_valid_slots=self.version_actual_existing_supply,
            consumed_valid_slots=self.version_actual_existing_consumed,
            learner_wait_seconds=observed_batch_wait,
            next_batch_latency_seconds=observed_batch_wait,
            policy_update_interval_seconds=interval_seconds,
            expired_trajectories=int(waste_delta["expired_trajectories"]),
            expired_actions=int(waste_delta["expired_actions"]),
            expired_tokens=expired_tokens,
            expired_tool_calls=int(waste_delta["expired_tool_calls"]),
            expired_tool_seconds=float(waste_delta["expired_tool_seconds"]),
            reprefill_tokens=int(
                request_metric_delta.get("router/rebuild_prefill_tokens", 0)
            ),
            prefill_tokens=int(
                request_metric_delta.get("vllm/request_prefill_tokens", 0)
            ),
            prefill_seconds=float(
                request_metric_delta.get("vllm/request_prefill_seconds", 0.0)
            ),
            scheduling_wait_seconds=scheduling_wait_seconds,
            scheduling_requests=scheduling_requests,
            scheduling_wait_mean_seconds=scheduling_wait_mean_seconds,
            inference_queue_seconds=inference_queue_seconds,
            inference_service_seconds=inference_service_seconds,
            queue_pressure=queue_overload_signal,
            starvation_fraction=starvation_signal,
            waste_fraction=expired_overload_signal,
            inference_ready_trajectories=inference_ready,
            tool_waiting_trajectories=tool_waiting,
            tool_wait_fraction=tool_wait_fraction,
            completion_eta_absolute_error_seconds=float(
                request_metric_delta.get(
                    "router/completion_eta_absolute_error_seconds", 0.0
                )
            ),
        )
        self.version_runtime_last_outcome = outcome
        self.version_runtime_outcomes.append(outcome)
        if len(self.version_runtime_outcomes) > self.version_runtime_max_outcomes:
            del self.version_runtime_outcomes[
                : -self.version_runtime_max_outcomes
            ]

        self.version_runtime_undersupply_ewma = self._update_ewma(
            self.version_runtime_undersupply_ewma, undersupply_signal
        )
        self.version_runtime_overload_ewma = self._update_ewma(
            self.version_runtime_overload_ewma, overload_signal
        )
        self.version_runtime_starvation_ewma = self._update_ewma(
            self.version_runtime_starvation_ewma, starvation_signal
        )
        self.version_runtime_waste_ewma = self._update_ewma(
            self.version_runtime_waste_ewma, expired_overload_signal
        )
        self.version_runtime_queue_pressure_ewma = self._update_ewma(
            self.version_runtime_queue_pressure_ewma, queue_overload_signal
        )
        self.version_runtime_tool_wait_fraction_ewma = self._update_ewma(
            self.version_runtime_tool_wait_fraction_ewma, tool_wait_fraction
        )
        self.version_runtime_inference_ready = inference_ready
        self.version_runtime_tool_waiting = tool_waiting
        self._merge_policy_update_trace(
            outcome.version,
            expired_tokens=outcome.expired_tokens,
            expired_actions=outcome.expired_actions,
            expired_tool_calls=outcome.expired_tool_calls,
            starvation_fraction=starvation_signal,
            waste_fraction=expired_overload_signal,
            queue_pressure=queue_overload_signal,
            inference_ready_trajectories=inference_ready,
            tool_waiting_trajectories=tool_waiting,
            tool_wait_fraction=tool_wait_fraction,
            reserve_before=self.adaptive_reserve,
        )
        self.runtime_estimator.observe_policy_interval(interval_seconds)
        self.runtime_estimator.observe_completed_records(
            self.consumed_records
        )
        self.runtime_estimator.observe_supply(
            salvageable_inflight=(
                self.version_salvageable_inflight_at_boundary
            ),
            completed_inflight=self._tracked_unfinished_completed,
            cohort_counts=self._tracked_unfinished_bucket_counts,
            completed_counts=self._tracked_unfinished_bucket_completed,
            prediction_error=outcome.supply_prediction_error,
        )
        self.adaptive_finish_ratio = self.runtime_estimator.finish_ratio
        self.dynamic_prediction_error_ewma = (
            self.runtime_estimator.supply_error_ewma
        )
        self.version_runtime_boundary_started_at = now
        self.version_runtime_last_learner_wait_seconds = 0.0

    def _reset_version_admission(self, create_step: int):
        if self.admission_policy != "version_adaptive" or not self.group_queue:
            return

        self.version_actual_existing_consumed = self._tracked_existing_consumed
        self.version_actual_existing_supply = (
            self.version_valid_ready_at_boundary + self._tracked_unfinished_completed
        )
        self.version_admission_prediction_error = (
            self.version_actual_existing_supply - self.version_expected_existing_supply
        )
        previous_version = (
            self.version_runtime_plan.version
            if self.version_runtime_plan is not None
            else None
        )
        supply = self._version_supply_snapshot(create_step)
        self._finalize_version_runtime_outcome(create_step, supply)
        self.version_runtime_reserve_before = self.adaptive_reserve
        self._update_dynamic_reserve(create_step)
        if previous_version is not None:
            reason = DYNAMIC_RESERVE_REASON_NAMES.get(
                self.dynamic_reserve_update_reason,
                f"reason_{self.dynamic_reserve_update_reason}",
            )
            shadow_reason = DYNAMIC_RESERVE_REASON_NAMES.get(
                self.dynamic_shadow_update_reason,
                f"reason_{self.dynamic_shadow_update_reason}",
            )
            self._merge_policy_update_trace(
                previous_version,
                reserve_after=self.adaptive_reserve,
                decision_reason=(
                    "shadow_not_applied"
                    if self.dynamic_reserve_shadow_mode
                    else reason
                ),
                shadow_mode=self.dynamic_reserve_shadow_mode,
                shadow_reserve_after=(
                    self.dynamic_shadow_reserve
                    if self.dynamic_reserve_shadow_mode
                    else self.adaptive_reserve
                ),
                shadow_decision_reason=(
                    shadow_reason
                    if self.dynamic_reserve_shadow_mode
                    else reason
                ),
            )

        self.version_runtime_revision = 0
        self.version_admission_version = create_step
        self.version_valid_ready_at_boundary = supply["valid_ready"]
        self.version_salvageable_inflight_at_boundary = supply["salvageable_inflight"]
        self.version_invested_inflight_at_boundary = supply["invested_inflight"]
        self.version_gpu_invested_inflight_at_boundary = supply[
            "gpu_invested_inflight"
        ]
        self.version_reset_only_inflight_at_boundary = supply[
            "reset_only_inflight"
        ]
        self.version_reserved_unstarted_at_boundary = supply["reserved_unstarted"]
        self.version_near_expiry_at_boundary = supply["near_expiry"]
        self.version_unfinished_bucket_counts = dict(supply["unfinished_bucket_counts"])
        self.version_progress_observed_candidates = supply[
            "unfinished_progress_observed_candidates"
        ]
        self.version_progress_mean_actions_sum = supply[
            "unfinished_progress_mean_actions_sum"
        ]
        self.version_progress_frontier_actions_sum = supply[
            "unfinished_progress_frontier_actions_sum"
        ]
        self.version_progress_max_actions = supply["unfinished_progress_max_actions"]
        self.version_runtime_forecast = self.runtime_estimator.build_forecast(
            create_step,
            ready_valid_slots=self.version_valid_ready_at_boundary,
            salvageable_inflight=self.version_salvageable_inflight_at_boundary,
            unfinished_bucket_counts=supply["unfinished_bucket_counts"],
        )
        self.version_expected_inflight_supply = (
            self.version_runtime_forecast.predicted_inflight_slots
        )
        self.version_bucket_learned_population = (
            self.version_runtime_forecast.learned_population
        )
        self.version_bucket_fallback_population = (
            self.version_runtime_forecast.fallback_population
        )
        self.version_expected_existing_supply = (
            self.version_runtime_forecast.expected_existing_supply
        )

        width = next(iter(self.group_queue.values())).admission_width
        self.version_runtime_plan = self._build_version_runtime_plan(
            create_step,
            supply,
            self.version_expected_existing_supply,
            admission_enabled=True,
        )
        self.version_admission_budget = self.version_runtime_plan.admission_budget
        self.version_admission_budget_trainable = (
            self.version_runtime_plan.admission_budget_trainable
        )
        self.version_admission_used = 0
        self.version_admission_remaining = self.version_admission_budget
        demand = self.rollout_batch_size + self.adaptive_reserve
        desired_groups = math.ceil(
            max(0.0, demand - self.version_expected_existing_supply) / self.group_size
        )
        if self.version_admission_budget // width < desired_groups:
            self.admission_throttled_total += 1

        self._tracked_existing_groups = supply["existing_groups"]
        self._tracked_unfinished_groups = supply["unfinished_groups"]
        self._tracked_unfinished_group_buckets = supply["unfinished_group_buckets"]
        self._tracked_unfinished_bucket_counts = supply["unfinished_bucket_counts"]
        self._tracked_unfinished_bucket_completed = {}
        self._tracked_existing_consumed = 0
        self._tracked_unfinished_consumed = 0
        self._tracked_unfinished_completed = 0
        self._admit_version_budget(create_step)

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
            "scheduler/version_priority_ready_bypass_total": sum(
                queue.version_priority_ready_bypass_total
                for queue in self.group_queue.values()
            ),
            "scheduler/watermark_admission_enabled": int(self.admission_policy == "outstanding_watermark"),
            "scheduler/version_adaptive_admission_enabled": int(
                self.admission_policy == "version_adaptive"
            ),
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
            "scheduler/version_admission_version": self.version_admission_version,
            "scheduler/version_admission_budget": self.version_admission_budget,
            "scheduler/version_admission_budget_trainable": self.version_admission_budget_trainable,
            "scheduler/version_admission_used": self.version_admission_used,
            "scheduler/version_admission_remaining": self.version_admission_remaining,
            "scheduler/version_runtime_plan_enabled": int(
                self.version_runtime_plan is not None
            ),
            "scheduler/version_runtime_revision": self.version_runtime_revision,
            "scheduler/version_runtime_admission_enabled": int(
                self.version_runtime_plan.admission_enabled
                if self.version_runtime_plan is not None else False
            ),
            "scheduler/version_runtime_admission_reason": (
                VERSION_RUNTIME_ADMISSION_REASON_CODES.get(
                    self.version_runtime_plan.admission_reason, -1
                )
                if self.version_runtime_plan is not None else -1
            ),
            "scheduler/version_runtime_admission_deficit": (
                self.version_runtime_plan.admission_deficit
                if self.version_runtime_plan is not None else 0.0
            ),
            "scheduler/version_runtime_admission_capacity": (
                self.version_runtime_plan.admission_capacity
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_priority_candidates": (
                len(self.version_runtime_plan.priority_candidate_groups)
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_rebuild_candidates": (
                len(self.version_runtime_plan.rebuild_candidate_groups)
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_rebuild_target": (
                self.version_runtime_plan.rebuild_target_trajectories
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_kv_feedback_requests": (
                self.version_runtime_plan.kv_feedback_requests
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_kv_feedback_hit_ratio": (
                self.version_runtime_plan.kv_feedback_hit_ratio
                if self.version_runtime_plan is not None else 0.0
            ),
            "scheduler/version_runtime_kv_feedback_resets": (
                self.version_runtime_plan.kv_feedback_resets
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_priority_deadline": (
                self.version_runtime_plan.priority_deadline_version
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_estimator_revision": (
                self.version_runtime_plan.estimator_revision
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_ready_valid_slots": (
                self.version_runtime_plan.ready_valid_slots
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_predicted_inflight_slots": (
                self.version_runtime_plan.predicted_inflight_slots
                if self.version_runtime_plan is not None else 0.0
            ),
            "scheduler/version_runtime_reserve_before": (
                self.version_runtime_plan.reserve_before
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_reserve_after": (
                self.version_runtime_plan.reserve_after
                if self.version_runtime_plan is not None else 0
            ),
            "scheduler/version_runtime_undersupply_signal": (
                self.version_runtime_plan.undersupply_signal
                if self.version_runtime_plan is not None else 0.0
            ),
            "scheduler/version_runtime_overload_signal": (
                self.version_runtime_plan.overload_signal
                if self.version_runtime_plan is not None else 0.0
            ),
            "scheduler/version_runtime_outcomes": len(
                self.version_runtime_outcomes
            ),
            "scheduler/version_runtime_supply_abs_error_ewma": (
                self.runtime_estimator.supply_abs_error_ewma or 0.0
            ),
            "scheduler/version_runtime_last_next_batch_latency_seconds": (
                self.version_runtime_last_outcome.next_batch_latency_seconds
                if self.version_runtime_last_outcome is not None else 0.0
            ),
            "scheduler/version_runtime_last_policy_update_interval_seconds": (
                self.version_runtime_last_outcome.policy_update_interval_seconds
                if self.version_runtime_last_outcome is not None else 0.0
            ),
            "scheduler/version_runtime_last_expired_tokens": (
                self.version_runtime_last_outcome.expired_tokens
                if self.version_runtime_last_outcome is not None else 0
            ),
            "scheduler/version_runtime_last_expired_actions": (
                self.version_runtime_last_outcome.expired_actions
                if self.version_runtime_last_outcome is not None else 0
            ),
            "scheduler/version_runtime_last_expired_tool_calls": (
                self.version_runtime_last_outcome.expired_tool_calls
                if self.version_runtime_last_outcome is not None else 0
            ),
            "scheduler/version_runtime_last_reprefill_tokens": (
                self.version_runtime_last_outcome.reprefill_tokens
                if self.version_runtime_last_outcome is not None else 0
            ),
            "scheduler/version_runtime_last_prefill_seconds": (
                self.version_runtime_last_outcome.prefill_seconds
                if self.version_runtime_last_outcome is not None else 0.0
            ),
            "scheduler/version_runtime_last_scheduling_wait_seconds": (
                self.version_runtime_last_outcome.scheduling_wait_seconds
                if self.version_runtime_last_outcome is not None else 0.0
            ),
            "scheduler/version_runtime_last_scheduling_wait_mean_seconds": (
                self.version_runtime_last_outcome.scheduling_wait_mean_seconds
                if self.version_runtime_last_outcome is not None else 0.0
            ),
            "scheduler/version_boundary_profiler_enabled": int(
                self.version_boundary_profiler_enabled
            ),
            "scheduler/version_boundary_events": len(self.version_boundary_events),
            "scheduler/version_boundary_observed_started": int(
                self.latest_version_boundary_summary.get(
                    "observed_started_trajectories", 0
                )
            ),
            "scheduler/version_boundary_unobserved_started": int(
                self.latest_version_boundary_summary.get(
                    "unobserved_started_trajectories", 0
                )
            ),
            "scheduler/version_boundary_completed_carryover": int(
                self.latest_version_boundary_summary.get(
                    "completed_carryover_trajectories", 0
                )
            ),
            "scheduler/version_boundary_cross_version": int(
                self.latest_version_boundary_summary.get("cross_version_trajectories", 0)
            ),
            "scheduler/version_boundary_survivors": int(
                self.latest_version_boundary_summary.get("survivor_trajectories", 0)
            ),
            "scheduler/version_boundary_expired": int(
                self.latest_version_boundary_summary.get("expired_trajectories", 0)
            ),
            "scheduler/version_boundary_context_tokens": int(
                self.latest_version_boundary_summary.get("current_context_tokens", 0)
            ),
            "scheduler/version_progress_topup_events": self.version_progress_topup_events,
            "scheduler/version_progress_topup_trajectories": (
                self.version_progress_topup_trajectories
            ),
            "scheduler/valid_ready_at_version_boundary": self.version_valid_ready_at_boundary,
            "scheduler/salvageable_inflight_at_version_boundary": (
                self.version_salvageable_inflight_at_boundary
            ),
            "scheduler/invested_inflight_at_version_boundary": (
                self.version_invested_inflight_at_boundary
            ),
            "scheduler/gpu_invested_inflight_at_version_boundary": (
                self.version_gpu_invested_inflight_at_boundary
            ),
            "scheduler/reset_only_inflight_at_version_boundary": (
                self.version_reset_only_inflight_at_boundary
            ),
            "scheduler/reserved_unstarted_at_version_boundary": (
                self.version_reserved_unstarted_at_boundary
            ),
            "scheduler/near_expiry_at_version_boundary": self.version_near_expiry_at_boundary,
            "scheduler/expected_existing_supply": self.version_expected_existing_supply,
            "scheduler/actual_existing_supply": self.version_actual_existing_supply,
            "scheduler/actual_existing_consumed": self.version_actual_existing_consumed,
            "scheduler/admission_prediction_error": self.version_admission_prediction_error,
            "scheduler/adaptive_finish_ratio": self.adaptive_finish_ratio,
            "scheduler/bucketed_finish_enabled": int(self.bucketed_finish_enabled),
            "scheduler/bucketed_expected_inflight_supply": self.version_expected_inflight_supply,
            "scheduler/bucketed_learned_population": self.version_bucket_learned_population,
            "scheduler/bucketed_fallback_population": self.version_bucket_fallback_population,
            "scheduler/boundary_progress_observed_candidates": (
                self.version_progress_observed_candidates
            ),
            "scheduler/boundary_progress_mean_actions_sum": (
                self.version_progress_mean_actions_sum
            ),
            "scheduler/boundary_progress_frontier_actions_sum": (
                self.version_progress_frontier_actions_sum
            ),
            "scheduler/boundary_progress_max_actions": self.version_progress_max_actions,
            "scheduler/dynamic_reserve_enabled": int(self.dynamic_reserve_enabled),
            "scheduler/dynamic_utility_controller_enabled": int(
                self.dynamic_reserve_controller == "utility_hill_climb"
            ),
            "scheduler/dynamic_reserve": self.adaptive_reserve,
            "scheduler/dynamic_reserve_update_reason": self.dynamic_reserve_update_reason,
            "scheduler/dynamic_reserve_pending_direction": self.dynamic_reserve_pending_direction,
            "scheduler/dynamic_reserve_pending_count": self.dynamic_reserve_pending_count,
            "scheduler/dynamic_reserve_cooldown_remaining": (
                self.dynamic_reserve_cooldown_remaining
            ),
            "scheduler/dynamic_learner_wait_ewma": self.dynamic_learner_wait_ewma or 0.0,
            "scheduler/dynamic_stale_ewma": self.dynamic_stale_ewma or 0.0,
            "scheduler/dynamic_prediction_error_ewma": self.dynamic_prediction_error_ewma or 0.0,
            "scheduler/dynamic_closed_loop_controller_enabled": int(
                self.dynamic_reserve_controller == "closed_loop_aimd"
            ),
            "scheduler/dynamic_undersupply_ewma": (
                self.version_runtime_undersupply_ewma or 0.0
            ),
            "scheduler/dynamic_overload_ewma": (
                self.version_runtime_overload_ewma or 0.0
            ),
            "scheduler/dynamic_reserve_increase_total": self.dynamic_reserve_increase_total,
            "scheduler/dynamic_reserve_decrease_total": self.dynamic_reserve_decrease_total,
            "scheduler/dynamic_reserve_hold_total": self.dynamic_reserve_hold_total,
            "scheduler/dynamic_utility_direction": self.dynamic_utility_direction,
            "scheduler/dynamic_utility_window_count": self.dynamic_utility_window_count,
            "scheduler/dynamic_utility_settle_remaining": self.dynamic_utility_settle_remaining,
            "scheduler/dynamic_utility_last_window": self.dynamic_utility_last_window,
            "scheduler/dynamic_utility_last_window_efficiency": (
                self.dynamic_utility_last_window_efficiency
            ),
            "scheduler/dynamic_utility_sample": self.dynamic_utility_sample,
            "scheduler/dynamic_useful_token_rate": self.dynamic_useful_token_rate,
            "scheduler/dynamic_stale_token_rate": self.dynamic_stale_token_rate,
            "scheduler/dynamic_compute_efficiency": self.dynamic_compute_efficiency,
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
        for bucket in FINISH_RATE_BUCKET_KEYS:
            filter_metrics[f"scheduler/finish_ratio/{bucket}"] = self.bucketed_finish_ratios.get(
                bucket, self.adaptive_finish_ratio
            )
            filter_metrics[f"scheduler/finish_ratio_samples/{bucket}"] = (
                self.bucketed_finish_sample_counts.get(bucket, 0)
            )
            filter_metrics[f"scheduler/carryover_at_boundary/{bucket}"] = (
                self.version_unfinished_bucket_counts.get(bucket, 0)
            )
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
        consumed_metrics, _ = self._aggregate_consumed_records(
            self.new_consumed_records, "scheduler/consumed"
        )
        filter_metrics.update(consumed_metrics)
        now = time.monotonic()
        observation_seconds = max(1e-6, now - self.dynamic_last_observation_time)
        self.dynamic_last_observation_time = now
        stale_tokens = 0
        new_stale_count = 0
        for record in new_discard_records:
            if not str(record.get("discard_reason", "")).startswith("version_"):
                continue
            record_key = (
                int(record.get("group_id", -1)),
                int(record.get("episode_id", -1)),
                int(record.get("env_id", -1)),
            )
            current_tokens = int(record.get("inference_tokens", 0))
            previous_tokens = self.dynamic_stale_record_tokens_seen.get(
                record_key, 0
            )
            stale_tokens += max(0, current_tokens - previous_tokens)
            self.dynamic_stale_record_tokens_seen[record_key] = max(
                previous_tokens, current_tokens
            )
            if record_key not in self.dynamic_stale_record_ids_seen:
                self.dynamic_stale_record_ids_seen.add(record_key)
                new_stale_count += 1
        consumed_tokens = int(
            consumed_metrics["scheduler/consumed/valid_inference_tokens"]
        )
        consumed_count = int(
            consumed_metrics["scheduler/consumed/valid_trajectories"]
        )
        stale_token_sample, stale_trajectory_fraction = (
            compute_stale_control_signal(
                stale_tokens,
                consumed_tokens,
                new_stale_count,
                consumed_count,
            )
        )
        if stale_token_sample is not None:
            stale_fraction = stale_token_sample
            stale_signal_source = 1
            self.dynamic_stale_ewma = self._update_ewma(
                self.dynamic_stale_ewma, stale_fraction
            )
        else:
            # A reset-only stale trajectory consumed no inference compute. Keep
            # its count observable, but do not mix a trajectory fraction into
            # the token-waste controller signal.
            stale_fraction = 0.0
            stale_signal_source = 0
        filter_metrics["scheduler/dynamic_stale_fraction"] = stale_fraction
        filter_metrics["scheduler/dynamic_stale_trajectory_fraction"] = (
            stale_trajectory_fraction
        )
        filter_metrics["scheduler/dynamic_stale_fraction_source_tokens"] = (
            stale_signal_source
        )
        filter_metrics["scheduler/dynamic_stale_new_trajectories"] = (
            new_stale_count
        )
        filter_metrics["scheduler/dynamic_stale_ewma"] = (
            self.dynamic_stale_ewma or 0.0
        )
        utility, useful_rate, stale_rate, compute_efficiency = compute_effective_rollout_utility(
            consumed_metrics["scheduler/consumed/valid_response_tokens"],
            consumed_metrics["scheduler/consumed/valid_inference_tokens"],
            stale_tokens,
            observation_seconds,
            self.dynamic_utility_waste_weight,
        )
        self.dynamic_utility_sample = utility
        self.dynamic_useful_token_rate = useful_rate
        self.dynamic_stale_token_rate = stale_rate
        self.dynamic_compute_efficiency = compute_efficiency
        if self.dynamic_reserve_enabled and self.dynamic_reserve_controller == "utility_hill_climb":
            should_record, self.dynamic_utility_settle_remaining = consume_utility_settle(
                self.dynamic_utility_settle_remaining
            )
            if should_record:
                self.dynamic_utility_window_sum += utility
                self.dynamic_utility_window_count += 1
                self.dynamic_utility_window_response_tokens += int(
                    consumed_metrics["scheduler/consumed/valid_response_tokens"]
                )
                self.dynamic_utility_window_consumed_tokens += int(
                    consumed_metrics["scheduler/consumed/valid_inference_tokens"]
                )
                self.dynamic_utility_window_stale_tokens += stale_tokens
                self.dynamic_utility_window_seconds += observation_seconds
        filter_metrics["scheduler/dynamic_observation_seconds"] = observation_seconds
        filter_metrics["scheduler/dynamic_utility_sample"] = utility
        filter_metrics["scheduler/dynamic_useful_token_rate"] = useful_rate
        filter_metrics["scheduler/dynamic_useful_response_token_rate"] = useful_rate
        filter_metrics["scheduler/dynamic_stale_token_rate"] = stale_rate
        filter_metrics["scheduler/dynamic_compute_efficiency"] = compute_efficiency
        filter_metrics["scheduler/dynamic_utility_window_count"] = self.dynamic_utility_window_count
        filter_metrics["scheduler/dynamic_utility_settle_remaining"] = (
            self.dynamic_utility_settle_remaining
        )
        self.new_consumed_records = []
        return filter_metrics

    @staticmethod
    def _aggregate_consumed_records(records: List[Dict[str, Any]], prefix: str):
        age_histogram: Dict[str, int] = {}
        actions_histogram: Dict[str, int] = {}
        for record in records:
            age = str(int(record.get("version_age", 0)))
            actions = str(int(record.get("actions_completed", 0)))
            age_histogram[age] = age_histogram.get(age, 0) + 1
            actions_histogram[actions] = actions_histogram.get(actions, 0) + 1

        count = len(records)
        metrics = {
            f"{prefix}/trajectories": count,
            f"{prefix}/valid_trajectories": sum(
                bool(record.get("trainable_valid", True)) for record in records
            ),
            f"{prefix}/placeholder_trajectories": sum(
                bool(record.get("placeholder", False)) for record in records
            ),
            f"{prefix}/reset_only_trajectories": sum(
                bool(record.get("reset_only", False)) for record in records
            ),
            f"{prefix}/version_age_sum": sum(int(record.get("version_age", 0)) for record in records),
            f"{prefix}/version_age_max": max(
                (int(record.get("version_age", 0)) for record in records), default=0
            ),
            f"{prefix}/actions": sum(int(record.get("actions_completed", 0)) for record in records),
            f"{prefix}/prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records),
            f"{prefix}/response_tokens": sum(int(record.get("response_tokens", 0)) for record in records),
            f"{prefix}/inference_tokens": sum(int(record.get("inference_tokens", 0)) for record in records),
            f"{prefix}/valid_inference_tokens": sum(
                int(record.get("inference_tokens", 0))
                for record in records
                if bool(record.get("trainable_valid", True))
            ),
            f"{prefix}/valid_response_tokens": sum(
                int(record.get("response_tokens", 0))
                for record in records
                if bool(record.get("trainable_valid", True))
            ),
        }
        for age in range(4):
            metrics[f"{prefix}/version_age_{age}"] = sum(
                int(record.get("version_age", 0)) == age for record in records
            )
        metrics[f"{prefix}/version_age_ge_4"] = sum(
            int(record.get("version_age", 0)) >= 4 for record in records
        )
        return metrics, {
            "version_age": age_histogram,
            "actions_completed": actions_histogram,
        }

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
            f"{prefix}/reset_only_trajectories": sum(
                bool(record.get("reset_only", False)) for record in records
            ),
            f"{prefix}/actions": sum(int(record.get("actions_completed", 0)) for record in records),
            f"{prefix}/inference_calls": sum(int(record.get("inference_calls", 0)) for record in records),
            f"{prefix}/tool_calls": sum(int(record.get("tool_calls", 0)) for record in records),
            f"{prefix}/prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records),
            f"{prefix}/response_tokens": sum(int(record.get("response_tokens", 0)) for record in records),
            f"{prefix}/inference_tokens": sum(int(record.get("inference_tokens", 0)) for record in records),
            f"{prefix}/generate_seconds": sum(
                float(record.get("generate_seconds", 0.0)) for record in records
            ),
            f"{prefix}/env_seconds": sum(float(record.get("env_seconds", 0.0)) for record in records),
            f"{prefix}/tool_wall_seconds": sum(
                float(record.get("tool_wall_seconds", 0.0))
                for record in records
            ),
            f"{prefix}/model_execute_seconds": sum(
                float(record.get("model_execute_seconds", 0.0))
                for record in records
            ),
            f"{prefix}/engine_step_seconds_attributed": sum(
                float(record.get("engine_step_seconds_attributed", 0.0))
                for record in records
            ),
            f"{prefix}/trajectory_wall_seconds": sum(
                float(record.get("trajectory_wall_seconds", 0.0)) for record in records
            ),
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

    @staticmethod
    def _completed_rollout_record(rollout: DataProto, group: GroupData) -> Dict[str, Any]:
        metric = GroupQueue._metric_by_suffix
        first_value = GroupQueue._first_non_tensor_value
        env_id = first_value(rollout, "env_ids", -1)
        placeholder = bool(
            rollout.meta_info.get("drop_flag", False)
            if rollout.meta_info
            else False
        )
        record = {
            "trajectory_id": str(first_value(rollout, "traj_id", "unknown")),
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
            "placeholder": placeholder,
            "trainable_valid": not placeholder,
            "truncated": bool(metric(rollout, "/traj_truncated", 0)),
            "actions_completed": int(metric(rollout, "/traj_actions_completed", 0)),
            "inference_calls": int(metric(rollout, "/traj_inference_calls", 0)),
            "tool_calls": int(metric(rollout, "/traj_tool_calls", 0)),
            "prompt_tokens": int(metric(rollout, "/traj_prompt_tokens_total", 0)),
            "response_tokens": int(metric(rollout, "/traj_response_tokens_total", 0)),
            "inference_tokens": int(metric(rollout, "/traj_inference_tokens_total", 0)),
            "generate_seconds": float(metric(rollout, "/traj_generate_seconds_total", 0)),
            "env_seconds": float(metric(rollout, "/traj_env_seconds_total", 0)),
            "tool_wall_seconds": float(
                metric(rollout, "/traj_tool_wall_seconds_total", 0)
            ),
            "request_queue_seconds": float(
                metric(rollout, "/traj_request_queue_seconds_total", 0)
            ),
            "router_scheduling_wait_seconds": float(
                metric(
                    rollout,
                    "/traj_router_scheduling_wait_seconds_total",
                    0,
                )
            ),
            "request_ttft_seconds": float(
                metric(rollout, "/traj_request_ttft_seconds_total", 0)
            ),
            "request_prefill_seconds": float(
                metric(rollout, "/traj_request_prefill_seconds_total", 0)
            ),
            "request_decode_seconds": float(
                metric(rollout, "/traj_request_decode_seconds_total", 0)
            ),
            "request_inference_seconds": float(
                metric(rollout, "/traj_request_inference_seconds_total", 0)
            ),
            "request_latency_seconds": float(
                metric(rollout, "/traj_request_latency_seconds_total", 0)
            ),
            "model_forward_seconds": float(
                metric(rollout, "/traj_model_forward_seconds_total", 0)
            ),
            "model_execute_seconds": float(
                metric(rollout, "/traj_model_execute_seconds_total", 0)
            ),
            "engine_step_seconds_attributed": float(
                metric(
                    rollout,
                    "/traj_engine_step_seconds_attributed_total",
                    0,
                )
            ),
            "prefill_engine_step_seconds_attributed": float(
                metric(
                    rollout,
                    "/traj_prefill_engine_step_seconds_attributed_total",
                    0,
                )
            ),
            "decode_engine_step_seconds_attributed": float(
                metric(
                    rollout,
                    "/traj_decode_engine_step_seconds_attributed_total",
                    0,
                )
            ),
            "engine_prefill_tokens": int(
                metric(rollout, "/traj_engine_prefill_tokens_total", 0)
            ),
            "engine_cached_prompt_tokens": int(
                metric(rollout, "/traj_engine_cached_prompt_tokens_total", 0)
            ),
            "max_actions": int(metric(rollout, "/traj_max_actions", 0)),
            "remaining_actions": int(
                metric(rollout, "/traj_remaining_actions", 0)
            ),
            "action_budget_progress": float(
                metric(rollout, "/traj_action_budget_progress", 0)
            ),
            "trajectory_started_at_unix": float(
                metric(rollout, "/traj_started_at_unix", 0)
            ),
            "trajectory_completed_at_unix": float(
                metric(rollout, "/traj_completed_at_unix", 0)
            ),
            "trajectory_wall_seconds": float(
                first_value(rollout, "traj_wall_seconds_total", 0)
            ),
        }
        tensor_progress = GroupQueue._tensor_progress(rollout)
        for field, value in tensor_progress.items():
            record[field] = max(record.get(field, 0), value)
        direct_progress_fields = {
            "actions_completed": "traj_actions_completed",
            "inference_calls": "traj_inference_calls",
            "tool_calls": "traj_tool_calls",
            "prompt_tokens": "traj_prompt_tokens_total",
            "response_tokens": "traj_response_tokens_total",
            "inference_tokens": "traj_inference_tokens_total",
            "generate_seconds": "traj_generate_seconds_total",
            "env_seconds": "traj_env_seconds_total",
            "tool_wall_seconds": "traj_tool_wall_seconds_total",
            "request_queue_seconds": "traj_request_queue_seconds_total",
            "router_scheduling_wait_seconds": (
                "traj_router_scheduling_wait_seconds_total"
            ),
            "request_ttft_seconds": "traj_request_ttft_seconds_total",
            "request_prefill_seconds": "traj_request_prefill_seconds_total",
            "request_decode_seconds": "traj_request_decode_seconds_total",
            "request_inference_seconds": (
                "traj_request_inference_seconds_total"
            ),
            "request_latency_seconds": "traj_request_latency_seconds_total",
            "model_forward_seconds": "traj_model_forward_seconds_total",
            "model_execute_seconds": "traj_model_execute_seconds_total",
            "engine_step_seconds_attributed": (
                "traj_engine_step_seconds_attributed_total"
            ),
            "prefill_engine_step_seconds_attributed": (
                "traj_prefill_engine_step_seconds_attributed_total"
            ),
            "decode_engine_step_seconds_attributed": (
                "traj_decode_engine_step_seconds_attributed_total"
            ),
            "engine_prefill_tokens": "traj_engine_prefill_tokens_total",
            "engine_cached_prompt_tokens": (
                "traj_engine_cached_prompt_tokens_total"
            ),
            "max_actions": "traj_max_actions",
            "remaining_actions": "traj_remaining_actions",
            "action_budget_progress": "traj_action_budget_progress",
            "trajectory_started_at_unix": "traj_started_at_unix",
            "trajectory_completed_at_unix": "traj_completed_at_unix",
            "trajectory_wall_seconds": "traj_wall_seconds_total",
        }
        for field, key in direct_progress_fields.items():
            value = first_value(rollout, key, 0)
            try:
                record[field] = max(record[field], float(value))
            except (TypeError, ValueError):
                continue
        record["reset_only"] = bool(
            record.get("reset_completed", False)
            and int(record.get("inference_calls", 0)) == 0
        )
        if record["max_actions"] > 0:
            record["remaining_actions"] = max(
                0, record["max_actions"] - record["actions_completed"]
            )
            record["action_budget_progress"] = min(
                1.0, record["actions_completed"] / record["max_actions"]
            )
        return record

    @staticmethod
    def _deduplicate_shutdown_records(
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Merge a worker snapshot with the same rollout submitted during shutdown."""
        category_priority = {
            "inflight_at_shutdown": 1,
            "completed_not_submitted": 2,
            "completed_unconsumed": 3,
        }
        progress_fields = (
            "actions_completed",
            "inference_calls",
            "tool_calls",
            "prompt_tokens",
            "response_tokens",
            "inference_tokens",
            "current_context_tokens",
            "generate_seconds",
            "environment_model_seconds",
            "env_seconds",
            "tool_wall_seconds",
            "request_queue_seconds",
            "router_scheduling_wait_seconds",
            "request_ttft_seconds",
            "request_prefill_seconds",
            "request_decode_seconds",
            "request_inference_seconds",
            "request_latency_seconds",
            "model_forward_seconds",
            "model_execute_seconds",
            "engine_step_seconds_attributed",
            "prefill_engine_step_seconds_attributed",
            "decode_engine_step_seconds_attributed",
            "engine_prefill_tokens",
            "engine_cached_prompt_tokens",
            "max_actions",
            "action_budget_progress",
            "trajectory_wall_seconds",
        )
        result: List[Dict[str, Any]] = []
        index_by_trajectory: Dict[str, int] = {}
        duplicates_removed = 0

        def preference(record: Dict[str, Any]):
            return (
                category_priority.get(str(record.get("category", "")), 0),
                int(bool(record.get("completed", False))),
                int(record.get("actions_completed", 0)),
                int(record.get("inference_calls", 0)),
            )

        for raw_record in records:
            record = dict(raw_record)
            trajectory_id = str(record.get("trajectory_id", ""))
            if not trajectory_id or trajectory_id == "unknown":
                result.append(record)
                continue

            existing_index = index_by_trajectory.get(trajectory_id)
            if existing_index is None:
                index_by_trajectory[trajectory_id] = len(result)
                result.append(record)
                continue

            existing = result[existing_index]
            preferred = record if preference(record) > preference(existing) else existing
            merged = dict(existing)
            merged.update(preferred)
            for field in progress_fields:
                try:
                    merged[field] = max(
                        float(existing.get(field, 0)),
                        float(record.get(field, 0)),
                    )
                except (TypeError, ValueError):
                    continue
            for field in ("completed", "truncated", "reset_completed"):
                merged[field] = bool(existing.get(field, False)) or bool(
                    record.get(field, False)
                )
            result[existing_index] = merged
            duplicates_removed += 1

        return result, duplicates_removed

    def collect_shutdown_waste(self, inflight_records: List[Dict[str, Any]]):
        records = []
        for group_queue in self.group_queue.values():
            for group in group_queue.groups.values():
                for rollout in group.rollouts:
                    if rollout is not None:
                        records.append(self._completed_rollout_record(rollout, group))
        records.extend(record for record in inflight_records if record is not None)
        records, duplicate_records_removed = self._deduplicate_shutdown_records(records)
        for record in records:
            max_actions = int(record.get("max_actions", 0))
            if max_actions > 0:
                actions = int(record.get("actions_completed", 0))
                record["remaining_actions"] = max(0, max_actions - actions)
                record["action_budget_progress"] = min(
                    1.0, actions / max_actions
                )
            record["reset_only"] = bool(
                record.get("reset_completed", False)
                and int(record.get("inference_calls", 0)) == 0
            )
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
            "terminal_waste/duplicate_records_removed": duplicate_records_removed,
            "terminal_waste/completed_unconsumed": sum(
                record.get("category") == "completed_unconsumed" for record in records
            ),
            "terminal_waste/completed_not_submitted": sum(
                record.get("category") == "completed_not_submitted" for record in records
            ),
            "terminal_waste/inflight": sum(
                record.get("category") == "inflight_at_shutdown" for record in records
            ),
            "terminal_waste/reset_only": sum(
                bool(record.get("reset_only", False)) for record in records
            ),
            "terminal_waste/actions": sum(int(record.get("actions_completed", 0)) for record in records),
            "terminal_waste/inference_calls": sum(int(record.get("inference_calls", 0)) for record in records),
            "terminal_waste/tool_calls": sum(int(record.get("tool_calls", 0)) for record in records),
            "terminal_waste/prompt_tokens": sum(int(record.get("prompt_tokens", 0)) for record in records),
            "terminal_waste/response_tokens": sum(int(record.get("response_tokens", 0)) for record in records),
            "terminal_waste/inference_tokens": sum(int(record.get("inference_tokens", 0)) for record in records),
            "terminal_waste/env_seconds": sum(float(record.get("env_seconds", 0.0)) for record in records),
            "terminal_waste/trajectory_wall_seconds": sum(
                float(record.get("trajectory_wall_seconds", 0.0)) for record in records
            ),
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
        consumed_metrics, consumed_histograms = self._aggregate_consumed_records(
            self.consumed_records, "consumed"
        )
        metrics.update(consumed_metrics)
        finished_at = self.rollout_finished_at or time.monotonic()
        started_at = self.rollout_started_at or finished_at
        goodput_metrics = summarize_rollout_goodput(
            self.consumed_records,
            async_discard_records,
            records,
            elapsed_seconds=max(0.0, finished_at - started_at),
            learner_wait_seconds=self.learner_wait_seconds_total,
            admitted_trajectories=self.admitted_trajectories_total,
        )
        goodput_metrics["learner/wait_events"] = self.learner_wait_events
        metrics.update(goodput_metrics)
        boundary_metrics = {
            "version_boundary/count": len(self.version_boundary_events),
            "version_boundary/observed_started_trajectories": sum(
                event["summary"]["observed_started_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/unobserved_started_trajectories": sum(
                event["summary"]["unobserved_started_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/completed_carryover_trajectories": sum(
                event["summary"]["completed_carryover_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/cross_version_trajectories": sum(
                event["summary"]["cross_version_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/cross_version_invested_trajectories": sum(
                event["summary"]["cross_version_invested_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/expired_trajectories": sum(
                event["summary"]["expired_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/survivor_trajectories": sum(
                event["summary"]["survivor_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/completed_survivor_trajectories": sum(
                event["summary"]["completed_survivor_trajectories"]
                for event in self.version_boundary_events
            ),
            "version_boundary/unfinished_actions": sum(
                event["summary"]["unfinished_actions"]
                for event in self.version_boundary_events
            ),
            "version_boundary/unfinished_logical_inference_tokens": sum(
                event["summary"]["unfinished_logical_inference_tokens"]
                for event in self.version_boundary_events
            ),
            "version_boundary/unfinished_current_context_tokens": sum(
                event["summary"]["unfinished_current_context_tokens"]
                for event in self.version_boundary_events
            ),
            "version_boundary/actions_completed": sum(
                event["summary"]["actions_completed"]
                for event in self.version_boundary_events
            ),
            "version_boundary/logical_inference_tokens": sum(
                event["summary"]["logical_inference_tokens"]
                for event in self.version_boundary_events
            ),
            "version_boundary/expired_actions": sum(
                event["summary"]["expired_actions"]
                for event in self.version_boundary_events
            ),
            "version_boundary/expired_logical_inference_tokens": sum(
                event["summary"]["expired_logical_inference_tokens"]
                for event in self.version_boundary_events
            ),
            "version_boundary/expired_tool_wall_seconds": sum(
                event["summary"].get("expired_tool_wall_seconds", 0.0)
                for event in self.version_boundary_events
            ),
            "version_boundary/expired_model_execute_seconds": sum(
                event["summary"].get("expired_model_execute_seconds", 0.0)
                for event in self.version_boundary_events
            ),
            "version_boundary/expired_engine_step_seconds_attributed": sum(
                event["summary"].get(
                    "expired_engine_step_seconds_attributed", 0.0
                )
                for event in self.version_boundary_events
            ),
        }
        metrics.update(boundary_metrics)
        runtime_state = {
            "plan": (
                self.version_runtime_plan.to_dict()
                if self.version_runtime_plan is not None
                else None
            ),
            "forecast": (
                self.version_runtime_forecast.to_dict()
                if self.version_runtime_forecast is not None
                else None
            ),
            "last_outcome": (
                self.version_runtime_last_outcome.to_dict()
                if self.version_runtime_last_outcome is not None
                else None
            ),
            "outcomes": [
                outcome.to_dict()
                for outcome in self.version_runtime_outcomes
            ],
            "policy_update_traces": [
                self.policy_update_traces[version].to_dict()
                for version in sorted(self.policy_update_traces)
            ],
            "dynamic_reserve": self.adaptive_reserve,
            "dynamic_reserve_shadow_mode": self.dynamic_reserve_shadow_mode,
            "dynamic_shadow_reserve": self.dynamic_shadow_reserve,
            "dynamic_shadow_decision_reason": DYNAMIC_RESERVE_REASON_NAMES.get(
                self.dynamic_shadow_update_reason,
                f"reason_{self.dynamic_shadow_update_reason}",
            ),
            "dynamic_learner_wait_ewma": self.dynamic_learner_wait_ewma or 0.0,
            "dynamic_stale_fraction_ewma": self.dynamic_stale_ewma or 0.0,
            "dynamic_prediction_error_ewma": (
                self.dynamic_prediction_error_ewma or 0.0
            ),
            "supply_abs_error_ewma": (
                self.runtime_estimator.supply_abs_error_ewma or 0.0
            ),
            "undersupply_ewma": (
                self.version_runtime_undersupply_ewma or 0.0
            ),
            "overload_ewma": self.version_runtime_overload_ewma or 0.0,
            "starvation_fraction_ewma": (
                self.version_runtime_starvation_ewma or 0.0
            ),
            "waste_fraction_ewma": self.version_runtime_waste_ewma or 0.0,
            "queue_pressure_ewma": (
                self.version_runtime_queue_pressure_ewma or 0.0
            ),
            "tool_wait_fraction_ewma": (
                self.version_runtime_tool_wait_fraction_ewma or 0.0
            ),
            "progress_topup_events": self.version_progress_topup_events,
            "progress_topup_trajectories": (
                self.version_progress_topup_trajectories
            ),
            "reserve_increase_total": self.dynamic_reserve_increase_total,
            "reserve_decrease_total": self.dynamic_reserve_decrease_total,
            "reserve_hold_total": self.dynamic_reserve_hold_total,
        }
        metrics.update(
            {
                "version_runtime/final_revision": self.version_runtime_revision,
                "version_runtime/final_dynamic_reserve": self.adaptive_reserve,
                "version_runtime/final_shadow_reserve": (
                    self.dynamic_shadow_reserve
                ),
                "version_runtime/progress_topup_events": (
                    self.version_progress_topup_events
                ),
                "version_runtime/progress_topup_trajectories": (
                    self.version_progress_topup_trajectories
                ),
            }
        )
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
            "consumed": {
                "metrics": consumed_metrics,
                "histograms": consumed_histograms,
                "records": self.consumed_records,
            },
            "rollout_goodput": goodput_metrics,
            "learner_wait": {
                "records": self.learner_wait_records,
            },
            "version_boundaries": {
                "metrics": boundary_metrics,
                "events": self.version_boundary_events,
            },
            "version_runtime": runtime_state,
        }

    def clear(self):
        self.rollout_complete = {}
        for get_task in self.pending_gets:
            get_task.cancel()
        self.pending_gets = set()
        for group_queue in self.group_queue.values():
            group_queue.clear()
        self.version_boundary_events.clear()
        self.latest_version_boundary_summary = {}
        self.rollout_started_at = None
        self.rollout_finished_at = None
        self.learner_wait_seconds_total = 0.0
        self.learner_wait_events = 0
        self.learner_wait_records.clear()
        self.policy_update_traces.clear()
        self.dynamic_stale_record_tokens_seen.clear()
        self.dynamic_stale_record_ids_seen.clear()
        self.version_runtime_outcomes.clear()
        self.version_runtime_last_outcome = None
        self.version_runtime_forecast = None
        self.version_runtime_boundary_started_at = None
        self.version_runtime_last_learner_wait_seconds = 0.0
        self.version_runtime_starvation_ewma = None
        self.version_runtime_waste_ewma = None
        self.version_runtime_queue_pressure_ewma = None
        self.version_runtime_tool_wait_fraction_ewma = None
        self.version_runtime_inference_ready = 0
        self.version_runtime_tool_waiting = 0
        self.version_runtime_waste_totals = {
            "expired_trajectories": 0,
            "expired_actions": 0,
            "expired_tokens": 0,
            "expired_tool_calls": 0,
            "expired_tool_seconds": 0.0,
        }
        self.version_runtime_consumed_tokens_total = 0
        self.version_runtime_request_metric_totals = {}

    def mark_rollout_end(self):
        if self.rollout_finished_at is None:
            self.rollout_finished_at = time.monotonic()

    def stop_admission(self):
        for group_queue in self.group_queue.values():
            group_queue.stop_admission()

    def advance_step(self, step, kv_feedback: Optional[Dict[str, Any]] = None):
        if self.rollout_started_at is None:
            self.rollout_started_at = time.monotonic()
        if kv_feedback is not None:
            self.latest_kv_feedback = dict(kv_feedback)
        current_versions = [
            queue.current_step
            for queue in self.group_queue.values()
            if queue.current_step is not None
        ]
        from_version = max(current_versions) if current_versions else None
        boundary_event = None
        if (
            self.version_boundary_profiler_enabled
            and from_version is not None
            and int(step) > int(from_version)
        ):
            boundary_event = self._capture_version_boundary(int(from_version), int(step))

        fixed_step_admission = self.admission_policy == "step"
        for group_queue in self.group_queue.values():
            group_queue.advance_step(step, admit_step_groups=fixed_step_admission)
        if self.admission_policy == "outstanding_watermark":
            self._refill_to_watermark(step)
        elif self.admission_policy == "version_adaptive":
            self._reset_version_admission(step)
        if self.admission_policy != "version_adaptive":
            self._refresh_nonadaptive_runtime_plan(step)
        runtime_plan = (
            self.version_runtime_plan.to_dict()
            if self.version_runtime_plan is not None
            else {"version": int(step)}
        )
        if boundary_event is not None:
            boundary_event["runtime_plan"] = dict(runtime_plan)
            boundary_event["post_boundary_outstanding"] = self._outstanding_snapshot(step)
            self.version_boundary_events.append(boundary_event)
            self.latest_version_boundary_summary = dict(boundary_event["summary"])
        return runtime_plan

    def update_trajectory_progress(self, snapshots: List[Dict[str, Any]]):
        for group_queue in self.group_queue.values():
            group_queue.update_progress_snapshots(snapshots)

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
        became_trainable = self.group_queue[group_id].put(episode_id, start_step, rollout)
        if became_trainable:
            self._record_version_adaptive_completion(group_id, episode_id)
            completed_group = self.group_queue[group_id].groups.get(episode_id)
            if completed_group is not None:
                self.runtime_estimator.observe_completed_records(
                    [
                        self._completed_rollout_record(item, completed_group)
                        for item in completed_group.rollouts[: self.group_size]
                        if item is not None
                    ]
                )
        self.waiting -= 1
        self.total += 1
        if self.admission_policy == "outstanding_watermark":
            current_step = self.group_queue[group_id].current_step
            if current_step is not None:
                self._refill_to_watermark(current_step)

    def _take_pending_group_gets(self) -> set[asyncio.Task]:
        """Return one pending get task for every unfinished group queue."""
        pending = {
            task
            for task in self.pending_gets
            if not task.cancelled()
            and task.get_name() not in self.rollout_complete
        }
        scheduled_group_ids = {task.get_name() for task in pending}
        for group_id, group_queue in self.group_queue.items():
            group_name = str(group_id)
            if (
                group_name not in self.rollout_complete
                and group_name not in scheduled_group_ids
            ):
                pending.add(
                    asyncio.create_task(group_queue.get(), name=group_name)
                )
        self.pending_gets = set()
        return pending

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
            self.current_batch_missing = (
                max(0, batch_size - len(ret)) if batch_size >= 0 else 0
            )

            if len(self.rollout_complete) == len(self.group_queue):
                break

            async def wait_a_episode():
                # Keep unfinished gets from the previous batch, but also
                # backfill queues whose get completed in an earlier batch.
                # Waiting only on a non-empty subset can strand ready work in
                # every other queue indefinitely.
                pending = self._take_pending_group_gets()

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
                        self._record_version_adaptive_consumption(group, len(group_rollout))
                        for rollout in group_rollout:
                            consumed_record = self._completed_rollout_record(rollout, group)
                            consumed_record.update(
                                category="consumed",
                                discard_reason="",
                                consumed_at_step=int(current_step),
                                trajectory_consumed_at_unix=time.time(),
                                version_age=max(0, int(current_step) - int(group.create_step)),
                            )
                            self.consumed_records.append(consumed_record)
                            self.new_consumed_records.append(consumed_record)
                        ret.extend(group_rollout)
                        progress_bar.update(len(group_rollout))

                    assert batch_size < 0 or (done and len(ret) >= batch_size) or (not done and len(ret) <= batch_size), f"{batch_size=}, {len(ret)=}, {done=}"
                    if done:
                        self.pending_gets.update(done)
                self.pending_gets.update(pending)
                self._refill_to_watermark(current_step)

            await wait_a_episode()
        get_batch_return_start_time = time.time()
        self.current_batch_missing = 0
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
    shutdown_grace_seconds = 5.0

    def __init__(self, config, env_manager_config: EnvManagerConfig, resource_manager, infer_cluster, mode, collator=None):
        self.config = config
        self.env_manager_config = env_manager_config
        self.resource_manager = resource_manager
        self.infer_cluster = infer_cluster
        self.mode = mode
        self.collator = collator

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        # Every environment can hold one long-lived get_episode_id call while
        # completed environments concurrently return results through put().
        # Keep both sets of calls runnable, plus get_batch/control headroom.
        queue_max_concurrency = 2 * env_num + 2
        self.env_output_queue = GroupQueueManager.options(
            name=f"GroupQueueManager-{mode}",
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False),
            max_concurrency=queue_max_concurrency,
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

        await self.env_output_queue.mark_rollout_end.remote()

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

        # Some environment threads may be waiting for the next episode rather
        # than inside generate(). Wake them without clearing buffered rollouts;
        # shutdown waste is collected after the rollout loop exits.
        await self.env_output_queue.stop_admission.remote()

        # Environment threads may be blocked in generate(). Abort those
        # requests before waiting for the rollout loop to observe running=False.
        try:
            await asyncio.wait_for(
                self.router_manager.shutdown.remote(),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            timeout_stages.append("router")
            logger.warning("timed out stopping rollout router")

        rollout_task_cancelled = False
        done, _ = await asyncio.wait(
            {self.rollout_task}, timeout=self.shutdown_grace_seconds
        )
        if not done:
            # External environments such as SWE sandboxes can remain blocked in
            # reset() after admission has stopped. The terminal snapshot is
            # already complete, so detach that orchestration wait instead of
            # delaying pipeline shutdown for work that cannot be consumed.
            rollout_task_cancelled = True
            self.rollout_task.cancel()
            done, _ = await asyncio.wait(
                {self.rollout_task}, timeout=timeout_seconds
            )

        if done:
            if not self.rollout_task.cancelled():
                self.rollout_task.result()
        else:
            timeout_stages.append("rollout_loop")
            logger.warning("timed out cancelling rollout loop during shutdown")

        inflight_records = [
            record
            for worker_records in worker_snapshots
            for record in worker_records
        ]
        shutdown_report = await self.env_output_queue.collect_shutdown_waste.remote(inflight_records)
        boundary_recovery, router_lifetime_metrics = await asyncio.gather(
            self.router_manager.collect_version_boundary_profile.remote(),
            self.router_manager.collect_lifetime_request_metrics.remote(),
        )
        shutdown_report["version_boundary_recovery"] = boundary_recovery
        for name, value in boundary_recovery.get("metrics", {}).items():
            shutdown_report["metrics"][f"version_boundary_recovery/{name}"] = value
        shutdown_report["router_lifetime"] = {
            "metrics": router_lifetime_metrics,
        }
        for name, value in router_lifetime_metrics.items():
            shutdown_report["metrics"][f"router_lifetime/{name}"] = value
        await self.env_output_queue.shutdown.remote()

        shutdown_report["shutdown"] = {
            "timeout_seconds": timeout_seconds,
            "timeout_stages": timeout_stages,
            "rollout_task_cancelled": rollout_task_cancelled,
        }
        shutdown_report["metrics"]["terminal_waste/shutdown_timeouts"] = len(timeout_stages)
        shutdown_report["metrics"]["terminal_waste/rollout_task_cancelled"] = int(
            rollout_task_cancelled
        )
        self.rollout_task = None
        return shutdown_report

    async def suspend(self):
        await self.router_manager.suspend.remote()
        await self._snapshot_trajectory_progress()
        await self.router_manager.abort_all.remote()
        await self.router_manager.wait_complete.remote()

    async def _snapshot_trajectory_progress(self):
        worker_snapshots, router_snapshots = await asyncio.gather(
            asyncio.gather(*self.es_manager.collect_trajectory_progress(blocking=False)),
            self.router_manager.collect_trajectory_progress.remote(),
        )
        progress_snapshots = [
            snapshot
            for worker_records in worker_snapshots
            for snapshot in worker_records
        ]
        progress_snapshots.extend(router_snapshots)
        await self.env_output_queue.update_trajectory_progress.remote(progress_snapshots)

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

        await self._snapshot_trajectory_progress()
        await asyncio.gather(*self.es_manager.update_step(global_step, inject_trace_context({}), blocking=False))
        kv_feedback = await self.router_manager.collect_runtime_feedback.remote()
        runtime_plan = await self.env_output_queue.advance_step.remote(
            global_step, kv_feedback
        )
        await self.router_manager.resume.remote(runtime_plan)

        learner_wait_start = time.time()
        get_task = asyncio.create_task(self._get_batch(batch_size, global_step))
        reconcile_interval = max(
            0.1,
            float(
                getattr(
                    self.config,
                    "version_runtime_reconcile_interval_seconds",
                    5.0,
                )
            ),
        )
        while not get_task.done():
            done, _ = await asyncio.wait(
                {get_task, self.rollout_task},
                timeout=reconcile_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self.rollout_task in done and self.rollout_task.exception() is not None:
                await self.rollout_task
            if get_task.done():
                break
            if self.mode == "train":
                revised_plan = (
                    await self.env_output_queue.reconcile_version_progress.remote(
                        global_step,
                        batch_size,
                        time.time() - learner_wait_start,
                    )
                )
                if revised_plan is not None:
                    await self.router_manager.update_runtime_plan.remote(revised_plan)
        data_batch = await get_task
        if self.mode == "train":
            await self.env_output_queue.record_learner_wait.remote(
                time.time() - learner_wait_start,
                global_step,
                batch_size,
            )
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
        metrics.update(await self.router_manager.collect_request_metrics.remote())
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

    async def record_policy_update_timing(
        self, version: int, timing: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Forward driver-side phase timing to the trajectory state owner."""
        return await self.env_output_queue.record_policy_update_timing.remote(
            int(version), dict(timing)
        )

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
