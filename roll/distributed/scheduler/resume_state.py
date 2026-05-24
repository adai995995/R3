"""Per-trajectory scheduling state: belief feedback + tool-wait EMA + pending lease."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from roll.distributed.scheduler.kv_lease_client import LookupResumeResult
    from roll.distributed.scheduler.trajectory_value import (
        BeliefConfig,
        LearningPenaltyWeights,
        LeaseTtlWeights,
        TrajectoryValueWeights,
    )


@dataclass
class TrajectorySchedulingRecord:
    p_hit_bias: float = 0.0
    t_tool_ema_s: float = 5.0
    tool_wait_samples: int = 0
    pending_tool_lease_ttl_s: Optional[float] = None
    pending_tool_lease_score: Optional[float] = None
    pending_tool_lease_backend_id: Optional[int] = None


@dataclass
class BeliefFeedbackConfig:
    alpha_hit: float = 0.05
    alpha_miss: float = 0.08
    bias_min: float = -0.2
    bias_max: float = 0.2
    hot_downgrade_bias: float = -0.15

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "BeliefFeedbackConfig":
        cfg = cfg or {}
        return cls(
            alpha_hit=float(cfg.get("alpha_hit", cls.alpha_hit)),
            alpha_miss=float(cfg.get("alpha_miss", cls.alpha_miss)),
            bias_min=float(cfg.get("bias_min", cls.bias_min)),
            bias_max=float(cfg.get("bias_max", cls.bias_max)),
            hot_downgrade_bias=float(cfg.get("hot_downgrade_bias", cls.hot_downgrade_bias)),
        )


@dataclass
class SchedulingWeightSnapshot:
    """Shared weights for Env suspend-lease and Router (set on router initialize)."""

    value_weights: "TrajectoryValueWeights"
    penalty_weights: "LearningPenaltyWeights"
    belief: "BeliefConfig"
    lease_weights: "LeaseTtlWeights"
    force_migrate_age_s: float = 30.0
    feedback_hot_downgrade_bias: float = -0.15


_WEIGHT_SNAPSHOT: Optional[SchedulingWeightSnapshot] = None


def set_scheduling_weight_snapshot(snapshot: SchedulingWeightSnapshot) -> None:
    global _WEIGHT_SNAPSHOT
    _WEIGHT_SNAPSHOT = snapshot


def get_scheduling_weight_snapshot() -> Optional[SchedulingWeightSnapshot]:
    return _WEIGHT_SNAPSHOT


class TrajectorySchedulingState:
    """Thread-safe store shared by TrajEnvManager (write) and Router (read/write)."""

    def __init__(self, *, default_t_tool_s: float = 5.0, tool_ema_alpha: float = 0.2):
        self._lock = Lock()
        self._records: Dict[str, TrajectorySchedulingRecord] = {}
        self._default_t_tool_s = max(0.1, float(default_t_tool_s))
        self._tool_ema_alpha = max(0.01, min(1.0, float(tool_ema_alpha)))

    def clear(self, trajectory_id: str) -> None:
        with self._lock:
            self._records.pop(trajectory_id, None)

    def get_p_hit_bias(self, trajectory_id: Optional[str]) -> float:
        if not trajectory_id:
            return 0.0
        with self._lock:
            rec = self._records.get(trajectory_id)
            return float(rec.p_hit_bias) if rec is not None else 0.0

    def get_t_tool_s(self, trajectory_id: Optional[str]) -> float:
        if not trajectory_id:
            return self._default_t_tool_s
        with self._lock:
            rec = self._records.get(trajectory_id)
            if rec is None or rec.tool_wait_samples <= 0:
                return self._default_t_tool_s
            return float(rec.t_tool_ema_s)

    def sync_from_route_meta(self, trajectory_id: str, route_meta: Dict[str, Any]) -> None:
        """Apply scheduling fields from env meta (cross-process Router/Env)."""
        with self._lock:
            rec = self._records.setdefault(trajectory_id, TrajectorySchedulingRecord())
            raw_t = route_meta.get("scheduling_t_tool_s")
            if raw_t is not None:
                try:
                    rec.t_tool_ema_s = max(0.0, float(raw_t))
                    rec.tool_wait_samples = max(rec.tool_wait_samples, 1)
                except (TypeError, ValueError):
                    pass

    def update_tool_wait(self, trajectory_id: str, external_wait_s: float) -> None:
        wait_s = max(0.0, float(external_wait_s))
        with self._lock:
            rec = self._records.setdefault(trajectory_id, TrajectorySchedulingRecord())
            rec.tool_wait_samples += 1
            if rec.tool_wait_samples <= 1:
                rec.t_tool_ema_s = wait_s
            else:
                a = self._tool_ema_alpha
                rec.t_tool_ema_s = a * wait_s + (1.0 - a) * rec.t_tool_ema_s

    def observe_resume_outcome(
        self,
        trajectory_id: str,
        *,
        affinity_hit: bool,
        context_class: str,
        prefill_ratio: float,
        feedback: BeliefFeedbackConfig,
    ) -> None:
        with self._lock:
            rec = self._records.setdefault(trajectory_id, TrajectorySchedulingRecord())
            if affinity_hit and context_class == "gpu_hit":
                rec.p_hit_bias = min(
                    feedback.bias_max,
                    rec.p_hit_bias + feedback.alpha_hit * (1.0 - rec.p_hit_bias),
                )
            elif context_class == "full_prefill" or prefill_ratio >= 0.9:
                rec.p_hit_bias = max(
                    feedback.bias_min,
                    rec.p_hit_bias - feedback.alpha_miss,
                )
            elif not affinity_hit and context_class == "cpu_reload":
                rec.p_hit_bias = max(
                    feedback.bias_min,
                    rec.p_hit_bias - feedback.alpha_miss * 0.5,
                )

    def observe_lookup_resume(
        self,
        trajectory_id: str,
        lookup: "LookupResumeResult",
        *,
        feedback: BeliefFeedbackConfig,
        history_len_tokens: float = 0.0,
    ) -> None:
        """L2: adjust p_hit_bias from engine lookup_resume before dispatch."""
        if not trajectory_id or not lookup.found:
            return
        with self._lock:
            rec = self._records.setdefault(trajectory_id, TrajectorySchedulingRecord())
            conf = max(0.0, min(1.0, float(lookup.cache_confidence)))
            h = max(1.0, float(history_len_tokens))
            prefill_ratio = float(lookup.estimated_prefill_tokens) / h
            if conf >= 0.5 and lookup.hit_tokens > 0:
                rec.p_hit_bias = min(
                    feedback.bias_max,
                    rec.p_hit_bias + feedback.alpha_hit * conf * (1.0 - rec.p_hit_bias),
                )
            elif prefill_ratio >= 0.9 or lookup.estimated_prefill_tokens >= h * 0.9:
                rec.p_hit_bias = max(
                    feedback.bias_min,
                    rec.p_hit_bias - feedback.alpha_miss,
                )
            elif lookup.hit_tokens <= 0 and conf < 0.2:
                rec.p_hit_bias = max(
                    feedback.bias_min,
                    rec.p_hit_bias - feedback.alpha_miss * 0.5,
                )

    def set_pending_tool_lease(
        self,
        trajectory_id: str,
        *,
        ttl_s: float,
        lease_score: float,
        backend_id: Optional[int],
    ) -> None:
        with self._lock:
            rec = self._records.setdefault(trajectory_id, TrajectorySchedulingRecord())
            rec.pending_tool_lease_ttl_s = max(0.0, float(ttl_s))
            rec.pending_tool_lease_score = max(0.0, min(1.0, float(lease_score)))
            rec.pending_tool_lease_backend_id = backend_id

    def pop_pending_tool_lease(
        self, trajectory_id: Optional[str]
    ) -> tuple[Optional[float], Optional[float], Optional[int]]:
        if not trajectory_id:
            return None, None, None
        with self._lock:
            rec = self._records.get(trajectory_id)
            if rec is None:
                return None, None, None
            ttl = rec.pending_tool_lease_ttl_s
            score = rec.pending_tool_lease_score
            backend = rec.pending_tool_lease_backend_id
            rec.pending_tool_lease_ttl_s = None
            rec.pending_tool_lease_score = None
            rec.pending_tool_lease_backend_id = None
            return ttl, score, backend


_GLOBAL_STATE: Optional[TrajectorySchedulingState] = None
_GLOBAL_LOCK = Lock()


def get_trajectory_scheduling_state() -> TrajectorySchedulingState:
    global _GLOBAL_STATE
    with _GLOBAL_LOCK:
        if _GLOBAL_STATE is None:
            _GLOBAL_STATE = TrajectorySchedulingState()
        return _GLOBAL_STATE


def reset_trajectory_scheduling_state(
    *, default_t_tool_s: float = 5.0, tool_ema_alpha: float = 0.2
) -> TrajectorySchedulingState:
    """Replace global store (tests / router initialize)."""
    global _GLOBAL_STATE
    with _GLOBAL_LOCK:
        _GLOBAL_STATE = TrajectorySchedulingState(
            default_t_tool_s=default_t_tool_s,
            tool_ema_alpha=tool_ema_alpha,
        )
        return _GLOBAL_STATE
