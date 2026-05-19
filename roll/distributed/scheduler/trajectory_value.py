"""Trajectory value and recoverability belief for resume-aware scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class BeliefLevel(str, Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class TrajectoryValueWeights:
    w_p: float = 1.0
    w_f: float = 0.5
    w_a: float = 0.3
    w_c: float = 0.8
    w_q: float = 0.5
    h_max: float = 32768.0
    a_max: float = 60.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "TrajectoryValueWeights":
        cfg = cfg or {}
        return cls(
            w_p=float(cfg.get("w_p", cfg.get("p", cls.w_p))),
            w_f=float(cfg.get("w_f", cfg.get("finish", cls.w_f))),
            w_a=float(cfg.get("w_a", cfg.get("age", cls.w_a))),
            w_c=float(cfg.get("w_c", cfg.get("recompute", cls.w_c))),
            w_q=float(cfg.get("w_q", cfg.get("load", cls.w_q))),
            h_max=float(cfg.get("h_max", cls.h_max)),
            a_max=float(cfg.get("a_max", cls.a_max)),
        )


@dataclass
class LearningPenaltyWeights:
    c_inv: float = 1.5
    c_loop: float = 1.0
    c_stall: float = 0.5
    c_term: float = 10.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "LearningPenaltyWeights":
        cfg = cfg or {}
        return cls(
            c_inv=float(cfg.get("c_inv", cfg.get("invalid", cls.c_inv))),
            c_loop=float(cfg.get("c_loop", cfg.get("loop", cls.c_loop))),
            c_stall=float(cfg.get("c_stall", cfg.get("stall", cls.c_stall))),
            c_term=float(cfg.get("c_term", cfg.get("terminated", cls.c_term))),
        )


@dataclass
class BeliefConfig:
    p_hot: float = 0.85
    p_warm: float = 0.45
    p_cold: float = 0.10
    hot_pause_age_s: float = 5.0
    cold_pause_age_s: float = 30.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "BeliefConfig":
        cfg = cfg or {}
        return cls(
            p_hot=float(cfg.get("p_hot", cls.p_hot)),
            p_warm=float(cfg.get("p_warm", cls.p_warm)),
            p_cold=float(cfg.get("p_cold", cls.p_cold)),
            hot_pause_age_s=float(cfg.get("hot_pause_age_s", cls.hot_pause_age_s)),
            cold_pause_age_s=float(cfg.get("cold_pause_age_s", cls.cold_pause_age_s)),
        )


def _norm_log(value: float, vmax: float) -> float:
    if vmax <= 0:
        return 0.0
    return math.log1p(max(0.0, value)) / math.log1p(vmax)


def _float_meta(route_meta: Dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = route_meta.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _indicator(route_meta: Dict[str, Any], key: str) -> float:
    return 1.0 if _float_meta(route_meta, key, 0.0) >= 0.5 else 0.0


def compute_learning_penalty(
    route_meta: Dict[str, Any],
    *,
    weights: LearningPenaltyWeights,
) -> float:
    """Negative-only learning-side adjustment (invalid / loop / stall / terminated)."""
    return -(
        weights.c_inv * _indicator(route_meta, "trajectory_invalid")
        + weights.c_loop * _indicator(route_meta, "trajectory_loop")
        + weights.c_stall * _indicator(route_meta, "trajectory_stall")
        + weights.c_term * _indicator(route_meta, "trajectory_terminated")
    )


def classify_belief(
    route_meta: Dict[str, Any],
    *,
    belief: BeliefConfig,
    force_migrate_age_s: float,
    last_worker_overloaded: bool = False,
) -> BeliefLevel:
    pause_age = _float_meta(route_meta, "pause_age_s")
    last_backend_id = route_meta.get("last_backend_id")

    if _indicator(route_meta, "trajectory_terminated") >= 0.5:
        return BeliefLevel.COLD
    if _indicator(route_meta, "trajectory_invalid") >= 0.5 or _indicator(route_meta, "trajectory_loop") >= 0.5:
        return BeliefLevel.COLD
    if pause_age >= max(force_migrate_age_s, belief.cold_pause_age_s):
        return BeliefLevel.COLD
    if last_backend_id is None:
        return BeliefLevel.COLD
    if last_worker_overloaded:
        return BeliefLevel.WARM
    if pause_age <= belief.hot_pause_age_s:
        return BeliefLevel.HOT
    return BeliefLevel.WARM


def belief_to_p_hit(level: BeliefLevel, belief: BeliefConfig) -> float:
    if level == BeliefLevel.HOT:
        return belief.p_hot
    if level == BeliefLevel.WARM:
        return belief.p_warm
    return belief.p_cold


def apply_p_hit_bias(
    p_hit: float,
    p_hit_bias: float,
    *,
    belief: BeliefConfig,
) -> float:
    return max(belief.p_cold, min(belief.p_hot, float(p_hit) + float(p_hit_bias)))


@dataclass
class LeaseTtlWeights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 0.5
    delta: float = 1.0
    t_tool_min: float = 2.0
    t_tool_max: float = 120.0
    v_traj_scale: float = 5.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "LeaseTtlWeights":
        cfg = cfg or {}
        return cls(
            alpha=float(cfg.get("alpha", cls.alpha)),
            beta=float(cfg.get("beta", cls.beta)),
            gamma=float(cfg.get("gamma", cls.gamma)),
            delta=float(cfg.get("delta", cls.delta)),
            t_tool_min=float(cfg.get("t_tool_min", cls.t_tool_min)),
            t_tool_max=float(cfg.get("t_tool_max", cls.t_tool_max)),
            v_traj_scale=float(cfg.get("v_traj_scale", cls.v_traj_scale)),
        )


def compute_lease_score(v_traj: float, *, weights: LeaseTtlWeights) -> float:
    scale = max(1e-6, weights.v_traj_scale)
    return max(0.0, min(1.0, float(v_traj) / scale))


def compute_lease_ttl(
    route_meta: Dict[str, Any],
    *,
    p_hit: float,
    v_traj: float,
    t_tool_s: float,
    belief_level: BeliefLevel,
    weights: LeaseTtlWeights,
) -> tuple[float, float]:
    """Return (ttl_s, lease_score) from trajectory value and belief."""
    h = _float_meta(route_meta, "history_len_tokens")
    remaining_ratio = _float_meta(route_meta, "remaining_steps_ratio", 1.0)
    if "remaining_steps" in route_meta and "max_steps" in route_meta:
        max_steps = max(1.0, _float_meta(route_meta, "max_steps", 1.0))
        remaining_ratio = _float_meta(route_meta, "remaining_steps") / max_steps
    n_h = _norm_log(h, 32768.0)
    n_r = max(0.0, min(1.0, remaining_ratio))
    lease_score = compute_lease_score(v_traj, weights=weights)
    penalty = _indicator(route_meta, "trajectory_invalid") + _indicator(route_meta, "trajectory_loop")
    ttl = (
        float(t_tool_s)
        + weights.alpha * lease_score * n_h
        + weights.beta * float(p_hit) * float(t_tool_s)
        + weights.gamma * (1.0 - n_r) * float(t_tool_s)
        - weights.delta * penalty * float(t_tool_s)
    )
    if belief_level == BeliefLevel.COLD or _indicator(route_meta, "trajectory_terminated") >= 0.5:
        ttl = min(ttl, weights.t_tool_min)
    ttl = max(weights.t_tool_min, min(weights.t_tool_max, ttl))
    return ttl, lease_score


def p_hit_for_worker(
    dp_rank: int,
    route_meta: Dict[str, Any],
    *,
    belief_level: BeliefLevel,
    belief: BeliefConfig,
) -> float:
    last_backend_id = route_meta.get("last_backend_id")
    base = belief_to_p_hit(belief_level, belief)
    if isinstance(last_backend_id, int) and last_backend_id == dp_rank:
        return base
    if belief_level == BeliefLevel.HOT:
        return belief.p_warm * 0.5
    if belief_level == BeliefLevel.WARM:
        return belief.p_cold
    return belief.p_cold * 0.5


def compute_v_sys(
    route_meta: Dict[str, Any],
    *,
    p_hit: float,
    worker_load: float = 0.0,
    weights: TrajectoryValueWeights,
) -> float:
    h = _float_meta(route_meta, "history_len_tokens")
    pause_age = _float_meta(route_meta, "pause_age_s")
    remaining_ratio = _float_meta(route_meta, "remaining_steps_ratio", 1.0)
    if "remaining_steps" in route_meta and "max_steps" in route_meta:
        max_steps = max(1.0, _float_meta(route_meta, "max_steps", 1.0))
        remaining_ratio = _float_meta(route_meta, "remaining_steps") / max_steps

    n_h = _norm_log(h, weights.h_max)
    n_a = _norm_log(pause_age, weights.a_max)
    n_r = max(0.0, min(1.0, remaining_ratio))
    n_q = max(0.0, worker_load)

    return (
        weights.w_p * p_hit * n_h
        + weights.w_f * (1.0 - n_r)
        + weights.w_a * n_a
        - weights.w_c * (1.0 - p_hit) * n_h
        - weights.w_q * n_q
    )


def compute_trajectory_value(
    route_meta: Dict[str, Any],
    *,
    p_hit: float,
    worker_load: float = 0.0,
    value_weights: TrajectoryValueWeights,
    penalty_weights: LearningPenaltyWeights,
) -> float:
    v_sys = compute_v_sys(
        route_meta,
        p_hit=p_hit,
        worker_load=worker_load,
        weights=value_weights,
    )
    v_learn = compute_learning_penalty(route_meta, weights=penalty_weights)
    return v_sys + v_learn


def compute_resume_priority(
    route_meta: Dict[str, Any],
    *,
    belief: BeliefConfig,
    force_migrate_age_s: float,
    value_weights: TrajectoryValueWeights,
    penalty_weights: LearningPenaltyWeights,
    last_worker_overloaded: bool = False,
    p_hit_bias: float = 0.0,
    feedback_hot_downgrade_bias: float = -0.15,
) -> tuple[float, BeliefLevel, float]:
    """Ordering score for resume requests. Returns (priority, belief_level, p_hit_effective)."""
    level = classify_belief(
        route_meta,
        belief=belief,
        force_migrate_age_s=force_migrate_age_s,
        last_worker_overloaded=last_worker_overloaded,
    )
    if level == BeliefLevel.HOT and p_hit_bias <= feedback_hot_downgrade_bias:
        level = BeliefLevel.WARM
    p_hit = apply_p_hit_bias(belief_to_p_hit(level, belief), p_hit_bias, belief=belief)
    priority = compute_trajectory_value(
        route_meta,
        p_hit=p_hit,
        worker_load=0.0,
        value_weights=value_weights,
        penalty_weights=penalty_weights,
    )
    return priority, level, p_hit


def compute_worker_route_score(
    dp_rank: int,
    route_meta: Dict[str, Any],
    *,
    belief_level: BeliefLevel,
    belief: BeliefConfig,
    worker_load: float,
    value_weights: TrajectoryValueWeights,
    penalty_weights: LearningPenaltyWeights,
) -> float:
    """Placement score for a candidate worker."""
    p_w = p_hit_for_worker(dp_rank, route_meta, belief_level=belief_level, belief=belief)
    return compute_trajectory_value(
        route_meta,
        p_hit=p_w,
        worker_load=worker_load,
        value_weights=value_weights,
        penalty_weights=penalty_weights,
    )


def should_send_preferred_header(belief_level: BeliefLevel, route_meta: Dict[str, Any]) -> bool:
    """Form B: only emit strong affinity hint when belief is HOT and hint exists."""
    if belief_level != BeliefLevel.HOT:
        return False
    last_backend_id = route_meta.get("last_backend_id")
    return isinstance(last_backend_id, int)