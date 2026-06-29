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


@dataclass
class SystemCostWeights:
    """System-only resume scheduling weights.

    These weights intentionally exclude semantic trajectory quality signals such
    as loop/stall/low reward. Lifecycle invalidation should be handled outside
    the score as a hard state-machine rule.
    """

    prefill_h_max: float = 32768.0
    queue_cost: float = 1.0
    load_cost: float = 0.5
    delay_regret: float = 1.0
    dispatch_value: float = 1.0
    age: float = 0.1
    age_norm_s: float = 10.0
    age_max: float = 5.0
    p_hit_decay_per_s: float = 0.02
    horizon_delta_s: float = 1.0
    kv_bytes_per_token: float = 2048.0
    memory_cost: float = 1.0e-9
    memory_pressure_default: float = 1.0

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "SystemCostWeights":
        cfg = cfg or {}
        return cls(
            prefill_h_max=float(cfg.get("prefill_h_max", cls.prefill_h_max)),
            queue_cost=float(cfg.get("queue_cost", cfg.get("lambda_q", cls.queue_cost))),
            load_cost=float(cfg.get("load_cost", cfg.get("lambda_load", cls.load_cost))),
            delay_regret=float(cfg.get("delay_regret", cfg.get("lambda_1", cls.delay_regret))),
            dispatch_value=float(cfg.get("dispatch_value", cfg.get("lambda_2", cls.dispatch_value))),
            age=float(cfg.get("age", cfg.get("lambda_3", cls.age))),
            age_norm_s=float(cfg.get("age_norm_s", cls.age_norm_s)),
            age_max=float(cfg.get("age_max", cls.age_max)),
            p_hit_decay_per_s=float(cfg.get("p_hit_decay_per_s", cls.p_hit_decay_per_s)),
            horizon_delta_s=float(cfg.get("horizon_delta_s", cls.horizon_delta_s)),
            kv_bytes_per_token=float(cfg.get("kv_bytes_per_token", cls.kv_bytes_per_token)),
            memory_cost=float(cfg.get("memory_cost", cfg.get("lambda_mem", cls.memory_cost))),
            memory_pressure_default=float(cfg.get("memory_pressure_default", cls.memory_pressure_default)),
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
    p_hit_base: Optional[float] = None,
) -> float:
    last_backend_id = route_meta.get("last_backend_id")
    base = p_hit_base if p_hit_base is not None else belief_to_p_hit(belief_level, belief)
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


def classify_system_cost_belief(
    route_meta: Dict[str, Any],
    *,
    belief: BeliefConfig,
    force_migrate_age_s: float,
    last_worker_overloaded: bool = False,
) -> BeliefLevel:
    """Belief for system-cost scheduling.

    Unlike `classify_belief`, this does not treat loop/stall/semantic invalid as
    a cache-locality signal. COLD is reserved for lifecycle or cache eligibility
    failures and very stale resumes.
    """
    pause_age = _float_meta(route_meta, "pause_age_s")
    last_backend_id = route_meta.get("last_backend_id")

    if _indicator(route_meta, "trajectory_terminated") >= 0.5:
        return BeliefLevel.COLD
    if _indicator(route_meta, "model_version_mismatch") >= 0.5:
        return BeliefLevel.COLD
    if _indicator(route_meta, "prefix_hash_mismatch") >= 0.5:
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


@dataclass
class EngineTelemetryConfig:
    """Phase C: engine-measured prefix hit for system-cost scoring."""

    measured_weight: float = 0.7
    hit_ratio_threshold: float = 0.3
    full_prefill_ratio: float = 0.85

    @classmethod
    def from_config(cls, cfg: Optional[Dict]) -> "EngineTelemetryConfig":
        cfg = cfg or {}
        return cls(
            measured_weight=float(cfg.get("engine_telemetry_measured_weight", cfg.get("measured_weight", cls.measured_weight))),
            hit_ratio_threshold=float(
                cfg.get("engine_telemetry_hit_ratio_threshold", cfg.get("hit_ratio_threshold", cls.hit_ratio_threshold))
            ),
            full_prefill_ratio=float(
                cfg.get("engine_telemetry_full_prefill_ratio", cfg.get("full_prefill_ratio", cls.full_prefill_ratio))
            ),
        )


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def apply_resume_engine_telemetry(
    route_meta: Dict[str, Any],
    out: Dict[str, Any],
    *,
    config: EngineTelemetryConfig,
) -> bool:
    """Copy worker prefix-cache telemetry into route_meta/out. Returns True if measured."""
    matched = _float_or_none(out.get("matched_prefix_tokens"))
    if matched is None:
        return False
    history_len = max(1.0, _float_meta(route_meta, "history_len_tokens"))
    prefill = _float_or_none(out.get("resume_prefill_tokens"))
    if prefill is None:
        prompt_tokens = _float_or_none(out.get("prompt_tokens"))
        if prompt_tokens is not None:
            prefill = max(0.0, prompt_tokens - matched)
        else:
            prefill = max(0.0, history_len - matched)
    prefill_ratio = prefill / history_len
    hit_ratio = max(0.0, min(1.0, matched / history_len))
    out["matched_prefix_tokens"] = matched
    out["resume_prefill_tokens"] = prefill
    out["prefill_ratio"] = prefill_ratio
    out["actual_hit"] = 1.0 if matched > 0 else 0.0
    out["engine_cache_confidence"] = hit_ratio
    route_meta["matched_prefix_tokens"] = matched
    route_meta["resume_prefill_tokens"] = prefill
    route_meta["prefill_ratio"] = prefill_ratio
    route_meta["actual_hit"] = out["actual_hit"]
    route_meta["engine_cache_confidence"] = hit_ratio
    route_meta["p_hit_measured"] = hit_ratio
    out["p_hit_measured"] = hit_ratio
    return True


def compute_p_hit_measured(route_meta: Dict[str, Any]) -> Optional[float]:
    raw = route_meta.get("p_hit_measured")
    if raw is None:
        raw = route_meta.get("engine_cache_confidence")
    return _float_or_none(raw)


def classify_resume_context_class(
    route_meta: Dict[str, Any],
    *,
    affinity_hit: bool,
    config: EngineTelemetryConfig,
) -> str:
    """Classify resume cache outcome from engine telemetry (not bare affinity)."""
    history_len = max(1.0, _float_meta(route_meta, "history_len_tokens"))
    matched = _float_or_none(route_meta.get("matched_prefix_tokens"))
    prefill_ratio = _float_or_none(route_meta.get("prefill_ratio"))
    if prefill_ratio is None and matched is not None:
        prefill_ratio = max(0.0, history_len - matched) / history_len
    if prefill_ratio is None:
        prefill_ratio = 1.0
    if matched is not None:
        hit_ratio = matched / history_len
    else:
        measured = compute_p_hit_measured(route_meta)
        hit_ratio = measured if measured is not None else 0.0
    if hit_ratio >= config.hit_ratio_threshold:
        return "gpu_hit"
    if prefill_ratio >= config.full_prefill_ratio:
        return "full_prefill"
    if affinity_hit and hit_ratio > 0:
        return "cpu_reload"
    return "full_prefill"


def merge_effective_p_hit(
    p_hit_belief: float,
    route_meta: Dict[str, Any],
    *,
    measured_weight: float,
    enabled: bool,
) -> float:
    measured = compute_p_hit_measured(route_meta)
    if not enabled or measured is None:
        return p_hit_belief
    w_m = max(0.0, min(1.0, float(measured_weight)))
    return w_m * measured + (1.0 - w_m) * p_hit_belief


def compute_history_prefill_cost(route_meta: Dict[str, Any], *, weights: SystemCostWeights) -> float:
    """Normalized proxy for the prefill cost saved by a KV/prefix hit."""
    return _norm_log(_float_meta(route_meta, "history_len_tokens"), weights.prefill_h_max)


def compute_system_dispatch_score(
    route_meta: Dict[str, Any],
    *,
    p_hit: float,
    worker_load: float,
    worker_queue_delay_s: float = 0.0,
    weights: SystemCostWeights,
) -> tuple[float, float]:
    """Return (dispatch_score, expected_prefill_saved)."""
    prefill_cost = compute_history_prefill_cost(route_meta, weights=weights)
    expected_saved = max(0.0, float(p_hit)) * prefill_cost
    score = (
        expected_saved
        - weights.queue_cost * max(0.0, float(worker_queue_delay_s))
        - weights.load_cost * max(0.0, float(worker_load))
    )
    return score, expected_saved


def compute_system_order_score(
    route_meta: Dict[str, Any],
    *,
    belief: BeliefConfig,
    force_migrate_age_s: float,
    weights: SystemCostWeights,
    worker_load: float = 0.0,
    worker_queue_delay_s: float = 0.0,
    queue_wait_s: float = 0.0,
    last_worker_overloaded: bool = False,
    p_hit_bias: float = 0.0,
    feedback_hot_downgrade_bias: float = -0.15,
    enable_engine_telemetry: bool = False,
    engine_telemetry_measured_weight: float = 0.7,
) -> tuple[float, BeliefLevel, float, float]:
    """Ordering score for resume requests under system-cost design.

    Returns (order_score, belief_level, p_hit_effective, dispatch_score).
    """
    level = classify_system_cost_belief(
        route_meta,
        belief=belief,
        force_migrate_age_s=force_migrate_age_s,
        last_worker_overloaded=last_worker_overloaded,
    )
    if level == BeliefLevel.HOT and p_hit_bias <= feedback_hot_downgrade_bias:
        level = BeliefLevel.WARM
    p_hit_belief = apply_p_hit_bias(belief_to_p_hit(level, belief), p_hit_bias, belief=belief)
    p_hit = merge_effective_p_hit(
        p_hit_belief,
        route_meta,
        measured_weight=engine_telemetry_measured_weight,
        enabled=enable_engine_telemetry,
    )
    route_meta["p_hit_belief"] = p_hit_belief
    route_meta["p_hit_effective"] = p_hit
    dispatch_score, expected_saved = compute_system_dispatch_score(
        route_meta,
        p_hit=p_hit,
        worker_load=worker_load,
        worker_queue_delay_s=worker_queue_delay_s,
        weights=weights,
    )
    delta = max(0.0, float(weights.horizon_delta_s))
    decay = max(0.0, float(weights.p_hit_decay_per_s)) * delta
    ttl_remaining = _float_meta(route_meta, "ttl_remaining_s", -1.0)
    if ttl_remaining >= 0.0:
        decay = max(decay, max(0.0, delta - ttl_remaining) / max(delta, 1e-6))
    p_after = max(belief.p_cold, p_hit - decay)
    delay_regret = max(0.0, p_hit - p_after) * compute_history_prefill_cost(route_meta, weights=weights)
    age_norm = max(1e-6, float(weights.age_norm_s))
    age_bonus = min(max(0.0, float(queue_wait_s)) / age_norm, max(0.0, float(weights.age_max)))
    order_score = (
        weights.delay_regret * delay_regret
        + weights.dispatch_value * max(0.0, dispatch_score)
        + weights.age * age_bonus
    )
    route_meta["system_delay_regret"] = delay_regret
    route_meta["expected_prefill_saved"] = expected_saved
    route_meta["dispatch_score"] = dispatch_score
    route_meta["order_score"] = order_score
    return order_score, level, p_hit, dispatch_score


def compute_system_worker_route_score(
    dp_rank: int,
    route_meta: Dict[str, Any],
    *,
    belief_level: BeliefLevel,
    belief: BeliefConfig,
    worker_load: float,
    weights: SystemCostWeights,
    p_hit_base: Optional[float] = None,
) -> float:
    base = p_hit_base if p_hit_base is not None else _float_or_none(route_meta.get("p_hit_effective"))
    p_w = p_hit_for_worker(
        dp_rank,
        route_meta,
        belief_level=belief_level,
        belief=belief,
        p_hit_base=base,
    )
    score, expected_saved = compute_system_dispatch_score(
        route_meta,
        p_hit=p_w,
        worker_load=worker_load,
        weights=weights,
    )
    if route_meta.get("last_backend_id") == dp_rank:
        route_meta["expected_prefill_saved"] = expected_saved
    return score


def compute_system_lease_ttl(
    route_meta: Dict[str, Any],
    *,
    p_hit: float,
    t_tool_s: float,
    belief_level: BeliefLevel,
    weights: SystemCostWeights,
    lease_weights: LeaseTtlWeights,
) -> tuple[float, float]:
    """System-cost TTL/lease score from prefill benefit minus memory byte-seconds."""
    t_min = max(0.0, lease_weights.t_tool_min)
    t_max = max(t_min, lease_weights.t_tool_max)
    # Do not let tiny t_tool_ema (e.g. local BM25 ms waits) drive TTL below t_tool_min.
    t_tool = max(t_min, max(0.1, float(t_tool_s)))
    candidates = sorted({
        t_min,
        min(t_max, t_tool),
        min(t_max, 2.0 * t_tool),
        t_max,
    })
    prefill_cost = compute_history_prefill_cost(route_meta, weights=weights)
    expected_saved = max(0.0, float(p_hit)) * prefill_cost
    history_len = _float_meta(route_meta, "history_len_tokens")
    kv_bytes = _float_meta(route_meta, "kv_bytes", history_len * weights.kv_bytes_per_token)
    memory_pressure = _float_meta(route_meta, "memory_pressure", weights.memory_pressure_default)
    best_ttl = t_min
    best_value = float("-inf")
    for ttl in candidates:
        tool_return_prob = max(0.0, min(1.0, ttl / t_tool))
        value = (
            tool_return_prob * expected_saved
            - weights.memory_cost * max(0.0, kv_bytes) * ttl * max(0.0, memory_pressure)
        )
        if value > best_value:
            best_value = value
            best_ttl = ttl
    if belief_level == BeliefLevel.COLD or _indicator(route_meta, "trajectory_terminated") >= 0.5:
        best_ttl = min(best_ttl, t_min)
    lease_score = max(0.0, min(1.0, best_value))
    route_meta["kv_bytes_proxy"] = max(0.0, kv_bytes)
    route_meta["memory_pressure"] = max(0.0, memory_pressure)
    return best_ttl, lease_score


RESUME_WASTE_METRIC_KEYS = (
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
)


def _first_float(source: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _float_or_none(source.get(key))
        if value is not None:
            return value
    return None


def annotate_resume_waste_metrics(route_meta: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
    """Attach resume-boundary waste metrics to route_meta and output.

    These metrics are intentionally conservative and observation-friendly:
    they do not change routing decisions, but make the four target wastes
    measurable from rollout logs.
    """
    if not isinstance(route_meta, dict) or not isinstance(out, dict):
        return {}

    merged: Dict[str, Any] = dict(route_meta)
    merged.update(out)

    history_len = _first_float(merged, "history_len_tokens", "resume_history_len_tokens")
    matched_tokens = _first_float(merged, "matched_prefix_tokens", "lookup_hit_tokens")
    prefill_tokens = _first_float(merged, "resume_prefill_tokens", "estimated_prefill_tokens")
    if prefill_tokens is None and history_len is not None and matched_tokens is not None:
        prefill_tokens = max(0.0, history_len - matched_tokens)

    p_hit_effective = _first_float(
        merged,
        "p_hit_effective",
        "p_hit_measured",
        "belief_p_hit",
        "lookup_cache_confidence",
        "cache_confidence",
    )
    kv_bytes = _first_float(merged, "kv_bytes", "kv_bytes_proxy")
    if kv_bytes is None and history_len is not None:
        kv_bytes_per_token = _first_float(merged, "kv_bytes_per_token")
        if kv_bytes_per_token is None:
            kv_bytes_per_token = 2048.0
        kv_bytes = max(0.0, history_len) * max(0.0, kv_bytes_per_token)
        metrics_kv_bytes_proxy = kv_bytes
    else:
        metrics_kv_bytes_proxy = None
    lease_ttl = _first_float(
        merged,
        "resume_lease_ttl_s",
        "pending_resume_lease_ttl_s",
        "ttl_remaining_s",
        "lookup_lease_remaining_s",
    )
    pause_age = _first_float(merged, "pause_age_s")
    prefill_time_ms = _first_float(merged, "prefill_time_ms")
    prefill_ms_per_token = _first_float(merged, "prefill_ms_per_token")
    if prefill_ms_per_token is None and prefill_time_ms is not None and prefill_tokens and prefill_tokens > 0:
        prefill_ms_per_token = max(0.0, prefill_time_ms / prefill_tokens)
    if prefill_ms_per_token is None and prefill_tokens and prefill_tokens > 0:
        engine_start_ts = _first_float(merged, "engine_start_ts")
        engine_first_token_ts = _first_float(merged, "engine_first_token_ts", "resume_first_token_ts")
        if engine_start_ts is not None and engine_first_token_ts is not None:
            ttft_ms = max(0.0, engine_first_token_ts - engine_start_ts) * 1000.0
            prefill_ms_per_token = ttft_ms / max(1.0, prefill_tokens)

    metrics: Dict[str, Any] = {}
    if metrics_kv_bytes_proxy is not None:
        metrics["kv_bytes_proxy"] = metrics_kv_bytes_proxy
    if matched_tokens is not None:
        metrics["saved_prefill_tokens"] = max(0.0, matched_tokens)
    if matched_tokens is not None and prefill_ms_per_token is not None:
        metrics["saved_prefill_ms"] = max(0.0, matched_tokens) * max(0.0, prefill_ms_per_token)

    effective_ttl = None
    if lease_ttl is not None and pause_age is not None:
        effective_ttl = max(0.0, min(lease_ttl, pause_age))
    elif lease_ttl is not None:
        effective_ttl = max(0.0, lease_ttl)
    elif pause_age is not None:
        effective_ttl = max(0.0, pause_age)
    if effective_ttl is not None:
        metrics["kv_lease_effective_ttl_s"] = effective_ttl
    if kv_bytes is not None and effective_ttl is not None:
        pinned_gb_seconds = max(0.0, kv_bytes) * max(0.0, effective_ttl) / (1024.0 ** 3)
        metrics["pinned_kv_gb_seconds"] = pinned_gb_seconds
        saved_prefill_ms = _float_or_none(metrics.get("saved_prefill_ms"))
        if saved_prefill_ms is not None and pinned_gb_seconds > 0:
            metrics["saved_prefill_ms_per_gb_second"] = saved_prefill_ms / pinned_gb_seconds

    if history_len is not None and p_hit_effective is not None and matched_tokens is not None:
        expected_hit_tokens = max(0.0, min(1.0, p_hit_effective)) * max(0.0, history_len)
        metrics["avoidable_reprefill_tokens"] = max(0.0, expected_hit_tokens - max(0.0, matched_tokens))

    actual_hit = _first_float(merged, "actual_hit")
    if actual_hit is None and matched_tokens is not None:
        actual_hit = 1.0 if matched_tokens > 0 else 0.0
    pinned_gb_seconds_value = _float_or_none(metrics.get("pinned_kv_gb_seconds"))
    if actual_hit is not None and actual_hit <= 0.0 and pinned_gb_seconds_value is not None:
        metrics["dead_pinned_kv_gb_seconds"] = pinned_gb_seconds_value

    belief_level = str(merged.get("belief_level", "")).lower()
    is_hot_resume = belief_level == "hot" or (pause_age is not None and pause_age <= 5.0)
    if is_hot_resume:
        metrics["hot_resume_miss_ratio"] = 0.0 if (actual_hit is not None and actual_hit > 0.0) else 1.0

    last_backend_id = merged.get("last_backend_id")
    selected_backend_id = merged.get("selected_backend_id")
    if isinstance(last_backend_id, int) and isinstance(selected_backend_id, int):
        metrics["locality_mismatch_count"] = 0.0 if last_backend_id == selected_backend_id else 1.0

    delay_regret = _first_float(merged, "system_delay_regret")
    if delay_regret is not None:
        metrics["queue_decay_loss_proxy"] = max(0.0, delay_regret)
        if history_len is not None and prefill_ms_per_token is not None:
            metrics["queue_decay_loss_ms"] = max(0.0, delay_regret) * max(0.0, history_len) * max(0.0, prefill_ms_per_token)

    for key, value in metrics.items():
        route_meta[key] = value
        out[key] = value
    return metrics


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


def plan_tool_suspend_lease(
    route_meta: Dict[str, Any],
    *,
    belief: BeliefConfig,
    force_migrate_age_s: float,
    value_weights: TrajectoryValueWeights,
    penalty_weights: LearningPenaltyWeights,
    lease_weights: LeaseTtlWeights,
    t_tool_s: float,
    p_hit_bias: float = 0.0,
    last_worker_overloaded: bool = False,
    feedback_hot_downgrade_bias: float = -0.15,
    use_system_cost: bool = False,
    system_cost_weights: Optional[SystemCostWeights] = None,
) -> tuple[float, float, BeliefLevel, float]:
    """Compute (ttl_s, lease_score, belief_level, v_traj) at tool-suspend boundary."""
    if use_system_cost:
        system_cost_weights = system_cost_weights or SystemCostWeights()
        priority, level, p_hit, _ = compute_system_order_score(
            route_meta,
            belief=belief,
            force_migrate_age_s=force_migrate_age_s,
            weights=system_cost_weights,
            worker_load=0.0,
            last_worker_overloaded=last_worker_overloaded,
            p_hit_bias=p_hit_bias,
            feedback_hot_downgrade_bias=feedback_hot_downgrade_bias,
        )
        ttl, lease_score = compute_system_lease_ttl(
            route_meta,
            p_hit=p_hit,
            t_tool_s=t_tool_s,
            belief_level=level,
            weights=system_cost_weights,
            lease_weights=lease_weights,
        )
        return ttl, lease_score, level, priority

    priority, level, p_hit = compute_resume_priority(
        route_meta,
        belief=belief,
        force_migrate_age_s=force_migrate_age_s,
        value_weights=value_weights,
        penalty_weights=penalty_weights,
        last_worker_overloaded=last_worker_overloaded,
        p_hit_bias=p_hit_bias,
        feedback_hot_downgrade_bias=feedback_hot_downgrade_bias,
    )
    ttl, lease_score = compute_lease_ttl(
        route_meta,
        p_hit=p_hit,
        v_traj=priority,
        t_tool_s=t_tool_s,
        belief_level=level,
        weights=lease_weights,
    )
    return ttl, lease_score, level, priority


def merge_resume_lease_ttl_score(
    route_meta: Dict[str, Any],
    *,
    store_pending_ttl: Optional[float] = None,
    store_pending_score: Optional[float] = None,
) -> tuple[Optional[float], Optional[float], bool]:
    """Merge resume-computed lease with env pending suspend lease (max TTL / score)."""
    resume_ttl = route_meta.get("resume_lease_ttl_s")
    resume_score = route_meta.get("resume_lease_score")
    pending_ttl = route_meta.get("pending_resume_lease_ttl_s")
    pending_score = route_meta.get("pending_resume_lease_score")
    from_pending_meta = pending_ttl is not None or pending_score is not None
    if store_pending_ttl is not None:
        pending_ttl = (
            store_pending_ttl
            if pending_ttl is None
            else max(float(pending_ttl), float(store_pending_ttl))
        )
    if store_pending_score is not None and pending_score is None:
        pending_score = store_pending_score
    candidates_ttl = [
        float(x) for x in (resume_ttl, pending_ttl) if isinstance(x, (int, float))
    ]
    candidates_score = [
        float(x) for x in (resume_score, pending_score) if isinstance(x, (int, float))
    ]
    final_ttl = max(candidates_ttl) if candidates_ttl else None
    final_score = max(candidates_score) if candidates_score else None
    used_pending = from_pending_meta or store_pending_ttl is not None
    if final_ttl is not None:
        route_meta["resume_lease_ttl_s"] = final_ttl
    if final_score is not None:
        route_meta["resume_lease_score"] = final_score
    return final_ttl, final_score, used_pending


def should_send_preferred_header(belief_level: BeliefLevel, route_meta: Dict[str, Any]) -> bool:
    """Form B: only emit strong affinity hint when belief is HOT and hint exists."""
    if belief_level != BeliefLevel.HOT:
        return False
    last_backend_id = route_meta.get("last_backend_id")
    return isinstance(last_backend_id, int)