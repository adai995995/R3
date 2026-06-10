import pytest

from roll.distributed.scheduler.trajectory_value import (
    BeliefConfig,
    BeliefLevel,
    EngineTelemetryConfig,
    LearningPenaltyWeights,
    LeaseTtlWeights,
    SystemCostWeights,
    TrajectoryValueWeights,
    apply_resume_engine_telemetry,
    classify_resume_context_class,
    classify_system_cost_belief,
    classify_belief,
    compute_learning_penalty,
    compute_resume_priority,
    compute_system_dispatch_score,
    compute_system_lease_ttl,
    compute_system_order_score,
    merge_effective_p_hit,
    compute_system_worker_route_score,
    should_send_preferred_header,
)
from roll.pipeline.agentic.trajectory_signals import compute_trajectory_signals


def test_learning_penalty_invalid_and_loop():
    route_meta = {
        "trajectory_invalid": 1.0,
        "trajectory_loop": 1.0,
        "trajectory_stall": 0.0,
        "trajectory_terminated": 0.0,
    }
    penalty = compute_learning_penalty(route_meta, weights=LearningPenaltyWeights())
    assert penalty < 0.0
    assert penalty <= -2.0


def test_classify_belief_hot_vs_cold():
    belief = BeliefConfig(hot_pause_age_s=5.0, cold_pause_age_s=30.0)
    hot_meta = {
        "pause_age_s": 1.0,
        "last_backend_id": 0,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_terminated": 0.0,
    }
    cold_meta = {
        "pause_age_s": 60.0,
        "last_backend_id": 0,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_terminated": 0.0,
    }
    assert classify_belief(hot_meta, belief=belief, force_migrate_age_s=30.0) == BeliefLevel.HOT
    assert classify_belief(cold_meta, belief=belief, force_migrate_age_s=30.0) == BeliefLevel.COLD


def test_invalid_forces_cold():
    belief = BeliefConfig()
    meta = {
        "pause_age_s": 0.5,
        "last_backend_id": 0,
        "trajectory_invalid": 1.0,
        "trajectory_loop": 0.0,
        "trajectory_terminated": 0.0,
    }
    assert classify_belief(meta, belief=belief, force_migrate_age_s=30.0) == BeliefLevel.COLD
    assert not should_send_preferred_header(BeliefLevel.COLD, meta)


def test_system_cost_belief_ignores_semantic_invalid_and_loop():
    belief = BeliefConfig()
    meta = {
        "pause_age_s": 0.5,
        "last_backend_id": 0,
        "trajectory_invalid": 1.0,
        "trajectory_loop": 1.0,
        "trajectory_terminated": 0.0,
    }
    assert classify_system_cost_belief(meta, belief=belief, force_migrate_age_s=30.0) == BeliefLevel.HOT


def test_resume_priority_penalized_below_clean():
    weights = TrajectoryValueWeights()
    penalty_weights = LearningPenaltyWeights()
    belief = BeliefConfig()
    clean = {
        "pause_age_s": 2.0,
        "history_len_tokens": 4096,
        "last_backend_id": 0,
        "remaining_steps_ratio": 0.5,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_stall": 0.0,
        "trajectory_terminated": 0.0,
    }
    dirty = dict(clean)
    dirty["trajectory_loop"] = 1.0
    p_clean, _, _ = compute_resume_priority(
        clean,
        belief=belief,
        force_migrate_age_s=30.0,
        value_weights=weights,
        penalty_weights=penalty_weights,
    )
    p_dirty, _, _ = compute_resume_priority(
        dirty,
        belief=belief,
        force_migrate_age_s=30.0,
        value_weights=weights,
        penalty_weights=penalty_weights,
    )
    assert p_dirty < p_clean


def test_system_cost_order_score_not_penalized_by_loop():
    belief = BeliefConfig()
    weights = SystemCostWeights()
    clean = {
        "pause_age_s": 2.0,
        "history_len_tokens": 4096,
        "last_backend_id": 0,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_stall": 0.0,
        "trajectory_terminated": 0.0,
    }
    dirty = dict(clean)
    dirty["trajectory_loop"] = 1.0
    p_clean, level_clean, hit_clean, _ = compute_system_order_score(
        clean,
        belief=belief,
        force_migrate_age_s=30.0,
        weights=weights,
    )
    p_dirty, level_dirty, hit_dirty, _ = compute_system_order_score(
        dirty,
        belief=belief,
        force_migrate_age_s=30.0,
        weights=weights,
    )
    assert p_dirty == p_clean
    assert level_dirty == level_clean
    assert hit_dirty == hit_clean


def test_system_cost_worker_route_prefers_less_loaded_when_no_hit_advantage():
    meta = {
        "pause_age_s": 60.0,
        "history_len_tokens": 4096,
        "last_backend_id": 0,
        "trajectory_terminated": 0.0,
    }
    belief = BeliefConfig()
    weights = SystemCostWeights(load_cost=1.0)
    score_busy = compute_system_worker_route_score(
        0,
        meta,
        belief_level=BeliefLevel.COLD,
        belief=belief,
        worker_load=10.0,
        weights=weights,
    )
    score_idle = compute_system_worker_route_score(
        1,
        meta,
        belief_level=BeliefLevel.COLD,
        belief=belief,
        worker_load=0.0,
        weights=weights,
    )
    assert score_idle > score_busy


def test_system_cost_lease_returns_bounded_ttl_and_score():
    meta = {"history_len_tokens": 8000}
    ttl, score = compute_system_lease_ttl(
        meta,
        p_hit=0.85,
        t_tool_s=5.0,
        belief_level=BeliefLevel.HOT,
        weights=SystemCostWeights(memory_cost=0.0),
        lease_weights=LeaseTtlWeights(t_tool_min=2.0, t_tool_max=20.0),
    )
    assert 2.0 <= ttl <= 20.0
    assert 0.0 <= score <= 1.0


def test_system_cost_lease_ttl_not_below_min_when_tool_wait_is_ms():
    meta = {"history_len_tokens": 8000}
    ttl, _ = compute_system_lease_ttl(
        meta,
        p_hit=0.85,
        t_tool_s=0.004,
        belief_level=BeliefLevel.HOT,
        weights=SystemCostWeights(memory_cost=1.0),
        lease_weights=LeaseTtlWeights(t_tool_min=2.0, t_tool_max=20.0),
    )
    assert ttl >= 2.0


def test_classify_resume_context_class_gpu_hit_from_measured_prefix():
    config = EngineTelemetryConfig(hit_ratio_threshold=0.3)
    meta = {"history_len_tokens": 1000, "matched_prefix_tokens": 400.0, "prefill_ratio": 0.6}
    assert classify_resume_context_class(meta, affinity_hit=True, config=config) == "gpu_hit"


def test_classify_resume_context_class_full_prefill_same_worker_no_hit():
    config = EngineTelemetryConfig(hit_ratio_threshold=0.3, full_prefill_ratio=0.85)
    meta = {"history_len_tokens": 1000, "matched_prefix_tokens": 0.0, "prefill_ratio": 1.0}
    assert classify_resume_context_class(meta, affinity_hit=True, config=config) == "full_prefill"


def test_classify_resume_context_class_cpu_reload_partial_hit():
    config = EngineTelemetryConfig(hit_ratio_threshold=0.3, full_prefill_ratio=0.85)
    meta = {"history_len_tokens": 1000, "matched_prefix_tokens": 100.0, "prefill_ratio": 0.5}
    assert classify_resume_context_class(meta, affinity_hit=True, config=config) == "cpu_reload"


def test_merge_effective_p_hit_blends_measured_and_belief():
    meta = {"p_hit_measured": 0.8}
    assert merge_effective_p_hit(0.2, meta, measured_weight=0.7, enabled=True) == pytest.approx(0.62)


def test_system_dispatch_score_differs_with_measured_p_hit():
    belief = BeliefConfig()
    weights = SystemCostWeights()
    meta = {"history_len_tokens": 4096, "pause_age_s": 2.0, "last_backend_id": 0}
    _, saved_low = compute_system_dispatch_score(meta, p_hit=0.1, worker_load=0.0, weights=weights)
    _, saved_high = compute_system_dispatch_score(meta, p_hit=0.9, worker_load=0.0, weights=weights)
    assert saved_high > saved_low


def test_system_order_score_uses_engine_telemetry_when_enabled():
    belief = BeliefConfig()
    weights = SystemCostWeights()
    meta = {
        "pause_age_s": 2.0,
        "history_len_tokens": 4096,
        "last_backend_id": 0,
        "p_hit_measured": 0.9,
    }
    _, _, hit_off, disp_off = compute_system_order_score(
        dict(meta),
        belief=belief,
        force_migrate_age_s=30.0,
        weights=weights,
        enable_engine_telemetry=False,
    )
    _, _, hit_on, disp_on = compute_system_order_score(
        dict(meta),
        belief=belief,
        force_migrate_age_s=30.0,
        weights=weights,
        enable_engine_telemetry=True,
        engine_telemetry_measured_weight=1.0,
    )
    assert hit_on > hit_off
    assert disp_on > disp_off


def test_apply_resume_engine_telemetry_populates_route_meta():
    config = EngineTelemetryConfig()
    route_meta = {"history_len_tokens": 500}
    out = {"matched_prefix_tokens": 200, "prompt_tokens": 500}
    assert apply_resume_engine_telemetry(route_meta, out, config=config)
    assert route_meta["p_hit_measured"] == pytest.approx(0.4)
    assert route_meta["resume_prefill_tokens"] == pytest.approx(300)


def test_trajectory_signals_loop_detection():
    history = []
    for _ in range(4):
        history.append({"llm_response": "same action", "reward": 0.0})
    signals = compute_trajectory_signals(
        history=history,
        step=4,
        max_steps=10,
        terminated=False,
        truncated=False,
        loop_window=3,
    )
    assert signals["trajectory_loop"] == 1.0
