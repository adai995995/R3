import pytest

from roll.distributed.scheduler.trajectory_value import (
    BeliefConfig,
    BeliefLevel,
    LearningPenaltyWeights,
    TrajectoryValueWeights,
    classify_belief,
    compute_learning_penalty,
    compute_resume_priority,
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
