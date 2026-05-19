from roll.distributed.scheduler.resume_state import (
    BeliefFeedbackConfig,
    TrajectorySchedulingState,
)
from roll.distributed.scheduler.trajectory_value import (
    BeliefConfig,
    BeliefLevel,
    LearningPenaltyWeights,
    LeaseTtlWeights,
    TrajectoryValueWeights,
    compute_lease_ttl,
    compute_resume_priority,
)


def test_tool_wait_ema():
    store = TrajectorySchedulingState(default_t_tool_s=5.0, tool_ema_alpha=1.0)
    store.update_tool_wait("t1", 2.0)
    store.update_tool_wait("t1", 8.0)
    assert store.get_t_tool_s("t1") == 8.0


def test_belief_feedback_bias():
    store = TrajectorySchedulingState()
    fb = BeliefFeedbackConfig(alpha_hit=0.1, alpha_miss=0.1)
    store.observe_resume_outcome("t1", affinity_hit=True, context_class="gpu_hit", prefill_ratio=0.1, feedback=fb)
    assert store.get_p_hit_bias("t1") > 0.0
    store.observe_resume_outcome("t1", affinity_hit=False, context_class="full_prefill", prefill_ratio=1.0, feedback=fb)
    assert store.get_p_hit_bias("t1") < 0.05


def test_lease_ttl_from_value():
    route_meta = {
        "history_len_tokens": 8000,
        "remaining_steps_ratio": 0.3,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_terminated": 0.0,
    }
    ttl, score = compute_lease_ttl(
        route_meta,
        p_hit=0.85,
        v_traj=2.0,
        t_tool_s=5.0,
        belief_level=BeliefLevel.HOT,
        weights=LeaseTtlWeights(),
    )
    assert ttl >= 2.0
    assert 0.0 <= score <= 1.0


def test_resume_priority_with_bias():
    route_meta = {
        "pause_age_s": 1.0,
        "history_len_tokens": 4096,
        "last_backend_id": 0,
        "remaining_steps_ratio": 0.5,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_stall": 0.0,
        "trajectory_terminated": 0.0,
    }
    p0, _, hit0 = compute_resume_priority(
        route_meta,
        belief=BeliefConfig(),
        force_migrate_age_s=30.0,
        value_weights=TrajectoryValueWeights(),
        penalty_weights=LearningPenaltyWeights(),
        p_hit_bias=0.0,
    )
    p1, _, hit1 = compute_resume_priority(
        route_meta,
        belief=BeliefConfig(),
        force_migrate_age_s=30.0,
        value_weights=TrajectoryValueWeights(),
        penalty_weights=LearningPenaltyWeights(),
        p_hit_bias=0.15,
    )
    assert hit1 >= hit0
    assert p1 >= p0
