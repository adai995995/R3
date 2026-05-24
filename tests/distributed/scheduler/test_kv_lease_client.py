from roll.distributed.scheduler.kv_lease_client import LookupResumeResult
from roll.distributed.scheduler.trajectory_value import (
    BeliefConfig,
    LearningPenaltyWeights,
    LeaseTtlWeights,
    TrajectoryValueWeights,
    plan_tool_suspend_lease,
)


def test_lookup_resume_result_from_json():
    r = LookupResumeResult.from_json(
        {
            "found": True,
            "hit_tokens": 128,
            "estimated_prefill_tokens": 32,
            "cache_confidence": 0.9,
            "lease_remaining_s": 12.5,
            "worker_url": "http://w0:8000",
        }
    )
    assert r.found is True
    assert r.hit_tokens == 128
    assert r.cache_confidence == 0.9


def test_lookup_resume_empty_defaults_not_found():
    r = LookupResumeResult.from_json({})
    assert r.found is False


def test_plan_tool_suspend_lease_positive_ttl():
    route_meta = {
        "pause_age_s": 1.0,
        "last_backend_id": 0,
        "history_len_tokens": 1000.0,
        "remaining_steps_ratio": 0.5,
        "trajectory_invalid": 0.0,
        "trajectory_loop": 0.0,
        "trajectory_stall": 0.0,
        "trajectory_terminated": 0.0,
    }
    ttl, score, level, v = plan_tool_suspend_lease(
        route_meta,
        belief=BeliefConfig(),
        force_migrate_age_s=30.0,
        value_weights=TrajectoryValueWeights(),
        penalty_weights=LearningPenaltyWeights(),
        lease_weights=LeaseTtlWeights(),
        t_tool_s=5.0,
    )
    assert ttl >= 2.0
    assert 0.0 <= score <= 1.0
    assert v != 0.0
