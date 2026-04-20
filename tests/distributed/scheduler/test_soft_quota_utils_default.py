from roll.distributed.scheduler.soft_quota_utils import (
    ResumeFallbackContext,
    bucketize_queue_wait_s,
    bucketize_score,
    parse_ratio,
    resume_fallback_reason,
)


def test_parse_ratio():
    assert parse_ratio("3:1") == (3, 1)
    assert parse_ratio("0:0") == (1, 1)
    assert parse_ratio("") == (1, 1)
    assert parse_ratio("bad") == (1, 1)


def test_bucketize_queue_wait_s():
    assert bucketize_queue_wait_s(0.0) == "lt_10ms"
    assert bucketize_queue_wait_s(0.02) == "lt_100ms"
    assert bucketize_queue_wait_s(0.2) == "lt_500ms"
    assert bucketize_queue_wait_s(2.0) == "lt_3s"
    assert bucketize_queue_wait_s(20.0) == "ge_10s"


def test_bucketize_score():
    assert bucketize_score(-10.0) == "lt_-5"
    assert bucketize_score(-2.0) == "lt_-1"
    assert bucketize_score(-0.1) == "lt_0"
    assert bucketize_score(0.1) == "lt_1"
    assert bucketize_score(2.0) == "lt_5"
    assert bucketize_score(10.0) == "ge_5"


def test_resume_fallback_reason_granularity():
    ctx = ResumeFallbackContext(
        enable_resume_priority=True,
        enable_resume_aware_routing=True,
        active_dp_ranks={0, 1},
        hint_present=True,
        hint_ready=True,
        hint_inflight=0,
        hint_queue_depth=0,
        overloaded_inflight_threshold=10,
        overloaded_queue_depth_threshold=10,
        max_running_requests_per_worker=2,
        hint_worker_load=0,
    )

    # Hit is independent of hint.
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": 1},
            selected_dp_rank=1,
            previous_rank=1,
            ctx=ctx,
            force_migrate_age_s=30.0,
        )
        == "hit"
    )

    # Forced migrate dominates.
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 100.0, "last_backend_id": 1},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=ctx,
            force_migrate_age_s=30.0,
        )
        == "forced_migrate"
    )

    # Disabled.
    disabled_ctx = ResumeFallbackContext(
        enable_resume_priority=False,
        enable_resume_aware_routing=True,
        active_dp_ranks={0, 1},
        hint_present=True,
        hint_ready=True,
        hint_inflight=0,
        hint_queue_depth=0,
        overloaded_inflight_threshold=10,
        overloaded_queue_depth_threshold=10,
        max_running_requests_per_worker=2,
        hint_worker_load=0,
    )
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": 1},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=disabled_ctx,
            force_migrate_age_s=30.0,
        )
        == "disabled"
    )

    # No hint.
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=ctx,
            force_migrate_age_s=30.0,
        )
        == "no_hint"
    )

    # Not found.
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": "bad"},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=ctx,
            force_migrate_age_s=30.0,
        )
        == "not_found"
    )

    # Inactive (not in active ranks or not present).
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": 9},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=ctx,
            force_migrate_age_s=30.0,
        )
        == "inactive"
    )

    # Not ready.
    not_ready_ctx = ResumeFallbackContext(
        enable_resume_priority=True,
        enable_resume_aware_routing=True,
        active_dp_ranks={0, 1},
        hint_present=True,
        hint_ready=False,
        hint_inflight=0,
        hint_queue_depth=0,
        overloaded_inflight_threshold=10,
        overloaded_queue_depth_threshold=10,
        max_running_requests_per_worker=2,
        hint_worker_load=0,
    )
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": 1},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=not_ready_ctx,
            force_migrate_age_s=30.0,
        )
        == "not_ready"
    )

    # Overloaded (inflight threshold).
    overloaded_ctx = ResumeFallbackContext(
        enable_resume_priority=True,
        enable_resume_aware_routing=True,
        active_dp_ranks={0, 1},
        hint_present=True,
        hint_ready=True,
        hint_inflight=10,
        hint_queue_depth=0,
        overloaded_inflight_threshold=10,
        overloaded_queue_depth_threshold=10,
        max_running_requests_per_worker=2,
        hint_worker_load=2,
    )
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": 1},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=overloaded_ctx,
            force_migrate_age_s=30.0,
        )
        == "overloaded"
    )

    # Selected other (hint OK, but picked another).
    assert (
        resume_fallback_reason(
            route_meta={"pause_age_s": 0.0, "last_backend_id": 1},
            selected_dp_rank=0,
            previous_rank=1,
            ctx=ctx,
            force_migrate_age_s=30.0,
        )
        == "selected_other"
    )

