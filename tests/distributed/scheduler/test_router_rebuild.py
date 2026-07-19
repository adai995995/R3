from roll.distributed.scheduler.router import (
    TrajectoryRuntimeState,
    build_runtime_priority_key,
    build_router_progress_snapshot,
    build_boundary_recovery_record,
    common_prefix_tokens,
    is_post_boundary_request,
    select_rebuild_worker,
    select_prefix_locality_worker,
    select_soft_locality_worker,
)


def test_router_progress_snapshot_preserves_boundary_scheduling_state():
    state = TrajectoryRuntimeState.from_priority(
        {
            "trajectory_id": "traj-7",
            "group_id": 2,
            "episode_id": 5,
            "env_id": 7,
            "policy_version": 3,
            "current_version": 4,
            "actions_completed": 6,
            "max_actions": 10,
        },
        "fallback",
    )

    snapshot = build_router_progress_snapshot(state, 4096)

    assert snapshot["env_id"] == 7
    assert snapshot["actions_completed"] == 6
    assert snapshot["remaining_actions"] == 4
    assert snapshot["current_context_tokens"] == 4096
    assert snapshot["completed"] is False


def test_boundary_recovery_record_distinguishes_reported_and_logical_prefill():
    state = TrajectoryRuntimeState(
        "trajectory-3",
        policy_version=2,
        current_version=4,
        version_age=2,
        actions_completed=6,
        max_actions=10,
        group_id=1,
        episode_id=7,
    )

    reported = build_boundary_recovery_record(
        state,
        cache_epoch=4,
        boundary_version=4,
        worker_rank=2,
        route_reason="rebuild",
        prompt_tokens=4096,
        response_metrics={
            "vllm/request_prefill_tokens": 3072,
            "vllm/request_scheduler_batch_id": 11,
            "vllm/request_scheduler_batch_size": 4,
        },
    )
    upper_bound = build_boundary_recovery_record(
        state,
        cache_epoch=4,
        boundary_version=4,
        worker_rank=2,
        route_reason="rebuild",
        prompt_tokens=4096,
        response_metrics={},
    )

    assert reported["logical_reprefill_exposure_tokens"] == 3072
    assert reported["reprefill_measurement"] == "engine_reported_prefill"
    assert reported["engine_scheduler_batch_id"] == 11
    assert reported["engine_scheduler_batch_size"] == 4
    assert upper_bound["logical_reprefill_exposure_tokens"] == 4096
    assert upper_bound["reprefill_measurement"] == "logical_prompt_upper_bound"


def test_boundary_recovery_uses_router_epoch_when_request_version_lags():
    state = TrajectoryRuntimeState(
        "trajectory-4",
        policy_version=2,
        current_version=2,
        version_age=0,
        actions_completed=5,
        max_actions=10,
    )

    assert is_post_boundary_request(state, boundary_version=3)
    record = build_boundary_recovery_record(
        state,
        cache_epoch=3,
        boundary_version=3,
        worker_rank=0,
        route_reason="rebuild",
        prompt_tokens=2048,
        response_metrics={"vllm/request_prefill_tokens": 2048},
    )

    assert record["version_age"] == 1


def test_common_prefix_tokens_is_bounded():
    assert common_prefix_tokens([1, 2, 3], [1, 2, 4], 8) == 2
    assert common_prefix_tokens([1, 2, 3], [1, 2, 3], 2) == 2


def test_rebuild_placement_prefers_prefix_diversity_within_worker():
    worker_prompts = {
        0: [(1, 2, 3, 4)],
        1: [(9, 8, 7, 6)],
    }
    selected, lcp = select_rebuild_worker(
        [1, 2, 3, 5],
        worker_prompts,
        {0, 1},
        {0: 1, 1: 1},
        prefix_limit=16,
    )

    assert selected == 1
    assert lcp == 0


def test_rebuild_placement_balances_empty_workers():
    selected, lcp = select_rebuild_worker(
        [1, 2, 3],
        {0: [(7, 8)], 1: []},
        {0, 1},
        {0: 1, 1: 0},
        prefix_limit=16,
    )

    assert selected == 1
    assert lcp == 0


def test_trajectory_priority_is_version_then_progress():
    older_unstarted = TrajectoryRuntimeState(
        "old", policy_version=2, current_version=3, version_age=1,
        actions_completed=0, max_actions=10,
    )
    newer_deep = TrajectoryRuntimeState(
        "new", policy_version=3, current_version=3, version_age=0,
        actions_completed=9, max_actions=10,
    )
    older_deep = TrajectoryRuntimeState(
        "old-deep", policy_version=2, current_version=3, version_age=1,
        actions_completed=8, max_actions=10,
    )

    assert older_unstarted.priority_key < newer_deep.priority_key
    assert older_deep.priority_key < older_unstarted.priority_key


def test_boundary_plan_candidates_precede_unplanned_requests():
    planned_second = TrajectoryRuntimeState(
        "planned-second",
        policy_version=3,
        current_version=4,
        version_age=1,
        actions_completed=2,
        max_actions=10,
        group_id=2,
        episode_id=4,
    )
    planned_first = TrajectoryRuntimeState(
        "planned-first",
        policy_version=3,
        current_version=4,
        version_age=1,
        actions_completed=2,
        max_actions=10,
        group_id=1,
        episode_id=5,
    )
    unplanned_older = TrajectoryRuntimeState(
        "unplanned",
        policy_version=2,
        current_version=4,
        version_age=2,
        actions_completed=9,
        max_actions=10,
        group_id=9,
        episode_id=9,
    )
    ranks = {"1:5": 0, "2:4": 1}

    assert build_runtime_priority_key(
        planned_first, ranks
    ) < build_runtime_priority_key(planned_second, ranks)
    assert build_runtime_priority_key(
        planned_second, ranks
    ) < build_runtime_priority_key(unplanned_older, ranks)


def test_soft_locality_keeps_affinity_within_load_slack():
    selected, reason = select_soft_locality_worker(
        affinity_rank=1,
        active_dp_ranks={0, 1},
        worker_pressure={0: 2, 1: 3},
        cache_valid=True,
        load_slack=1,
    )

    assert selected == 1
    assert reason == "affinity"


def test_soft_locality_allows_load_override():
    selected, reason = select_soft_locality_worker(
        affinity_rank=1,
        active_dp_ranks={0, 1},
        worker_pressure={0: 1, 1: 4},
        cache_valid=True,
        load_slack=1,
    )

    assert selected == 0
    assert reason == "load_override"


def test_runtime_state_carries_boundary_group_identity():
    state = TrajectoryRuntimeState.from_priority(
        {
            "trajectory_id": "trajectory-7",
            "group_id": 3,
            "episode_id": 11,
            "policy_version": 4,
            "current_version": 5,
            "actions_completed": 2,
            "max_actions": 10,
        },
        "fallback",
    )

    assert state.group_key == "3:11"
    assert state.version_age == 1


def test_working_set_routing_uses_prefix_when_load_is_safe():
    selected, reason, cached = select_prefix_locality_worker(
        [1, 2, 3, 9],
        {0: [(1, 2, 8)], 1: [(1, 2, 3, 4)]},
        {0, 1},
        {0: 1, 1: 2},
        load_slack=1,
        prefix_limit=16,
    )

    assert selected == 1
    assert reason == "prefix_locality"
    assert cached == 3


def test_working_set_routing_obeys_load_override():
    selected, reason, cached = select_prefix_locality_worker(
        [1, 2, 3, 9],
        {0: [(9, 9)], 1: [(1, 2, 3, 4)]},
        {0, 1},
        {0: 0, 1: 4},
        load_slack=1,
        prefix_limit=16,
    )

    assert selected == 0
    assert reason == "prefix_load_override"
    assert cached == 0
