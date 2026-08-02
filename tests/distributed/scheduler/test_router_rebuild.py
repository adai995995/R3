import pytest

from roll.distributed.scheduler.router import (
    TrajectoryRuntimeState,
    build_runtime_priority_key,
    build_router_progress_snapshot,
    build_boundary_recovery_record,
    build_prefix_directory_keys,
    build_prefix_fingerprints,
    build_refresh_request_record,
    common_prefix_tokens,
    is_post_boundary_request,
    select_rebuild_worker,
    select_prefix_locality_worker,
    select_prefix_directory_worker,
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
            "vllm/request_ttft_seconds": 0.4,
            "vllm/request_model_execute_seconds": 0.25,
            "vllm/request_engine_step_seconds_attributed": 0.2,
            "vllm/request_prefill_engine_step_seconds_attributed": 0.15,
            "vllm/request_decode_engine_step_seconds_attributed": 0.05,
        },
        boundary_resumed_at=10.0,
        request_dispatched_at=10.2,
        request_completed_at=11.0,
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
    assert reported["first_token_after_boundary_seconds"] == pytest.approx(0.6)
    assert reported["finish_after_boundary_seconds"] == pytest.approx(1.0)
    assert reported["request_model_execute_seconds"] == pytest.approx(0.25)
    assert reported[
        "request_engine_step_seconds_attributed"
    ] == pytest.approx(0.2)
    assert reported[
        "request_prefill_engine_step_seconds_attributed"
    ] == pytest.approx(0.15)
    assert reported[
        "request_decode_engine_step_seconds_attributed"
    ] == pytest.approx(0.05)
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


def test_prefix_fingerprints_are_block_aligned_and_deterministic():
    tokens = list(range(300))

    first = build_prefix_fingerprints(tokens, depths=(128, 256))
    second = build_prefix_fingerprints(tokens, depths=(128, 256))

    assert first == second
    assert [item["prefix_tokens"] for item in first] == [128, 256, 288]
    assert len({item["fingerprint"] for item in first}) == 3


def test_prefix_directory_prefers_deepest_ready_owner_without_prompt_scan():
    prompt = list(range(320))
    keys = build_prefix_directory_keys(prompt, prefix_limit=256)
    ready = {
        keys[0]: {0},
        keys[-1]: {1},
    }

    selected, reason, cached = select_prefix_directory_worker(
        keys,
        ready,
        {0, 1},
        {0: 0, 1: 1},
        load_slack=1,
    )

    assert selected == 1
    assert reason == "prefix_directory"
    assert cached == 256


def test_refresh_request_record_is_aligned_to_boundary():
    state = TrajectoryRuntimeState(
        "trajectory-5",
        policy_version=2,
        current_version=3,
        version_age=1,
        actions_completed=4,
        max_actions=10,
    )

    record = build_refresh_request_record(
        state,
        cache_epoch=4,
        boundary_version=3,
        worker_rank=1,
        route_reason="rebuild",
        prompt_tokens=list(range(256)),
        response_metrics={
            "vllm/request_cached_prompt_tokens": 0,
            "vllm/request_prefill_tokens": 256,
            "vllm/request_decode_tokens": 32,
            "vllm/request_ttft_seconds": 0.5,
            "vllm/request_scheduler_batch_id": 7,
        },
        request_dispatched_at=10.25,
        request_completed_at=11.0,
        boundary_resumed_at=10.0,
        first_epoch_request=True,
    )

    assert record["survivor_request"] is True
    assert record["dispatch_after_boundary_seconds"] == pytest.approx(0.25)
    assert record["first_token_after_boundary_seconds"] == pytest.approx(0.75)
    assert record["finish_after_boundary_seconds"] == pytest.approx(1.0)
    assert record["prefill_tokens"] == 256
    assert record["decode_tokens"] == 32
    assert record["prefix_fingerprints"]


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
