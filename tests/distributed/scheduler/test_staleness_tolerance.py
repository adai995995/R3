import asyncio
from types import SimpleNamespace

from roll.distributed.scheduler.rollout_scheduler import (
    GroupData,
    GroupQueue,
    GroupQueueManager,
    VersionAwareRuntimeController,
    VersionRuntimeOutcome,
    VersionRuntimeState,
    compute_admission_control_step,
    compute_admission_search_step,
    update_latency_hill_climb,
)
from roll.pipeline.agentic.agentic_config import EnvMonitorConfig


class NeverFilter:
    def __init__(self, *args, **kwargs):
        pass

    def filter(self, **kwargs):
        return False


def make_adaptive_manager(
    *,
    rollout_batch_size=4,
    max_outstanding_trajectories=8,
    finish_ratio=0.5,
    ewma_alpha=0.5,
    safe_forecast=False,
    bootstrap_reserve_groups=0,
):
    config = SimpleNamespace(
        async_generation_ratio=1,
        trajectory_staleness_tolerance=2,
        trajectory_scheduling_policy="version_priority",
        trajectory_admission_policy="version_adaptive",
        max_outstanding_trajectories=max_outstanding_trajectories,
        adaptive_admission_reserve_trajectories=0,
        adaptive_admission_initial_finish_ratio=finish_ratio,
        adaptive_admission_ewma_alpha=ewma_alpha,
        adaptive_admission_safe_forecast_enabled=safe_forecast,
        adaptive_admission_forecast_confidence_z=1.0,
        adaptive_admission_bootstrap_reserve_groups=(
            bootstrap_reserve_groups
        ),
        rollout_batch_size=rollout_batch_size,
        env_monitor=EnvMonitorConfig(enable=False),
    )
    env_manager_config = SimpleNamespace(
        group_size=1,
        group_size_redundancy=0,
        max_traj_per_env=1,
        group_filter_cls="tests.distributed.scheduler.test_staleness_tolerance.NeverFilter",
        env_configs={0: {0: {"group_id": 0}}},
    )
    manager_class = GroupQueueManager.__ray_metadata__.modified_class
    return manager_class(config, env_manager_config, "train")


def make_fixed_add_manager(*, trajectories_per_step=3, num_groups=4):
    config = SimpleNamespace(
        async_generation_ratio=1,
        trajectory_staleness_tolerance=2,
        trajectory_scheduling_policy="fifo",
        trajectory_admission_policy="step",
        fixed_step_admission_trajectories=trajectories_per_step,
        max_outstanding_trajectories=None,
        rollout_batch_size=4,
        env_monitor=EnvMonitorConfig(enable=False),
    )
    env_manager_config = SimpleNamespace(
        group_size=1,
        group_size_redundancy=0,
        max_traj_per_env=1,
        group_filter_cls="tests.distributed.scheduler.test_staleness_tolerance.NeverFilter",
        env_configs={
            0: {env_id: {"group_id": env_id} for env_id in range(num_groups)}
        },
    )
    manager_class = GroupQueueManager.__ray_metadata__.modified_class
    return manager_class(config, env_manager_config, "train")


def test_staleness_tolerance_is_independent_from_rollout_ahead():
    queue = GroupQueue(
        group_id=0,
        progress_bar=None,
        group_size=1,
        group_size_redundancy=0,
        max_traj_per_env=1,
        async_generation_ratio=1,
        staleness_tolerance=2,
        group_filter=NeverFilter(),
    )

    queue.advance_step(0)
    assert len(queue.groups) == 2

    queue.advance_step(1)
    queue.advance_step(2)
    assert 0 in queue.groups
    assert 1 in queue.groups

    queue.advance_step(3)
    assert 0 not in queue.groups
    assert 1 not in queue.groups


def test_fixed_step_admission_uses_one_global_budget_without_changing_env_count():
    manager = make_fixed_add_manager(trajectories_per_step=3, num_groups=4)

    manager.advance_step(0)
    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 6
    assert manager.admitted_trajectories_total == 6

    manager.advance_step(1)
    assert manager._outstanding_snapshot(1)["outstanding_trajectories"] == 9
    assert manager.admitted_trajectories_total == 9
    group_counts = [len(queue.groups) for queue in manager.group_queue.values()]
    assert max(group_counts) - min(group_counts) <= 1


def test_version_priority_assigns_oldest_policy_version_first():
    queue = GroupQueue(
        group_id=0,
        progress_bar=None,
        group_size=1,
        group_size_redundancy=0,
        max_traj_per_env=1,
        async_generation_ratio=0,
        staleness_tolerance=4,
        group_filter=NeverFilter(),
        scheduling_policy="version_priority",
    )
    queue.groups[0] = GroupData(group_id=0, episode_id=0, create_step=3)
    queue.groups[1] = GroupData(group_id=0, episode_id=1, create_step=1)

    episode_id = asyncio.run(queue.get_episode_id())

    assert episode_id == 1


def test_version_priority_consumes_oldest_ready_group_without_head_blocking():
    async def run_test():
        queue = GroupQueue(
            group_id=0,
            progress_bar=None,
            group_size=1,
            group_size_redundancy=0,
            max_traj_per_env=1,
            async_generation_ratio=0,
            staleness_tolerance=4,
            group_filter=NeverFilter(),
            scheduling_policy="version_priority",
        )
        queue.groups[0] = GroupData(
            group_id=0,
            episode_id=0,
            create_step=0,
            rollouts=[],
            running_rollouts=1,
        )
        ready_rollout = object()
        queue.groups[1] = GroupData(
            group_id=0,
            episode_id=1,
            create_step=1,
            rollouts=[ready_rollout],
            running_rollouts=1,
        )

        group = await asyncio.wait_for(queue.get(), timeout=0.1)

        assert group.episode_id == 1
        assert group.rollouts == [ready_rollout]
        assert 0 in queue.groups
        assert queue.version_priority_ready_bypass_total == 1

    asyncio.run(run_test())


def test_gpu_rebuild_candidates_are_exact_unfinished_trajectories():
    queue = GroupQueue(
        group_id=2,
        progress_bar=None,
        group_size=2,
        group_size_redundancy=0,
        max_traj_per_env=1,
        async_generation_ratio=0,
        staleness_tolerance=4,
        group_filter=NeverFilter(),
        scheduling_policy="version_priority",
    )
    group = GroupData(group_id=2, episode_id=7, create_step=1)
    queue.groups[7] = group
    queue.update_progress_snapshots(
        [
            {
                "trajectory_id": "invested",
                "group_id": 2,
                "episode_id": 7,
                "env_id": 0,
                "actions_completed": 4,
                "inference_calls": 4,
                "completed": False,
            },
            {
                "trajectory_id": "reset-only",
                "group_id": 2,
                "episode_id": 7,
                "env_id": 1,
                "actions_completed": 0,
                "inference_calls": 0,
                "completed": False,
            },
            {
                "trajectory_id": "already-complete",
                "group_id": 2,
                "episode_id": 7,
                "env_id": 2,
                "actions_completed": 5,
                "inference_calls": 5,
                "completed": True,
            },
        ]
    )

    assert queue.gpu_invested_trajectory_candidates(group) == [
        ("invested", 4, 0)
    ]


def test_router_progress_does_not_overwrite_environment_runtime_phase():
    queue = GroupQueue(
        group_id=2,
        progress_bar=None,
        group_size=1,
        group_size_redundancy=0,
        max_traj_per_env=1,
        async_generation_ratio=0,
        staleness_tolerance=4,
        group_filter=NeverFilter(),
        scheduling_policy="version_priority",
    )
    queue.update_progress_snapshots(
        [
            {
                "trajectory_id": "trajectory-0",
                "group_id": 2,
                "episode_id": 7,
                "env_id": 0,
                "runtime_phase": "tool_or_environment",
                "actions_completed": 2,
            },
            {
                "trajectory_id": "trajectory-0",
                "group_id": 2,
                "episode_id": 7,
                "env_id": 0,
                "runtime_phase": "router_last_request",
                "progress_source": "router",
                "actions_completed": 2,
            },
        ]
    )

    snapshot = queue.progress_snapshots[(7, 0)]
    assert snapshot["runtime_phase"] == "tool_or_environment"
    assert snapshot["router_runtime_phase"] == "router_last_request"


def test_outstanding_snapshot_counts_ready_running_reserved_and_retired():
    queue = GroupQueue(
        group_id=0,
        progress_bar=None,
        group_size=2,
        group_size_redundancy=1,
        max_traj_per_env=1,
        async_generation_ratio=0,
        staleness_tolerance=2,
        group_filter=NeverFilter(),
    )
    queue.current_step = 3
    queue.groups[0] = GroupData(
        group_id=0,
        episode_id=0,
        create_step=1,
        rollouts=[object()],
        running_rollouts=2,
    )
    queue.retired_groups[1] = GroupData(
        group_id=0,
        episode_id=1,
        create_step=0,
        rollouts=[object()],
        running_rollouts=2,
    )

    snapshot = queue.outstanding_snapshot()

    assert snapshot["ready_trajectories"] == 1
    assert snapshot["running_trajectories"] == 1
    assert snapshot["reserved_trajectories"] == 1
    assert snapshot["retired_running_trajectories"] == 1
    assert snapshot["outstanding_trajectories"] == 4
    assert snapshot["oldest_version_age"] == 3


def test_watermark_admission_caps_global_producer_debt():
    config = SimpleNamespace(
        async_generation_ratio=2,
        trajectory_staleness_tolerance=2,
        trajectory_scheduling_policy="version_priority",
        trajectory_admission_policy="outstanding_watermark",
        max_outstanding_trajectories=5,
        rollout_batch_size=2,
        env_monitor=EnvMonitorConfig(enable=False),
    )
    env_manager_config = SimpleNamespace(
        group_size=1,
        group_size_redundancy=0,
        max_traj_per_env=1,
        group_filter_cls="tests.distributed.scheduler.test_staleness_tolerance.NeverFilter",
        env_configs={
            0: {
                0: {"group_id": 0},
                1: {"group_id": 1},
            }
        },
    )
    manager_class = GroupQueueManager.__ray_metadata__.modified_class
    manager = manager_class(config, env_manager_config, "train")

    manager.advance_step(0)
    initial = manager._outstanding_snapshot(0)

    assert initial["outstanding_trajectories"] == 5
    assert sum(len(queue.groups) for queue in manager.group_queue.values()) == 5

    first_queue = manager.group_queue[0]
    first_episode = min(first_queue.groups)
    first_queue.groups.pop(first_episode)
    admitted = manager._refill_to_watermark(0)
    refilled = manager._outstanding_snapshot(0)

    assert admitted == 1
    assert refilled["outstanding_trajectories"] == 5
    assert manager.admitted_trajectories_total == 6


def test_version_adaptive_computes_one_budget_per_version_without_refill():
    manager = make_adaptive_manager()

    manager.advance_step(0)

    assert manager.version_admission_budget == 4
    assert manager.version_admission_used == 4
    assert manager.version_admission_trainable_used == 4
    assert manager.version_admission_remaining == 0
    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 4

    queue = manager.group_queue[0]
    queue.groups.pop(min(queue.groups))
    manager._refill_to_watermark(0)

    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 3
    assert manager.admitted_trajectories_total == 4


def test_version_adaptive_bootstraps_one_group_without_supply_history():
    manager = make_adaptive_manager(
        safe_forecast=True,
        bootstrap_reserve_groups=1,
    )

    manager.advance_step(0)

    assert manager.version_bootstrap_reserve == 1
    assert manager.version_admission_budget == 5
    assert manager.version_runtime_plan.safety_reserve == 1


def test_progress_reconcile_publishes_an_online_plan_revision():
    manager = make_adaptive_manager(finish_ratio=1.0)
    manager.version_adaptive_progress_floor_enabled = True
    manager.version_runtime_reconcile_wait_seconds = 0.0
    manager.version_runtime_max_revisions_per_version = 4
    manager.advance_step(0)
    manager.current_batch_missing = 4

    revised = manager.reconcile_version_progress(0, 4, 1.0)

    assert revised["version"] == 0
    assert revised["revision"] == 1
    assert revised["admission_reason"] == "progress_reconcile"
    assert revised["admission_delta_trajectories"] == 1
    assert revised["admission_budget"] == 5
    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 5
    report = manager.collect_shutdown_waste([])
    assert report["version_runtime"]["plan"]["revision"] == 1
    assert report["metrics"]["version_runtime/final_revision"] == 1


def test_progress_reconcile_waits_for_observed_progress_before_repeating():
    manager = make_adaptive_manager(finish_ratio=1.0)
    manager.version_adaptive_progress_floor_enabled = True
    manager.version_runtime_reconcile_wait_seconds = 0.0
    manager.version_runtime_max_revisions_per_version = 4
    manager.advance_step(0)
    manager.current_batch_missing = 4

    first = manager.reconcile_version_progress(0, 4, 1.0)
    repeated = manager.reconcile_version_progress(0, 4, 2.0)

    assert first is not None
    assert repeated is None
    assert manager.version_progress_topup_pending_hold_total == 1

    manager.current_batch_missing = 3
    manager.version_runtime_last_topup_at = 0.0
    after_progress = manager.reconcile_version_progress(0, 4, 3.0)

    assert after_progress is not None
    assert after_progress["revision"] == 2


def test_state_feedback_acts_on_latest_update_not_lagging_starvation_ewma():
    manager = make_adaptive_manager()
    manager.dynamic_reserve_enabled = True
    manager.dynamic_reserve_controller = "state_feedback"
    manager.dynamic_reserve_shadow_mode = False
    manager.dynamic_reserve_signal_patience = 1
    manager.dynamic_reserve_cooldown_versions = 0
    manager.dynamic_reserve_warmup_versions = 0
    manager.dynamic_reserve_min = 0
    manager.dynamic_reserve_max = 8
    manager.dynamic_reserve_additive_step = 1
    manager.dynamic_starvation_high = 0.1
    manager.dynamic_reserve_stale_high = 0.25
    manager.dynamic_queue_high = 0.2
    manager.dynamic_tool_wait_high = 0.5
    manager.version_runtime_starvation_ewma = 0.8
    manager.version_runtime_last_outcome = VersionRuntimeOutcome(
        plan_id="version-0-revision-0",
        forecast_id="version-0-estimator-0",
        version=0,
        final_revision=0,
        predicted_existing_supply=4,
        actual_existing_valid_slots=4,
        admitted_trajectories=4,
        completed_valid_slots=4,
        consumed_valid_slots=4,
        learner_wait_seconds=0,
        next_batch_latency_seconds=0,
        starvation_fraction=0,
        waste_fraction=0,
        queue_pressure=0,
        inference_ready_trajectories=4,
    )

    manager._update_dynamic_reserve(1)

    assert manager.adaptive_reserve == 0
    assert manager.dynamic_reserve_update_reason == 0


def test_state_feedback_requires_sustained_evidence_before_downscaling():
    manager = make_adaptive_manager()
    manager.dynamic_reserve_enabled = True
    manager.dynamic_reserve_controller = "state_feedback"
    manager.dynamic_reserve_shadow_mode = False
    manager.adaptive_reserve = 4
    manager.dynamic_reserve_signal_patience = 1
    manager.dynamic_reserve_downscale_patience = 3
    manager.dynamic_reserve_cooldown_versions = 0
    manager.dynamic_reserve_warmup_versions = 0
    manager.dynamic_reserve_min = 0
    manager.dynamic_reserve_max = 8
    manager.dynamic_reserve_additive_step = 1
    manager.dynamic_starvation_high = 0.02
    manager.dynamic_reserve_stale_high = 0.25
    manager.dynamic_queue_high = 0.2
    manager.dynamic_tool_wait_high = 0.5
    manager.version_runtime_last_outcome = VersionRuntimeOutcome(
        plan_id="version-0-revision-0",
        forecast_id="version-0-estimator-0",
        version=0,
        final_revision=0,
        predicted_existing_supply=4,
        actual_existing_valid_slots=4,
        admitted_trajectories=4,
        completed_valid_slots=4,
        consumed_valid_slots=4,
        learner_wait_seconds=0,
        next_batch_latency_seconds=0,
        starvation_fraction=0,
        waste_fraction=0.5,
        queue_pressure=0,
        inference_ready_trajectories=4,
        projected_next_batch_supply=4,
    )

    manager._update_dynamic_reserve(1)
    manager._update_dynamic_reserve(2)
    assert manager.adaptive_reserve == 4

    manager._update_dynamic_reserve(3)
    assert manager.adaptive_reserve == 3


def test_state_feedback_does_not_downscale_when_next_batch_is_undersupplied():
    manager = make_adaptive_manager()
    manager.dynamic_reserve_enabled = True
    manager.dynamic_reserve_controller = "state_feedback"
    manager.dynamic_reserve_shadow_mode = False
    manager.adaptive_reserve = 4
    manager.dynamic_reserve_signal_patience = 1
    manager.dynamic_reserve_downscale_patience = 1
    manager.dynamic_reserve_cooldown_versions = 0
    manager.dynamic_reserve_warmup_versions = 0
    manager.dynamic_reserve_min = 0
    manager.dynamic_reserve_max = 8
    manager.dynamic_reserve_additive_step = 1
    manager.dynamic_starvation_high = 0.02
    manager.dynamic_reserve_stale_high = 0.25
    manager.dynamic_queue_high = 0.2
    manager.dynamic_tool_wait_high = 0.5
    manager.version_runtime_last_outcome = VersionRuntimeOutcome(
        plan_id="version-0-revision-0",
        forecast_id="version-0-estimator-0",
        version=0,
        final_revision=0,
        predicted_existing_supply=2,
        actual_existing_valid_slots=2,
        admitted_trajectories=4,
        completed_valid_slots=2,
        consumed_valid_slots=2,
        learner_wait_seconds=0,
        next_batch_latency_seconds=0,
        starvation_fraction=0,
        waste_fraction=0.5,
        queue_pressure=0,
        inference_ready_trajectories=2,
        projected_next_batch_supply=2,
    )

    manager._update_dynamic_reserve(1)

    assert manager.adaptive_reserve == 4
    assert manager.dynamic_reserve_update_reason == 12


def test_shutdown_records_merge_snapshot_submitted_during_collection():
    manager_class = GroupQueueManager.__ray_metadata__.modified_class
    records, duplicates_removed = manager_class._deduplicate_shutdown_records(
        [
            {
                "trajectory_id": "traj-1",
                "category": "completed_unconsumed",
                "completed": True,
                "actions_completed": 7,
                "inference_tokens": 700,
            },
            {
                "trajectory_id": "traj-1",
                "category": "inflight_at_shutdown",
                "runtime_phase": "policy_inference",
                "completed": False,
                "actions_completed": 6,
                "inference_tokens": 650,
                "trajectory_wall_seconds": 12.0,
            },
        ]
    )

    assert duplicates_removed == 1
    assert len(records) == 1
    assert records[0]["category"] == "completed_unconsumed"
    assert records[0]["actions_completed"] == 7
    assert records[0]["inference_tokens"] == 700
    assert records[0]["trajectory_wall_seconds"] == 12.0


def test_runtime_controller_owns_reconcile_delta_without_mutating_state():
    controller = VersionAwareRuntimeController()
    state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=0,
        outstanding_trajectories=4,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=(),
    )
    active_plan = controller.decide(state)
    reconcile_state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=1,
        outstanding_trajectories=5,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=(),
    )

    revised = controller.decide(
        reconcile_state,
        active_plan=active_plan,
        missing_trajectories=4,
        current_batch_missing=4,
        learner_wait_seconds=2,
        reconcile_wait_seconds=1,
        max_revisions_per_version=4,
        max_admission_groups=1,
    )

    assert active_plan is not None
    assert revised is not None
    assert active_plan.revision == 0
    assert active_plan.admission_budget == 0
    assert revised.revision == 1
    assert revised.admission_delta_trajectories == 1
    assert revised.admission_budget == 1
    assert revised.expected_existing_supply == 1
    assert revised.outstanding_trajectories == 5
    assert revised.admission_deficit == 3
    assert revised.admission_capacity == 3


def test_admission_reconcile_preserves_scheduling_and_kv_decisions():
    controller = VersionAwareRuntimeController()
    state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=0,
        outstanding_trajectories=4,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=((1, 1, 1, 2, 1),),
        gpu_invested_candidate_groups=((1, 1, 1, 2, 1),),
        gpu_invested_candidate_trajectories=(
            ("trajectory-1", 1, 1, 0, 1, 2),
        ),
        exact_rebuild_cohort=True,
    )
    active_plan = controller.decide(state)
    reconcile_state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=1,
        outstanding_trajectories=5,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=((2, 2, 2, 7, 1),),
        gpu_invested_candidate_groups=((2, 2, 2, 7, 1),),
        gpu_invested_candidate_trajectories=(
            ("trajectory-2", 2, 2, 0, 1, 7),
        ),
        exact_rebuild_cohort=True,
    )

    revised = controller.decide(
        reconcile_state,
        active_plan=active_plan,
        missing_trajectories=4,
        current_batch_missing=4,
        learner_wait_seconds=2,
        reconcile_wait_seconds=1,
        max_revisions_per_version=4,
        max_admission_groups=1,
    )

    assert active_plan is not None
    assert revised is not None
    assert revised.admission_budget == active_plan.admission_budget + 1
    assert revised.priority_candidate_groups == (
        active_plan.priority_candidate_groups
    )
    assert revised.rebuild_candidate_groups == (
        active_plan.rebuild_candidate_groups
    )
    assert revised.rebuild_candidate_trajectories == (
        active_plan.rebuild_candidate_trajectories
    )


def test_runtime_controller_reconciles_measured_deficit_after_wait_deadline():
    controller = VersionAwareRuntimeController()
    state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=0,
        outstanding_trajectories=4,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=(),
    )
    active_plan = controller.decide(state)
    reconcile_state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        # This is an eventual-completion forecast, not proof that the current
        # batch deficit will resolve within its wait budget.
        expected_existing_supply=8,
        outstanding_trajectories=4,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=(),
    )

    before_deadline = controller.decide(
        reconcile_state,
        active_plan=active_plan,
        missing_trajectories=4,
        current_batch_missing=2,
        learner_wait_seconds=0.5,
        reconcile_wait_seconds=1,
        max_revisions_per_version=4,
        max_admission_groups=1,
    )
    revised = controller.decide(
        reconcile_state,
        active_plan=active_plan,
        missing_trajectories=4,
        current_batch_missing=2,
        learner_wait_seconds=2,
        reconcile_wait_seconds=1,
        max_revisions_per_version=4,
        max_admission_groups=1,
    )

    assert before_deadline is None
    assert revised is not None
    assert revised.admission_reason == "progress_reconcile"
    assert revised.admission_delta_trajectories == 1
    assert revised.revision == 1


def test_runtime_controller_holds_topup_for_imminent_supply_or_gpu_queue():
    controller = VersionAwareRuntimeController()
    state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=0,
        outstanding_trajectories=4,
        max_outstanding_trajectories=8,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=(),
    )
    active_plan = controller.decide(state)

    supply_hold = controller.decide(
        state,
        active_plan=active_plan,
        missing_trajectories=4,
        current_batch_missing=2,
        learner_wait_seconds=2,
        reconcile_wait_seconds=1,
        timely_existing_supply=2,
        queue_pressure=0.0,
        queue_high=0.2,
        max_revisions_per_version=4,
    )
    queue_hold = controller.decide(
        state,
        active_plan=active_plan,
        missing_trajectories=4,
        current_batch_missing=2,
        learner_wait_seconds=2,
        reconcile_wait_seconds=1,
        timely_existing_supply=0,
        queue_pressure=0.5,
        queue_high=0.2,
        max_revisions_per_version=4,
    )

    assert supply_hold is None
    assert queue_hold is None


def test_runtime_admission_keeps_workers_supplied_without_manual_reserve():
    controller = VersionAwareRuntimeController()
    state = VersionRuntimeState(
        version=3,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=4,
        outstanding_trajectories=1,
        max_outstanding_trajectories=12,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=(),
        worker_count=4,
        running_requests=1,
        queued_requests=0,
        inference_ready_trajectories=1,
    )

    plan = controller.decide(state)

    assert plan is not None
    assert plan.safety_reserve == 0
    assert plan.admission_reason == "worker_saturation_floor"
    assert plan.admission_budget == 3
    assert plan.worker_floor_trajectories == 3


def test_runtime_admission_low_yield_is_bounded_by_operating_target():
    controller = VersionAwareRuntimeController()

    def decide(capacity):
        return controller.decide(
            VersionRuntimeState(
                version=3,
                learner_demand=4,
                safety_reserve=4,
                expected_existing_supply=0,
                outstanding_trajectories=0,
                max_outstanding_trajectories=capacity,
                admission_width=1,
                group_size=1,
                staleness_tolerance=2,
                invested_candidate_groups=(),
                admission_yield_probability=0.001,
            )
        )

    plan_32 = decide(32)
    plan_128 = decide(128)

    assert plan_32 is not None
    assert plan_128 is not None
    assert plan_32.admission_budget == 8
    assert plan_128.admission_budget == 8
    assert plan_32.admission_operating_target == 8
    assert plan_128.admission_operating_target == 8


def test_runtime_admission_holds_at_operating_target():
    plan = VersionAwareRuntimeController().decide(
        VersionRuntimeState(
            version=3,
            learner_demand=4,
            safety_reserve=4,
            expected_existing_supply=0,
            outstanding_trajectories=8,
            max_outstanding_trajectories=128,
            admission_width=1,
            group_size=1,
            staleness_tolerance=2,
            invested_candidate_groups=(),
            admission_yield_probability=0.0,
        )
    )

    assert plan is not None
    assert plan.admission_budget == 0
    assert plan.admission_reason == "operating_target"


def test_latency_hill_climb_uses_step_time_as_primary_objective():
    first, direction, reason = update_latency_hill_climb(
        4,
        1,
        10.0,
        None,
        0.0,
        0.0,
        0.0,
        reserve_min=0,
        reserve_max=16,
        additive_step=1,
        improvement_margin=0.03,
        starvation_high=0.02,
        waste_high=0.25,
        queue_idle=0.001,
        queue_high=0.20,
    )
    improved, direction, improved_reason = update_latency_hill_climb(
        first,
        direction,
        9.0,
        10.0,
        0.0,
        0.10,
        0.0,
        reserve_min=0,
        reserve_max=16,
        additive_step=1,
        improvement_margin=0.03,
        starvation_high=0.02,
        waste_high=0.25,
        queue_idle=0.001,
        queue_high=0.20,
    )
    reversed_reserve, reversed_direction, reversed_reason = (
        update_latency_hill_climb(
            improved,
            direction,
            11.0,
            9.0,
            0.0,
            0.10,
            0.0,
            reserve_min=0,
            reserve_max=16,
            additive_step=1,
            improvement_margin=0.03,
            starvation_high=0.02,
            waste_high=0.25,
            queue_idle=0.001,
            queue_high=0.20,
        )
    )

    assert (first, reason) == (5, 13)
    assert (improved, improved_reason) == (6, 14)
    assert reversed_reserve == 5
    assert reversed_direction == -1
    assert reversed_reason == 15


def test_latency_hill_climb_does_not_trade_starvation_for_token_efficiency():
    reserve, direction, reason = update_latency_hill_climb(
        12,
        1,
        14.0,
        13.8,
        0.20,
        0.50,
        0.05,
        reserve_min=0,
        reserve_max=32,
        additive_step=1,
        improvement_margin=0.03,
        starvation_high=0.02,
        waste_high=0.25,
        queue_idle=0.001,
        queue_high=0.20,
    )

    assert reserve == 13
    assert direction == 1
    assert reason == 16


def test_latency_hill_climb_expands_when_starved_and_queue_is_idle():
    reserve, direction, reason = update_latency_hill_climb(
        12,
        1,
        15.0,
        13.0,
        0.20,
        0.50,
        0.0001,
        reserve_min=0,
        reserve_max=32,
        additive_step=2,
        improvement_margin=0.03,
        starvation_high=0.02,
        waste_high=0.25,
        queue_idle=0.001,
        queue_high=0.20,
    )

    assert reserve == 14
    assert direction == 1
    assert reason == 16


def test_latency_hill_climb_reverses_regression_after_queue_forms():
    reserve, direction, reason = update_latency_hill_climb(
        12,
        1,
        15.0,
        13.0,
        0.20,
        0.50,
        0.05,
        reserve_min=0,
        reserve_max=32,
        additive_step=2,
        improvement_margin=0.03,
        starvation_high=0.02,
        waste_high=0.25,
        queue_idle=0.001,
        queue_high=0.20,
    )

    assert reserve == 10
    assert direction == -1
    assert reason == 15


def test_admission_control_step_scales_with_batch_and_preserves_group_units():
    assert compute_admission_control_step(16, 1, 0, 0.25) == 4
    assert compute_admission_control_step(64, 1, 0, 0.25) == 16
    assert compute_admission_control_step(32, 4, 0, 0.25) == 8
    assert compute_admission_control_step(32, 4, 5, 0.25) == 8


def test_admission_search_step_uses_one_group_in_automatic_mode():
    assert compute_admission_search_step(1, 0) == 1
    assert compute_admission_search_step(4, 0) == 4
    assert compute_admission_search_step(4, 5) == 8


def test_admission_yield_uses_the_target_batch_deadline():
    manager = make_adaptive_manager()
    manager.advance_step(0)
    current_key, next_key = list(manager._tracked_admitted_groups)[:2]
    manager._tracked_admitted_targets[current_key] = "current_batch"
    manager._tracked_admitted_targets[next_key] = "next_batch"
    manager._tracked_admitted_completed_unix[current_key] = 10.0
    manager._tracked_admitted_completed_unix[next_key] = 10.0
    manager.version_current_batch_closed_unix = 5.0

    assert manager._timely_admitted_valid_slots(20.0) == 1


def test_next_batch_supply_excludes_invested_work_beyond_horizon():
    manager = make_adaptive_manager()
    manager.advance_step(0)
    group_keys = list(manager._tracked_admitted_groups)
    near_key, far_key = group_keys[:2]
    supply = {
        "valid_ready": 1,
        "unfinished_groups": {near_key, far_key},
        "unfinished_group_buckets": {
            near_key: "age_0__actions_0",
            far_key: "age_0__actions_0",
        },
        "candidate_estimates": (
            SimpleNamespace(
                group_key=f"{near_key[0]}:{near_key[1]}",
                feasible=True,
                eta_seconds=1.0,
                completion_probability=0.5,
            ),
            SimpleNamespace(
                group_key=f"{far_key[0]}:{far_key[1]}",
                feasible=True,
                eta_seconds=10.0,
                completion_probability=0.5,
            ),
        ),
    }

    predicted = manager._predict_next_batch_supply(
        supply, horizon_seconds=2.0
    )

    assert predicted == 1.5


def test_version_adaptive_discounts_carry_over_after_a_missed_batch_deadline():
    manager = make_adaptive_manager()
    manager.advance_step(0)
    queue = manager.group_queue[0]
    first_group = queue.groups[min(queue.groups)]
    first_group.rollouts.append(object())

    manager.advance_step(1)

    # The previous admission cohort missed its target batch, so its timely
    # yield is discounted before the three unfinished trajectories are used
    # to predict near-term supply.
    assert manager.version_valid_ready_at_boundary == 1
    assert manager.version_salvageable_inflight_at_boundary == 3
    assert manager.version_invested_inflight_at_boundary == 0
    assert manager.version_reserved_unstarted_at_boundary == 3
    assert manager.version_expected_existing_supply == 1.75
    assert manager.version_admission_budget_trainable == 0
    assert manager._outstanding_snapshot(1)["outstanding_trajectories"] == 4


def test_version_adaptive_attributes_new_admission_yield_to_its_plan():
    manager = make_adaptive_manager(finish_ratio=0.5, ewma_alpha=0.5)
    manager.advance_step(0)
    admitted_groups = list(manager._tracked_admitted_groups)
    for group_id, episode_id in admitted_groups[:2]:
        manager._record_version_adaptive_completion(group_id, episode_id)

    manager.advance_step(1)

    outcome = manager.version_runtime_last_outcome
    assert outcome.version == 0
    assert outcome.admitted_trainable_slots == 4
    assert outcome.timely_admitted_valid_slots == 2
    assert outcome.observed_admission_yield == 0.5
    assert manager.runtime_estimator.admission_yield_sample_count == 4


def test_version_adaptive_prepares_supply_during_learner_compute():
    manager = make_adaptive_manager(
        finish_ratio=0.5,
        max_outstanding_trajectories=16,
    )
    manager.advance_step(0)
    queue = manager.group_queue[0]
    for episode_id in sorted(queue.groups)[:4]:
        queue.groups.pop(episode_id)

    revised = manager.prepare_next_batch_supply(0, 4)

    assert revised is not None
    assert revised["admission_reason"] == "next_batch_supply"
    assert revised["admission_delta_trajectories"] == 4
    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 4
    assert manager.version_progress_topup_events == 1


def test_version_adaptive_separates_invested_and_unstarted_carry_over():
    manager = make_adaptive_manager()
    manager.advance_step(0)
    groups = list(manager.group_queue[0].groups.values())
    groups[0].running_rollouts = 1

    manager.advance_step(1)

    assert manager.version_salvageable_inflight_at_boundary > 0
    assert manager.version_invested_inflight_at_boundary == 1
    assert manager.version_reserved_unstarted_at_boundary == (
        manager.version_salvageable_inflight_at_boundary - 1
    )


def test_version_adaptive_updates_supply_ratio_from_next_batch_consumption():
    manager = make_adaptive_manager(finish_ratio=0.25, ewma_alpha=1.0)
    manager.advance_step(0)
    manager.advance_step(1)

    queue = manager.group_queue[0]
    tracked_groups = list(manager._tracked_unfinished_groups)
    for group_id, episode_id in tracked_groups[
        : len(tracked_groups) // 2
    ]:
        group = manager.group_queue[group_id].groups[episode_id]
        manager._record_version_adaptive_consumption(group, 1)

    manager.advance_step(2)

    assert manager.version_actual_existing_supply == (
        len(tracked_groups) // 2
    )
    assert manager.adaptive_finish_ratio == 0.5
    assert manager.collect_metrics()["scheduler/adaptive_finish_ratio"] == 0.5
