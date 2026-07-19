import asyncio
from types import SimpleNamespace

from roll.distributed.scheduler.rollout_scheduler import (
    GroupData,
    GroupQueue,
    GroupQueueManager,
    VersionAwareRuntimeController,
    VersionRuntimeState,
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
    manager._refill_to_watermark(0)
    refilled = manager._outstanding_snapshot(0)

    assert refilled["outstanding_trajectories"] == 5
    assert manager.admitted_trajectories_total == 6


def test_version_adaptive_computes_one_budget_per_version_without_refill():
    manager = make_adaptive_manager()

    manager.advance_step(0)

    assert manager.version_admission_budget == 4
    assert manager.version_admission_used == 4
    assert manager.version_admission_remaining == 0
    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 4

    queue = manager.group_queue[0]
    queue.groups.pop(min(queue.groups))
    manager._refill_to_watermark(0)

    assert manager._outstanding_snapshot(0)["outstanding_trajectories"] == 3
    assert manager.admitted_trajectories_total == 4


def test_progress_reconcile_publishes_an_online_plan_revision():
    manager = make_adaptive_manager()
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

    revised = controller.decide(
        state,
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
    assert active_plan.admission_budget == 4
    assert revised.revision == 1
    assert revised.admission_delta_trajectories == 1
    assert revised.admission_budget == 5


def test_version_adaptive_reduces_new_work_when_carry_over_can_finish():
    manager = make_adaptive_manager()
    manager.advance_step(0)
    queue = manager.group_queue[0]
    first_group = queue.groups[min(queue.groups)]
    first_group.rollouts.append(object())

    manager.advance_step(1)

    # One ready trajectory plus three unfinished trajectories at a 0.5 finish ratio
    # provide 2.5 expected samples, so only two new groups are admitted for a batch of four.
    assert manager.version_valid_ready_at_boundary == 1
    assert manager.version_salvageable_inflight_at_boundary == 3
    assert manager.version_invested_inflight_at_boundary == 0
    assert manager.version_reserved_unstarted_at_boundary == 3
    assert manager.version_expected_existing_supply == 2.5
    assert manager.version_admission_budget_trainable == 2
    assert manager._outstanding_snapshot(1)["outstanding_trajectories"] == 6


def test_version_adaptive_separates_invested_and_unstarted_carry_over():
    manager = make_adaptive_manager()
    manager.advance_step(0)
    groups = list(manager.group_queue[0].groups.values())
    groups[0].running_rollouts = 1

    manager.advance_step(1)

    assert manager.version_salvageable_inflight_at_boundary == 4
    assert manager.version_invested_inflight_at_boundary == 1
    assert manager.version_reserved_unstarted_at_boundary == 3


def test_version_adaptive_updates_finish_ratio_from_completed_carry_over():
    manager = make_adaptive_manager(finish_ratio=0.25, ewma_alpha=1.0)
    manager.advance_step(0)
    manager.advance_step(1)

    queue = manager.group_queue[0]
    tracked_groups = list(manager._tracked_unfinished_groups)
    for group_id, episode_id in tracked_groups[:2]:
        manager._record_version_adaptive_completion(group_id, episode_id)

    manager.advance_step(2)

    assert manager.version_actual_existing_supply == 2
    assert manager.adaptive_finish_ratio == 0.5
    assert manager.collect_metrics()["scheduler/adaptive_finish_ratio"] == 0.5
