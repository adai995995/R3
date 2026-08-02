import asyncio
from collections import defaultdict
from types import SimpleNamespace

from roll.distributed.scheduler.router import (
    build_engine_request_priority,
    completion_eta_observation,
    EnvAffinityRouter,
    RouterManager,
    summarize_request_metric_totals,
    TrajectoryRuntimeState,
    select_completion_eta_worker,
)


class RouterManagerStub:
    pass


def test_engine_priority_queue_metrics_are_aggregated_separately_from_router_gate():
    metrics = summarize_request_metric_totals(
        {
            "vllm/request_priority_enabled": 10,
            "vllm/request_priority_queued": 4,
            "vllm/request_priority_queue_seconds": 1.5,
        },
        scope="lifetime",
    )

    assert metrics["router/engine_priority_requests"] == 10
    assert metrics["router/engine_priority_queued_requests"] == 4
    assert metrics["router/engine_priority_queued_ratio"] == 0.4
    assert metrics["router/engine_priority_queue_seconds"] == 1.5


def _runtime_state(
    trajectory_id,
    *,
    policy_version=2,
    actions_completed=4,
    max_actions=10,
    group_id=1,
    episode_id=1,
):
    return TrajectoryRuntimeState(
        trajectory_id=trajectory_id,
        policy_version=policy_version,
        current_version=4,
        version_age=max(0, 4 - policy_version),
        actions_completed=actions_completed,
        max_actions=max_actions,
        group_id=group_id,
        episode_id=episode_id,
    )


def test_engine_priority_preserves_runtime_ordering():
    planned = {"1:1": 0}
    candidate = build_engine_request_priority(_runtime_state("candidate"), planned)
    outside_plan = build_engine_request_priority(
        _runtime_state("outside", group_id=2), planned
    )
    assert candidate < outside_plan

    older = build_engine_request_priority(
        _runtime_state("older", policy_version=1), {}
    )
    newer = build_engine_request_priority(
        _runtime_state("newer", policy_version=3), {}
    )
    assert older < newer

    invested = build_engine_request_priority(
        _runtime_state("invested", actions_completed=4), {}
    )
    unstarted = build_engine_request_priority(
        _runtime_state("unstarted", actions_completed=0), {}
    )
    assert invested < unstarted


def test_completion_eta_error_uses_queue_and_request_service_time():
    actual, error = completion_eta_observation(
        predicted_seconds=4.5,
        scheduling_wait_seconds=1.5,
        request_service_seconds=2.0,
    )

    assert actual == 3.5
    assert error == 1.0


def test_runtime_feedback_includes_cumulative_request_costs():
    manager = RouterManager.__new__(RouterManager)
    manager.router = SimpleNamespace(
        collect_runtime_feedback=lambda: {"requests": 3, "resets": 1}
    )
    manager.request_metric_lifetime_totals = defaultdict(
        float,
        {
            "vllm/request_prefill_tokens": 120,
            "vllm/request_prefill_seconds": 1.5,
            "router/scheduling_wait_seconds": 0.25,
        },
    )

    feedback = manager.collect_runtime_feedback()

    assert feedback["requests"] == 3
    assert feedback["resets"] == 1
    assert feedback["request_metrics"]["vllm/request_prefill_tokens"] == 120
    assert feedback["request_metrics"]["router/scheduling_wait_seconds"] == 0.25


def test_completion_eta_placement_balances_queue_and_prefix_cost():
    selected, reason, estimate = select_completion_eta_worker(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        worker_prompts={0: [(1, 2, 3, 4)], 1: []},
        active_dp_ranks={0, 1},
        worker_pressure={0: 2, 1: 0},
        worker_service_seconds={0: 2.0, 1: 2.0},
        token_seconds=0.1,
        prefix_limit=16,
    )

    assert selected == 1
    assert reason == "completion_eta_load"
    assert estimate["prefill_tokens"] == 6

    selected, reason, estimate = select_completion_eta_worker(
        prompt_tokens=[1, 2, 3, 4, 5, 6],
        worker_prompts={0: [(1, 2, 3, 4)], 1: []},
        active_dp_ranks={0, 1},
        worker_pressure={0: 0, 1: 0},
        worker_service_seconds={0: 2.0, 1: 2.0},
        token_seconds=0.1,
        prefix_limit=16,
    )

    assert selected == 0
    assert reason == "completion_eta_prefix"
    assert estimate["cached_tokens"] == 4


def test_env_affinity_router_serves_oldest_version_when_saturated():
    async def run_test():
        router = EnvAffinityRouter(
            router_manager=RouterManagerStub(),
            workers=[object()],
            model_path=None,
            router_args=SimpleNamespace(max_running_requests=1),
        )
        await router.initialize()

        running = await router._acquire_priority_slot(
            0, priority=5, request_id="running"
        )
        assert running == (True, 0, False, False, False)
        newer = asyncio.create_task(
            router._acquire_priority_slot(0, priority=3, request_id="newer")
        )
        await asyncio.sleep(0)
        older = asyncio.create_task(
            router._acquire_priority_slot(0, priority=1, request_id="older")
        )
        await asyncio.sleep(0)

        await router._release_priority_slot(0)
        done, _ = await asyncio.wait({newer, older}, return_when=asyncio.FIRST_COMPLETED)

        assert older in done
        assert newer not in done
        older_result = await older
        assert older_result[2] is True
        assert older_result[4] is True

        await router._release_priority_slot(0)
        newer_result = await newer
        assert newer_result[2] is True
        await router._release_priority_slot(0)

    asyncio.run(run_test())


def test_env_affinity_router_bypasses_priority_queue_when_disabled():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(), [object()], None, SimpleNamespace(max_running_requests=1)
        )
        await router.initialize()
        assert await router._acquire_priority_slot(0, None, "request") == (
            False,
            0,
            False,
            False,
            False,
        )

    asyncio.run(run_test())


def test_env_affinity_router_delegates_queueing_to_engine_priority_scheduler():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=1,
                router_config={
                    "engine_priority_scheduling_enabled": True,
                    "priority_max_running_requests": 1,
                },
            ),
        )
        await router.initialize()

        assert await router._acquire_priority_slot(0, 1, "request") == (
            False,
            0,
            False,
            False,
            False,
        )
        assert router.priority_inflight == [0]

    asyncio.run(run_test())


def test_epoch_observation_is_per_trajectory_even_with_worker_affinity():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(max_running_requests=1, router_config={}),
        )
        await router.initialize()
        router.on_version_resume({"version": 3})
        router.src_rank2_dp_rank[7] = 0

        assert router._observe_first_epoch_request("trajectory-a") is True
        assert router._observe_first_epoch_request("trajectory-a") is False
        assert router._observe_first_epoch_request("trajectory-b") is True

    asyncio.run(run_test())


def test_version_resume_installs_and_clears_planned_priority_cohort():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(max_running_requests=1, router_config={}),
        )
        await router.initialize()

        router.on_version_resume({
            "version": 3,
            "priority_enabled": True,
            "priority_candidate_groups": ["1:2", "3:4"],
        })
        assert router.priority_candidate_ranks == {"1:2": 0, "3:4": 1}

        router.on_version_resume({
            "version": 4,
            "priority_enabled": False,
            "priority_candidate_groups": ["9:9"],
        })
        assert router.priority_candidate_ranks == {}

    asyncio.run(run_test())


def test_online_plan_revision_does_not_advance_cache_epoch():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(max_running_requests=8, router_config={}),
        )
        await router.initialize()
        router.on_version_resume({"version": 3, "revision": 0})
        cache_epoch = router.cache_epoch

        router.on_runtime_plan_update(
            {
                "version": 3,
                "revision": 1,
                "priority_enabled": True,
                "priority_candidate_groups": ["5:6"],
                "rebuild_candidate_groups": ["5:6"],
            }
        )

        assert router.cache_epoch == cache_epoch
        assert router.runtime_plan["revision"] == 1
        assert router.priority_candidate_ranks == {"5:6": 0}

    asyncio.run(run_test())


def test_online_plan_revision_replaces_rebuild_cohort():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "post_update_rebuild_enabled": True,
                    "post_update_rebuild_requests": 2,
                },
            ),
        )
        await router.initialize()
        router.on_version_resume(
            {
                "version": 3,
                "revision": 0,
                "rebuild_candidate_groups": ["1:1"],
                "rebuild_candidate_trajectories": ["old-trajectory"],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 1,
            }
        )

        router.on_runtime_plan_update(
            {
                "version": 3,
                "revision": 1,
                "rebuild_candidate_groups": ["2:2"],
                "rebuild_candidate_trajectories": ["new-trajectory"],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 2,
            }
        )

        assert router.rebuild_candidate_groups == {"2:2"}
        assert router.rebuild_candidate_trajectories == {"new-trajectory"}
        assert router.rebuild_target == 2
        assert router.rebuild_remaining == 2

    asyncio.run(run_test())


def test_exact_rebuild_cohort_rejects_uninvested_trajectory_in_same_group():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "post_update_rebuild_enabled": True,
                    "post_update_rebuild_requests": 1,
                },
            ),
        )
        await router.initialize()
        router.on_version_resume(
            {
                "version": 3,
                "rebuild_candidate_groups": ["1:2"],
                "rebuild_candidate_trajectories": ["invested"],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 1,
            }
        )
        uninvested = TrajectoryRuntimeState(
            "uninvested",
            policy_version=2,
            current_version=3,
            version_age=1,
            actions_completed=0,
            max_actions=10,
            group_id=1,
            episode_id=2,
        )
        invested = TrajectoryRuntimeState(
            "invested",
            policy_version=2,
            current_version=3,
            version_age=1,
            actions_completed=4,
            max_actions=10,
            group_id=1,
            episode_id=2,
        )

        _, uninvested_assignment = await router._prepare_first_epoch_request(
            uninvested, "uninvested", [1, 2]
        )
        _, invested_assignment = await router._prepare_first_epoch_request(
            invested, "invested", [1, 2]
        )

        assert uninvested_assignment is None
        # The exact candidate consumes the rebuild budget, but a singleton
        # cluster deliberately falls back to normal affinity because it has no
        # follower that could reuse the reconstructed prefix.
        assert invested_assignment is None
        assert router.rebuild_remaining == 0

    asyncio.run(run_test())


def test_version_change_releases_pending_rebuild_waiter():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "post_update_rebuild_enabled": True,
                    "post_update_rebuild_requests": 1,
                    "post_update_rebuild_coalesce_seconds": 10.0,
                },
            ),
        )
        await router.initialize()
        router.on_version_resume(
            {
                "version": 3,
                "rebuild_candidate_trajectories": ["trajectory-a"],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 1,
            }
        )
        state = TrajectoryRuntimeState(
            "trajectory-a",
            policy_version=2,
            current_version=3,
            version_age=1,
            actions_completed=3,
            max_actions=10,
        )
        request = asyncio.create_task(
            router._prepare_first_epoch_request(
                state, "trajectory-a", [1, 2, 3]
            )
        )
        await asyncio.sleep(0)
        flush_task = router.rebuild_flush_task
        assert len(router.rebuild_pending) == 1

        router.on_version_resume(
            {
                "version": 4,
                "rebuild_candidate_trajectories": [],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 0,
            }
        )
        first_epoch_request, assignment = await asyncio.wait_for(
            request, timeout=0.1
        )
        await asyncio.gather(flush_task, return_exceptions=True)

        assert first_epoch_request is True
        assert assignment is None
        assert router.rebuild_pending == []

    asyncio.run(run_test())


def test_priority_coalesce_compares_arrivals_before_dispatch():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "priority_max_running_requests": 1,
                    "priority_coalesce_seconds": 0.02,
                },
            ),
        )
        await router.initialize()

        running = await router._acquire_priority_slot(
            0, priority=5, request_id="running"
        )
        assert running[0] is True

        newer = asyncio.create_task(
            router._acquire_priority_slot(0, priority=3, request_id="newer")
        )
        await asyncio.sleep(0)
        older = asyncio.create_task(
            router._acquire_priority_slot(0, priority=1, request_id="older")
        )
        await asyncio.sleep(0)
        assert not newer.done()
        assert not older.done()

        await router._release_priority_slot(0)
        done, _ = await asyncio.wait(
            {newer, older}, return_when=asyncio.FIRST_COMPLETED
        )

        assert older in done
        assert newer not in done
        older_result = await older
        assert older_result[2] is True
        assert older_result[3] is True
        assert older_result[4] is True
        await router._release_priority_slot(0)
        newer_result = await newer
        assert newer_result[3] is True
        await router._release_priority_slot(0)

    asyncio.run(run_test())


def test_priority_coalesce_is_work_conserving_when_capacity_is_available():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "priority_max_running_requests": 1,
                    "priority_coalesce_seconds": 10.0,
                },
            ),
        )
        await router.initialize()

        result = await asyncio.wait_for(
            router._acquire_priority_slot(
                0, priority=1, request_id="uncontended"
            ),
            timeout=0.1,
        )

        assert result == (True, 0, False, False, False)
        await router._release_priority_slot(0)

    asyncio.run(run_test())


def test_rebuild_is_bypassed_below_worker_execution_window():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object(), object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "post_update_rebuild_enabled": True,
                    "post_update_rebuild_requests": 4,
                    "post_update_rebuild_min_outstanding_per_worker": 4,
                },
            ),
        )
        await router.initialize()

        router.on_version_resume(
            {
                "version": 3,
                "revision": 0,
                "worker_count": 2,
                "outstanding_trajectories": 7,
                "rebuild_candidate_trajectories": ["a", "b"],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 2,
            }
        )
        assert router.rebuild_load_eligible is False
        assert router.rebuild_target == 0
        assert router.rebuild_remaining == 0

        router.on_version_resume(
            {
                "version": 4,
                "revision": 0,
                "worker_count": 2,
                "outstanding_trajectories": 8,
                "rebuild_candidate_trajectories": ["a", "b"],
                "rebuild_cohort_exact": True,
                "rebuild_target_trajectories": 2,
            }
        )
        assert router.rebuild_load_eligible is True
        assert router.rebuild_target == 2
        assert router.rebuild_remaining == 2

    asyncio.run(run_test())


def test_rebuild_capacity_can_burst_without_relaxing_normal_priority_limit():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "priority_max_running_requests": 1,
                    "priority_rebuild_max_running_requests": 2,
                },
            ),
        )
        await router.initialize()

        first = await router._acquire_priority_slot(
            0, priority=1, request_id="first-rebuild", max_running_requests=2
        )
        second = await router._acquire_priority_slot(
            0, priority=1, request_id="second-rebuild", max_running_requests=2
        )
        assert first[0] is True
        assert second[0] is True
        assert router.priority_inflight[0] == 2

        normal = asyncio.create_task(
            router._acquire_priority_slot(0, priority=1, request_id="normal")
        )
        await asyncio.sleep(0)
        assert not normal.done()

        await router._release_priority_slot(0)
        await asyncio.sleep(0)
        assert not normal.done()
        await router._release_priority_slot(0)
        assert (await normal)[0] is True
        await router._release_priority_slot(0)

    asyncio.run(run_test())


def test_engine_cache_reset_invalidates_router_shadow_state():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object(), object()],
            None,
            SimpleNamespace(max_running_requests=8, router_config={}),
        )
        await router.initialize()
        router.src_rank2_dp_rank.update({"a": 0, "b": 1})
        router.src_rank_cache_epoch.update({"a": 1, "b": 1})
        router.src_rank_last_prompt_tokens.update({"a": 8, "b": 8})
        router.working_set_worker_prompts[0].append((1, 2, 3))
        router.working_set_worker_prompts[1].append((4, 5, 6))

        observed, resets, invalidated = router._apply_engine_kv_feedback(
            0,
            {
                "vllm/engine_prefix_cache_requests_delta": 1,
                "vllm/engine_prefix_cache_resets_delta": 1,
            },
        )

        assert observed is True
        assert resets == 1
        assert invalidated == 1
        assert "a" not in router.src_rank2_dp_rank
        assert router.src_rank2_dp_rank["b"] == 1
        assert 0 not in router.working_set_worker_prompts
        assert router.working_set_worker_prompts[1] == [(4, 5, 6)]

    asyncio.run(run_test())


def test_boundary_profile_proves_engine_cobatching_per_worker():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object()],
            None,
            SimpleNamespace(max_running_requests=8, router_config={}),
        )
        await router.initialize()
        base = {
            "logical_prompt_tokens": 10,
            "logical_reprefill_exposure_tokens": 10,
            "reported_prefill_tokens": 10,
            "engine_scheduler_batch_id": 4,
            "engine_scheduler_batch_size": 2,
            "cache_epoch": 3,
            "worker_rank": 0,
        }
        router.boundary_recovery_records = [dict(base), dict(base)]

        metrics = router.collect_version_boundary_profile()["metrics"]

        assert metrics["engine_scheduler_batch_records"] == 2
        assert metrics["engine_scheduler_batches"] == 1
        assert metrics["engine_batches_with_multiple_survivors"] == 1
        assert metrics["engine_cobatched_survivor_requests"] == 2
        assert metrics["engine_scheduler_batch_size_max"] == 2

    asyncio.run(run_test())


def test_rebuild_wave_preserves_affinity_when_no_reuse_is_possible():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object(), object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "post_update_rebuild_enabled": True,
                    "post_update_rebuild_requests": 2,
                    "post_update_rebuild_coalesce_seconds": 0.01,
                },
            ),
        )
        await router.initialize()
        router.on_version_resume(
            {
                "version": 4,
                "rebuild_candidate_groups": ["1:1", "2:2"],
                "rebuild_target_trajectories": 2,
            }
        )
        states = [
            TrajectoryRuntimeState("a", 3, 4, 1, 5, 10, 1, 1),
            TrajectoryRuntimeState("b", 3, 4, 1, 2, 10, 2, 2),
        ]
        first, second = await asyncio.gather(
            router._prepare_first_epoch_request(states[0], "a", [1, 2, 3]),
            router._prepare_first_epoch_request(states[1], "b", [1, 2, 4]),
        )

        assert first[0] is True and second[0] is True
        assert first[1] is None and second[1] is None

    asyncio.run(run_test())


def test_version_resume_preserves_placement_as_rebuild_fallback():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object(), object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={"post_update_rebuild_enabled": True},
            ),
        )
        await router.initialize()
        router.src_rank2_dp_rank["trajectory-a"] = 1
        router.src_rank_cache_epoch["trajectory-a"] = 0

        router.on_version_resume({"version": 4})

        assert router.src_rank2_dp_rank["trajectory-a"] == 1
        assert router.src_rank_cache_epoch["trajectory-a"] == 0

    asyncio.run(run_test())


def test_rebuild_follower_waits_until_same_prefix_seed_is_ready():
    async def run_test():
        router = EnvAffinityRouter(
            RouterManagerStub(),
            [object(), object()],
            None,
            SimpleNamespace(
                max_running_requests=8,
                router_config={
                    "post_update_rebuild_enabled": True,
                    "post_update_rebuild_requests": 3,
                    "post_update_rebuild_coalesce_seconds": 0.01,
                    "post_update_rebuild_seed_slots_per_worker": 1,
                },
            ),
        )
        await router.initialize()
        router.on_version_resume(
            {
                "version": 4,
                "rebuild_candidate_groups": ["1:1", "2:2", "3:3"],
                "rebuild_target_trajectories": 3,
            }
        )
        states = [
            TrajectoryRuntimeState("a", 3, 4, 1, 5, 10, 1, 1),
            TrajectoryRuntimeState("b", 3, 4, 1, 4, 10, 2, 2),
            TrajectoryRuntimeState("c", 3, 4, 1, 3, 10, 3, 3),
        ]
        prompt = list(range(256))
        tasks = [
            asyncio.create_task(
                router._prepare_first_epoch_request(state, state.trajectory_id, prompt)
            )
            for state in states
        ]
        await asyncio.sleep(0.03)

        completed = [task for task in tasks if task.done()]
        waiting = [task for task in tasks if not task.done()]
        assert len(completed) == 2
        assert len(waiting) == 1
        seed_assignment = completed[0].result()[1]
        assert seed_assignment[4] == "seed"

        await router._complete_rebuild_seed(
            cache_epoch=seed_assignment[5],
            cluster_key=seed_assignment[3],
            dp_rank=seed_assignment[0],
            prompt_tokens=prompt,
            success=True,
        )
        follower_assignment = (await waiting[0])[1]

        assert follower_assignment[4] == "follower"
        assert follower_assignment[0] == seed_assignment[0]
        assert follower_assignment[1] == 256

    asyncio.run(run_test())
