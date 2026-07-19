import asyncio
from types import SimpleNamespace

from roll.distributed.scheduler.router import (
    EnvAffinityRouter,
    TrajectoryRuntimeState,
)


class RouterManagerStub:
    pass


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

        newer = asyncio.create_task(
            router._acquire_priority_slot(0, priority=3, request_id="newer")
        )
        await asyncio.sleep(0)
        older = asyncio.create_task(
            router._acquire_priority_slot(0, priority=1, request_id="older")
        )
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


def test_rebuild_wave_coalesces_and_places_first_epoch_requests_together():
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
        assert first[1][2] == 2 and second[1][2] == 2
        assert {first[1][0], second[1][0]} == {0, 1}

    asyncio.run(run_test())
