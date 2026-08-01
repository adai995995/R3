from roll.distributed.scheduler.version_runtime_testbed import (
    TestbedConfig as RuntimeTestbedConfig,
    TestbedPhase,
    TraceTrajectory,
    compare_policies,
    generate_trace,
    run_testbed,
)


def test_synthetic_trace_is_deterministic():
    assert generate_trace(16, 7) == generate_trace(16, 7)
    assert generate_trace(16, 7) != generate_trace(16, 8)


def test_same_trace_drives_both_policies_without_mutation():
    trace = generate_trace(128, 11, min_actions=3, max_actions=9)
    original = list(trace)
    result = compare_policies(
        trace,
        RuntimeTestbedConfig(
            versions=12,
            learner_demand=4,
            service_actions_per_version=16,
            staleness_tolerance=2,
            max_outstanding=20,
            safety_reserve=4,
            workers=2,
            rebuild_budget=4,
        ),
    )

    assert trace == original
    assert result["fixed_fifo"]["policy"] == "fixed_fifo"
    assert result["unified"]["policy"] == "unified"
    assert len(result["fixed_fifo"]["boundaries"]) == 12
    assert len(result["unified"]["boundaries"]) == 12


def test_version_priority_salvages_invested_long_trajectories():
    trace = [
        TraceTrajectory(
            trajectory_id=f"trajectory-{index}",
            total_actions=6 if index % 2 == 0 else 3,
            prefix_id=index % 4,
            prefix_tokens=512,
            tokens_per_action=96,
            response_tokens_per_action=48,
        )
        for index in range(128)
    ]
    config = RuntimeTestbedConfig(
        versions=16,
        learner_demand=4,
        service_actions_per_version=14,
        staleness_tolerance=2,
        max_outstanding=24,
        safety_reserve=6,
        workers=2,
        rebuild_budget=4,
    )
    fixed = run_testbed(trace, config, "fixed_fifo")["metrics"]
    unified = run_testbed(trace, config, "unified")["metrics"]

    assert unified["stale_inference_tokens"] <= fixed["stale_inference_tokens"]
    assert unified["consumed_trajectories"] >= fixed["consumed_trajectories"]


def test_working_set_rebuild_produces_cache_reuse():
    trace = [
        TraceTrajectory(
            trajectory_id=f"trajectory-{index}",
            total_actions=5,
            prefix_id=index % 2,
            prefix_tokens=1024,
            tokens_per_action=128,
            response_tokens_per_action=64,
        )
        for index in range(96)
    ]
    result = run_testbed(
        trace,
        RuntimeTestbedConfig(
            versions=10,
            learner_demand=4,
            service_actions_per_version=12,
            staleness_tolerance=2,
            max_outstanding=20,
            safety_reserve=4,
            workers=2,
            rebuild_budget=4,
        ),
        "unified",
    )["metrics"]

    assert result["rebuild_requests"] > 0
    assert result["saved_prefill_tokens"] > 0
    assert result["prefill_saved_ratio"] > 0


def test_phase_trace_injects_tool_slowdown_and_learner_demand_change():
    trace = [
        TraceTrajectory(
            trajectory_id=f"trajectory-{index}",
            total_actions=3,
            prefix_id=index % 2,
            prefix_tokens=256,
            tokens_per_action=64,
            response_tokens_per_action=32,
            tool_seconds_per_action=0.5,
        )
        for index in range(64)
    ]
    result = run_testbed(
        trace,
        RuntimeTestbedConfig(
            versions=3,
            learner_demand=2,
            service_actions_per_version=4,
            staleness_tolerance=4,
            max_outstanding=24,
            safety_reserve=2,
            phases=(
                TestbedPhase(start_version=0),
                TestbedPhase(
                    start_version=1,
                    learner_demand=4,
                    tool_delay_scale=10.0,
                ),
            ),
        ),
        "unified",
    )

    assert result["boundaries"][0]["service_actions"] == 4
    assert result["boundaries"][1]["service_actions"] == 0
    assert result["boundaries"][1]["learner_demand"] == 4


def test_closed_loop_reserve_reacts_to_undersupply_then_expiration():
    trace = [
        TraceTrajectory(
            trajectory_id=f"trajectory-{index}",
            total_actions=10,
            prefix_id=index % 2,
            prefix_tokens=256,
            tokens_per_action=64,
            response_tokens_per_action=32,
        )
        for index in range(128)
    ]
    result = run_testbed(
        trace,
        RuntimeTestbedConfig(
            versions=4,
            learner_demand=2,
            service_actions_per_version=2,
            staleness_tolerance=0,
            max_outstanding=32,
            safety_reserve=4,
            adaptive_reserve=True,
            reserve_min=0,
            reserve_max=12,
            reserve_additive_step=2,
            reserve_decay=0.5,
            reserve_ewma_alpha=1.0,
            reserve_wait_high=0.5,
            reserve_overload_high=0.1,
        ),
        "unified",
    )
    boundaries = result["boundaries"]

    assert boundaries[0]["reserve_update_reason"] == 1
    assert boundaries[0]["reserve_after"] > boundaries[0]["reserve_before"]
    assert any(
        boundary["reserve_update_reason"] == 2
        and boundary["reserve_after"] < boundary["reserve_before"]
        for boundary in boundaries[1:]
    )
