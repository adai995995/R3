import asyncio
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import ray
import numpy as np
import pytest
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from roll.distributed.scheduler.rollout_scheduler import (
    apply_dynamic_reserve_hysteresis,
    compute_progress_topup_groups,
    compute_effective_rollout_utility,
    compute_stale_control_signal,
    finish_rate_bucket,
    consume_utility_settle,
    GroupQueue,
    RolloutScheduler,
    GroupQueueManager,
    RuntimeEstimator,
    RuntimeCandidateEstimate,
    PolicyUpdateTrace,
    VersionRuntimeOutcome,
    compute_closed_loop_reserve,
    compute_state_feedback_reserve,
    runtime_finish_rate_bucket,
    compute_dynamic_reserve,
    update_utility_hill_climb,
    update_constrained_utility_hill_climb,
    update_bucketed_finish_ratios,
    predict_bucketed_finish_supply,
    build_version_runtime_plan,
    build_learner_wait_record,
    summarize_version_boundary_records,
    summarize_rollout_goodput,
)


def test_rollout_goodput_separates_raw_trainable_and_stale_tokens():
    metrics = summarize_rollout_goodput(
        [{"completed": True, "response_tokens": 20, "inference_tokens": 100}],
        [
            {
                "completed": True,
                "discard_reason": "version_stale_at_consume",
                "response_tokens": 10,
                "inference_tokens": 50,
                "model_execute_seconds": 2,
                "engine_step_seconds_attributed": 1.5,
            }
        ],
        [
            {
                "completed": False,
                "response_tokens": 5,
                "inference_tokens": 25,
                "model_execute_seconds": 1,
                "engine_step_seconds_attributed": 0.5,
            }
        ],
        elapsed_seconds=5,
        learner_wait_seconds=1,
        admitted_trajectories=5,
    )

    assert metrics["rollout/raw_response_tokens_per_second"] == 7
    assert metrics["rollout/trainable_response_tokens_per_second"] == 4
    assert metrics["rollout/stale_logical_token_fraction"] == 50 / 175
    assert metrics["rollout/stale_trajectory_fraction"] == 1 / 5
    assert metrics["rollout/stale_trajectory_fraction_of_recorded"] == 1 / 3
    assert metrics["rollout/stale_model_execute_fraction"] == 2 / 3
    assert metrics["rollout/stale_engine_step_fraction"] == 3 / 4
    assert metrics["learner/wait_fraction"] == 0.2


def test_rollout_goodput_excludes_consumed_placeholders_from_trainable_work():
    metrics = summarize_rollout_goodput(
        [
            {
                "completed": True,
                "trainable_valid": True,
                "response_tokens": 20,
                "inference_tokens": 100,
            },
            {
                "completed": True,
                "placeholder": True,
                "trainable_valid": False,
                "response_tokens": 7,
                "inference_tokens": 30,
            },
        ],
        [],
        [],
        elapsed_seconds=2,
        learner_wait_seconds=0,
    )

    assert metrics["rollout/learner_consumed_trajectories"] == 2
    assert metrics["rollout/trainable_trajectories"] == 1
    assert metrics["rollout/placeholder_trajectories"] == 1
    assert metrics["rollout/trainable_response_tokens"] == 20
    assert metrics["rollout/trainable_logical_inference_tokens"] == 100


def test_learner_wait_record_attributes_stale_work_to_the_batch_step():
    record = build_learner_wait_record(
        step=7,
        wait_seconds=2.5,
        batch_size=4,
        consumed_records=[
            {
                "consumed_at_step": 7,
                "version_age": 1,
                "trainable_valid": True,
            },
            {
                "consumed_at_step": 6,
                "version_age": 0,
                "trainable_valid": True,
            },
        ],
        discard_records=[
            {
                "discarded_at_step": 7,
                "discard_reason": "version_stale_at_consume",
                "actions_completed": 8,
                "response_tokens": 320,
                "tool_calls": 5,
                "tool_wall_seconds": 0.4,
                "engine_step_seconds_attributed": 1.25,
            },
            {
                "discarded_at_step": 6,
                "discard_reason": "version_stale_at_consume",
                "actions_completed": 100,
            },
            {
                "discarded_at_step": 7,
                "discard_reason": "redundancy_trim",
                "actions_completed": 100,
            },
        ],
        recorded_at_unix=123.0,
    )

    assert record == {
        "step": 7,
        "batch_size": 4,
        "wait_seconds": 2.5,
        "consumed_trajectories": 1,
        "valid_trajectories": 1,
        "valid_version_age_mean": 1.0,
        "valid_version_age_max": 1,
        "stale_discarded_trajectories": 1,
        "stale_discarded_actions": 8,
        "stale_discarded_output_tokens": 320,
        "stale_discarded_tool_calls": 5,
        "stale_discarded_tool_wall_seconds": 0.4,
        "stale_discarded_engine_step_seconds_attributed": 1.25,
        "consumed_queue_seconds": 0.0,
        "consumed_queue_mean_seconds": 0.0,
        "consumed_queue_p95_seconds": 0.0,
        "consumed_queue_max_seconds": 0.0,
        "consumed_tool_seconds": 0.0,
        "consumed_tool_mean_seconds": 0.0,
        "consumed_tool_p95_seconds": 0.0,
        "consumed_tool_max_seconds": 0.0,
        "consumed_generate_seconds": 0.0,
        "consumed_generate_mean_seconds": 0.0,
        "consumed_generate_p95_seconds": 0.0,
        "consumed_generate_max_seconds": 0.0,
        "batch_closing_trajectory_id": "",
        "batch_closing_completion_unix": 0.0,
        "batch_closing_queue_seconds": 0.0,
        "batch_closing_tool_seconds": 0.0,
        "batch_closing_generate_seconds": 0.0,
        "recorded_at_unix": 123.0,
    }


def test_reset_only_stale_count_does_not_enter_token_waste_signal():
    token_fraction, trajectory_fraction = compute_stale_control_signal(
        stale_tokens=0,
        consumed_tokens=0,
        stale_trajectories=3,
        consumed_trajectories=0,
    )

    assert token_fraction is None
    assert trajectory_fraction == 1.0


def test_learner_wait_record_identifies_batch_closing_trajectory():
    record = build_learner_wait_record(
        step=3,
        wait_seconds=4.0,
        batch_size=2,
        consumed_records=[
            {
                "consumed_at_step": 3,
                "trainable_valid": True,
                "trajectory_id": "early",
                "trajectory_completed_at_unix": 10.0,
                "request_queue_seconds": 1.0,
                "tool_wall_seconds": 2.0,
                "generate_seconds": 3.0,
            },
            {
                "consumed_at_step": 3,
                "trainable_valid": True,
                "trajectory_id": "closer",
                "trajectory_completed_at_unix": 20.0,
                "request_queue_seconds": 4.0,
                "tool_wall_seconds": 5.0,
                "generate_seconds": 6.0,
            },
        ],
        discard_records=[],
    )

    assert record["consumed_queue_seconds"] == 5.0
    assert record["consumed_queue_mean_seconds"] == 2.5
    assert record["consumed_queue_p95_seconds"] == 4.0
    assert record["batch_closing_trajectory_id"] == "closer"
    assert record["batch_closing_queue_seconds"] == 4.0
    assert record["batch_closing_tool_seconds"] == 5.0
    assert record["batch_closing_generate_seconds"] == 6.0
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.protocol import DataProto
from roll.pipeline.agentic.agentic_pipeline import GroupFilter
from roll.pipeline.agentic.agentic_config import EnvMonitorConfig


FULL_DATASET_ITER=4


def test_version_boundary_summary_separates_survivors_and_expired_work():
    records = [
        {
            "trajectory_id": "survivor",
            "version_age_at_boundary": 2,
            "will_expire": False,
            "completed": False,
            "actions_completed": 6,
            "inference_calls": 6,
            "tool_calls": 2,
            "inference_tokens": 1200,
            "current_context_tokens": 400,
        },
        {
            "trajectory_id": "expired",
            "version_age_at_boundary": 3,
            "will_expire": True,
            "completed": False,
            "actions_completed": 8,
            "inference_calls": 8,
            "tool_calls": 4,
            "inference_tokens": 2400,
            "current_context_tokens": 800,
        },
        {
            "trajectory_id": "ready",
            "version_age_at_boundary": 1,
            "will_expire": False,
            "completed": True,
            "actions_completed": 10,
            "inference_calls": 10,
            "inference_tokens": 3000,
            "current_context_tokens": 1000,
        },
    ]

    summary = summarize_version_boundary_records(
        records,
        from_version=4,
        to_version=5,
        staleness_tolerance=2,
        reserved_unstarted=3,
        unobserved_started=1,
    )

    assert summary["cross_version_trajectories"] == 2
    assert summary["cross_version_invested_trajectories"] == 2
    assert summary["survivor_trajectories"] == 1
    assert summary["completed_carryover_trajectories"] == 1
    assert summary["completed_survivor_trajectories"] == 1
    assert summary["expired_trajectories"] == 1
    assert summary["unfinished_started_trajectories"] == 2
    assert summary["expired_actions"] == 8
    assert summary["expired_logical_inference_tokens"] == 2400
    assert summary["current_context_tokens"] == 2200
    assert summary["reserved_unstarted_trajectories"] == 3


def test_version_runtime_plan_unifies_admission_deadline_and_rebuild_cohort():
    plan = build_version_runtime_plan(
        version=7,
        learner_demand=4,
        safety_reserve=2,
        expected_existing_supply=2.5,
        outstanding_trajectories=8,
        max_outstanding_trajectories=12,
        admission_width=2,
        group_size=2,
        staleness_tolerance=2,
        invested_candidate_groups=[
            (3, 9, 1, 7, 1),
            (2, 4, 2, 1, 2),
            (1, 5, 2, 6, 1),
        ],
    )

    assert plan.admission_budget == 4
    assert plan.admission_budget_trainable == 4
    assert plan.admission_reason == "supply_deficit"
    assert plan.admission_deficit == 3.5
    assert plan.priority_deadline_version == 9
    assert plan.priority_candidate_groups == ("1:5", "2:4", "3:9")
    assert plan.rebuild_candidate_groups == ("1:5", "2:4", "3:9")
    assert plan.rebuild_target_trajectories == 4
    assert plan.revision == 0
    assert plan.admission_delta_trajectories == 4
    assert plan.plan_id == "version-7-revision-0"
    assert plan.forecast_id == "version-7-estimator-0"


def test_version_runtime_plan_uses_shared_completion_estimates_for_priority():
    plan = build_version_runtime_plan(
        version=4,
        learner_demand=2,
        safety_reserve=0,
        expected_existing_supply=0,
        outstanding_trajectories=2,
        max_outstanding_trajectories=4,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=[
            (1, 1, 2, 8, 1),
            (1, 2, 1, 3, 1),
            (1, 3, 2, 9, 1),
        ],
        candidate_estimates=[
            RuntimeCandidateEstimate(
                "1:1", 0.9, 2.0, 1.0, True, 2, 8, 1
            ),
            RuntimeCandidateEstimate(
                "1:2", 0.8, 1.0, 4.0, True, 1, 3, 1
            ),
            RuntimeCandidateEstimate(
                "1:3", 0.1, 20.0, -10.0, False, 2, 9, 1
            ),
        ],
    )

    assert plan.priority_candidate_groups == ("1:1", "1:2", "1:3")
    assert [
        estimate.group_key
        for estimate in plan.priority_candidate_estimates
    ] == ["1:1", "1:2", "1:3"]


def test_runtime_estimator_attributes_forecast_and_updates_from_outcome():
    estimator = RuntimeEstimator(
        initial_finish_ratio=0.5,
        ewma_alpha=0.5,
        bucketed_finish_enabled=True,
        bucket_min_samples=1,
    )
    forecast = estimator.build_forecast(
        3,
        ready_valid_slots=2,
        salvageable_inflight=4,
        unfinished_bucket_counts={"age_1__actions_2_3": 4},
    )

    assert forecast.expected_existing_supply == 4
    assert forecast.forecast_id == "version-3-estimator-0"

    estimator.observe_supply(
        salvageable_inflight=4,
        completed_inflight=3,
        cohort_counts={"age_1__actions_2_3": 4},
        completed_counts={"age_1__actions_2_3": 3},
        prediction_error=-1.0,
    )
    next_forecast = estimator.build_forecast(
        4,
        ready_valid_slots=0,
        salvageable_inflight=4,
        unfinished_bucket_counts={"age_1__actions_2_3": 4},
    )

    assert estimator.revision == 1
    assert next_forecast.predicted_inflight_slots == 3
    assert estimator.supply_error_ewma == -1
    assert estimator.supply_abs_error_ewma == 1


def test_runtime_estimator_uses_readiness_bucket_with_coarse_fallback():
    estimator = RuntimeEstimator(
        initial_finish_ratio=0.25,
        ewma_alpha=1.0,
        bucketed_finish_enabled=True,
        bucket_min_samples=2,
    )
    ready_bucket = runtime_finish_rate_bucket(
        1, 3, {"inference_ready": 2, "tool_waiting": 0}
    )
    tool_bucket = runtime_finish_rate_bucket(
        1, 3, {"inference_ready": 0, "tool_waiting": 2}
    )
    estimator.observe_supply(
        salvageable_inflight=2,
        completed_inflight=2,
        cohort_counts={ready_bucket: 2},
        completed_counts={ready_bucket: 2},
    )

    predicted, learned, fallback, by_bucket = (
        estimator.predict_unfinished_supply(
            salvageable_inflight=2,
            unfinished_bucket_counts={tool_bucket: 2},
        )
    )

    assert predicted == 2
    assert learned == 2
    assert fallback == 0
    assert by_bucket[tool_bucket] == 2


def test_runtime_estimator_adds_cross_version_reprefill_cost_to_eta():
    estimator = RuntimeEstimator(
        initial_finish_ratio=0.5,
        ewma_alpha=1.0,
        bucketed_finish_enabled=False,
        bucket_min_samples=1,
    )
    estimator.observe_policy_interval(10.0)
    estimator.observe_completed_records(
        [
            {
                "completed": True,
                "group_id": 1,
                "episode_id": 2,
                "env_id": 3,
                "version_start": 0,
                "inference_calls": 2,
                "actions_completed": 2,
                "generate_seconds": 2.0,
                "engine_prefill_tokens": 100,
                "request_prefill_seconds": 2.0,
            }
        ]
    )

    estimate = estimator.estimate_candidate(
        group_key="1:2",
        version_age=1,
        staleness_tolerance=2,
        progress_actions=2,
        invested_trajectories=1,
        finish_bucket="age_1__actions_2_3",
        runtime_summary={
            "frontier_remaining_actions": 2,
            "mean_context_tokens": 50,
            "inference_ready": 1,
            "tool_waiting": 0,
        },
    )

    assert estimate.eta_seconds == 3.0
    assert estimate.feasible is True


def test_version_runtime_outcome_exposes_signed_forecast_residual():
    outcome = VersionRuntimeOutcome(
        plan_id="version-2-revision-1",
        forecast_id="version-2-estimator-1",
        version=2,
        final_revision=1,
        predicted_existing_supply=6.5,
        actual_existing_valid_slots=4,
        admitted_trajectories=2,
        completed_valid_slots=4,
        consumed_valid_slots=4,
        learner_wait_seconds=3.0,
        next_batch_latency_seconds=3.0,
        reprefill_tokens=800,
        prefill_tokens=1200,
        prefill_seconds=0.75,
        scheduling_wait_seconds=0.25,
        scheduling_requests=4,
        scheduling_wait_mean_seconds=0.0625,
    )

    assert outcome.supply_prediction_error == -2.5
    assert outcome.to_dict()["supply_prediction_error"] == -2.5
    assert outcome.to_dict()["reprefill_tokens"] == 800
    assert outcome.to_dict()["scheduling_wait_mean_seconds"] == 0.0625


def test_policy_update_trace_checks_activation_to_activation_decomposition():
    trace = PolicyUpdateTrace(
        version=4,
        batch_wait_seconds=2.0,
        learner_compute_seconds=5.0,
        publish_activate_seconds=1.0,
        other_seconds=0.5,
        update_interval_seconds=8.5,
        finalized=True,
    )

    assert trace.decomposition_error_seconds == 0.0
    assert trace.to_dict()["decomposition_error_seconds"] == 0.0


def test_record_policy_update_timing_accepts_serialized_trace_version():
    manager_cls = GroupQueueManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.policy_update_traces = {}

    result = manager.record_policy_update_timing(
        4,
        {
            "version": 4,
            "batch_wait_seconds": 2.0,
            "learner_compute_seconds": 5.0,
        },
    )

    assert result["version"] == 4
    assert result["batch_wait_seconds"] == 2.0
    assert result["learner_compute_seconds"] == 5.0


def test_version_runtime_plan_rebuilds_only_gpu_invested_working_set():
    plan = build_version_runtime_plan(
        version=7,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=2,
        outstanding_trajectories=8,
        max_outstanding_trajectories=12,
        admission_width=2,
        group_size=2,
        staleness_tolerance=2,
        invested_candidate_groups=[
            (1, 2, 2, 0, 2),
            (3, 4, 1, 5, 2),
        ],
        gpu_invested_candidate_groups=[
            (3, 4, 1, 5, 1),
        ],
        gpu_invested_candidate_trajectories=[
            ("trajectory-9", 3, 4, 9, 1, 5),
        ],
        revision=2,
    )

    assert plan.priority_candidate_groups == ("1:2", "3:4")
    assert plan.rebuild_candidate_groups == ("3:4",)
    assert plan.rebuild_candidate_trajectories == ("trajectory-9",)
    assert plan.rebuild_cohort_exact is True
    assert plan.rebuild_target_trajectories == 1
    assert plan.revision == 2


def test_version_runtime_plan_carries_engine_feedback_into_decision_snapshot():
    plan = build_version_runtime_plan(
        version=8,
        learner_demand=2,
        safety_reserve=0,
        expected_existing_supply=2,
        outstanding_trajectories=2,
        max_outstanding_trajectories=4,
        admission_width=1,
        group_size=1,
        staleness_tolerance=2,
        invested_candidate_groups=[],
        gpu_invested_candidate_trajectories=[],
        kv_feedback_requests=20,
        kv_feedback_hit_ratio=0.625,
        kv_feedback_resets=4,
    )

    assert plan.kv_feedback_requests == 20
    assert plan.kv_feedback_hit_ratio == 0.625
    assert plan.kv_feedback_resets == 4


def test_version_runtime_plan_keeps_priority_and_kv_when_admission_is_disabled():
    plan = build_version_runtime_plan(
        version=5,
        learner_demand=4,
        safety_reserve=0,
        expected_existing_supply=1,
        outstanding_trajectories=12,
        max_outstanding_trajectories=12,
        admission_width=2,
        group_size=2,
        staleness_tolerance=1,
        invested_candidate_groups=[
            (2, 3, 1, 4, 2),
            (1, 7, 0, 8, 1),
        ],
        admission_enabled=False,
        priority_enabled=True,
    )

    assert plan.admission_budget == 0
    assert plan.admission_reason == "disabled"
    assert plan.priority_candidate_groups == ("2:3", "1:7")
    assert plan.rebuild_candidate_groups == ("2:3", "1:7")


def test_version_runtime_plan_explains_outstanding_capacity_limit():
    plan = build_version_runtime_plan(
        version=5,
        learner_demand=8,
        safety_reserve=0,
        expected_existing_supply=0,
        outstanding_trajectories=12,
        max_outstanding_trajectories=12,
        admission_width=2,
        group_size=2,
        staleness_tolerance=1,
        invested_candidate_groups=[],
    )

    assert plan.admission_budget == 0
    assert plan.admission_capacity == 0
    assert plan.admission_reason == "outstanding_cap"


def test_discard_metrics_fall_back_to_trajectory_data():
    trajectory_data = {
        "version_info": {"version_start": 2, "version_end": 3, "version_age": 1},
        "waste_info": {
            "completed": True,
            "actions_completed": 7,
            "inference_calls": 7,
            "tool_calls": 2,
            "prompt_tokens_total": 1200,
            "response_tokens_total": 300,
            "inference_tokens_total": 1500,
            "env_seconds_total": 4.5,
            "generate_seconds_total": 3.0,
        },
    }
    rollout = DataProto.from_single_dict({
        "input_ids": torch.zeros((2, 1), dtype=torch.long),
        "trajectory_data": np.array([None, json.dumps(trajectory_data)], dtype=object),
    })

    assert GroupQueue._metric_by_suffix(rollout, "/traj_actions_completed") == 7
    assert GroupQueue._metric_by_suffix(rollout, "/traj_actions_ge_4") == 1
    assert GroupQueue._metric_by_suffix(rollout, "/traj_inference_tokens_total") == 1500
    assert GroupQueue._metric_by_suffix(rollout, "/traj_version_start") == 2

    rollout.meta_info["metrics"] = {
        "env/WebShopEnv/traj_actions_completed": [7.0, 7.0],
    }
    assert GroupQueue._metric_by_suffix(rollout, "/traj_actions_completed") == 7


def test_discard_record_uses_boundary_snapshot_and_deduplicates():
    queue = GroupQueue.__new__(GroupQueue)
    queue.group_id = 3
    queue.current_step = 4
    queue.discard_records = []
    queue.discard_record_indices = {}
    queue.dirty_discard_indices = set()
    queue.progress_snapshots = {}

    group = type("Group", (), {"group_id": 3, "episode_id": 5, "create_step": 1})()
    rollout = DataProto.from_single_dict({
        "input_ids": torch.zeros((1, 1), dtype=torch.long),
        "env_ids": np.array([9], dtype=object),
        "traj_id": np.array(["trajectory-9"], dtype=object),
    })
    queue.update_progress_snapshots([{
        "group_id": 3,
        "episode_id": 5,
        "env_id": 9,
        "version_start": 1,
        "version_end": 3,
        "reset_completed": True,
        "completed": False,
        "truncated": False,
        "actions_completed": 6,
        "inference_calls": 6,
        "tool_calls": 0,
        "prompt_tokens": 900,
        "response_tokens": 240,
        "inference_tokens": 1140,
        "generate_seconds": 2.0,
        "env_seconds": 3.0,
        "trajectory_started_at_unix": 100.0,
        "trajectory_completed_at_unix": None,
    }])
    queue.update_progress_snapshots([{
        "group_id": 3,
        "episode_id": 5,
        "env_id": 9,
        "actions_completed": 0,
        "inference_calls": 0,
        "inference_tokens": 0,
    }])

    queue.record_discarded_rollout(rollout, group, "version_expired_buffered", 4)
    queue.record_discarded_rollout(rollout, group, "version_expired_late_return", 4)

    assert len(queue.discard_records) == 1
    assert queue.discard_records[0]["actions_completed"] == 6
    assert queue.discard_records[0]["inference_tokens"] == 1140
    assert queue.discard_records[0]["trajectory_started_at_unix"] == 100.0
    assert queue.discard_records[0]["trajectory_completed_at_unix"] == 0.0
    assert queue.discard_records[0]["discard_reason"] == "version_expired_late_return"


def test_tensor_progress_recovers_completed_rollout_without_metadata():
    rollout = DataProto.from_single_dict({
        "response_mask": torch.tensor([[0, 1, 1, 0, 0, 1, 1]], dtype=torch.long),
        "attention_mask": torch.ones((1, 7), dtype=torch.long),
    })

    assert GroupQueue._tensor_progress(rollout) == {
        "actions_completed": 2,
        "inference_calls": 2,
        "prompt_tokens": 6,
        "response_tokens": 4,
        "inference_tokens": 10,
        "current_context_tokens": 7,
    }


def test_consumed_metrics_preserve_version_age_and_progress_distribution():
    records = [
        {
            "version_age": 0,
            "actions_completed": 5,
            "prompt_tokens": 100,
            "response_tokens": 20,
            "inference_tokens": 120,
        },
        {
            "version_age": 2,
            "actions_completed": 10,
            "prompt_tokens": 300,
            "response_tokens": 40,
            "inference_tokens": 340,
        },
    ]

    metrics, histograms = GroupQueueManager._aggregate_consumed_records(records, "consumed")

    assert metrics["consumed/trajectories"] == 2
    assert metrics["consumed/version_age_sum"] == 2
    assert metrics["consumed/version_age_max"] == 2
    assert metrics["consumed/actions"] == 15
    assert metrics["consumed/inference_tokens"] == 460
    assert histograms["version_age"] == {"0": 1, "2": 1}
    assert histograms["actions_completed"] == {"5": 1, "10": 1}


def test_completed_rollout_record_recovers_tensor_progress():
    group = type("Group", (), {"group_id": 1, "episode_id": 2, "create_step": 3})()
    rollout = DataProto.from_single_dict({
        "response_mask": torch.tensor([[0, 1, 1, 0, 1]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.long),
        "env_ids": np.array([4], dtype=object),
        "traj_id": np.array(["trajectory-4"], dtype=object),
    })

    record = GroupQueueManager._completed_rollout_record(rollout, group)

    assert record["actions_completed"] == 2
    assert record["inference_calls"] == 2
    assert record["response_tokens"] == 3
    assert record["inference_tokens"] == 8


def test_dynamic_reserve_increases_on_wait_and_decays_on_waste():
    common = {
        "reserve_min": 0,
        "reserve_max": 8,
        "additive_step": 2,
        "multiplicative_decay": 0.5,
        "warmup_versions": 2,
        "wait_high": 2.0,
        "stale_high": 0.25,
        "prediction_error_margin": 1.0,
    }

    assert compute_dynamic_reserve(4, 1, 8.0, 0.0, 0.0, **common) == (4, 4)
    assert compute_dynamic_reserve(4, 2, 8.0, 0.0, 0.0, **common) == (6, 1)
    assert compute_dynamic_reserve(6, 3, 0.5, 1.0, 0.0, **common) == (2, 2)
    assert compute_dynamic_reserve(4, 3, 0.5, 0.0, -2.0, **common) == (2, 3)
    assert compute_dynamic_reserve(4, 3, 1.0, 0.0, 0.0, **common) == (4, 0)


def test_closed_loop_reserve_separates_under_supply_from_forecast_error():
    common = {
        "reserve_min": 0,
        "reserve_max": 8,
        "additive_step": 2,
        "multiplicative_decay": 0.5,
        "warmup_versions": 0,
        "wait_high": 2.0,
        "overload_high": 0.25,
    }

    assert compute_closed_loop_reserve(2, 3, 4.0, 0.0, **common) == (
        4,
        1,
    )
    assert compute_closed_loop_reserve(6, 3, 4.0, 0.5, **common) == (
        2,
        2,
    )
    assert compute_closed_loop_reserve(4, 3, 1.0, 0.0, **common) == (
        4,
        0,
    )


def test_state_feedback_reserve_distinguishes_supply_and_queue_pressure():
    common = {
        "reserve_min": 0,
        "reserve_max": 8,
        "additive_step": 1,
        "warmup_versions": 0,
        "starvation_high": 0.10,
        "waste_high": 0.25,
        "queue_high": 0.20,
        "tool_wait_high": 0.50,
        "learner_demand": 4,
    }

    assert compute_state_feedback_reserve(
        3, 4, 0.30, 0.05, 0.05, 1, 0.20, **common
    ) == (4, 1)
    assert compute_state_feedback_reserve(
        3, 4, 0.30, 0.05, 0.30, 1, 0.20, **common
    ) == (3, 7)
    assert compute_state_feedback_reserve(
        3, 4, 0.30, 0.40, 0.05, 1, 0.20, **common
    ) == (3, 8)
    assert compute_state_feedback_reserve(
        3, 4, 0.30, 0.05, 0.05, 1, 0.80, **common
    ) == (4, 9)
    assert compute_state_feedback_reserve(
        3, 4, 0.01, 0.40, 0.05, 4, 0.20, **common
    ) == (2, 2)
    assert compute_state_feedback_reserve(
        3, 4, 0.01, 0.05, 0.30, 4, 0.20, **common
    ) == (2, 10)


def test_dynamic_reserve_trace_converges_into_deadband():
    common = {
        "reserve_min": 0,
        "reserve_max": 8,
        "additive_step": 2,
        "multiplicative_decay": 0.5,
        "warmup_versions": 0,
        "wait_high": 2.0,
        "stale_high": 0.25,
        "prediction_error_margin": 1.0,
    }
    trace = [
        (8.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (0.5, 1.0, -2.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ]
    reserve = 0
    observed = []
    for version, (wait, stale, error) in enumerate(trace):
        reserve, _ = compute_dynamic_reserve(
            reserve, version, wait, stale, error, **common
        )
        observed.append(reserve)

    assert observed == [2, 4, 2, 2, 2]


def test_dynamic_reserve_hysteresis_requires_confirmation_and_cools_down():
    state = (4, 0, 0, 0)
    observed = []
    for candidate, reason in [(6, 1), (6, 1), (8, 1), (8, 1), (2, 3), (2, 3)]:
        reserve, pending_direction, pending_count, cooldown = state
        updated, applied_reason, pending_direction, pending_count, cooldown = (
            apply_dynamic_reserve_hysteresis(
                reserve,
                candidate,
                reason,
                pending_direction,
                pending_count,
                cooldown,
                signal_patience=2,
                cooldown_versions=2,
            )
        )
        state = (updated, pending_direction, pending_count, cooldown)
        observed.append((updated, applied_reason, pending_count, cooldown))

    assert observed == [
        (4, 6, 1, 0),
        (6, 1, 0, 2),
        (6, 6, 0, 1),
        (6, 6, 0, 0),
        (6, 6, 1, 0),
        (2, 3, 0, 2),
    ]


def test_effective_rollout_utility_allows_idle_but_penalizes_stale_work():
    utility, useful_rate, stale_rate, compute_efficiency = compute_effective_rollout_utility(
        consumed_response_tokens=100,
        consumed_inference_tokens=1000,
        stale_inference_tokens=200,
        elapsed_seconds=2.0,
        waste_weight=1.5,
    )

    assert useful_rate == 50.0
    assert stale_rate == 100.0
    assert compute_efficiency == pytest.approx(1000 / 1300)
    assert utility == pytest.approx(50 * 1000 / 1300)


def test_utility_hill_climb_reverses_one_step_after_regression():
    first = update_utility_hill_climb(
        reserve=4,
        direction=1,
        utility=100.0,
        previous_utility=None,
        reserve_min=0,
        reserve_max=8,
        additive_step=2,
        improvement_margin=0.05,
    )
    regressed = update_utility_hill_climb(
        reserve=first[0],
        direction=first[1],
        utility=80.0,
        previous_utility=100.0,
        reserve_min=0,
        reserve_max=8,
        additive_step=2,
        improvement_margin=0.05,
    )

    assert first == (6, 1, 7)
    assert regressed == (4, -1, 9)


def test_constrained_utility_controller_reduces_reserve_below_efficiency_floor():
    updated = update_constrained_utility_hill_climb(
        reserve=6,
        direction=1,
        utility=120.0,
        previous_utility=100.0,
        compute_efficiency=0.80,
        min_compute_efficiency=0.95,
        reserve_min=0,
        reserve_max=8,
        additive_step=2,
        improvement_margin=0.05,
    )

    assert updated == (4, -1, 11)


def test_constrained_utility_controller_optimizes_after_efficiency_guard_passes():
    updated = update_constrained_utility_hill_climb(
        reserve=4,
        direction=1,
        utility=110.0,
        previous_utility=100.0,
        compute_efficiency=0.98,
        min_compute_efficiency=0.95,
        reserve_min=0,
        reserve_max=8,
        additive_step=2,
        improvement_margin=0.05,
    )

    assert updated == (6, 1, 8)


def test_progress_floor_only_admits_minimum_bounded_supply():
    assert compute_progress_topup_groups(4, 0, 0, 24, 2, 2) == 2
    assert compute_progress_topup_groups(4, 2, 22, 24, 2, 2) == 1
    assert compute_progress_topup_groups(4, 4, 0, 24, 2, 2) == 0


def test_finish_rate_bucket_captures_version_age_and_action_progress():
    assert finish_rate_bucket(0, 0) == "age_0__actions_0"
    assert finish_rate_bucket(2, 3) == "age_2__actions_2_3"
    assert finish_rate_bucket(3, 7) == "age_3__actions_4_7"
    assert finish_rate_bucket(9, 12) == "age_ge_4__actions_ge_8"


def test_bucketed_finish_predictor_learns_and_falls_back_per_cohort():
    ratios = {}
    samples = {}
    update_bucketed_finish_ratios(
        ratios,
        samples,
        {"age_1__actions_0": 4, "age_1__actions_4_7": 4},
        {"age_1__actions_0": 0, "age_1__actions_4_7": 4},
        ewma_alpha=0.5,
    )

    expected, learned, fallback = predict_bucketed_finish_supply(
        {
            "age_1__actions_0": 2,
            "age_1__actions_4_7": 2,
            "age_2__actions_1": 2,
        },
        ratios,
        samples,
        fallback_ratio=0.5,
        min_bucket_samples=4,
    )

    assert ratios == {
        "age_1__actions_0": 0.0,
        "age_1__actions_4_7": 1.0,
    }
    assert samples == {"age_1__actions_0": 4, "age_1__actions_4_7": 4}
    assert expected == 3.0
    assert learned == 4
    assert fallback == 2


def test_version_adaptive_completion_is_attributed_before_learner_consumption():
    manager_cls = GroupQueueManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.admission_policy = "version_adaptive"
    manager.group_size = 2
    manager._tracked_unfinished_groups = {(3, 7)}
    manager._tracked_unfinished_group_buckets = {
        (3, 7): "age_1__actions_4_7",
    }
    manager._tracked_unfinished_completed = 0
    manager._tracked_unfinished_bucket_completed = {}

    manager._record_version_adaptive_completion(3, 7)

    assert manager._tracked_unfinished_completed == 2
    assert manager._tracked_unfinished_bucket_completed == {
        "age_1__actions_4_7": 2,
    }


def test_pending_group_gets_backfill_missing_queues():
    async def run_test():
        manager_cls = GroupQueueManager.__ray_metadata__.modified_class
        manager = manager_cls.__new__(manager_cls)
        manager.rollout_complete = {}

        gate = asyncio.Event()

        class BlockingQueue:
            async def get(self):
                await gate.wait()

        manager.group_queue = {0: BlockingQueue(), 14: BlockingQueue()}
        existing = asyncio.create_task(
            manager.group_queue[14].get(), name="14"
        )
        manager.pending_gets = {existing}

        pending = manager._take_pending_group_gets()

        assert {task.get_name() for task in pending} == {"0", "14"}
        assert manager.pending_gets == set()

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(run_test())


def test_trainable_frontier_uses_group_size_order_statistic():
    queue = GroupQueue.__new__(GroupQueue)
    queue.group_id = 1
    queue.group_size = 2
    queue.progress_snapshots = {}
    group = type(
        "Group",
        (),
        {"group_id": 1, "episode_id": 8, "create_step": 0, "rollouts": []},
    )()
    queue.update_progress_snapshots([
        {"group_id": 1, "episode_id": 8, "env_id": 0, "actions_completed": 2},
        {"group_id": 1, "episode_id": 8, "env_id": 1, "actions_completed": 7},
        {"group_id": 1, "episode_id": 8, "env_id": 2, "actions_completed": 10},
    ])

    # With group_size=2 and one redundant candidate, the second-highest progress
    # is the frontier that determines whether the group can become trainable.
    assert queue.trainable_frontier_actions(group) == 7
    assert queue.trainable_progress_summary(group) == {
        "mean_actions": 8,
        "frontier_actions": 7,
        "max_actions": 10,
        "observed_candidates": 2,
        "gpu_invested_candidates": 0,
    }


def test_utility_settle_excludes_transition_observations():
    assert consume_utility_settle(2) == (False, 1)
    assert consume_utility_settle(1) == (False, 0)
    assert consume_utility_settle(0) == (True, 0)

class MockGroupFilter(GroupFilter):
    def filter(self, group_id: int, episode_id: int, group: list[DataProto]):
        return episode_id % 3 == 0

@dataclass
class MockAgenticConfig:
    async_generation_ratio: int
    rollout_batch_size: int
    env_monitor: EnvMonitorConfig = field(default_factory=lambda: EnvMonitorConfig(enable=False))

class MockEnvManagerConfig:
    def __init__(
        self,
        world_size,
        env_groups,
        group_size,
        group_size_redundancy,
        rollout_batch_size,
        enable_filter,
        enable_redundancy,
    ):
        self.world_size = world_size
        self.env_groups = env_groups
        self.group_size = group_size
        self.group_size_redundancy = group_size_redundancy if enable_redundancy else 0
        self.final_group_size = group_size + self.group_size_redundancy
        if enable_filter:
            self.group_filter_cls = "tests.distributed.scheduler.test_rollout_scheduler.MockGroupFilter"
        else:
            self.group_filter_cls = "roll.pipeline.agentic.agentic_pipeline.GroupFilter"

        train_env_num = self.env_groups * self.group_size

        self.max_traj_per_env = (rollout_batch_size + train_env_num - 1) // train_env_num

        self.max_env_num_per_worker = self.env_groups * self.final_group_size
        self.env_num = self.world_size * self.max_env_num_per_worker
        self.env_configs = {0: {i: {"group_id": i} for i in range(env_groups)}}
        print(f"config: {self.env_num=} {self.world_size=} {self.max_env_num_per_worker=} {self.max_traj_per_env=}")

class MockEnvironmentWorker:
    def __init__(self, thread_id, gropu_id, output_queue):
        self.thread_id = thread_id
        self.group_id = gropu_id
        self.output_queue = output_queue
        self.current_step = None

    def run_rollout_loop(self, full_dataset):
        iter = 0
        while True:
            iter += 1
            episode_id = ray.get(self.output_queue.get_episode_id.remote(self.group_id))
            if episode_id is None:
                print("Env worker exit on episode_id is None")
                break
            elif full_dataset and episode_id == FULL_DATASET_ITER:
                print("Env worker exit on traverse all dataset")
                break
            else:
                start_step = self.current_step
            assert start_step is not None
            rollout = DataProto(meta_info={"rollout": (start_step, episode_id)})
            ray.get(self.output_queue.put.remote(self.group_id, episode_id, start_step, rollout))
        ray.get(self.output_queue.put.remote(self.group_id, episode_id, start_step, None))

class MockEnvManager(Worker):
    def __init__(self, env_manager_config, env_output_queue, full_dataset):
        assert env_manager_config.world_size == 1
        self.output_queue = env_output_queue
        self.full_dataset = full_dataset
        self.workers = [
            MockEnvironmentWorker(thread_id=i, gropu_id=i // env_manager_config.final_group_size, output_queue=env_output_queue)
            for i in range(env_manager_config.env_num)
        ]

    def stop(self, blocking=False):
        async def _stop():
            for worker in self.workers:
                pass
        return [_stop()]

    def update_step(self, step, blocking=False):
        async def _update_step():
            for worker in self.workers:
                worker.current_step = step
        return [_update_step()]

    def run_rollout_loop(self, seed, blocking=False):
        async def _run_rollout_loop():
            loop = asyncio.get_event_loop()
            pool = ThreadPoolExecutor(max_workers=len(self.workers))
            await asyncio.gather(*[loop.run_in_executor(pool, worker.run_rollout_loop, self.full_dataset) for worker in self.workers])
            pool.shutdown()
        return [_run_rollout_loop()]

@ray.remote
class MockRequestScheduler:
    def __init__(self, *args, **kwargs):
        pass

    async def suspend(self):
        pass

    async def resume(self):
        pass

    async def abort_request(self):
        pass

@ray.remote
class MockRolloutScheduler(RolloutScheduler):
    def __init__(self, config, env_manager_config, mode):
        self.config = config
        self.env_manager_config = env_manager_config
        self.mode = mode

        env_num = self.env_manager_config.world_size * self.env_manager_config.max_env_num_per_worker

        self.env_output_queue = GroupQueueManager.options(
            max_concurrency=2 * env_num + 2,
        ).remote(
            self.config,
            self.env_manager_config,
            mode
        )

        self.generate_scheduler = MockRequestScheduler.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(),
                    soft=False,
                ),
                max_concurrency = env_num + 1 # reserve extra one for suspend/resume
            ).remote()

        self.es_manager = MockEnvManager(
            env_manager_config=self.env_manager_config,
            env_output_queue=self.env_output_queue,
            full_dataset=config.rollout_batch_size<=0,
        )

        self.rollout_task = None

    async def suspend(self):
        await self.generate_scheduler.suspend.remote()

    async def shutdown(self):
        if self.rollout_task is None:
            return
        await asyncio.gather(*self.es_manager.stop(blocking=False))
        await self.env_output_queue.shutdown.remote()
        await self.rollout_task
        self.rollout_task = None

    # FIXME use RolloutScheduler.get_batch
    async def get_batch(self, data: DataProto, batch_size):
        global_step = data.meta_info["global_step"]

        # start env manager
        if self.rollout_task is None:
            seed = random.randint(0, 1000000) if self.mode == "train" else self.config.seed
            self.rollout_task = asyncio.create_task(self._run_rollout_loop(seed))

        await asyncio.gather(*self.es_manager.update_step(global_step, blocking=False))
        await self.env_output_queue.advance_step.remote(global_step)
        await self.generate_scheduler.resume.remote()

        get_task = asyncio.create_task(self._get_batch(batch_size, global_step))
        await asyncio.wait({get_task, self.rollout_task}, return_when=asyncio.FIRST_COMPLETED)
        if self.rollout_task.done() and self.rollout_task.exception() is not None:
            await self.rollout_task
            assert False
        data_batch = await get_task
        if batch_size <= 0:
            await self.rollout_task
            self.rollout_task = None
            await self.env_output_queue.clear.remote()
        return data_batch

async def async_test_GroupQueueManager(rollout_batch_size, async_generation_ratio, enable_filter=True, enable_redundancy=True):
    print(f">>>>>>>>>>>>>>>>>>>>>>>> TEST rollout_batch_size {rollout_batch_size} async_generation_ratio {async_generation_ratio}")
    config = MockAgenticConfig(rollout_batch_size=rollout_batch_size, async_generation_ratio=async_generation_ratio)

    env_manager_config = MockEnvManagerConfig(
        world_size=1,
        env_groups=2,
        group_size=8, # grpo
        group_size_redundancy=4,
        rollout_batch_size=rollout_batch_size,
        enable_filter=enable_filter,
        enable_redundancy=enable_redundancy,
    )

    scheduler = MockRolloutScheduler.remote(config, env_manager_config, "train")

    for i in range(10):
        current_step = i
        data = DataProto(meta_info={"global_step": current_step})
        await scheduler.suspend.remote()
        batch = await scheduler.get_batch.remote(data=data, batch_size=rollout_batch_size)

        rollout_steps = [rollout.meta_info["rollout"][0] for rollout in batch]
        print(f"batch on step({current_step}): {rollout_steps}")
        expected = FULL_DATASET_ITER * env_manager_config.env_groups * env_manager_config.group_size if rollout_batch_size <= 0 else rollout_batch_size
        assert len(batch) == expected, f"{len(batch)=} expected={expected}"
        assert all(step == rollout_steps[0] for step in rollout_steps), "Not all start_step are equal"
        assert (
            all(max(0, current_step - async_generation_ratio) == step for step in rollout_steps)
        ), f"current_step({current_step}) - rollout_step({rollout_steps[0]}) exceed async_generation_ratio"

        await asyncio.sleep(1)
    await scheduler.shutdown.remote()

async def _run_GroupQueueManager():
    # default_setting:
    #   env_num=16

    # batch_size = -1
    await async_test_GroupQueueManager(-1, 0, enable_filter=False, enable_redundancy=False)

    # sync training
    await async_test_GroupQueueManager(16, 0)
    await async_test_GroupQueueManager(8, 0)
    await async_test_GroupQueueManager(24, 0)
    await async_test_GroupQueueManager(32, 0)
    await async_test_GroupQueueManager(64, 0)

    # async training: 2
    await async_test_GroupQueueManager(16, 2)
    await async_test_GroupQueueManager(8, 2)
    # do not test batch_size 12, because 12 % group_size != 0
    await async_test_GroupQueueManager(24, 2)
    await async_test_GroupQueueManager(32, 2)
    await async_test_GroupQueueManager(64, 2)

    # async training: 7
    await async_test_GroupQueueManager(16, 7)
    await async_test_GroupQueueManager(8, 7)

    # async training: 1
    await async_test_GroupQueueManager(16, 1)
    await async_test_GroupQueueManager(8, 1)
    await async_test_GroupQueueManager(24, 1)
    await async_test_GroupQueueManager(32, 1)
    await async_test_GroupQueueManager(64, 1)


# Takes about 10 minutes in NPU CI, so skip this full queue-manager matrix there.
@pytest.mark.skip_on_npu
def test_GroupQueueManager():
    asyncio.run(_run_GroupQueueManager())

if __name__ == "__main__":
    test_GroupQueueManager()
