from collections import defaultdict
from types import SimpleNamespace

import pytest

from roll.distributed.scheduler.router import (
    summarize_request_metric_totals,
    update_request_metric_totals,
)
from roll.distributed.scheduler.runtime_observation import (
    finalize_runtime_observation_report,
)


def _config(**overrides):
    values = {
        "exp_name": "matched-full",
        "rollout_batch_size": 8,
        "async_generation_ratio": 2,
        "trajectory_staleness_tolerance": 1,
        "trajectory_admission_policy": "version_adaptive",
        "trajectory_scheduling_policy": "version_priority",
        "max_outstanding_trajectories": 32,
        "adaptive_admission_reserve_trajectories": 4,
        "adaptive_admission_bucketed_finish_enabled": True,
        "version_adaptive_progress_floor_enabled": True,
        "dynamic_admission_reserve_enabled": False,
        "enable_checkpointing": False,
        "save_final_checkpoint": False,
        "router_args": {
            "router_config": {
                "post_update_rebuild_enabled": True,
                "working_set_routing_enabled": True,
                "soft_locality_enabled": True,
            }
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_finalize_report_uses_actual_consumed_count_and_labels_full_runtime():
    report = {
        "metrics": {
            "consumed/trajectories": 13,
            "terminal_waste/trajectories": 4,
            "async_waste/trajectories": 3,
        }
    }
    finalize_runtime_observation_report(report, _config(), 2)

    assert report["metrics"]["terminal_waste/consumed_trajectories"] == 13
    assert report["metrics"]["terminal_waste/waste_to_consumed_ratio"] == pytest.approx(
        4 / 13
    )
    assert report["experiment"]["runtime_variant"] == "full"
    assert report["experiment"]["version_adaptive_progress_floor_enabled"] is True
    assert report["experiment"]["checkpointing_enabled"] is False
    assert report["experiment"]["save_final_checkpoint"] is False


def test_finalize_report_labels_component_disabled_run_as_baseline():
    config = _config(
        trajectory_admission_policy="outstanding_watermark",
        trajectory_scheduling_policy="fifo",
        router_args={"router_config": {}},
    )
    report = {"metrics": {}}
    finalize_runtime_observation_report(report, config, 3)

    assert report["metrics"]["terminal_waste/consumed_trajectories"] == 24
    assert report["experiment"]["runtime_variant"] == "baseline"
    assert not any(report["experiment"]["runtime_components"].values())


def test_finalize_report_separates_placeholders_and_flags_metric_anomaly():
    report = {
        "metrics": {
            "consumed/trajectories": 4,
            "consumed/placeholder_trajectories": 2,
            "consumed/valid_inference_tokens": 0,
            "router_lifetime/vllm/request_prompt_tokens": 1000,
        }
    }
    finalize_runtime_observation_report(report, _config(), 1)

    assert report["metrics"]["consumed/valid_trajectories"] == 2
    assert report["metrics"]["consumed/valid_fraction"] == pytest.approx(0.5)
    assert report["data_quality"]["valid_consumed_zero_inference_tokens"] is True



def test_lifetime_router_metrics_survive_interval_collection():
    interval = defaultdict(float)
    lifetime = defaultdict(float)
    requests = [
        {
            "vllm/request_prompt_tokens": 100,
            "vllm/request_cached_prompt_tokens": 40,
            "vllm/request_kv_hit_ratio": 0.4,
            "vllm/engine_prefix_cache_query_blocks_delta": 10,
            "vllm/engine_prefix_cache_hit_blocks_delta": 4,
            "vllm/engine_prefix_cache_query_tokens_delta": 160,
            "vllm/engine_prefix_cache_cached_tokens_delta": 64,
            "router/scheduling_decisions": 1,
            "router/scheduling_wait_seconds": 0.2,
            "router/priority_queued_requests": 0,
            "router/priority_coalesced_requests": 1,
            "router/priority_reordered_requests": 0,
            "vllm/request_scheduler_batch_id": 7,
            "vllm/request_scheduler_batch_reported": 1,
            "vllm/request_scheduler_batch_size": 3,
        },
        {
            "vllm/request_prompt_tokens": 60,
            "vllm/request_cached_prompt_tokens": 20,
            "vllm/request_prefill_tokens": 40,
            "vllm/request_kv_hit_ratio": 1 / 3,
            "router/post_update_rebuild_request": 1,
            "router/scheduling_decisions": 1,
            "router/scheduling_wait_seconds": 0.4,
            "router/priority_queued_requests": 1,
            "router/priority_coalesced_requests": 1,
            "router/priority_reordered_requests": 1,
            "vllm/request_scheduler_batch_id": 8,
            "vllm/request_scheduler_batch_reported": 1,
            "vllm/request_scheduler_batch_size": 2,
        },
    ]
    for request in requests:
        update_request_metric_totals(interval, request)
        update_request_metric_totals(lifetime, request)

    interval_summary = summarize_request_metric_totals(
        dict(interval), scope="interval"
    )
    interval.clear()
    lifetime_summary = summarize_request_metric_totals(
        dict(lifetime), scope="lifetime"
    )

    assert interval_summary["vllm/interval_kv_hit_ratio"] == pytest.approx(60 / 160)
    assert lifetime_summary["vllm/lifetime_kv_hit_ratio"] == pytest.approx(60 / 160)
    assert "vllm/request_kv_hit_ratio" not in lifetime_summary
    assert lifetime_summary["router/rebuild_prompt_tokens"] == 60
    assert lifetime_summary["router/rebuild_prefill_tokens"] == 40
    assert lifetime_summary["router/kv_block_hit_ratio"] == pytest.approx(0.4)
    assert lifetime_summary["router/scheduling_wait_seconds_mean"] == pytest.approx(0.3)
    assert lifetime_summary["router/priority_queued_ratio"] == 0.5
    assert lifetime_summary["router/priority_coalesced_ratio"] == 1
    assert lifetime_summary["router/priority_reordered_ratio"] == 0.5
    assert lifetime_summary["vllm/request_scheduler_batch_size_mean"] == 2.5
    assert "vllm/request_scheduler_batch_id" not in lifetime_summary
