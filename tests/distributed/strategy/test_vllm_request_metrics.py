from types import SimpleNamespace

import pytest

from roll.distributed.strategy.vllm_strategy import request_timing_metrics


def test_request_timing_metrics_separates_queue_ttft_decode_and_model_time():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1, 2, 3, 4])],
        metrics=SimpleNamespace(
            arrival_time=10.0,
            first_scheduled_time=10.2,
            first_token_time=11.5,
            finished_time=13.0,
            time_in_queue=0.25,
            scheduler_time=0.1,
            model_forward_time=0.8,
            model_execute_time=1.2,
        ),
    )

    metrics = request_timing_metrics(output)

    assert metrics["vllm/request_output_tokens"] == 4
    assert metrics["vllm/request_queue_seconds"] == pytest.approx(0.25)
    assert metrics["vllm/request_ttft_seconds"] == pytest.approx(1.5)
    assert metrics["vllm/request_decode_seconds"] == pytest.approx(1.5)
    assert metrics["vllm/request_latency_seconds"] == pytest.approx(3.0)
    assert metrics["vllm/request_model_execute_seconds"] == pytest.approx(1.2)
    assert metrics["vllm/request_decode_tokens_per_second"] == pytest.approx(2.0)


def test_request_timing_metrics_derives_queue_time_from_vllm_v1_timestamps():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1, 2])],
        metrics=SimpleNamespace(
            arrival_time=10.0,
            first_scheduled_time=12.5,
            first_token_time=13.0,
            finished_time=14.0,
            time_in_queue=0.0,
            scheduler_time=None,
            model_forward_time=None,
            model_execute_time=None,
        ),
        roll_request_timing_metrics={"queue_seconds": 0.0},
    )

    metrics = request_timing_metrics(output)

    assert metrics["vllm/request_queue_seconds"] == pytest.approx(2.5)


def test_request_timing_metrics_uses_v1_frontend_and_engine_hooks():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[1, 2, 3])],
        metrics=None,
        roll_request_timing_metrics={
            "queue_seconds": 0.2,
            "ttft_seconds": 0.7,
            "prefill_seconds": 0.4,
            "decode_seconds": 0.5,
            "inference_seconds": 0.9,
            "latency_seconds": 1.2,
        },
        roll_request_engine_metrics={
            "engine_step_seconds_attributed": 0.6,
            "prefill_engine_step_seconds_attributed": 0.3,
            "decode_engine_step_seconds_attributed": 0.3,
            "scheduled_tokens": 103,
        },
    )

    metrics = request_timing_metrics(output)

    assert metrics["vllm/request_ttft_seconds"] == pytest.approx(0.7)
    assert metrics["vllm/request_prefill_seconds"] == pytest.approx(0.4)
    assert metrics["vllm/request_inference_seconds"] == pytest.approx(0.9)
    assert metrics[
        "vllm/request_engine_step_seconds_attributed"
    ] == pytest.approx(0.6)
    assert metrics[
        "vllm/request_prefill_engine_step_seconds_attributed"
    ] == pytest.approx(0.3)
