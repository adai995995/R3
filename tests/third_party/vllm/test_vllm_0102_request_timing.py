from types import SimpleNamespace

import pytest

from roll.third_party.vllm.vllm_0_10_2 import request_state_timing_metrics


def test_request_state_timing_metrics_exposes_engine_queue_events():
    metrics = request_state_timing_metrics(
        SimpleNamespace(
            queued_ts=10.0,
            scheduled_ts=12.5,
            first_token_ts=13.0,
            last_token_ts=15.0,
            first_token_latency=3.0,
        )
    )

    assert metrics["queue_seconds"] == pytest.approx(2.5)
    assert metrics["prefill_seconds"] == pytest.approx(0.5)
    assert metrics["decode_seconds"] == pytest.approx(2.0)
    assert metrics["inference_seconds"] == pytest.approx(2.5)
    assert metrics["ttft_seconds"] == pytest.approx(3.0)


def test_request_state_timing_metrics_ignores_unobserved_events():
    metrics = request_state_timing_metrics(
        SimpleNamespace(
            queued_ts=0.0,
            scheduled_ts=0.0,
            first_token_ts=0.0,
            last_token_ts=0.0,
            first_token_latency=0.0,
        )
    )

    assert metrics == {}
