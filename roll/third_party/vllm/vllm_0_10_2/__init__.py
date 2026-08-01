"""Compatibility hooks for vLLM 0.10.2."""

from typing import Any, Dict

from vllm.outputs import RequestOutput
from vllm.v1.engine.output_processor import RequestState


def request_state_timing_metrics(stats: Any) -> Dict[str, float]:
    """Expose V1 queue and model-stage timings kept on RequestState."""
    if stats is None:
        return {}

    queued_ts = float(getattr(stats, "queued_ts", 0.0) or 0.0)
    scheduled_ts = float(getattr(stats, "scheduled_ts", 0.0) or 0.0)
    first_token_ts = float(getattr(stats, "first_token_ts", 0.0) or 0.0)
    last_token_ts = float(getattr(stats, "last_token_ts", 0.0) or 0.0)
    timing: Dict[str, float] = {}

    if queued_ts > 0.0 and scheduled_ts > 0.0:
        timing["queue_seconds"] = max(0.0, scheduled_ts - queued_ts)
    if scheduled_ts > 0.0 and first_token_ts > 0.0:
        timing["prefill_seconds"] = max(0.0, first_token_ts - scheduled_ts)
        if last_token_ts > 0.0:
            timing["inference_seconds"] = max(
                0.0, last_token_ts - scheduled_ts
            )
    if first_token_ts > 0.0 and last_token_ts > 0.0:
        timing["decode_seconds"] = max(0.0, last_token_ts - first_token_ts)

    first_token_latency = float(
        getattr(stats, "first_token_latency", 0.0) or 0.0
    )
    if first_token_latency > 0.0:
        timing["ttft_seconds"] = first_token_latency
    return timing


def _install_request_timing_patch() -> None:
    original = RequestState._new_request_output
    if getattr(original, "_roll_timing_patch", False):
        return

    def _new_request_output(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        if isinstance(output, RequestOutput):
            timing = request_state_timing_metrics(getattr(self, "stats", None))
            if timing:
                output.roll_request_timing_metrics = timing
        return output

    setattr(_new_request_output, "_roll_timing_patch", True)
    RequestState._new_request_output = _new_request_output


_install_request_timing_patch()
