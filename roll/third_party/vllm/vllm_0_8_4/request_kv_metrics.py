"""Backport per-request prefix-cache accounting for vLLM 0.8.4 V1."""

import logging
from typing import Any, Optional

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore


logger = logging.getLogger(__name__)

_CACHE_STATE_ATTR = "_roll_request_cached_tokens"
_PATCH_MARKER = "_roll_request_kv_metrics_patch"
_UTILITY_METHOD = "roll_pop_request_cached_tokens"


def record_scheduled_request_cached_tokens(
    scheduler: Any,
    scheduler_output: Any,
) -> None:
    """Record the exact initial computed-prefix length for new requests."""
    cached_tokens = getattr(scheduler, _CACHE_STATE_ATTR, None)
    if cached_tokens is None:
        cached_tokens = {}
        setattr(scheduler, _CACHE_STATE_ATTR, cached_tokens)

    for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        cached_tokens[request.req_id] = max(
            0,
            int(request.num_computed_tokens),
        )


def pop_request_cached_tokens(engine_core: Any, request_id: str) -> Optional[int]:
    """Return and remove one request's initial cached-token measurement."""
    cached_tokens = getattr(engine_core.scheduler, _CACHE_STATE_ATTR, None)
    if cached_tokens is None:
        return None
    value = cached_tokens.pop(request_id, None)
    return None if value is None else int(value)


def clear_request_cached_tokens(engine_core: Any, request_ids: list[str]) -> None:
    """Remove measurements for requests aborted before their first output."""
    cached_tokens = getattr(engine_core.scheduler, _CACHE_STATE_ATTR, None)
    if cached_tokens is None:
        return
    for request_id in request_ids:
        cached_tokens.pop(request_id, None)


async def populate_request_cached_tokens(
    async_llm: Any,
    request_id: str,
    request_output: Any,
) -> None:
    """Populate RequestOutput.num_cached_tokens before its first yield."""
    if getattr(request_output, "num_cached_tokens", None) is not None:
        return

    try:
        cached_tokens = await async_llm.engine_core.call_utility_async(
            _UTILITY_METHOD,
            request_id,
        )
    except Exception:
        if not getattr(async_llm, "_roll_kv_metric_warning_emitted", False):
            logger.warning(
                "Unable to fetch per-request cached-token metrics from "
                "the vLLM engine core.",
                exc_info=True,
            )
            async_llm._roll_kv_metric_warning_emitted = True
        return

    if cached_tokens is not None:
        request_output.num_cached_tokens = max(0, int(cached_tokens))


def install_request_kv_metrics_patch() -> None:
    """Install the vLLM 0.8.4 V1 measurement hooks once per process."""
    if getattr(Scheduler.schedule, _PATCH_MARKER, False):
        return

    original_schedule = Scheduler.schedule
    original_abort_requests = EngineCore.abort_requests

    def schedule_with_request_kv_metrics(self, *args, **kwargs):
        scheduler_output = original_schedule(self, *args, **kwargs)
        record_scheduled_request_cached_tokens(self, scheduler_output)
        return scheduler_output

    def abort_requests_with_kv_cleanup(self, request_ids):
        result = original_abort_requests(self, request_ids)
        clear_request_cached_tokens(self, request_ids)
        return result

    setattr(schedule_with_request_kv_metrics, _PATCH_MARKER, True)
    Scheduler.schedule = schedule_with_request_kv_metrics
    EngineCore.abort_requests = abort_requests_with_kv_cleanup
    setattr(EngineCore, _UTILITY_METHOD, pop_request_cached_tokens)


__all__ = [
    "clear_request_cached_tokens",
    "install_request_kv_metrics_patch",
    "populate_request_cached_tokens",
    "pop_request_cached_tokens",
    "record_scheduled_request_cached_tokens",
]
