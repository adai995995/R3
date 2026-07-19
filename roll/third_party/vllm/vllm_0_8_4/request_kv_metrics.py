"""Backport per-request prefix-cache accounting for vLLM 0.8.4 V1."""

import logging
import time
from typing import Any, Optional

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCore, EngineCoreProc


logger = logging.getLogger(__name__)

_CACHE_STATE_ATTR = "_roll_request_cached_tokens"
_BATCH_STATE_ATTR = "_roll_request_batch_metadata"
_BATCH_COUNTER_ATTR = "_roll_scheduler_batch_id"
_PATCH_MARKER = "_roll_request_kv_metrics_patch"
_UTILITY_METHOD = "roll_pop_request_kv_metrics"
_ENGINE_INPUT_PATCH_MARKER = "_roll_rebuild_input_coalesce_patch"
MARK_REBUILD_REQUEST_UTILITY = "roll_mark_rebuild_request"
_REBUILD_REQUEST_IDS_ATTR = "_roll_rebuild_request_ids"
REBUILD_INPUT_COALESCE_SECONDS = 0.005


def mark_rebuild_request(engine_core: Any, request_id: str) -> bool:
    """Mark a request for one idle-engine admission coalescing window."""
    request_ids = getattr(engine_core, _REBUILD_REQUEST_IDS_ATTR, None)
    if request_ids is None:
        request_ids = set()
        setattr(engine_core, _REBUILD_REQUEST_IDS_ATTR, request_ids)
    request_ids.add(str(request_id))
    return True


def consume_rebuild_add_message(engine_core: Any, message: Any) -> bool:
    """Consume the marker when its EngineCore ADD message arrives."""
    if not isinstance(message, (tuple, list)) or len(message) < 2:
        return False
    request = message[1]
    request_id = getattr(request, "request_id", None)
    request_ids = getattr(engine_core, _REBUILD_REQUEST_IDS_ATTR, None)
    if request_id is None or not request_ids or request_id not in request_ids:
        return False
    request_ids.remove(request_id)
    return True


def record_scheduled_request_cached_tokens(
    scheduler: Any,
    scheduler_output: Any,
) -> None:
    """Record the exact initial computed-prefix length for new requests."""
    cached_tokens = getattr(scheduler, _CACHE_STATE_ATTR, None)
    if cached_tokens is None:
        cached_tokens = {}
        setattr(scheduler, _CACHE_STATE_ATTR, cached_tokens)

    new_requests = list(getattr(scheduler_output, "scheduled_new_reqs", ()))
    if not new_requests:
        return
    batch_id = int(getattr(scheduler, _BATCH_COUNTER_ATTR, 0)) + 1
    setattr(scheduler, _BATCH_COUNTER_ATTR, batch_id)
    batch_metadata = getattr(scheduler, _BATCH_STATE_ATTR, None)
    if batch_metadata is None:
        batch_metadata = {}
        setattr(scheduler, _BATCH_STATE_ATTR, batch_metadata)

    for request in new_requests:
        cached_tokens[request.req_id] = max(
            0,
            int(request.num_computed_tokens),
        )
        batch_metadata[request.req_id] = {
            "scheduler_batch_id": batch_id,
            "scheduler_batch_size": len(new_requests),
        }


def pop_request_cached_tokens(engine_core: Any, request_id: str) -> Optional[int]:
    """Return and remove one request's initial cached-token measurement."""
    cached_tokens = getattr(engine_core.scheduler, _CACHE_STATE_ATTR, None)
    if cached_tokens is None:
        return None
    value = cached_tokens.pop(request_id, None)
    return None if value is None else int(value)


def pop_request_kv_metrics(engine_core: Any, request_id: str) -> dict[str, Any]:
    """Return one request's cache hit and actual scheduler-batch metadata."""
    cached_tokens = pop_request_cached_tokens(engine_core, request_id)
    batch_metadata = getattr(engine_core.scheduler, _BATCH_STATE_ATTR, None)
    batch = batch_metadata.pop(request_id, {}) if batch_metadata is not None else {}
    return {
        "cached_tokens": cached_tokens,
        "scheduler_batch_id": batch.get("scheduler_batch_id"),
        "scheduler_batch_size": batch.get("scheduler_batch_size"),
    }


def clear_request_cached_tokens(engine_core: Any, request_ids: list[str]) -> None:
    """Remove measurements for requests aborted before their first output."""
    cached_tokens = getattr(engine_core.scheduler, _CACHE_STATE_ATTR, None)
    if cached_tokens is not None:
        for request_id in request_ids:
            cached_tokens.pop(request_id, None)
    batch_metadata = getattr(engine_core.scheduler, _BATCH_STATE_ATTR, None)
    if batch_metadata is not None:
        for request_id in request_ids:
            batch_metadata.pop(request_id, None)


async def populate_request_cached_tokens(
    async_llm: Any,
    request_id: str,
    request_output: Any,
) -> None:
    """Populate RequestOutput.num_cached_tokens before its first yield."""
    try:
        request_metrics = await async_llm.engine_core.call_utility_async(
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

    if not isinstance(request_metrics, dict):
        request_metrics = {"cached_tokens": request_metrics}
    cached_tokens = request_metrics.get("cached_tokens")
    if (
        getattr(request_output, "num_cached_tokens", None) is None
        and cached_tokens is not None
    ):
        request_output.num_cached_tokens = max(0, int(cached_tokens))
    batch_id = request_metrics.get("scheduler_batch_id")
    batch_size = request_metrics.get("scheduler_batch_size")
    request_output.roll_scheduler_batch_id = (
        None if batch_id is None else int(batch_id)
    )
    request_output.roll_scheduler_batch_size = (
        None if batch_size is None else max(1, int(batch_size))
    )


def install_request_kv_metrics_patch() -> None:
    """Install the vLLM 0.8.4 V1 measurement hooks once per process."""
    if getattr(Scheduler.schedule, _PATCH_MARKER, False):
        return

    original_schedule = Scheduler.schedule
    original_abort_requests = EngineCore.abort_requests
    original_add_request = EngineCore.add_request
    original_process_input_queue = EngineCoreProc._process_input_queue

    def schedule_with_request_kv_metrics(self, *args, **kwargs):
        scheduler_output = original_schedule(self, *args, **kwargs)
        record_scheduled_request_cached_tokens(self, scheduler_output)
        return scheduler_output

    def abort_requests_with_kv_cleanup(self, request_ids):
        result = original_abort_requests(self, request_ids)
        clear_request_cached_tokens(self, request_ids)
        return result

    def add_request_with_rebuild_marker_cleanup(self, request):
        request_ids = getattr(self, _REBUILD_REQUEST_IDS_ATTR, None)
        if request_ids is not None:
            request_ids.discard(request.request_id)
        return original_add_request(self, request)

    def process_input_queue_with_rebuild_coalescing(self):
        if self.global_unfinished_reqs or self.scheduler.has_requests():
            return original_process_input_queue(self)

        rebuild_request_received = False
        while not self.global_unfinished_reqs and not self.scheduler.has_requests():
            message = self.input_queue.get()
            rebuild_request_received = (
                consume_rebuild_add_message(self, message)
                or rebuild_request_received
            )
            self._handle_client_request(*message)

        if rebuild_request_received:
            time.sleep(REBUILD_INPUT_COALESCE_SECONDS)
        while not self.input_queue.empty():
            message = self.input_queue.get_nowait()
            consume_rebuild_add_message(self, message)
            self._handle_client_request(*message)

    setattr(schedule_with_request_kv_metrics, _PATCH_MARKER, True)
    Scheduler.schedule = schedule_with_request_kv_metrics
    EngineCore.abort_requests = abort_requests_with_kv_cleanup
    EngineCore.add_request = add_request_with_rebuild_marker_cleanup
    setattr(
        process_input_queue_with_rebuild_coalescing,
        _ENGINE_INPUT_PATCH_MARKER,
        True,
    )
    EngineCoreProc._process_input_queue = process_input_queue_with_rebuild_coalescing
    setattr(EngineCore, _UTILITY_METHOD, pop_request_kv_metrics)
    setattr(EngineCore, MARK_REBUILD_REQUEST_UTILITY, mark_rebuild_request)


__all__ = [
    "clear_request_cached_tokens",
    "consume_rebuild_add_message",
    "install_request_kv_metrics_patch",
    "populate_request_cached_tokens",
    "pop_request_kv_metrics",
    "pop_request_cached_tokens",
    "record_scheduled_request_cached_tokens",
    "mark_rebuild_request",
    "MARK_REBUILD_REQUEST_UTILITY",
]
