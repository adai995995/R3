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
_LAST_SCHEDULE_ATTR = "_roll_last_schedule_token_counts"
_ENGINE_TIMING_ATTR = "_roll_request_engine_step_timing"
_PATCH_MARKER = "_roll_request_kv_metrics_patch"
_ENGINE_STEP_PATCH_MARKER = "_roll_engine_step_timing_patch"
_UTILITY_METHOD = "roll_pop_request_kv_metrics"
POP_ENGINE_TIMING_UTILITY = "roll_pop_request_engine_timing"
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
    scheduled_request_data = {}
    requests = getattr(scheduler, "requests", {})
    for request_id, tokens in getattr(
        scheduler_output, "num_scheduled_tokens", {}
    ).items():
        request_id = str(request_id)
        scheduled_tokens = max(0, int(tokens))
        request = requests.get(request_id)
        # vLLM advances num_computed_tokens immediately before returning the
        # SchedulerOutput. Reconstruct its value at the start of this step.
        computed_before = max(
            0,
            int(getattr(request, "num_computed_tokens", scheduled_tokens))
            - scheduled_tokens,
        )
        prompt_tokens = int(
            getattr(request, "num_prompt_tokens", computed_before)
        )
        scheduled_request_data[request_id] = {
            "tokens": scheduled_tokens,
            "prefill_tokens": max(
                0,
                min(scheduled_tokens, prompt_tokens - computed_before),
            ),
        }
    setattr(scheduler, _LAST_SCHEDULE_ATTR, scheduled_request_data)
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


def record_engine_step_timing(engine_core: Any, elapsed_seconds: float) -> None:
    """Attribute one deduplicated engine step by scheduled-token share."""
    scheduled = getattr(engine_core.scheduler, _LAST_SCHEDULE_ATTR, {})
    setattr(engine_core.scheduler, _LAST_SCHEDULE_ATTR, {})
    total_tokens = sum(int(item.get("tokens", 0)) for item in scheduled.values())
    if total_tokens <= 0:
        return

    timings = getattr(engine_core, _ENGINE_TIMING_ATTR, None)
    if timings is None:
        timings = {}
        setattr(engine_core, _ENGINE_TIMING_ATTR, timings)
    elapsed = max(0.0, float(elapsed_seconds))
    for request_id, item in scheduled.items():
        scheduled_tokens = int(item.get("tokens", 0))
        token_share = scheduled_tokens / total_tokens
        attributed = elapsed * token_share
        prefill_tokens = min(
            scheduled_tokens,
            max(0, int(item.get("prefill_tokens", 0))),
        )
        prefill_share = (
            prefill_tokens / scheduled_tokens if scheduled_tokens else 0.0
        )
        request_timing = timings.setdefault(
            request_id,
            {
                "engine_step_seconds_attributed": 0.0,
                "prefill_engine_step_seconds_attributed": 0.0,
                "decode_engine_step_seconds_attributed": 0.0,
                "scheduled_tokens": 0,
            },
        )
        request_timing["engine_step_seconds_attributed"] += attributed
        request_timing[
            "prefill_engine_step_seconds_attributed"
        ] += attributed * prefill_share
        request_timing[
            "decode_engine_step_seconds_attributed"
        ] += attributed * (1.0 - prefill_share)
        request_timing["scheduled_tokens"] += scheduled_tokens


def pop_request_engine_timing(
    engine_core: Any, request_id: str
) -> dict[str, Any]:
    """Return and remove token-share-attributed V1 engine-step timing."""
    timings = getattr(engine_core, _ENGINE_TIMING_ATTR, None)
    if timings is None:
        return {}
    return dict(timings.pop(str(request_id), {}))


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
    timings = getattr(engine_core, _ENGINE_TIMING_ATTR, None)
    if timings is not None:
        for request_id in request_ids:
            timings.pop(request_id, None)


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
    original_engine_step = EngineCore.step
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

    def engine_step_with_timing(self, *args, **kwargs):
        started = time.perf_counter()
        output = original_engine_step(self, *args, **kwargs)
        record_engine_step_timing(self, time.perf_counter() - started)
        return output

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
    setattr(engine_step_with_timing, _ENGINE_STEP_PATCH_MARKER, True)
    EngineCore.step = engine_step_with_timing
    setattr(
        process_input_queue_with_rebuild_coalescing,
        _ENGINE_INPUT_PATCH_MARKER,
        True,
    )
    EngineCoreProc._process_input_queue = process_input_queue_with_rebuild_coalescing
    setattr(EngineCore, _UTILITY_METHOD, pop_request_kv_metrics)
    setattr(EngineCore, POP_ENGINE_TIMING_UTILITY, pop_request_engine_timing)
    setattr(EngineCore, MARK_REBUILD_REQUEST_UTILITY, mark_rebuild_request)


__all__ = [
    "clear_request_cached_tokens",
    "consume_rebuild_add_message",
    "install_request_kv_metrics_patch",
    "populate_request_cached_tokens",
    "pop_request_kv_metrics",
    "pop_request_cached_tokens",
    "pop_request_engine_timing",
    "record_engine_step_timing",
    "record_scheduled_request_cached_tokens",
    "mark_rebuild_request",
    "MARK_REBUILD_REQUEST_UTILITY",
    "POP_ENGINE_TIMING_UTILITY",
]
