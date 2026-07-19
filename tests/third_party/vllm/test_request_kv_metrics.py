from types import SimpleNamespace

import pytest

from roll.third_party.vllm.vllm_0_8_4.request_kv_metrics import (
    clear_request_cached_tokens,
    consume_rebuild_add_message,
    mark_rebuild_request,
    populate_request_cached_tokens,
    pop_request_cached_tokens,
    pop_request_kv_metrics,
    record_scheduled_request_cached_tokens,
)


def test_rebuild_engine_message_marker_is_request_scoped_and_one_shot():
    engine_core = SimpleNamespace()
    rebuild = ("add", SimpleNamespace(request_id="rebuild"))
    normal = ("add", SimpleNamespace(request_id="normal"))

    assert mark_rebuild_request(engine_core, "rebuild") is True
    assert consume_rebuild_add_message(engine_core, normal) is False
    assert consume_rebuild_add_message(engine_core, rebuild) is True
    assert consume_rebuild_add_message(engine_core, rebuild) is False
    assert consume_rebuild_add_message(engine_core, ("utility",)) is False


def test_records_only_new_request_initial_cached_tokens():
    scheduler = SimpleNamespace()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="new-hit", num_computed_tokens=96),
            SimpleNamespace(req_id="new-miss", num_computed_tokens=0),
        ],
        scheduled_cached_reqs=[
            SimpleNamespace(req_id="resumed", num_computed_tokens=160),
        ],
        finished_req_ids=set(),
    )

    record_scheduled_request_cached_tokens(scheduler, scheduler_output)

    assert scheduler._roll_request_cached_tokens == {
        "new-hit": 96,
        "new-miss": 0,
    }
    assert scheduler._roll_request_batch_metadata == {
        "new-hit": {"scheduler_batch_id": 1, "scheduler_batch_size": 2},
        "new-miss": {"scheduler_batch_id": 1, "scheduler_batch_size": 2},
    }


def test_abort_cleanup_and_pop_are_one_shot():
    scheduler = SimpleNamespace(
        _roll_request_cached_tokens={"aborted": 64, "live": 128},
        _roll_request_batch_metadata={
            "aborted": {"scheduler_batch_id": 1, "scheduler_batch_size": 2},
            "live": {"scheduler_batch_id": 1, "scheduler_batch_size": 2},
        },
    )
    engine_core = SimpleNamespace(scheduler=scheduler)
    clear_request_cached_tokens(engine_core, ["aborted"])

    assert pop_request_cached_tokens(engine_core, "aborted") is None
    assert pop_request_cached_tokens(engine_core, "live") == 128
    assert pop_request_cached_tokens(engine_core, "live") is None
    assert pop_request_kv_metrics(engine_core, "aborted")["scheduler_batch_id"] is None


@pytest.mark.asyncio
async def test_populates_request_output_from_engine_core():
    class FakeEngineCore:
        def __init__(self):
            self.calls = []

        async def call_utility_async(self, method, request_id):
            self.calls.append((method, request_id))
            return {
                "cached_tokens": 80,
                "scheduler_batch_id": 7,
                "scheduler_batch_size": 3,
            }

    engine_core = FakeEngineCore()
    async_llm = SimpleNamespace(engine_core=engine_core)
    request_output = SimpleNamespace(num_cached_tokens=None)

    await populate_request_cached_tokens(async_llm, "request-7", request_output)

    assert request_output.num_cached_tokens == 80
    assert request_output.roll_scheduler_batch_id == 7
    assert request_output.roll_scheduler_batch_size == 3
    assert engine_core.calls == [("roll_pop_request_kv_metrics", "request-7")]


@pytest.mark.asyncio
async def test_preserves_native_cached_token_measurement_and_adds_batch_metadata():
    class NativeEngineCore:
        async def call_utility_async(self, method, request_id):
            return {
                "cached_tokens": 12,
                "scheduler_batch_id": 9,
                "scheduler_batch_size": 4,
            }

    async_llm = SimpleNamespace(engine_core=NativeEngineCore())
    request_output = SimpleNamespace(num_cached_tokens=48)

    await populate_request_cached_tokens(async_llm, "request-8", request_output)

    assert request_output.num_cached_tokens == 48
    assert request_output.roll_scheduler_batch_id == 9
    assert request_output.roll_scheduler_batch_size == 4
