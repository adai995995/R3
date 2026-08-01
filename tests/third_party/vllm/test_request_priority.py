from roll.distributed.strategy.vllm_strategy import (
    request_priority_from_payload,
    request_priority_observation,
)


def test_request_priority_defaults_to_fcfs_value():
    assert request_priority_from_payload({}) == 0


def test_request_priority_reads_router_metadata():
    assert request_priority_from_payload({"_roll_request_priority": 37}) == 37


def test_request_priority_observation_distinguishes_engine_queueing():
    assert request_priority_observation(
        {"_roll_request_priority": 37},
        {"vllm/request_queue_seconds": 0.25},
    ) == {
        "vllm/request_priority": 37.0,
        "vllm/request_priority_enabled": 1,
        "vllm/request_priority_queued": 1,
        "vllm/request_priority_queue_seconds": 0.25,
    }


def test_request_priority_observation_ignores_handoff_noise_and_fcfs():
    assert request_priority_observation(
        {"_roll_request_priority": 0},
        {"vllm/request_queue_seconds": 0.0009},
    )["vllm/request_priority_queued"] == 0
    assert request_priority_observation(
        {}, {"vllm/request_queue_seconds": 1.0}
    )["vllm/request_priority_queue_seconds"] == 0.0
