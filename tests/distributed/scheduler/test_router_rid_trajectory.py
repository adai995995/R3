"""Ensure generate rid matches trajectory_id for Phase D KV lease binding."""

import torch
from tensordict import TensorDict

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.scheduler.router import RouterClient


def _make_client() -> RouterClient:
    return RouterClient(
        proxy=None,
        meta={"strategy_name": "sglang", "eos_token_id": 1, "pad_token_id": 0},
    )


def _make_req(*, trajectory_id: str | None = "GSM8K_traj_1") -> DataProto:
    meta_info = {
        "generation_config": {
            "num_return_sequences": 1,
            "max_new_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "stop_strings": [],
        },
    }
    if trajectory_id is not None:
        meta_info["trajectory_id"] = trajectory_id
    return DataProto(
        batch=TensorDict(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            },
            batch_size=[1],
        ),
        meta_info=meta_info,
    )


def test_preprocess_generate_uses_trajectory_id_as_rid():
    client = _make_client()
    payload, rid = client._preprocess_generate(_make_req(), request_id=None)
    assert rid == "GSM8K_traj_1"
    assert payload["rid"] == "GSM8K_traj_1"
    assert payload["_roll_route_meta"]["trajectory_id"] == "GSM8K_traj_1"


def test_preprocess_generate_explicit_request_id_wins():
    client = _make_client()
    payload, rid = client._preprocess_generate(_make_req(), request_id="explicit-rid")
    assert rid == "explicit-rid"
    assert payload["rid"] == "explicit-rid"


def test_preprocess_generate_falls_back_to_uuid_without_trajectory_id():
    client = _make_client()
    _, rid = client._preprocess_generate(_make_req(trajectory_id=None), request_id=None)
    assert isinstance(rid, str)
    assert rid != ""
    assert rid != "GSM8K_traj_1"


def test_postprocess_generate_forwards_lookup_meta():
    client = _make_client()
    req = _make_req()
    response = {
        "finish_reasons": ["stop"],
        "output_token_ids": [[1, 2]],
        "lookup_resume_found": 1.0,
        "lookup_hit_tokens": 512.0,
        "lookup_cache_confidence": 0.8,
    }
    out = client._postprocess_generate(req, response)
    assert out.meta_info["lookup_resume_found"] == 1.0
    assert out.meta_info["lookup_hit_tokens"] == 512.0
    assert out.meta_info["lookup_cache_confidence"] == 0.8


def test_lookup_route_meta_response_keys_cover_phase_d_telemetry():
    from roll.distributed.scheduler.router import _LOOKUP_ROUTE_META_RESPONSE_KEYS

    assert "lookup_resume_found" in _LOOKUP_ROUTE_META_RESPONSE_KEYS
    assert "lookup_hit_tokens" in _LOOKUP_ROUTE_META_RESPONSE_KEYS
