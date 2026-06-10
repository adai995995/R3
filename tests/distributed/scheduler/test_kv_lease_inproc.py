"""Inproc SGLang worker URL helpers and system-cost TTL floor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from roll.distributed.scheduler.kv_lease_client import (
    is_inproc_worker_url,
    lookup_result_from_worker_payload,
)
from roll.distributed.scheduler.router import EnvAffinityRouter
from roll.distributed.scheduler.trajectory_value import (
    BeliefLevel,
    LeaseTtlWeights,
    SystemCostWeights,
    compute_system_lease_ttl,
)


def test_is_inproc_worker_url():
    assert is_inproc_worker_url("inproc://sglang-engine/dp3")
    assert not is_inproc_worker_url("http://127.0.0.1:8000")
    assert not is_inproc_worker_url(None)


def test_lookup_result_from_worker_payload():
    out = lookup_result_from_worker_payload(
        {"found": True, "hit_tokens": 12, "cache_confidence": 0.9, "lease_remaining_s": 1.5}
    )
    assert out.found is True
    assert out.hit_tokens == 12
    assert out.cache_confidence == pytest.approx(0.9)


def test_system_cost_lease_ttl_respects_t_tool_min_for_tiny_waits():
    meta = {"history_len_tokens": 8000}
    ttl, _ = compute_system_lease_ttl(
        meta,
        p_hit=0.85,
        t_tool_s=0.003,
        belief_level=BeliefLevel.HOT,
        weights=SystemCostWeights(memory_cost=1.0, memory_pressure_default=1.0),
        lease_weights=LeaseTtlWeights(t_tool_min=2.0, t_tool_max=20.0),
    )
    assert ttl >= 2.0


def test_env_affinity_router_inproc_push_and_lookup():
    worker = MagicMock()
    worker.set_kv_lease.remote = AsyncMock(
        return_value={"ok": True, "trajectory_id": "traj_a"}
    )
    worker.lookup_kv_resume.remote = AsyncMock(
        return_value={
            "found": True,
            "trajectory_id": "traj_a",
            "lease_remaining_s": 1.0,
            "hit_tokens": 0,
            "cache_confidence": 0.85,
        }
    )

    router = EnvAffinityRouter.__new__(EnvAffinityRouter)
    router.workers = [worker]
    router.worker_urls = ["inproc://sglang-engine/dp0"]

    async def _run():
        ok = await router._push_kv_lease_inproc(
            0,
            trajectory_id="traj_a",
            ttl_s=2.0,
            lease_score=0.5,
        )
        assert ok is True
        worker.set_kv_lease.remote.assert_called_once()

        result = await router._lookup_resume_inproc(0, "traj_a")
        assert result.found is True
        assert result.cache_confidence == pytest.approx(0.85)

    asyncio.run(_run())
