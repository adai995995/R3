import time

import pytest


pytest.importorskip("ray")

from roll.distributed.scheduler.router import EnvAffinityRouter, PendingTrajectoryRequest


def _make_router_for_test() -> EnvAffinityRouter:
    router = EnvAffinityRouter.__new__(EnvAffinityRouter)
    router.pending_resume_requests = {}
    router.pending_normal_requests = {}
    router.request_wait_aging_weight = 0.0
    router._quota_resume_target = 2
    router._quota_normal_target = 1
    router._quota_cursor = 0
    router.enable_adaptive_quota = False
    router._last_adaptive_quota_ts = 0.0
    router.adaptive_quota_update_interval_s = 9999.0
    router.normal_max_queue_wait_s = 0.0
    router.resume_max_queue_wait_s = 0.0
    router.enable_resume_aware_routing = False
    router.active_dp_ranks = {0}
    router.running_requests = [set()]
    router.max_running_requests_per_worker = 0
    router.src_rank2_dp_rank = {}

    def _select_worker_for_request(_pending):
        return 0

    router._select_worker_for_request = _select_worker_for_request  # type: ignore[attr-defined]
    return router


def _pending(request_id: str, request_type: str, enqueue_seq: int, enqueue_ts: float | None = None) -> PendingTrajectoryRequest:
    return PendingTrajectoryRequest(
        request_id=request_id,
        uid=0,
        request_type=request_type,
        route_meta={},
        enqueue_ts=enqueue_ts if enqueue_ts is not None else time.time(),
        enqueue_seq=enqueue_seq,
        base_priority=0.0,
    )


def test_skip_on_empty():
    router = _make_router_for_test()
    router.pending_resume_requests["r1"] = _pending("r1", "resume", 0)
    selected, dp_rank = router._pick_next_dispatchable_request()
    assert selected is not None
    assert dp_rank == 0
    assert selected.request_id == "r1"


def test_quota_cycle_resume_resume_normal_fifo_normal():
    router = _make_router_for_test()
    router.pending_resume_requests["r1"] = _pending("r1", "resume", 0)
    router.pending_resume_requests["r2"] = _pending("r2", "resume", 1)
    router.pending_normal_requests["n1"] = _pending("n1", "normal", 2)
    router.pending_normal_requests["n2"] = _pending("n2", "normal", 3)

    s1, _ = router._pick_next_dispatchable_request()
    assert s1 and s1.request_id == "r1"
    router.pending_resume_requests.pop(s1.request_id)

    s2, _ = router._pick_next_dispatchable_request()
    assert s2 and s2.request_id == "r2"
    router.pending_resume_requests.pop(s2.request_id)

    s3, _ = router._pick_next_dispatchable_request()
    assert s3 and s3.request_id == "n1"
    router.pending_normal_requests.pop(s3.request_id)


def test_timeout_escape_hatch_prefers_normal():
    router = _make_router_for_test()
    now = time.time()
    router.normal_max_queue_wait_s = 0.01
    router.pending_resume_requests["r1"] = _pending("r1", "resume", 0, enqueue_ts=now)
    router.pending_normal_requests["n1"] = _pending("n1", "normal", 1, enqueue_ts=now - 10.0)

    selected, _ = router._pick_next_dispatchable_request()
    assert selected is not None
    assert selected.request_id == "n1"

