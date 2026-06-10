"""Lease backend resolution for EnvAffinityRouter (env_id -> dp_rank)."""


class _FakeSchedulingState:
    def __init__(self):
        self._backend = {}

    def get_lease_backend_id(self, tid):
        return self._backend.get(tid)

    def set_last_lease_backend_id(self, tid, bid):
        self._backend[tid] = bid


class _RouterStub:
    def __init__(self):
        self.src_rank2_dp_rank = {7: 2, 8: 4}
        self.scheduling_state = _FakeSchedulingState()

    def _trajectory_id_from_route_meta(self, route_meta):
        return route_meta.get("trajectory_id")

    # Bind real methods from EnvAffinityRouter
    from roll.distributed.scheduler.router import EnvAffinityRouter

    _resolve_lease_backend_id = EnvAffinityRouter._resolve_lease_backend_id
    _prepare_lease_route_meta = EnvAffinityRouter._prepare_lease_route_meta


def test_resolve_lease_backend_from_env_id():
    r = _RouterStub()
    route_meta = {"trajectory_id": "t1", "env_id": 7}
    assert r._resolve_lease_backend_id(route_meta) == 2


def test_prepare_lease_route_meta_sets_last_backend_id():
    r = _RouterStub()
    route_meta = {
        "trajectory_id": "t1",
        "env_id": 8,
        "pending_resume_lease_ttl_s": 5.0,
        "pending_resume_lease_score": 0.4,
    }
    r._prepare_lease_route_meta(route_meta)
    assert route_meta["last_backend_id"] == 4
    assert route_meta["resume_lease_ttl_s"] == 5.0
