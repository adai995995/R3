import asyncio

from roll.distributed.scheduler.rollout_scheduler import RolloutScheduler


class RemoteMethod:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        return self.function(*args)


class FakeEnvironmentCluster:
    def collect_trajectory_progress(self, blocking=False):
        assert not blocking
        return [asyncio.sleep(0, result=[{"trajectory_id": "inflight"}])]

    def stop(self, blocking=False):
        assert not blocking
        return [asyncio.sleep(0)]


class FakeQueue:
    def __init__(self, on_stop_admission=None):
        self.on_stop_admission = on_stop_admission
        self.mark_rollout_end = RemoteMethod(lambda: asyncio.sleep(0))
        self.stop_admission = RemoteMethod(self._stop_admission)
        self.collect_shutdown_waste = RemoteMethod(self._collect_shutdown_waste)
        self.shutdown = RemoteMethod(lambda: asyncio.sleep(0))

    async def _stop_admission(self):
        if self.on_stop_admission is not None:
            self.on_stop_admission()

    @staticmethod
    async def _collect_shutdown_waste(records):
        return {"metrics": {}, "records": records}


class FakeRouter:
    def __init__(self, on_shutdown=None):
        self.on_shutdown = on_shutdown
        self.shutdown = RemoteMethod(self._shutdown)
        self.collect_version_boundary_profile = RemoteMethod(
            lambda: asyncio.sleep(0, result={"metrics": {}})
        )
        self.collect_lifetime_request_metrics = RemoteMethod(
            lambda: asyncio.sleep(0, result={})
        )

    async def _shutdown(self):
        if self.on_shutdown is not None:
            self.on_shutdown()


def make_scheduler(router, rollout_task, queue=None):
    scheduler = object.__new__(RolloutScheduler)
    scheduler.shutdown_timeout_seconds = 0.01
    scheduler.shutdown_grace_seconds = 0.01
    scheduler.es_manager = FakeEnvironmentCluster()
    scheduler.env_output_queue = queue or FakeQueue()
    scheduler.router_manager = router
    scheduler.rollout_task = rollout_task
    return scheduler


async def run_graceful_shutdown_test():
    release_rollout = asyncio.Event()
    shutdown_order = []

    def stop_admission():
        shutdown_order.append("stop_admission")

    def stop_router():
        shutdown_order.append("router")
        release_rollout.set()

    scheduler = make_scheduler(
        FakeRouter(on_shutdown=stop_router),
        asyncio.create_task(release_rollout.wait()),
        queue=FakeQueue(on_stop_admission=stop_admission),
    )

    report = await scheduler.shutdown()

    assert report["shutdown"]["timeout_stages"] == []
    assert report["shutdown"]["rollout_task_cancelled"] is False
    assert report["metrics"]["terminal_waste/shutdown_timeouts"] == 0
    assert report["metrics"]["terminal_waste/rollout_task_cancelled"] == 0
    assert shutdown_order == ["stop_admission", "router"]
    assert scheduler.rollout_task is None


async def run_cancel_test():
    scheduler = make_scheduler(
        FakeRouter(), asyncio.create_task(asyncio.sleep(60))
    )

    report = await scheduler.shutdown()

    assert report["records"] == [{"trajectory_id": "inflight"}]
    assert report["shutdown"]["timeout_stages"] == []
    assert report["shutdown"]["rollout_task_cancelled"] is True
    assert report["metrics"]["terminal_waste/shutdown_timeouts"] == 0
    assert report["metrics"]["terminal_waste/rollout_task_cancelled"] == 1
    assert scheduler.rollout_task is None


async def run_timeout_test():
    release_rollout = asyncio.Event()

    async def resist_cancellation():
        while not release_rollout.is_set():
            try:
                await release_rollout.wait()
            except asyncio.CancelledError:
                continue

    rollout_task = asyncio.create_task(resist_cancellation())
    scheduler = make_scheduler(FakeRouter(), rollout_task)

    report = await scheduler.shutdown()

    assert report["shutdown"]["timeout_stages"] == ["rollout_loop"]
    assert report["shutdown"]["rollout_task_cancelled"] is True
    assert report["metrics"]["terminal_waste/shutdown_timeouts"] == 1
    assert report["metrics"]["terminal_waste/rollout_task_cancelled"] == 1
    assert scheduler.rollout_task is None

    release_rollout.set()
    await rollout_task


def test_rollout_shutdown_releases_generate_before_waiting_for_loop():
    asyncio.run(run_graceful_shutdown_test())


def test_rollout_shutdown_cancels_blocked_loop():
    asyncio.run(run_cancel_test())


def test_rollout_shutdown_timeout():
    asyncio.run(run_timeout_test())


if __name__ == "__main__":
    test_rollout_shutdown_timeout()
    print("shutdown timeout test passed")
