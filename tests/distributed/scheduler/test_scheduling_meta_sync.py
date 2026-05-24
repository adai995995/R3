from roll.distributed.scheduler.resume_state import (
    get_trajectory_scheduling_state,
    reset_trajectory_scheduling_state,
)


def test_sync_from_route_meta_t_tool():
    reset_trajectory_scheduling_state(default_t_tool_s=5.0)
    store = get_trajectory_scheduling_state()
    store.sync_from_route_meta("tid-1", {"scheduling_t_tool_s": 18.5})
    assert abs(store.get_t_tool_s("tid-1") - 18.5) < 1e-6
