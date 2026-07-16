# Stage A: Unified Runtime Scheduling Prototype

## Implementation

Each inference turn now carries a structured trajectory runtime state:

```text
trajectory_id, policy_version, current_version, version_age,
actions_completed, inference_calls, max_actions, remaining_actions
```

The per-worker priority queue uses a stable lexicographic key:

```text
(policy_version, is_unstarted, remaining_actions, FIFO sequence)
```

This serves older policy versions first, then prefers already-invested trajectories
and trajectories closer to their hard action limit. It uses no reward, success
probability or training-value signal.

Soft locality compares only two choices: the trajectory's current-version affinity
worker and the least-loaded active worker. Affinity is retained while its pressure is
within a configurable load slack; otherwise load overrides locality. A cache epoch
incremented at every version resume invalidates old-version locality hints.

## Validation

- Container compilation passed on Python 3.10.
- Six focused Router tests passed.
- Normal WebShop run: 4/4 real train/update steps completed.
- Pressure WebShop run (`4 slots/worker`, `load_slack=0`): 4/4 steps completed.
- Queue WebShop run (`1 slot/worker`): 2/2 steps completed.
- No version-stale trajectories were discarded in these short validation runs.

The four-step pressure run executed 156 scheduling decisions:

```text
affinity selected:       119
load overrides:            7
least-loaded first route:  4
router decision CPU time:  8.01 ms total, 51 us/request
```

The two-step single-slot run forced the priority path:

```text
scheduling decisions: 88
requests queued:      73 (83.0%)
aggregate wait:       67.32 s
router decision CPU:   2.62 ms total, 30 us/request
completed train steps: 2/2
```

The single-slot result is a functional stress test, not a recommended production
configuration or a throughput comparison. It proves that structured trajectory state
crosses the environment manager, proxy and Ray Router boundaries and reaches the
real priority queue.

## Raw Logs

- `webshop_stage_a_4step.log`
- `webshop_stage_a_pressure_4step.log`
- `webshop_stage_a_queue_2step.log`
