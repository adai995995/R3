# Version-Adaptive Admission: Initial Pressure Experiment

## Setup

- Environment: FrozenLake, up to 10 agent actions per trajectory.
- Model: Qwen2.5-0.5B-Instruct.
- Runtime: 4 Megatron training GPUs and 4 vLLM rollout GPUs.
- Training steps: 10.
- Rollout batch size: 16 trajectories per learner step.
- Train environment concurrency: 4.
- Staleness tolerance: 2 policy versions.
- Scheduling: version priority for both runs.
- Seed: 42.
- Checkpoint saving: disabled.

The only experimental variable was admission:

```text
Static:   outstanding_watermark=48
Adaptive: version_adaptive, max_outstanding=48, reserve=16
```

## Results

| Metric | Static watermark | Version adaptive |
|---|---:|---:|
| Consumed trajectories | 160 | 160 |
| Total admitted trajectories | 256 | 176 |
| Mean outstanding trajectories | 48.0 | 17.2 |
| Version-stale discarded trajectories | 48 | 0 |
| Terminal completed-unconsumed trajectories | 37 | 13 |
| Discarded actions | 0 | 0 |
| Discarded inference calls | 0 | 0 |
| Discarded inference tokens | 0 | 0 |
| Sum of training-step time | 65.2 s | 70.0 s |
| Effective consumed non-prompt tokens/s | 2070.1 | 1957.6 |

Adaptive admission reduced total admitted trajectories by 31.25%, reduced mean outstanding
trajectories by 64.2%, and eliminated 48 version-stale trajectory slots. It also changed the
per-version trainable admission budget instead of adding a fixed number:

```text
32, 16, 20, 16, 12, 16, 16, 20, 12, 16
```

Unfinished carry-over was observed at version boundaries 2, 3 and 7, with 4, 4 and 8
trajectories respectively. The estimated carry-over finish ratio adapted from 0.5 to 0.9375.

## Interpretation

This experiment validates the controller mechanics, but it does not yet validate the main
compute-salvage hypothesis. All 48 stale trajectories in the static run had zero completed
actions, zero inference calls and zero inference tokens. They were admitted but unstarted
reserved slots, not trajectories with invested GPU or environment work.

The static run therefore spent no measured inference work on those stale slots. Its larger ready
backlog reduced rollout waiting time and produced about 5.4% higher effective consumed non-prompt
token throughput. The current workload demonstrates a queue-size and oversampling reduction, not
an end-to-end GPU-compute saving.

## Next Required Change

The runtime and metrics distinguish three classes at each policy-version boundary:

```text
ready:              complete and learner-consumable
invested_partial:   started and has running/completed turns or inference work
reserved_unstarted: admitted but has not received inference service
```

Future admission decisions should report these classes separately. Version-priority salvage should
protect `invested_partial`; excess `reserved_unstarted` work can be cancelled or not renewed without
claiming saved GPU computation.

Post-experiment instrumentation now exports
`scheduler/invested_inflight_at_version_boundary` and
`scheduler/reserved_unstarted_at_version_boundary`. The current controller still uses their combined
expected supply; separating their control weights is the next controller change.

The next experiment should use a heterogeneous multi-turn workload where at least some stale
trajectories have non-zero actions and inference tokens. Only then can the primary claim be measured:

```text
saved stale inference tokens and environment seconds
without reducing effective trainable-token throughput
```
