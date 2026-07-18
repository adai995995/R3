# Version Boundary Profiler: Initial Real-System Experiment

## Purpose

Measure what exists at each policy update in a fully asynchronous AgenticRL run:

- unfinished trajectories that must cross the update;
- completed trajectories that only wait for learner consumption;
- invested actions and context in unfinished trajectories;
- stale trajectories eventually discarded;
- the first survivor requests that face an empty post-update KV cache.

The profiler does not use reward, training value, or trajectory outcome to make decisions.

## Implementation

The profiler now combines two low-frequency snapshots at each version boundary:

1. Environment state: reset, inference, environment step, or completed state.
2. Router state: trajectory ID, version, completed actions, remaining actions, and current prompt length already carried by normal inference requests.

This avoids adding a control-plane RPC after every action. The scheduler queries and merges the state once per learner/version boundary.

The report separates:

- `completed_carryover_trajectories`: completed data waiting in the rollout queue;
- `cross_version_trajectories`: unfinished trajectories that continue across an update;
- `survivor_trajectories`: unfinished trajectories still within the staleness limit;
- `unfinished_expired_trajectories`: unfinished trajectories that expire at the boundary.

## Validation Setup

- Environment: WebShop
- Model: Qwen3-4B-Instruct-2507
- Hardware: one node, 8 GPUs
- Actor training: 4 GPUs
- vLLM rollout: 4 GPUs
- Learner steps: 4
- Rollout batch size: 4 trajectories
- Maximum actions per trajectory: 20
- Staleness tolerance: 1 version
- Admission reserve: 12 trajectories
- Maximum outstanding trajectories: 48
- Sequence length: 16,384
- Checkpoint saving: disabled (`save_steps: -1`)

Config: `agent_val_webshop_qwen3_4b_boundary_profile_longtraj_validate_4step.yaml`

## Result

Across three policy updates:

| Metric | Value |
| --- | ---: |
| Unfinished cross-version observations | 11 |
| Unobserved started observations | 2 |
| Actions already completed by unfinished trajectories | 98 |
| Current context tokens of unfinished trajectories | 84,675 |
| Completed carry-over observations | 23 |
| Stale trajectories eventually discarded | 9 |
| Actions in discarded trajectories | 95 |
| Logical inference tokens in discarded trajectories | 616,928 |
| Trajectories consumed by the learner | 16 |
| Survivor first requests after an update | 2 |
| Logical post-update re-prefill exposure | 22,212 prompt tokens |

The 11 unfinished observations were not shallow. At the first boundary, four survivors had already completed 14, 15, 17, and 18 actions. At later boundaries, survivors had completed between 2 and 7 actions.

The nine stale trajectories also contained substantial work: four had at least eight actions, and three had 16, 19, or 20 actions.

## Interpretation

This run validates the central system observation:

1. Policy updates happen while nontrivial agent trajectories are still executing.
2. Some of those trajectories later become stale after consuming many inference and environment steps.
3. Surviving trajectories issue large full-context requests after the cache epoch changes.
4. Completed queue entries and unfinished survivors must be measured separately; only the latter need KV recovery and urgency scheduling.

The experiment intentionally creates pressure. It demonstrates that the problem exists, but it is not yet a throughput comparison or a claim that these parameters are optimal.

## Remaining Measurement Gaps

- `logical inference tokens` measure logical model input/output work, not exact GPU kernel time.
- vLLM did not report per-request prefill tokens in this run, so 22,212 is a logical prompt upper bound.
- WebShop environment-step time is close to zero locally and is not representative of a remote tool or sandbox.
- Two started trajectories still lacked a boundary snapshot, likely because they had been assigned but had not issued their first inference request.

## Artifacts

- Validation JSON: `terminal_waste.boundary_profile_longtraj_validate.step_4.json`
- Pre-fix pressure JSON: `terminal_waste.boundary_profile_longtraj.step_6.json`
- Earlier pressure JSON: `terminal_waste.boundary_profile_pressure.step_8.json`
