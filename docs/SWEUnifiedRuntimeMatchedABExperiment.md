# SWE-bench Unified Runtime Matched A/B Experiment

## Question

This experiment checks whether the unified version-aware runtime works on a
real, expensive coding-agent workload. The central question is whether doing
less speculative rollout work can produce trainable data faster while avoiding
the loss of trajectories that already consumed inference and sandbox resources.

## Setup

- Workload: three cached SWE-bench Verified tasks covering Flask, pytest, and
  Django.
- Agent: Qwen3-4B-Instruct-2507 with real code-search, edit, test, and sandbox
  tool calls.
- Hardware: one 8-GPU node. GPUs 0-3 train while GPUs 4-7 generate rollouts.
- Training: five real updates, four consumed trajectories per update.
- Asynchrony: rollout and learner run on disjoint GPUs; staleness tolerance is
  two policy versions.
- Trajectory limits: at most 24 outstanding trajectories and 10 actions per
  trajectory.
- Generation: at most 2,048 new tokens per action.
- Seed: 48 for training, rollout, and environment group assignment.
- Checkpoints: disabled, including the final checkpoint.

The matched runs differ only in runtime policy:

1. `fifo`: fixed outstanding watermark, FIFO trajectory scheduling, and no KV
   working-set rebuilding.
2. `full_floor`: version-adaptive admission, version-priority trajectory
   scheduling, a progress-based learner supply floor, post-update KV
   working-set rebuilding, working-set routing, and soft locality.

## End-to-End Results

| Runtime | Elapsed | Raw trajectories | Raw response tok/s | Trainable response tok/s | Trainable traj/s | Learner wait | Stale logical-token fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIFO | 1,301.05 s | 42 | 25.256 | 14.873 | 0.01537 | 80.37% | 30.12% |
| Full with floor | 863.39 s | 28 | 22.412 | 22.412 | 0.02316 | 76.43% | 0.00% |

Both runs delivered the same 20 trajectories and 19,350 response tokens to the
learner. The complete runtime generated 14 fewer trajectories and had 11.26%
lower raw response-token throughput, but increased trainable response-token and
trajectory throughput by 50.69%. The same five updates finished 437.66 seconds,
or 33.64%, sooner.

This is the intended goodput result: a busier rollout producer did not advance
training faster. The complete runtime removed work that was unlikely to remain
usable and converted a larger fraction of generation into learner input.

## Avoided Waste

| Asynchronous stale waste | FIFO | Full with floor |
|---|---:|---:|
| Trajectories | 12 | 0 |
| Inference calls | 66 | 0 |
| Tool calls | 55 | 0 |
| Logical inference tokens | 1,217,423 | 0 |
| Environment time | 670.57 s | 0 s |
| Trajectories with at least 4 actions | 8 | 0 |
| Trajectories with at least 8 actions | 4 | 0 |

The discarded FIFO trajectories were not shallow requests. Eight had already
performed at least four actions and four had performed at least eight actions.
At shutdown, FIFO also left three invested trajectories with 16 actions, 14
tool calls, and 337,697 logical tokens. The complete runtime left eight
reset-only trajectories with no inference, tool, token, or measured environment
work, so its terminal tail had not yet consumed rollout compute.

## Cross-Version Behavior

The complete runtime observed 17 cross-version trajectories at the four policy
boundaries and expired none of them. It consumed 12 trajectories at version age
one and four at version age two. This confirms that admission and priority did
not require all data to come from the newest version; they salvaged valid work
within the configured tolerance.

FIFO observed more speculative state at boundaries: 32 cross-version
trajectories versus 17. It also exposed 1,409,906 logical inference tokens at
boundaries, compared with 997,150 for the complete runtime.

## KV Result

| Runtime | KV block hit | Cache resets | Rebuild selections | Rebuild prefill tokens |
|---|---:|---:|---:|---:|
| FIFO | 86.36% | 17 | 0 | 0 |
| Full with floor | 83.28% | 14 | 22 | 16,553 |

The KV working-set path was active: it selected 22 rebuild requests and routed
68 requests through the version runtime candidate plan. However, the aggregate
KV block-hit ratio fell by 3.56% relative to FIFO, and the rebuild cohort hit
only 9.20%. This run does not establish a KV benefit. The observed end-to-end
gain should be attributed to admission and trajectory progress control until a
component ablation demonstrates otherwise.

## Interpretation

The prototype works end to end on this SWE-bench workload. It admits less work,
eliminates measured asynchronous stale waste, reduces learner waiting in both
seconds and fraction, and completes the same number of training updates faster.
The result directly supports the distinction between raw rollout throughput and
trainable goodput.

It is not yet a paper-scale result. The comparison uses one seed, five updates,
and three cached tasks. SWE sandbox setup is noisy, and asynchronous execution
does not guarantee identical trajectory realizations even with matched seeds.
The next validation should run at least three seeds, increase the number of
tasks and updates, and add admission-only, priority-only, and KV-only ablations.

Both original A/B runs reported a rollout-loop shutdown timeout after the
terminal snapshot. This lifecycle issue is now fixed. Shutdown gives the loop a
five-second grace period, then explicitly cancels a loop that is still blocked
in environment reset; only a task that resists cancellation is reported as a
real timeout.

A one-step, real ROCK SWE-bench smoke run validated the normal path. It consumed
four trajectories, exited without saving a checkpoint, and reported
`timeout_stages=[]`, `terminal_waste/shutdown_timeouts=0`, and
`rollout_task_cancelled=false`, meaning the loop stopped inside the grace
period. Unit tests additionally cover the blocked-but-cancellable path and the
true cancellation-timeout path; all three shutdown tests pass.

## Artifacts

- FIFO report: `output/agentic_swe_qwen3_4b_unified_ab_fifo_seed48_5step/terminal_waste.step_5.json`
- Complete-runtime report: `output/agentic_swe_qwen3_4b_unified_ab_full_floor_seed48_5step/terminal_waste.step_5.json`
- Shutdown-fix smoke report: `output/agentic_swe_qwen3_4b_unified_ab_full_floor_shutdown_v2_1step/terminal_waste.step_1.json`
- Comparison: `output/swe_unified_ab_seed48_5step_analysis/`
- Analyzer: `scripts/analyze_version_runtime_ab.py`
- Configs: `examples/agentic_demo/agent_val_rock_swe_qwen3_4b_unified_ab_*_seed48_5step.yaml`
