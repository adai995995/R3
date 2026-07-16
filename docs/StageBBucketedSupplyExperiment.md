# Stage B: Bucketed Carry-Over Supply Prediction

## Goal

Stage B replaces the single global carry-over finish EWMA with a predictor conditioned on
policy-version age and trajectory progress. The predictor estimates how much existing work will
become trainable during the next policy-version window and feeds that estimate into the existing
version-adaptive admission budget.

It does not use reward, success probability, advantage or training value, and it does not change
trajectory sampling or learner semantics.

## Implementation

The runtime uses stable buckets:

```text
version age:     0, 1, 2, 3, >=4
action progress: 0, 1, 2-3, 4-7, >=8
```

Progress is computed at the trainable-group granularity. Among all candidates, the runtime takes
the top `group_size` action counts and uses their integer mean for bucketing. It also reports the
`group_size`-th action count as a conservative frontier diagnostic.

Each bucket maintains an EWMA completion ratio and cumulative sample count. A bucket falls back to
the global finish EWMA until `adaptive_admission_bucket_min_samples` is reached.

The learning target is a `became_trainable` event emitted when `GroupQueue.put()` first reaches
`group_size` and the group passes filtering. Learner consumption is tracked separately. Using
consumption as the target would incorrectly label completed data as unfinished whenever the ready
queue exceeds one learner batch.

## Validation Setup

```text
workload:                 WebShop, real multi-turn environment
model:                    Qwen3-4B
hardware:                 one node, 8 GPUs
training / rollout:       4 Megatron GPUs / 4 vLLM GPUs
execution:                fully separated, asynchronous rollout and training
train steps:              8
rollout_batch_size:       4
group_size:               2
trajectory tolerance:     2 policy versions
max outstanding:          24 trajectories
admission reserve:        8 trajectories, fixed for isolation
bucket minimum samples:   2 trajectories
checkpoint saving:        disabled
```

Thirteen focused scheduler/controller tests passed in the `xxl_test` container. They cover bucket
mapping, fallback behavior, EWMA updates, group progress aggregation, completion-event attribution,
progress-floor behavior and existing reserve-controller regressions.

## Iteration Findings

The first 12-step run used the `group_size`-th action count as the progress label. Every carry-over
group fell into `actions_0`, even though terminal records contained many deep trajectories. A second
8-step diagnostic showed why: the group frontier remained zero while one candidate had often
already reached 5-10 actions. The top-`group_size` mean preserved this invested-work signal and
produced `actions_1`, `actions_2_3` and `actions_4_7` cohorts.

The initial implementation also learned from learner consumption. This was changed to group
completion before the final run. In the final experiment, actual available supply differed from
actual consumed carry-over in six of eight versions; for example, step 5 had eight existing samples
become available while the learner consumed four. Consumption would therefore have biased the
finish estimator downward by 50% in that window.

## Final Functional Result

```text
versions with non-empty carry-over buckets: 7 / 8
versions using at least one learned bucket:  6 / 8
versions where bucket estimate != global:   6 / 8
versions where available supply != consumed: 6 / 8
```

The bucketed predictor changed the one-shot admission decision after group-size rounding:

```text
step  global-EWMA budget  bucketed budget  delta
3                       6                8     +2
4                       6               10     +4
5                       0                4     +4
6                       8                6     -2
```

At step 6, learned completion ratios were materially different across states:

```text
age 1, actions 0:   0.083 over 20 observations
age 2, actions 0:   0.500 over 4 observations
age 2, actions 4-7: 1.000 over 4 observations
```

These are small-sample mechanism results, not calibrated probabilities. They do demonstrate that a
single global finish ratio discards runtime state that is predictive of timely completion and that
the learned state now affects real admission decisions.

The final short run consumed 32 trajectories. It also discarded 11 stale trajectories carrying
342,700 logical inference tokens and ended with five completed-but-unconsumed trajectories. This is
not evidence of an end-to-end improvement: the run has no matched baseline, uses a deliberately
large fixed reserve of 8, and is only eight steps long. Stage B is functionally complete, while
performance evaluation requires a longer matched A/B with the same seed and controller settings.

## Matched 50-Step A/B

A longer experiment compared the global EWMA predictor with the Stage B bucketed predictor. Both
runs used seed 44 and the same WebShop task stream, model, hardware, staleness tolerance and fixed
admission reserve. The only controller difference was
`adaptive_admission_bucketed_finish_enabled`.

```text
train steps:              50
rollout_batch_size:       4
group_size:               2
trajectory tolerance:     2 policy versions
max outstanding:          24 trajectories
admission reserve:        4 trajectories, fixed
bucket minimum samples:   4 trajectories
dynamic reserve:          disabled
checkpoint saving:        disabled
```

| Metric | Global EWMA | Bucketed | Change |
| --- | ---: | ---: | ---: |
| Existing-supply prediction MAE | 0.356 traj. | 0.058 traj. | -83.6% |
| Zero-error version boundaries | 19 / 49 | 43 / 49 | +24 |
| Learned-bucket carry-over coverage | 0% | 81.4% | +81.4 pp |
| Total admitted trajectories | 204 | 204 | 0 |
| Consumed trajectories | 200 | 200 | 0 |
| Stale-discarded trajectories | 0 | 0 | 0 |
| Mean consumed policy-version age | 1.31 | 1.05 | -19.8% |
| Useful response-token rate | 124.5 token/s | 134.8 token/s | +8.2% |
| Steady-state mean step time | 11.68 s | 11.45 s | -2.0% |
| Steady-state median step time | 10.30 s | 10.76 s | +4.4% |
| Consumed logical inference tokens | 3.45 M | 4.02 M | +16.7% |
| Terminal completed-but-unconsumed | 4 traj. | 3 traj. | -1 traj. |
| Terminal waste inference tokens | 18.6 K | 19.6 K | +5.3% |

The two policies selected different admission budgets at 25 of 50 version boundaries, with an
accumulated absolute difference of 50 trajectories, while admitting exactly 204 trajectories in
total. Stage B therefore changed *when* work entered the system rather than simply increasing load.

The primary positive result is prediction quality. The bucketed model reduced existing-supply MAE
by 83.6% and produced exact predictions at 43 of 49 evaluated boundaries. The learned population
was concentrated in four `age_1` progress buckets (`actions_0`, `actions_1`, `actions_2_3` and
`actions_4_7`); all converged to a near-one-window completion ratio in this workload. The global
EWMA mixed these states with earlier observations and reacted more slowly.

The lower mean consumed version age is also encouraging: Stage B supplied the same number of
training trajectories with fresher policy versions. However, this single-seed run is not clean
throughput evidence. The bucketed run happened to complete 9.3% more actions and consume 16.7% more
logical inference tokens, while median step time and learner-wait EWMA did not improve. Asynchronous
ordering changed the realized trajectory mix even under the same task seed. The reported token-rate
increase must therefore remain an observation, not an attributed system speedup.

Neither policy produced stale discards with reserve 4, so this operating point validates prediction
and admission retiming but cannot measure waste reduction. The next evaluation should repeat the A/B
over multiple seeds and add a controlled pressure sweep that increases reserve or outstanding work
until the baseline produces measurable staleness.

## Metrics Added

```text
scheduler/bucketed_expected_inflight_supply
scheduler/bucketed_learned_population
scheduler/bucketed_fallback_population
scheduler/finish_ratio/<age-progress-bucket>
scheduler/finish_ratio_samples/<age-progress-bucket>
scheduler/carryover_at_boundary/<age-progress-bucket>
scheduler/boundary_progress_observed_candidates
scheduler/boundary_progress_mean_actions_sum
scheduler/boundary_progress_frontier_actions_sum
scheduler/boundary_progress_max_actions
scheduler/actual_existing_consumed
```

## Raw Artifacts

- `webshop_stage_b_bucketed_12step.log`: initial frontier-based run.
- `webshop_stage_b_progress_8step.log`: top-`group_size` mean diagnostic run.
- `webshop_stage_b_final_8step.log`: final completion-event run.
- `terminal_waste.stage_b_final.step_8.json`: final terminal and stale-work report.
- `webshop_stage_b_ab_global_seed44_50step.log`: matched global-EWMA baseline.
- `webshop_stage_b_ab_bucketed_seed44_50step.log`: matched Stage B run.
- `terminal_waste.stage_b_ab_global_seed44.step_50.json`: baseline terminal report.
- `terminal_waste.stage_b_ab_bucketed_seed44.step_50.json`: Stage B terminal report.
