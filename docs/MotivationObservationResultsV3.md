# Motivation Observation Results V3

## Experiment setup

All runs use the native FIFO scheduler and outstanding-watermark admission. The
version-aware admission and priority controller are disabled, so these are
baseline observations rather than system speedup results.

| Item | Setting |
| --- | --- |
| Model | Qwen3-4B-Instruct-2507 |
| GPUs | 8 total: 4 learner + 4 rollout |
| Execution | Fully asynchronous, learner and rollout disaggregated |
| Learner batch | 4 trajectories per policy update |
| Training length | 20 policy steps; first 2 excluded as warm-up |
| Staleness tolerance | 2 versions, except the diagnostic relaxed runs |
| Scheduler | FIFO |
| Admission | Fixed maximum outstanding trajectories |
| Checkpoints | Disabled |
| AppWorld-fast | `C={8,16,32}`, seeds 83/84/85, max 16 actions |
| tau-bench airline | `C={8,16,32}`, seeds 74/75/76, max 20 actions |
| Staleness diagnostic | AppWorld, `C={16,32}`, tolerance `{2,1000}`, 3 seeds |

The primary training metric is mean end-to-end policy-step time. Rollout output
tokens per second measures producer activity. Learner token-consumption rate and
updates per hour are deliberately not used as headline metrics.

## Observation 1: More rollout work does not imply faster training

### AppWorld-fast

| Max in-flight C | Rollout output tok/s | Mean step (s) | Batch wait (s) | Stale fraction |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 186.7 | **14.49** | 0.68 | 10.2% |
| 16 | 255.1 | 15.05 | 0.84 | 33.5% |
| 32 | **464.1** | 16.17 | 2.50 | 56.7% |

From C=8 to C=32, rollout throughput increases by 148.8%, while mean policy-step
time becomes 11.9% longer. The stale fraction increases by 46.6 percentage
points.

### tau-bench airline

| Max in-flight C | Rollout output tok/s | Mean step (s) | Batch wait (s) | Stale fraction |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 241.0 | 22.61 | 9.00 | 12.9% |
| 16 | 407.2 | **18.71** | **4.95** | 35.4% |
| 32 | **558.0** | 22.28 | 8.58 | 50.7% |

This workload exposes the expected sweet spot. From C=8 to C=16, throughput
increases by 69.0% and step time falls by 17.2%. Increasing C again from 16 to
32 raises throughput by another 37.1%, but step time rises by 19.1% and batch
wait rises by 3.62 seconds.

The result is not "oversampling is always harmful." Moderate oversampling hides
rollout latency, while excessive oversampling creates enough stale churn and
resource competition to reverse the end-to-end gain.

## Observation 2: Bounded staleness converts excess concurrency into churn

The same AppWorld workload was run with the normal tolerance of 2 and a
diagnostic tolerance of 1000.

| Tolerance | C | Rollout tok/s | Mean step (s) | Batch wait (s) | Stale fraction | Stale rollout-compute share |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16 | 292.2 | 14.95 | 0.60 | 35.7% | 39.0% |
| 2 | 32 | 453.4 | 16.39 | 1.80 | 56.7% | 59.3% |
| 1000 | 16 | 190.1 | 14.11 | 0.023 | 0% | 0% |
| 1000 | 32 | 219.6 | 13.98 | 0.022 | 0% | 0% |

Under tolerance 2, C=16 to C=32 raises raw throughput by 55.7%, but makes the
step 9.9% slower, raises stale fraction by 21.0 percentage points, and raises
the stale rollout-compute share by 20.3 percentage points.

With the diagnostic relaxed tolerance, the same concurrency change makes the
step 0.95% faster and creates no stale discard. At C=32, relaxing staleness
reduces mean step time by 2.41 seconds even though measured raw throughput falls
from 453.4 to 219.6 tok/s. The strict run's high raw throughput was partly
discard-and-replace churn, not useful progress.

Tolerance 1000 is a causal diagnostic, not a proposed production policy: it
changes the accepted policy-age distribution and therefore cannot be used as
the system solution.

## Observation 3: Outstanding count alone is not enough state

The exact optimum differs across workloads: C=8 for AppWorld-fast and C=16 for
airline. However, C=16 is still within 3.8% of the AppWorld optimum, so these two
workloads alone do not prove that every static C must perform badly.

The boundary-aligned state data gives a more direct result. For each version
boundary, `to_version=v` is aligned with the time needed to form learner batch
`step=v`.

Across three seeds, there are 54 measured states per workload with exactly
`outstanding=16`:

| State with the same outstanding count | AppWorld-fast | tau-bench airline |
| --- | ---: | ---: |
| Next-batch mean wait | 0.84 s | 4.95 s |
| Next-batch median wait | 0.023 s | 4.41 s |
| Ready trajectories | 13.07 | 5.30 |
| Running trajectories | 2.93 | 9.65 |
| Mean running progress | 62.9% | 27.2% |
| Mean remaining actions | 4.38 | 11.86 |

The same count therefore corresponds to a 5.9x difference in mean future batch
formation time. The count cannot distinguish mostly-ready work from many
early-stage trajectories. Admission needs at least ready supply, trajectory
progress/version age, and remaining work; tool-wait state becomes necessary
when tools have nontrivial latency.

This proves state insufficiency. It does not yet prove that the current adaptive
controller beats the best static C end to end; that comparison remains the next
system experiment.

## Cost of discarded trajectories

The discarded trajectories are not shallow.

| Workload at C=32 | Total per 20-step run | Mean per discarded trajectory |
| --- | ---: | ---: |
| AppWorld output tokens | 91,540 | 623 |
| AppWorld actions | 1,829 | 12.44 |
| AppWorld tool calls | 1,662 | 11.30 |
| Airline output tokens | 167,218 | 1,276 |
| Airline actions | 1,879 | 14.32 |
| Airline tool calls | 843 | 6.43 |

Tool-call counts are meaningful here. The measured tool wall time is not
representative of remote production tools because these environments execute
tools locally.

## Evidence status

1. **Supported across two real workloads and three seeds:** rollout throughput
   can keep rising after end-to-end training speed has peaked.
2. **Supported by a controlled staleness intervention:** bounded freshness is a
   major mechanism that turns excess concurrency into discarded work.
3. **Supported at the state-information level:** equal outstanding counts can
   have very different future trainable-sample supply.
4. **Not yet established:** the adaptive controller's end-to-end advantage over
   the best static C, and whether that advantage persists through training
   convergence.

## Artifacts

- `output/motivation_v3_fixed_cap_3seed_analysis/`
- `output/motivation_v3_appworld_causal_3seed_analysis/`
- `output/motivation_v3_count_state_sufficiency_3seed/`
- `scripts/analyze_motivation_observation.py`
- `scripts/summarize_motivation_runs.py`
- `scripts/analyze_count_state_sufficiency.py`
