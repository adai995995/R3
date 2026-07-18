# Native Fully-Async AgenticRL Observation Experiment Plan

## 1. Purpose

This document defines the observation experiments required to motivate a version-aware AgenticRL
runtime. The experiments first characterize the unmodified asynchronous producer-consumer behavior;
they do not evaluate or tune the proposed controller.

The central claim is not that rollout throughput is unimportant. It is that, under fully asynchronous
and fully disaggregated execution, maximizing raw rollout production does not necessarily maximize
the rate at which the learner receives trainable data.

The experiments test four hypotheses:

1. Raw rollout throughput and learner-consumed goodput separate under excess load.
2. A fixed oversampling amount has no workload-independent optimum.
3. Count-based queue limits and hard version thresholds act on incomplete state or act too late.
4. Policy updates impose measurable stale-work and survivor re-prefill costs.

These observations motivate three runtime mechanisms:

```text
Adaptive admission
Version-aware trajectory scheduling
Post-update KV working-set rebuilding and placement
```

## 2. Scope and Non-Goals

The observation stage uses only native asynchronous scheduling plus measurement instrumentation.
It must not enable:

```text
adaptive admission
version-priority scheduling
post-update KV rebuilding
working-set-aware placement
reward-aware or training-value-aware scheduling
```

This stage does not attempt to find the final controller parameters or demonstrate end-to-end
speedup from the proposed system. It establishes the deficiencies that the system must address.

Partial-rollout and multi-version-serving systems are related alternatives, but they are not required
as executable baselines in the first observation stage. Their algorithmic and memory trade-offs will
be discussed separately.

## 3. Baseline and Reproducibility

### 3.1 Code baseline

Use clean ROLL commit:

```text
ba7092f
```

Create a dedicated metrics-only observation branch or worktree from this commit. Port only:

```text
trajectory lifecycle events
version-boundary profiler
learner consumption and wait metrics
per-request vLLM cached/prefill token accounting
request queue wait and routing telemetry
terminal drain/report generation
```

The current `version_driven` branch must not be used as the native baseline because it already
contains controller and Router policy changes.

### 3.2 Primary setup

```text
Model: Qwen3-4B-Instruct-2507
Workload: WebShop
Maximum actions: 20
Hardware: one node, 8 GPUs
Learner GPUs: 4
Rollout GPUs: 4
Inference backend: vLLM
Checkpoints: disabled
Seeds: 42, 43, 44
```

WebShop is the first workload because it provides multi-turn trajectories and natural length
variation at manageable cost. A second workload should validate that the result is not WebShop-
specific:

```text
HotpotQA long-context pressure, or
ReTool/tool-sandbox workload with non-trivial tool time
```

### 3.3 Fixed controls

Unless explicitly swept, keep the following unchanged:

```text
model and initial checkpoint
dataset and prompt order
learner batch size
group_size and group semantics
number of environment workers
max_running_request
GPU allocation
sampling parameters
staleness tolerance
parameter synchronization frequency
```

Set `group_size_redundancy=0` during the admission sweep. Redundancy is a separate oversampling
mechanism and would otherwise confound the result.

## 4. Definitions and Metric Semantics

Let:

```text
N = trajectories required by one learner batch
K = newly admitted trainable trajectories per policy version
S = maximum accepted policy-version lag
```

`K` must be a multiple of `group_size`; complete GRPO groups must never be split.

The native configuration couples several controls. For a fair `K` sweep, keep environment-worker
parallelism fixed and vary only the number of new episode groups admitted per version. If the native
configuration cannot express this independently, add a baseline-only fixed
`admission_groups_per_step` knob. This knob must remain static and must not inspect runtime state.

### 4.1 Primary metrics

```text
Raw response throughput
  = all generated response tokens / rollout wall time

Trainable response-token goodput
  = response tokens from learner-consumed trajectories / rollout wall time

Conversion efficiency
  = learner-consumed response tokens / all generated response tokens

Training progress rate
  = completed learner updates / wall time

Stale trajectory fraction
  = version-expired trajectories / all completed or expired trajectories

Stale token fraction
  = actual tokens spent on expired trajectories / all actual rollout tokens
```

Report actual prefill tokens and generated response tokens separately. Do not present logical prompt
or logical inference tokens as actual GPU computation. Under continuous batching, per-request kernel
time is difficult to attribute exactly; use engine-reported prefill/decode tokens plus interval-level
GPU busy time instead of inventing per-request GPU time.

### 4.2 Required telemetry

For every trajectory:

```text
trajectory, group and episode IDs
admission, first-service, completion, consumption and discard timestamps
creation policy version and version at every boundary
actions completed, maximum actions and current context length
prompt, cached, actual prefill and response tokens per inference call
inference calls and tool/sandbox calls
request queue wait, generation wall time and tool/sandbox time
final state: consumed, stale, completed-unconsumed or terminal-inflight
```

For every policy update:

```text
update timestamp and new version
in-flight, ready and completed-unconsumed trajectory counts
version-age histogram
progress histogram
expired and surviving trajectory counts
first post-update survivor requests and their actual prefill tokens
```

For the learner and system, sampled at every update and at a one-second interval:

```text
consumed trajectory IDs and tokens
batch formation and learner wait time
learner update start/end timestamps
request queue depth and running-request count
rollout and learner GPU utilization
```

## 5. Experiment E1: Throughput-Goodput Separation

### Question

Does increasing fixed rollout admission continue to improve raw production after learner-consumed
goodput has saturated?

### Sweep

```text
K / N = {0.50, 0.75, 1.00, 1.25, 1.50, 2.00}
S = 1
native asynchronous completion-driven scheduling
```

### Measurements

```text
raw response tokens/s
actual prefill tokens/s
completed trajectories/s
learner-consumed response tokens/s
learner updates/hour and time-to-50-updates
conversion efficiency
stale trajectory and token fractions
learner idle fraction
mean and p95 request queue wait
mean and p95 in-flight trajectory count
```

### Main figure

Plot all of the following against `K/N`:

```text
raw response tokens/s
learner-consumed response tokens/s
learner updates/hour
stale token fraction
```

Evidence for the hypothesis is a load region where raw throughput remains high or increases while
trainable goodput and learner progress saturate or decline, accompanied by increasing queue wait or
stale work. Report the complete curve even if no such region appears.

## 6. Experiment E2: No Universal Fixed Oversampling Ratio

### Question

Does the best static admission amount change with trajectory and environment service time?

### Workload phases

Construct or identify two reproducible phases:

```text
Short phase: shorter trajectories, smaller contexts and low tool latency
Long phase: longer trajectories, larger contexts or higher tool latency
```

Prefer natural workload buckets collected in a profiling run. If controlled tool delay is used, it
must be applied uniformly within the declared phase and documented as a testbed condition.

### Sweep and result

Repeat a reduced `K/N` sweep in both phases and report the `K` that maximizes:

```text
learner-consumed response tokens/s
learner updates/hour
```

Evidence for the hypothesis is a meaningful shift in the goodput-maximizing `K` between phases.
This result motivates runtime adaptation; it does not imply that a smaller `K` is always better.

## 7. Experiment E3: Hard Staleness Threshold Is Reactive

### Question

How much work has already been invested when a hard version threshold finally rejects a trajectory?

### Sweep

Choose one pressure-producing `K` from E1 and run:

```text
S = {0, 1, 2, 4}
50 learner updates
3 seeds
```

### Measurements at discard

```text
actions and inference calls completed
generated response tokens
actual prefill tokens
current context tokens
request queue wait
generation wall time
tool/sandbox calls and time
version age
```

### Figures

Produce one CDF or complementary CDF per metric:

```text
action depth at discard
generated tokens at discard
context length at discard
accumulated service time at discard
```

Also show the trade-off across `S`:

```text
discarded invested work
consumed sample version age
learner progress rate
```

The intended conclusion is that a hard threshold can bound learner-visible staleness but cannot
recover resources already spent before rejection.

## 8. Experiment E4: Count-Based Control Is State-Blind

### Question

Do two system states with the same outstanding-trajectory count provide different near-future
trainable supply?

### Analysis method

Reuse E1-E3 traces. Bin snapshots by a narrow outstanding-count range, for example:

```text
outstanding trajectories in [38, 42]
```

For each snapshot, calculate trainable response tokens and completed trainable trajectories produced
over the next 30 seconds and over the next learner-update interval.

Annotate each snapshot with:

```text
mean and maximum version age
mean, frontier and maximum action progress
fraction waiting for tools
mean current context length
completed-unconsumed supply
```

### Figures and model comparison

Plot:

```text
x-axis: outstanding trajectories
y-axis: future learner-consumed response tokens
color: mean progress or mean version age
```

Compare held-out-seed prediction error for:

```text
Count-only model:
  future_goodput ~ outstanding

Runtime-state model:
  future_goodput ~ outstanding + progress + version_age
                    + tool_wait + context_length + ready_supply
```

Report MAE and R-squared without using reward, advantage or success probability. A substantial and
repeatable error reduction from system-only state demonstrates that queue length alone is
insufficient for admission control.

## 9. Experiment E5: Policy-Update KV Recovery Cost

### Question

How much exact prefill work is repeated by surviving trajectories after a policy update flushes the
inference cache?

This experiment piggybacks on E1-E3 and requires no additional training runs.

For every survivor's first post-update request, record:

```text
actions already completed
logical prompt tokens
engine-reported cached prompt tokens
engine-reported actual prefill tokens
route and worker ID
```

Compare:

```text
first post-update survivor request
same-version continuation requests
```

Plot actual prefill tokens against survivor context length and report total survivor re-prefill tokens
per boundary. The existing four-step smoke observed seven engine-reported survivor requests with
44,566 logical prompt tokens and 44,566 actual prefill tokens; the formal experiment must establish
the distribution over 50 updates and three seeds.

## 10. Execution Schedule

### Phase A: pilot and pressure-point selection

```text
Workload: WebShop
Steps: 20
Seeds: 1
K/N: {0.50, 1.00, 1.50, 2.00}
S: 1
Runs: 4
```

Use this phase only to verify telemetry and locate the load knee. Do not draw paper conclusions from
one seed.

### Phase B: formal load sweep

```text
Workload: WebShop
Steps: 50
Seeds: {42, 43, 44}
K/N: four representative points selected from Phase A
Runs: 12
```

The selected points must include an underloaded point, the apparent knee and at least two overloaded
points.

### Phase C: staleness sweep

```text
Workload: WebShop
Steps: 50
Seeds: {42, 43, 44}
S: {0, 1, 2, 4}
K: one pressure-producing point from Phase A
Runs: 12
```

### Phase D: cross-workload validation

Run a reduced set on HotpotQA long-context pressure or a ReTool/tool-sandbox workload:

```text
K/N: {underloaded, knee, overloaded}
Seeds: at least 1 initially, then 3 for the final selected comparison
```

## 11. Warmup, Stop and Drain Protocol

Exclude the first five learner updates from steady-state rate calculations. Report both full-run and
steady-state values.

At the target learner update:

1. Stop admitting new trajectories.
2. Freeze the primary measurement horizon and record time-to-target.
3. Continue a bounded drain until admitted trajectories are consumed, expire or reach the declared
   drain timeout.
4. Classify timeout survivors as terminal-inflight, not stale waste.

Do not count completed-but-unconsumed trajectories at abrupt process shutdown as wasted work.

## 12. Artifact Layout

Store each run under:

```text
output/observations/native_async/<workload>/
  k_<ratio>/tolerance_<S>/seed_<seed>/
    resolved_config.yaml
    trajectory_events.jsonl
    boundary_events.jsonl
    system_snapshots.jsonl
    terminal_report.json
    summary.json
```

Record the exact commit, container image, model path, GPU mapping and command line in `summary.json`.

Produce the following paper-oriented figures:

```text
observation_raw_vs_trainable_goodput.pdf
observation_training_progress_vs_load.pdf
observation_stale_investment_cdf.pdf
observation_count_state_ambiguity.pdf
observation_boundary_reprefill.pdf
```

## 13. Interpretation Rules

1. Do not equate logical inference tokens with actual GPU computation.
2. Do not call terminal-inflight trajectories stale unless the version policy rejected them.
3. Do not compare samples/s without also reporting response-token goodput.
4. Do not change `num_env_groups`, worker count or `group_size` while claiming a pure admission sweep.
5. Do not tune each workload until the desired conclusion appears; preserve and report negative
   results.
6. Report mean, standard deviation and individual seeds for formal runs.
7. Separate system claims from reward or convergence claims.

## 14. Decision After the Observation Stage

Proceed to the proposed runtime comparison only after the native experiments establish which of the
following phenomena are real and material:

```text
throughput-goodput separation
an admission load knee that changes with workload state
deep or expensive trajectories rejected by hard freshness thresholds
large future-supply variance at similar queue counts
substantial post-update survivor re-prefill cost
```

The subsequent evaluation should compare native asynchronous scheduling, fixed admission at the best
offline setting, count-capped admission, individual runtime mechanisms and the unified runtime. The
observation data, not WebShop-specific parameter tuning, should determine that comparison matrix.
