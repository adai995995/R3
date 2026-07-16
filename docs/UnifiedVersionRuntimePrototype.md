# Unified Version-Aware Runtime Prototype

## Scope

This stage combines cross-version admission, intra-version trajectory priority and
version-scoped KV placement behind one policy-boundary decision. It also adds a deterministic
trace-driven testbed so mechanisms can be evaluated without tuning a real environment.

The runtime uses only system state. It does not inspect reward, advantage, success probability or
training value, and it does not alter trajectory actions, group semantics or learner sampling.

## Unified Boundary Plan

`GroupQueueManager.advance_step()` now produces one `VersionRuntimePlan` after taking the boundary
snapshot. The plan contains:

```text
policy version
learner demand and safety reserve
predicted existing trainable supply
current outstanding work
new-trajectory admission budget
freshness deadline
ordered invested carry-over groups
KV rebuild target trajectory count
```

Carry-over groups are ordered by version age and invested action progress. The rebuild target uses
the actual number of invested candidates in each group rather than assuming every group is full.

The same plan has three consumers:

1. `GroupQueueManager` applies its admission budget.
2. Per-request trajectory state supplies the matching deadline/progress priority to the Router.
3. `RouterManager.resume(plan)` starts the new cache epoch and installs the planned rebuild cohort.

The old scalar `resume(version)` path remains supported for configurations that do not enable
version-adaptive admission.

## Working-Set Rebuilding

At a cache epoch change, `EnvAffinityRouter` now:

1. Invalidates old trajectory affinity and cached-prefix observations.
2. Installs the invested carry-over group keys from the boundary plan.
3. Selects one first request per planned trajectory for the bounded rebuild wave.
4. Places rebuild prompts to minimize prefix overlap within each worker while balancing assignments.
5. Records successful prompts as the current-version worker working set.
6. Routes later first-touch trajectories to the worker with the longest observed current-version
   prefix, subject to the existing load-slack override.
7. Keeps continuation requests on their valid current-version affinity worker.

If there is no carry-over cohort, the configured rebuild budget seeds a cold-start working set. If
planned candidates do not arrive within the bounded observation window, the router falls back to
other first-touch requests instead of leaving the warm-up phase stuck.

This is an online Router-level first wave. It does not add a strict barrier inside vLLM or force all
rebuild requests into one physical engine batch. Avoiding that barrier keeps environment threads and
rollout GPUs from waiting solely for cache warm-up.

## New Metrics

```text
scheduler/version_runtime_plan_enabled
scheduler/version_runtime_rebuild_candidates
scheduler/version_runtime_rebuild_target
scheduler/version_runtime_priority_deadline
router/version_runtime_plan_request
router/rebuild_candidate_request
router/rebuild_candidate_request_ratio
router/working_set_prefix_selected
router/working_set_prefix_selected_ratio
```

Existing actual KV metrics remain the source of truth for cache benefit:

```text
router/kv_saved_prefill_tokens
router/kv_cacheable_reprefill_tokens
router/kv_block_hit_ratio
router/kv_cache_resets
```

## Trace-Driven Testbed

`roll/distributed/scheduler/version_runtime_testbed.py` is a dependency-free simulator with both a
Python API and CLI. A trace fixes every trajectory's:

```text
action length
prefix class and prefix tokens
tokens per action
response tokens per action
tool delay per action
```

The experiment independently controls:

```text
number of policy versions
learner demand per version
inference actions available per version
staleness tolerance
maximum outstanding trajectories
safety reserve
number of rollout workers
post-update rebuild budget
```

The exact same immutable trace is used for `fixed_fifo` and `unified`. The testbed reports admission,
completion, learner shortfall, consumed version age, stale actions/tokens, prefill tokens, saved
prefill tokens, rebuild requests and prefix-locality routes. Results are mechanism-level predictions,
not GPU throughput measurements.

Example:

```bash
python -m roll.distributed.scheduler.version_runtime_testbed \
  --versions 30 --learner-demand 8 --service-actions 32 \
  --tolerance 2 --max-outstanding 40 --reserve 8 \
  --workers 4 --rebuild-budget 8 --seed 42 \
  --trajectories 1024 --write-trace output/version_runtime_trace_seed42.json \
  --output output/version_runtime_testbed_seed42.json
```

## Validation

Focused production and testbed tests passed in `xxl_test`:

```text
18 passed
Python 3.10 compilation passed
```

The 30-version fixed-trace pressure experiment produced:

| Metric | Fixed FIFO | Unified Runtime |
| --- | ---: | ---: |
| Admitted trajectories | 412 | 297 |
| Completed/consumed trajectories | 54 | 165 |
| Learner shortfall trajectories | 186 | 75 |
| Stale trajectories | 322 | 106 |
| Stale logical inference tokens | 845,152 | 111,520 |
| Prefill saved ratio | 37.2% | 76.9% |
| Mean consumed version age | 1.43 | 1.88 |

The result validates the controlled mechanism and exposes a real trade-off: salvaging invested work
can improve resource conversion while increasing the age of consumed samples. It must not be read as
a production speedup.

The real Qwen3-4B WebShop smoke completed 4/4 learner steps on eight GPUs. It observed:

```text
boundary plan attached to every routed request
step 2: 2 carry-over groups, 4 invested trajectories
candidate-aware rebuild requests after the boundary
step 3: 2 post-rebuild prefix-locality placements
0 stale discards during the short smoke
0 tracebacks
```

The run ended with two completed-but-unconsumed trajectories and one known rollout-loop shutdown
timeout warning. The training process still exited successfully; this warning is part of the existing
shutdown path and is not evidence of a runtime decision failure.

## Artifacts

- `output/version_runtime_trace_seed42.json`
- `output/version_runtime_testbed_seed42.json`
- `output/experiment_logs/version_runtime/webshop_unified_runtime_4step.log`
- `output/webshop_qwen3_4b_unified_runtime_4step/terminal_waste.step_4.json`
- `examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_unified_runtime_4step.yaml`
