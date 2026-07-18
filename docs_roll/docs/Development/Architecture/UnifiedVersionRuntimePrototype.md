# Unified Version-Aware Runtime Prototype

## Scope

This stage combines cross-version admission, intra-version trajectory priority and
version-scoped KV placement behind one policy-boundary decision. It also adds a deterministic
trace-driven testbed so mechanisms can be evaluated without tuning a real environment.

The runtime uses only system state. It does not inspect reward, advantage, success probability or
training value, and it does not alter trajectory actions, group semantics or learner sampling.

## Unified Boundary Plan

`GroupQueueManager.advance_step()` now produces one `VersionRuntimePlan` after taking the boundary
snapshot. The control path is explicitly split into:

```text
VersionRuntimeState -> VersionAwareRuntimeController.decide() -> VersionRuntimePlan
```

The observed state contains only system information. The plan contains:

```text
policy version
learner demand and safety reserve
predicted existing trainable supply
current outstanding work
whether adaptive admission and version priority are enabled
new-trajectory admission budget
admission deficit, available capacity and an explicit decision reason
freshness deadline
ordered invested carry-over groups for trajectory priority
KV rebuild target trajectory count
```

Carry-over groups are ordered by version age and invested action progress. The rebuild target uses
the actual number of invested candidates in each group rather than assuming every group is full.

The same plan has three consumers:

1. `GroupQueueManager` applies its admission budget.
2. `EnvAffinityRouter` combines its ordered priority cohort with each request's latest
   version/progress state.
3. `RouterManager.resume(plan)` starts the new cache epoch and installs the planned rebuild cohort.

One plan is published for every training version. When adaptive admission is disabled for an
ablation, the plan reports `admission_reason=disabled` and an admission budget of zero, while its
priority and KV cohorts remain available. The old scalar `resume(version)` path remains supported
for callers outside this control loop.

Admission reasons are deliberately small and deterministic:

```text
disabled
existing_supply_sufficient
supply_deficit
partial_capacity
outstanding_cap
```

This makes it possible to explain each boundary decision without reconstructing controller state
from several unrelated metrics.

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
scheduler/version_runtime_admission_enabled
scheduler/version_runtime_admission_reason
scheduler/version_runtime_admission_deficit
scheduler/version_runtime_admission_capacity
scheduler/version_runtime_priority_candidates
scheduler/version_runtime_rebuild_candidates
scheduler/version_runtime_rebuild_target
scheduler/version_runtime_priority_deadline
router/version_runtime_plan_request
router/planned_priority_candidate_request
router/planned_priority_candidate_request_ratio
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

### Per-request vLLM 0.8.4 accounting

The bundled vLLM 0.8.4 V1 output type declares a cached-token field but does not populate it. ROLL
now installs a compatibility hook that records each new request's exact initial
`num_computed_tokens` in the engine-core scheduler. Before the first output is yielded,
`CustomAsyncLLM` retrieves that value once through the existing engine utility channel and populates
the request output. Aborted requests discard unconsumed measurements.

This closes the per-request accounting path without changing request order, generated tokens or
learner-visible samples. It adds one local control RPC per generation in this compatibility version;
an upgrade to a vLLM release with native per-request prefill statistics can remove that RPC.

The request-level metrics are:

```text
vllm/request_prompt_tokens
vllm/request_cached_prompt_tokens
vllm/request_prefill_tokens
vllm/request_kv_hit_ratio
```

At a policy boundary, the first survivor request is associated using the Router's active cache epoch
rather than the trajectory's locally retained execution version. This is measurement-only: it does
not change freshness acceptance or the trajectory version. Boundary records now contain the exact
cached and prefill tokens together with action progress, worker placement and route reason.

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

The latest controller, Router and testbed regression tests passed in `xxl_test`:

```text
38 scheduler/Router tests passed, 1 legacy Ray-heavy test deselected
4 trace-driven testbed tests passed
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

After the explicit state/controller refactor, a second 4-step WebShop smoke completed successfully
with no traceback. Its boundary decisions were:

| Step | Admission deficit | Priority groups | Rebuild groups | Planned requests served |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 0 | 0 | 0 |
| 1 | 4 | 0 | 0 | 0 |
| 2 | 6 | 2 | 2 | 6 |
| 3 | 2 | 1 | 1 | 4 |

All four admission decisions reported `supply_deficit`. The terminal report contained 16 learner-
consumed trajectories, 6,268 trainable response tokens, 105.3 trainable response tokens/s and no
stale discard. Four additional completed trajectories remained unconsumed at the short terminal
horizon. This run validates control-path integration only; it is too short to estimate goodput
improvement.

The vLLM 0.8.4 KV-accounting validation then completed another real 4-step Qwen3-4B WebShop run. It
observed three policy boundaries and seven survivor first requests. All seven records were supplied
by the inference engine:

| Metric | Value |
| --- | ---: |
| Survivor first requests | 7 |
| Engine-reported records | 7 |
| Logical survivor prompt tokens | 44,566 |
| Engine-reported cached tokens | 0 |
| Engine-reported prefill tokens | 44,566 |

The equality between logical prompt and engine-reported prefill tokens confirms full re-prefill on
these first post-update requests after cache invalidation. The sample included two deeply invested
survivors at 17 and 18 completed actions; they separately re-prefilled 14,461 and 15,440 tokens.
This is direct boundary-cost measurement, not an estimate based on logical context length.

The short run also produced 16 learner-consumed trajectories and 9 stale trajectories. Stale work
contained 68 actions and 382,319 logical inference tokens. These values characterize this smoke's
pressure and should not be interpreted as a tuned policy comparison. The existing rollout-loop
shutdown timeout warning occurred once; the pipeline still completed and released all GPU workers.

## Artifacts

- `output/version_runtime_trace_seed42.json`
- `output/version_runtime_testbed_seed42.json`
- `output/experiment_logs/version_runtime/webshop_unified_runtime_4step.log`
- `output/webshop_qwen3_4b_unified_runtime_4step/terminal_waste.step_4.json`
- `examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_unified_runtime_4step.yaml`
- `output/webshop_qwen3_4b_controller_smoke_4step/terminal_waste.step_4.json`
- `examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_controller_smoke_4step.yaml`
- `output/webshop_qwen3_4b_kv_metrics_validate_4step/terminal_waste.step_4.json`
- `examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_kv_metrics_validate_4step.yaml`
