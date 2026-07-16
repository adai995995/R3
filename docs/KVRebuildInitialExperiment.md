# Post-Update KV Rebuild: Initial WebShop Experiment

## Setup

- Runtime: fully separated Megatron learner and vLLM rollout.
- Hardware: one node, eight GPUs; GPUs 0-3 train and GPUs 4-7 rollout.
- Model: Qwen3-4B.
- Environment: WebShop, up to 10 actions per trajectory.
- Training: four real learner steps with parameter synchronization.
- Seed: 42.
- Baseline: `EnvAffinityRouter` with FIFO post-update placement.
- Treatment: 16-request post-update rebuild wave, placing prompts to minimize
  within-worker common-prefix overlap before restoring environment affinity.
- Checkpoint saving: disabled.

## Correct KV Metric

vLLM 0.8.4 V1 does not populate `RequestOutput.num_cached_tokens`. The experiment
therefore reads `PrefixCacheStats` directly from the V1 scheduler. For block size
16:

```text
saved_prefill_tokens = prefix_cache_hit_blocks * 16
cacheable_reprefill_tokens = (query_blocks - hit_blocks) * 16
```

This counts reusable full blocks exactly. Incomplete tails and vLLM's mandatory
last-block recomputation are not included in the cacheable-block denominator.

## Results

| Policy | Cacheable query tokens | Saved prefill tokens | Block hit rate | Consumed response tokens | Step wall time | Response tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FIFO | 393,408 | 275,120 | 69.93% | 7,281 | 58.40 s | 124.67 |
| Rebuild | 447,360 | 314,112 | 70.21% | 7,067 | 55.54 s | 127.25 |

The rebuild prototype improved normalized block hit rate by 0.28 percentage
points and observed response-token throughput by 2.1%. This validates the
mechanism and measurement path, but is not yet evidence of a stable performance
gain: the run contains only one seed and four steps, and asynchronous scheduling
changes the realized prompt workload.

WebShop already achieves about 70% block hits under FIFO because requests share a
large system and task-format prefix. The current diversity policy therefore has
limited headroom in this workload. The next evaluation should use multiple seeds
and include a workload with less dominant global-prefix reuse.

## Raw Logs

- `kv_fifo_metrics_v2.log`
- `kv_rebuild_metrics_v2.log`
