# Version-Boundary-Aware AgenticRL Runtime

## 1. Idea Summary

在全异步、训练与推理全分离的 AgenticRL 系统中，Actor 参数会周期性地从版本 `v` 更新到版本 `v+1`。这一版本边界同时破坏两类已经投入的系统资源：

1. 已经执行多个 turn 的 partial trajectories 可能因 version staleness 超过容忍阈值而被淘汰，已消耗的 GPU 计算和环境交互成本无法转化为训练样本。
2. 推理引擎在更新参数后会 flush 旧版本 KV cache。仍可继续执行的多轮轨迹在新版本第一次生成时需要重新进行全量 prefill。

传统 Agent Serving 虽然也有多轮请求、工具调用和 KV cache 管理，但通常不存在训练驱动的周期性参数版本推进，也不存在由 policy version tolerance 决定样本是否可训练的 freshness deadline。

因此，本系统将 **policy version boundary** 作为一等系统事件，在两个时间尺度上协同优化：

- 跨版本：自适应控制新增轨迹数量，并主动重建新版本的 KV working set。
- 版本内：优先推进接近过期或接近完成的 partial trajectories，使已经投入的计算尽快转化为可训练样本。

## 2. Problem Setting

系统采用以下执行模式：

- Actor learner 与 rollout inference engine 使用不同 GPU，训练和推理全分离。
- Rollout 持续异步生成轨迹，learner 每收集到一个训练 batch 后执行参数更新。
- 每条轨迹记录其开始生成时的 policy version。
- 轨迹只有在 version age 不超过 `trajectory_staleness_tolerance` 时才能进入训练 batch。
- 推理引擎加载新参数后，旧版本 KV cache 全部失效。

定义：

```text
version_age(trajectory) = current_policy_version - start_policy_version
version_slack(trajectory) = staleness_tolerance - version_age(trajectory)
```

系统优化目标不是改变 RL 算法，也不是选择“训练价值更高”的样本，而是提高已投入系统资源转化为可训练样本的比例。

## 3. Core Observation

参数更新不是普通的模型部署事件，而是一次同时影响 trajectory freshness 和 KV residency 的全局 discontinuity：

```text
                         policy update: v -> v+1
                                  |
                 +----------------+----------------+
                 |                                 |
       trajectory freshness changes          old KV is flushed
                 |                                 |
       some partial trajectories             surviving trajectories
       become stale and unusable              require full re-prefill
```

现有 FIFO 或固定 rollout-ahead 策略通常没有显式处理这两个损失：

- 每个训练 step 固定启动一批新轨迹，可能在已有大量 old-version partial trajectories 时继续扩大 producer debt。
- 轨迹按照普通到达顺序获得推理服务，已经执行较深且即将过期的轨迹不一定能及时完成。
- 参数更新后的第一批请求没有以 cache reconstruction 为目标进行组织，可能产生重复或低覆盖率的 full prefill。

## 4. System Design

系统由三个相互配合的机制组成。

### 4.1 Cross-Version Adaptive Trajectory Admission

每次进入新 policy version 时，不再固定增加相同数量的新轨迹。系统根据上一版本和当前遗留状态计算本版本的 admission budget。

输入信号包括：

- 当前未完成轨迹数量。
- 各 version age bucket 中的轨迹数量。
- 上一版本完成、被消费和被淘汰的轨迹数量。
- learner 下一批需要的样本数量。
- 最近的 trajectory completion rate 与 learner consumption rate。
- rollout worker 利用率和排队情况。

抽象决策为：

```text
new_trajectories(v)
    = safe_rollout_capacity(v)
    - committed_work_of_surviving_trajectories(v)
```

第一版可以表示为动态 target outstanding：

```python
target_outstanding = controller(runtime_history, version_age_distribution)
new_count = max(0, target_outstanding - current_outstanding)
```

基本行为：

- 遗留的 old-version partial trajectories 较多时，减少或暂停新增轨迹。
- 遗留轨迹较少且 rollout GPU 有空闲时，增加新轨迹。
- 新增量始终受最小、最大 outstanding bound 限制，避免 GPU starvation 或 producer debt 无界增长。

该机制控制的是每轮新增轨迹数量，而不是改变哪些已完成样本进入 RL 算法。

### 4.2 Post-Update Cache-Rebuild Batching

参数更新后，旧 KV cache 已被 flush。新版本第一次组 batch 时，系统不直接采用普通 FIFO，而是显式构造 cache-rebuild batch。

目标是让第一波不可避免的 full prefill 建立覆盖范围更大的新版本 KV working set：

```text
first batch after update
    -> select requests with complementary/non-overlapping prefixes
    -> rebuild multiple useful KV branches in parallel
    -> later requests reuse the rebuilt prefixes
```

其核心指标不是 request-level cache hit，而是：

```text
saved_prefill_tokens
re-prefill_tokens_after_update
first-wave_prefix_coverage
post-update_cache_warmup_time
```

这一策略依赖一个需要实验验证的推理引擎行为：如果同时进入同一 batch 的相同前缀请求无法复用该 batch 内尚未完成构建的 KV，那么第一批选择不同前缀可以避免并行重复 prefill，并为后续请求建立更多可复用分支。如果引擎支持 batch 内即时 KV 共享，则 batch 构造目标需要相应调整。

Cache-rebuild batching 仅在版本更新后的 warm-up window 生效。进入稳定阶段后恢复普通的 version-aware scheduling。

### 4.3 Intra-Version Urgency Scheduling

在同一个 policy version 内，系统优先推进已经投入计算且面临 freshness deadline 的轨迹。

优先级主要考虑：

1. Version slack：距离 staleness threshold 越近，优先级越高。
2. Remaining system work：在相同 freshness 下，预计更接近完成的轨迹优先，尽快形成可训练样本。
3. KV locality：若轨迹的当前版本 KV 已在某个 worker 上，优先继续在该 worker 执行。
4. Worker load：在 freshness 和 locality 相近时进行负载均衡。

概念上的优先级为：

```python
priority = (
    version_slack,
    estimated_remaining_system_work,
    -cached_prefix_tokens,
    worker_queue_depth,
)
```

该优先级不使用 reward、advantage、成功概率或训练价值。轨迹进度只用于估计完成它还需要多少系统服务，不用于判断样本的算法价值。

## 5. End-to-End Workflow

```text
learner finishes one training step
              |
              v
policy version v -> v+1
              |
              +--> classify surviving/stale partial trajectories
              |
              +--> compute adaptive admission budget
              |
              +--> flush old-version KV and build first cache-rebuild batch
              |
              v
run intra-version urgency scheduling
              |
              +--> prioritize near-expiry / near-completion trajectories
              +--> preserve current-version KV locality when possible
              +--> admit new trajectories using remaining capacity
              |
              v
produce the next learner-consumable batch
```

## 6. Design Principles and Non-Goals

### Design Principles

- Version is the primary system signal.
- Already committed environment and GPU work should be converted into trainable samples whenever it is still feasible.
- Admission, scheduling and KV placement must operate together; optimizing only one can move waste to another stage.
- Control-plane overhead must remain substantially smaller than saved inference and environment cost.
- All policies require FIFO/static fallbacks for controlled A/B experiments.

### Non-Goals

- 不根据 reward、advantage、trajectory success probability 或训练价值选择轨迹。
- 不改变 RL loss、采样概率或 group construction 等算法语义。
- 不通过延迟参数更新来等待 rollout 完成。
- 不要求新旧版本 KV 在同一个推理引擎中共存。
- 不将 sample/s 作为唯一性能指标；主要关注有效训练 token 和被浪费的计算量。

## 7. Required Metrics

### Trajectory Freshness and Admission

```text
outstanding_trajectories
outstanding_version_age_{0,1,2,3,ge_4}
near_expiry_trajectories
adaptive_new_trajectories
effective_outstanding_target
admission_throttled_total
trajectory_completion_rate
learner_consumption_rate
```

### Wasted Work

```text
stale_discarded_trajectories
discarded_actions
discarded_inference_calls
discarded_tool_calls
discarded_prompt_tokens
discarded_response_tokens
discarded_env_seconds
completed_but_unconsumed_trajectories
```

### KV and Prefill

```text
post_update_full_prefill_tokens
post_update_saved_prefill_tokens
post_update_prefix_coverage
cache_rebuild_batch_size
cache_rebuild_time
trajectory_kv_locality_hit_rate
```

### End-to-End Efficiency

```text
trainable_output_tokens_per_second
consumed_trajectories_per_gpu_hour
wasted_inference_tokens_per_consumed_trajectory
wasted_env_seconds_per_consumed_trajectory
learner_idle_time
rollout_gpu_utilization
controller_overhead_seconds
```

## 8. Current Implementation Status

The current MVP has implemented and validated:

- `trajectory_scheduling_policy: fifo | version_priority`.
- Version age-aware ordering in GroupQueue and Router request queues.
- `trajectory_admission_policy: step | outstanding_watermark`.
- A global static outstanding trajectory bound.
- Outstanding, version-age, stale-discard and terminal-waste metrics.
- A real 8-GPU AgenticRL smoke run with separated Megatron training and vLLM rollout.

The following components remain to be implemented:

- Cross-version adaptive calculation of the number of newly admitted trajectories.
- Post-update cache-rebuild batch construction.
- KV prefix metadata and current-version trajectory-to-worker affinity.
- Joint version urgency, KV locality and worker-load routing.
- Long-running A/B experiments across multiple staleness tolerances and seeds.

## 9. Main Experimental Questions

The design should answer four questions independently:

1. Does adaptive cross-version admission reduce stale trajectory work without causing learner or rollout GPU starvation?
2. Does post-update cache-rebuild batching reduce full re-prefill tokens after each model update?
3. Does intra-version urgency scheduling convert more partial trajectories into learner-consumable samples?
4. Are the end-to-end savings larger than the controller, metadata and routing overhead?

The final comparison should include at least:

```text
Baseline: fixed admission + FIFO scheduling
A: adaptive admission only
B: version-priority scheduling only
C: cache-rebuild batching only
Full: adaptive admission + cache rebuild + version-priority scheduling
```

The primary result should report saved/wasted GPU and environment work, rather than only samples per second.
