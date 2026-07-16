# Version-Boundary-Aware AgenticRL Runtime

> Implementation status: the unified boundary-plan, plan-driven KV working-set rebuild and
> trace-driven testbed are implemented and validated. See `UnifiedVersionRuntimePrototype.md` for
> the current code paths, metrics and initial controlled results.

## 1. Idea Summary

在全异步、训练与推理全分离的 AgenticRL 系统中，Actor 参数会周期性地从版本 `v` 更新到版本 `v+1`。这一版本边界同时破坏两类已经投入的系统资源：

1. 已经执行多个 turn 的 partial trajectories 可能因 version staleness 超过容忍阈值而被淘汰，已消耗的 GPU 计算和环境交互成本无法转化为训练样本。
2. 推理引擎在更新参数后会 flush 旧版本 KV cache。仍可继续执行的多轮轨迹在新版本第一次生成时需要重新进行全量 prefill。

传统 Agent Serving 虽然也有多轮请求、工具调用和 KV cache 管理，但通常不存在训练驱动的周期性参数版本推进，也不存在由 policy version tolerance 决定样本是否可训练的 freshness deadline。

因此，本系统将 **policy version boundary** 作为一等系统事件，在两个时间尺度上协同优化：

- 跨版本：通过闭环负载控制动态调节新增轨迹数量，并主动重建新版本的 KV working set。
- 版本内：优先推进接近过期或接近完成的 partial trajectories，使已经投入的计算尽快转化为可训练样本。

这里的核心不是单纯减少 in-flight 轨迹，而是持续跟踪一个动态工作点：rollout producer
提供的有效完成速率应贴近 learner 的消费速率，同时保留足够小的 rollout-ahead buffer
来隐藏环境和推理延迟。

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

#### 4.1.1 Version-Age and Progress-Bucketed Supply Prediction

单一全局 finish ratio 会把不同状态的 carry-over trajectory 混为一类。一个刚启动、
`version_age=1` 的 group 与一个已经完成多次 action、`version_age=2` 的 group，在下一个
版本窗口内成为可训练数据的概率并不相同。Stage B 因此按以下二维状态学习完成率：

```text
version age:     0, 1, 2, 3, >=4
action progress: 0, 1, 2-3, 4-7, >=8
```

由于 learner 按 trajectory group 原子消费，progress 也在 group 粒度定义。系统从所有
候选中取进度最高的 `group_size` 条轨迹，并使用其平均 action 数作为 bucket；同时记录
第 `group_size` 高的 frontier 作为诊断下界。这样既考虑 redundancy，又不会让单条深轨迹
完全掩盖其余未启动候选。

每个 bucket 维护独立 EWMA：

```text
finish_ratio[age, progress]
    = EWMA(groups becoming trainable in the next version window / cohort size)

predicted_existing_supply
    = valid_ready
    + sum(bucket_population * finish_ratio[age, progress])
```

监督事件必须是 `became_trainable`，而不是 `learner_consumed`。后者受 batch size 和 ready
queue 排队截断，会把已经完成但尚未被 learner 取走的数据错误标记为未完成。样本数不足的
bucket 回退到全局 finish EWMA，防止冷启动时的小样本比例直接控制 admission。

该预测器只估计 runtime supply，不改变 trajectory 的动作、reward、训练权重或 group 语义。

#### 4.1.2 Closed-Loop Rollout Load Control

Admission 本质上是 producer-consumer 之间的闭环负载控制，而不是固定 queue-size
调参。系统需要同时避免两个方向的失衡：

```text
load too low
    -> insufficient trainable supply
    -> learner waits for rollout
    -> training GPU idle and end-to-end throughput drops

load too high
    -> partial trajectories accumulate across policy versions
    -> more trajectories exceed the freshness deadline
    -> inference tokens, environment interactions and KV state are wasted
```

控制目标可以写为：

```text
timely_rollout_completion_rate ~= learner_consumption_rate

subject to:
    learner_wait_time <= wait_target
    stale_waste_rate <= waste_target
    min_outstanding <= outstanding <= max_outstanding
```

每个版本的 admission budget 由三部分组成：

```text
new_admission(v)
    = learner_demand(v)
    - predicted_timely_supply_from_existing_work(v)
    + dynamic_safety_reserve(v)
```

其中 `dynamic_safety_reserve` 不能是永久固定的超量采样常数。它只用于吸收轨迹长度、
工具延迟和推理服务时间的短期波动，并根据反馈动态变化：

- learner rollout wait 上升且 rollout 侧仍有可用服务能力时，逐步提高 reserve；仅 inference GPU 空闲不能触发扩容。
- stale discard、near-expiry backlog 或负 admission prediction error 上升时，降低 reserve。
- completion prediction 持续准确且系统稳定时，缓慢调整 reserve，避免频繁振荡。
- reserve 和每版本 admission delta 均设置上下界，避免一次观测噪声造成队列突增或清空。

第一版闭环控制器可以使用带 deadband 的 AIMD/EWMA 组合：

```python
if learner_wait_ewma > wait_high and stale_rate_ewma < stale_low:
    reserve = min(reserve + additive_step, reserve_max)
elif stale_rate_ewma > stale_high or prediction_error_ewma < -error_margin:
    reserve = max(reserve * multiplicative_decay, reserve_min)
```

后续可以替换为 PI controller 或 model-predictive controller，但控制信号仍必须是系统指标，
不能使用 reward、advantage 或训练价值。控制器还应监测 consumed trajectory 的 version age、
长度和任务分布，确认完成时间与调度优先级的相关性没有在有限训练窗口内造成不可接受的
隐式采样偏差。

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
              +--> estimate timely supply and learner demand
              +--> update dynamic reserve from wait/waste feedback
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
- Rollout load must be controlled around a dynamic operating point; neither minimum queue size nor maximum GPU utilization is the objective by itself.
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
dynamic_safety_reserve
timely_rollout_completion_rate
learner_consumption_rate
admission_prediction_error
admission_prediction_absolute_error
controller_wait_error
controller_waste_error
admission_throttled_total
trajectory_completion_rate
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
rollout_wait_time_per_step
rollout_gpu_utilization
controller_overhead_seconds
```

## 8. Current Implementation Status

The current MVP has implemented and validated:

- `trajectory_scheduling_policy: fifo | version_priority`.
- Version age-aware ordering in GroupQueue and Router request queues.
- `trajectory_admission_policy: step | outstanding_watermark | version_adaptive`.
- A global static outstanding trajectory bound.
- A one-shot admission budget computed at each policy-version boundary.
- Initial adaptive supply estimation using valid ready trajectories, unfinished carry-over
  trajectories, an EWMA finish ratio and a configurable reserve.
- Stage B completion prediction bucketed by policy-version age and group action progress, with a
  global EWMA cold-start fallback and minimum per-bucket sample threshold.
- Group-level progress uses the mean of the top `group_size` candidates; frontier, maximum action
  and snapshot coverage are retained as boundary diagnostics.
- Finish-rate supervision now comes from `became_trainable` events rather than learner consumption,
  so ready-queue backlog does not bias completion probability downward.
- Per-version admission budget, usage, expected supply, actual supply and prediction-error metrics.
- A dynamic reserve controller using learner-wait, stale-discard and supply-prediction EWMAs.
- Hysteresis controls that require persistent same-direction pressure and impose a cooldown after
  each reserve change, preventing one noisy version from causing repeated admission changes.
- Consumed-trajectory version-age, action and logical token distributions for detecting implicit workload shifts.
- Outstanding, version-age, stale-discard and terminal-waste metrics.
- A real 8-GPU AgenticRL smoke run with separated Megatron training and vLLM rollout. In the
  initial adaptive run, the admitted trainable trajectory count changed from 24 in version 0 to
  16 in versions 1 and 2 because each later boundary already had 8 valid ready trajectories.

WebShop 50-step experiments show why admission must be closed-loop. With seed 44 and otherwise
identical Full Runtime settings, a fixed reserve sweep produced:

```text
reserve 0:  admitted 200, stale 0,   steady step 17.21 s, rollout wait 7.86 s
reserve 4:  admitted 204, stale 0,   steady step 10.53 s, rollout wait 1.25 s
reserve 8:  admitted 316, stale 96,  steady step 11.84 s, rollout wait 2.44 s
reserve 12: admitted 362, stale 142, steady step 12.01 s, rollout wait 2.59 s
```

`reserve=0` starved the learner, while `reserve=8/12` recreated producer debt and stale waste.
`reserve=4` was the static operating point for this workload, not a universal constant. Model,
environment, tool latency and training speed changes can move this point during one training run.

The first 50-step dynamic-controller run exposed an important negative result:

```text
policy:       admitted  stale  wasted inference tokens  steady step  rollout wait
fixed R=4:         204      0                       0        10.53 s       1.25 s
fixed R=8:         316     96               2,500,702        11.84 s       2.44 s
naive dynamic:     220     15                 432,884        13.57 s       4.77 s
```

The naive per-version AIMD controller repeatedly traversed
`0 -> 2 -> 4 -> 6 -> 8 -> 4 -> 2 -> 0`. It reduced waste relative to fixed `reserve=8`, but its
reaction to startup wait and single-version prediction errors caused oscillation and worse learner
wait than fixed `reserve=4`. The runtime now uses a longer startup warmup plus signal patience and
post-update cooldown. This hysteretic version has passed focused tests but still requires a new
end-to-end 50-step run before it can be claimed as an improvement.

### Utility-Driven Controller Experiments

The next controller stopped treating learner wait as a direct expansion command. It used a
windowed perturb-and-observe loop, with reserve as the control knob and the following effective
output objective:

```text
useful_response_tokens_per_second * compute_efficiency

compute_efficiency = consumed_inference_work
                     / (consumed_inference_work + stale_inference_work)
```

This explicitly permits short rollout or learner idle periods when they reduce stale work enough
to improve end-to-end useful output. Full prompt-prefill tokens are not counted as useful output;
they only contribute to the consumed-versus-wasted compute-efficiency term.

The first 20-step run validated the control path but also showed that reserve changes have delayed
effects: stale work generated under the previous reserve can arrive after the controller has moved
to a new reserve. The runtime therefore excludes a configurable settling interval after each
change. A version-adaptive progress floor was also added: if valid ready plus salvageable in-flight
supply cannot produce the missing portion of the current learner batch, the runtime admits only
the minimum bounded number of groups needed to preserve forward progress.

A 40-step WebShop v3 run completed successfully, including the step where the previous run had
stalled. However, the progress floor did not trigger in that stochastic rerun, so end-to-end
deadlock recovery still needs a forced-underload test. The throughput result was also negative:

```text
policy (first 40 steps)  stale  wasted inference  step wall  rollout wait  effective response tok/s
fixed reserve=4              0                 0    10.71 s        1.46 s                    111.5
fixed reserve=8             74         1,976,189    11.71 s        2.30 s                     86.7
naive dynamic               9           274,037    13.94 s        5.20 s                    105.1
utility v3                 13           314,261    10.00 s        0.98 s                     76.3
```

The v3 controller spent 30 of 40 steps at reserve 6 or 8. Bursty response-token output sometimes
outweighed the stale penalty inside a short window, so unconstrained hill climbing explored high
load too aggressively. The next controller should use constrained optimization: maximize useful
response throughput only while a stale-work or minimum-compute-efficiency budget is satisfied.

### Constrained Admission and Progress-Floor Validation

The runtime now uses constrained utility hill climbing. A reserve increase is rejected when the
aggregate observation window falls below a configurable minimum compute efficiency. Window values
are computed from total useful tokens, stale tokens and elapsed time, rather than averaging
per-version ratios. This prevents a short or small version from receiving the same weight as a
long, expensive version.

The progress-floor supply estimate also discounts salvageable in-flight trajectories by the
observed finish ratio. A forced-underload WebShop run used `reserve=0`, `finish_ratio=0`, a maximum
of 12 outstanding trajectories and four real train/update steps. The floor triggered twice and
admitted eight trajectories in total. All four steps completed, 16 trajectories were consumed and
no stale trajectory was discarded. This validates the deadlock-recovery path under real separated
Megatron training and vLLM rollout.

### Post-Update KV Rebuild Prototype

At each version resume, `EnvAffinityRouter` now opens a bounded rebuild wave. During this wave it
places prompts online so that requests with a large common prefix are not assigned concurrently to
the same worker. After the wave, normal environment-to-worker affinity resumes. The policy changes
only runtime placement; it does not change trajectory admission, sampled actions or learner data.

The initial vLLM request-level metric was invalid: vLLM 0.8.4 V1 declares
`RequestOutput.num_cached_tokens` but does not populate it. The runtime now reads the scheduler's
native prefix-cache block counters through a custom stat logger. The authoritative metrics are:

```text
router/kv_query_blocks
router/kv_hit_blocks
router/kv_saved_prefill_tokens       = hit_blocks * block_size
router/kv_cacheable_reprefill_tokens = (query_blocks - hit_blocks) * block_size
router/kv_block_hit_ratio
router/kv_cache_resets
```

`saved_prefill_tokens` is exact for reusable full cache blocks. The incomplete prompt tail and the
mandatory last-block recomputation in vLLM are outside the cacheable-block denominator. Request-level
cached-token metrics are omitted when the engine cannot provide them; they are no longer reported
as a misleading zero.

A real four-step, single-seed WebShop A/B with training and one parameter update per step produced:

```text
policy    cacheable query tok  saved prefill tok  block hit  response tok/s  step wall
FIFO                 393,408            275,120      69.93%          124.67     58.40 s
rebuild              447,360            314,112      70.21%          127.25     55.54 s
```

After normalization, rebuild improved block hit rate by only 0.28 percentage points and observed
response-token throughput by 2.1%. This is a mechanism-validation result, not a performance claim:
the run is short, asynchronous completion changes the realized prompt workload, and WebShop has a
large shared system prefix that already yields about 70% block hits under FIFO. Multi-seed and
lower-baseline-hit workloads are needed to establish whether rebuild batching has useful headroom.

### Stage A Unified Scheduling Prototype

The runtime now propagates structured trajectory state on every inference turn and orders saturated
per-worker queues lexicographically by policy version, whether the trajectory has already started,
remaining hard action budget and FIFO sequence. This is a system-only priority: reward, task success
and estimated training value are not inputs.

Soft locality is epoch-scoped. A trajectory's affinity is valid only after a successful request in
the current cache epoch; every version resume increments the epoch. For each request the router
compares the affinity worker with the least-loaded worker. Affinity wins while its request pressure
is within `soft_locality_load_slack`; otherwise load overrides locality. This keeps the decision
local and avoids querying all inference-engine caches.

Three real WebShop validations completed: a normal four-step run, a four-step load-override run and
a two-step single-slot queue stress run. The load run observed seven affinity overrides over 156
decisions and about 51 microseconds of routing CPU time per request. The single-slot run forced 73
of 88 requests into the priority path and completed both train steps. These validate functionality,
not performance; the next experiment must compare FIFO, version-only, version-progress and full
routing under the same sustained stale-pressure workload.

The following components remain to be implemented:

- End-to-end validation and tuning of the hysteretic dynamic reserve controller.
- Longer multi-seed calibration of the Stage B bucket estimator and workload-dependent bucket edges.
- Adaptive capacity prediction that also incorporates worker load and tool-latency state.
- Multi-seed validation and first-wave-specific KV accounting for post-update rebuild routing.
- Current-version KV residency metadata beyond environment-to-worker affinity.
- End-to-end ablation of version-progress priority, soft locality and worker-load routing.
- Broader long-running A/B experiments across additional staleness tolerances, workloads and seeds.

## 9. Main Experimental Questions

The design should answer four questions independently:

1. Does closed-loop cross-version admission track workload changes and reduce stale trajectory work without causing learner or rollout GPU starvation?
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
It must also report both sides of the control tradeoff: useful output/inference tokens per second and
stale work per consumed training token. A policy that removes waste by starving the learner, or raises
raw token throughput by overproducing unusable trajectories, is not an improvement.
