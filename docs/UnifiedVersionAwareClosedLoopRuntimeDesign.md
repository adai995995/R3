# Unified Version-Aware Closed-Loop Runtime Design

状态：设计草案，基于 2026-07-31 的当前实现

## 1. 文档目的

本文档定义如何将当前 version-aware runtime 从“共享一份计划的多个启发式机制”推进为真正的反馈闭环。设计遵循增量演进原则：保留已经跑通的 admission、trajectory priority、KV reconstruction、Router feedback 和在线 top-up 路径，不重写 AgenticRL pipeline，也不改变 RL 算法、轨迹动作或 learner 的样本有效性判定。

本文档回答四个问题：

1. 当前代码已经形成了什么闭环；
2. 当前闭环为什么仍不完整；
3. admission、scheduling、placement 和 KV reconstruction 如何共享同一套预测和反馈；
4. 下一阶段应修改哪些接口、代码路径、指标和测试。

## 2. 目标与非目标

### 2.1 目标

系统的首要目标是缩短下一批 freshness-valid 训练数据的形成时间，同时减少无法被 learner 消费的已投入工作。

采用分层目标，而不是将多个指标任意加权成一个难以解释的分数：

1. 满足 learner 下一批有效样本需求；
2. 在满足需求的候选中，减少 trajectory expiration；
3. 在有效完成能力相近时，减少排队、full re-prefill 和重复 KV reconstruction；
4. 使用 FIFO sequence 作为最终公平性规则，避免长期饥饿。

`rollout tokens/s`、GPU utilization 和 request concurrency 是诊断信号，不是最终优化目标。论文中的主要端到端目标仍是 policy-update interval；资源浪费由 expired tokens、actions、tool calls 和 tool time 衡量。

### 2.2 非目标

- 不使用 reward、advantage、任务成功概率或训练价值进行 admission 或 scheduling；
- 不改变 task/prompt 的原始到达顺序和采样概率；
- 不改变 GRPO group construction、PPO sample semantics 或 learner loss；
- 不延迟 policy update 来等待旧轨迹完成；
- 不维护新旧模型版本 KV 共存；
- 不迁移 worker 之间的物理 KV payload，也不引入 CPU KV offload；
- 第一阶段不实现昂贵的全局最优匹配或新的 RL scheduler。

## 3. 当前实现基线

### 3.1 已有执行闭环

当前代码已经能够重复执行：

```text
trajectory / learner / KV state snapshot
                 |
                 v
VersionRuntimeState
                 |
                 v
VersionAwareRuntimeController.decide()
                 |
                 v
VersionRuntimePlan
       |                 |
       v                 v
GroupQueueManager   EnvAffinityRouter
admission/top-up    priority/placement/rebuild
       |                 |
       +--------+--------+
                v
       completion / expiration / KV metrics
                |
                v
       next policy-version decision
```

主要代码映射如下：

| 组件 | 当前代码位置 | 当前职责 |
| --- | --- | --- |
| Version state/plan/controller | `roll/distributed/scheduler/rollout_scheduler.py` | 边界快照、admission budget、priority/rebuild cohort、在线 revision |
| Trajectory pool | `GroupQueueManager` / `GroupQueue` | group 生命周期、valid-ready supply、staleness、admission |
| Request metadata | agentic env managers | 每次 generation boundary 附带 version 和 action progress |
| Request routing | `RouterManager` / `EnvAffinityRouter` | worker 选择、每 worker priority queue、KV shadow directory |
| Physical KV | vLLM rollout workers | KV block 分配、prefix cache、reset 和真实命中计数 |
| Policy activation | `RolloutScheduler` / Agentic pipeline | 暂停 dispatch、更新权重、激活新版本并发布 runtime plan |

### 3.2 已经存在的反馈

当前实现并非纯开环：

- unfinished groups 按 `version_age x action_progress` 分桶；
- 每个 bucket 使用 EWMA 学习下一版本窗口内成为 trainable 的比例；
- 下一边界记录 predicted existing supply 与 actual existing supply；
- learner wait、stale fraction 和 supply prediction error 可调整 safety reserve；
- learner 等待超过阈值时，同一版本内允许每次追加一个 group；
- vLLM reset 和 prefix-cache 计数反馈会使 Router 失效对应 KV shadow state。

### 3.3 当前仍未闭合的部分

当前系统是“执行闭环”，但还不是“统一决策闭环”：

1. Admission 预测的是 bucket completion ratio，scheduler 使用固定字典序，placement 使用固定 KV/load threshold，三者没有共享 completion-time/cost estimate。
2. `expected_existing_supply` 只描述可能完成多少，不描述什么时候完成；因此无法直接判断它能否降低当前 next-batch latency。
3. `remaining_actions = max_actions - actions_completed` 是 action budget 余量，不是真实 remaining service time。
4. tool-wait、context length、worker queue delay 和 re-prefill cost 尚未进入 supply predictor。
5. KV 命中反馈目前只用于统计和 shadow invalidation，不用于在线校准 placement cost。
6. priority 执行结果没有回流来校准“哪些轨迹实际上可被挽救”。
7. supply prediction error 同时被当作模型误差和负载控制信号。二者含义不同：预测过于乐观可能要求增大补量，也可能来源于过载导致完成率下降。必须先用实际 wait/waste 区分，不能仅凭误差符号决定 reserve 方向。
8. 当前 experimental utility controller 使用 token-rate utility。由于 trajectory 长度方差很大，它不适合作为最终主控制目标，应保留为实验 baseline，而不是默认闭环。

## 4. 闭环定义

只有同时满足以下条件，才称为完整闭环：

1. 每个 version/revision 生成唯一、可追踪的 plan；
2. plan 中保存决策时的供给、延迟和成本预测；
3. admission、priority、placement 和 reconstruction 使用同一份预测状态；
4. completion、expiration、learner wait、prefill 和 queue delay 能归因到对应 plan；
5. actual outcome 与 prediction 的残差会更新 estimator；
6. 更新后的 estimator 会影响下一版本的 admission 和同版本后续 request routing；
7. 所有决策均有 hard cap、group alignment、FIFO fallback 和可关闭的 ablation 开关。

目标数据流为：

```text
RuntimeStateSnapshot
        |
        v
Shared RuntimeEstimator
        |
        v
VersionRuntimeForecast
        |
        v
Unified RuntimeController
        |
        +--> admission budget
        +--> priority cohort / ranks
        +--> rebuild cohort
        +--> routing model snapshot
        |
        v
Runtime execution
        |
        v
VersionRuntimeOutcome
        |
        v
Estimator.observe() + reserve correction
        |
        +----------------------> next decision
```

## 5. 统一计量单位与语义

### 5.1 Learner demand

当前代码中的 `rollout_batch_size`、`valid_ready` 和 `expected_existing_supply` 以 trainable trajectory slots 计量。对于一个 group：

```text
trainable contribution = group_size
admission cost          = group_size + group_size_redundancy
```

设计继续使用该计量方式，以避免改动现有 batch 语义：

- PPO 或独立 trajectory：`group_size = 1`；
- GRPO：只有 group 满足现有 valid/group filter 条件时，才贡献 `group_size` 个 trainable slots；
- supply predictor 必须在 group 层判断 trainability，不能简单累加互不独立的 trajectory completion probability。

### 5.2 Freshness deadline

```text
version_age_i   = active_version - start_version_i
version_slack_i = staleness_tolerance - version_age_i
```

`version_slack` 是版本单位的硬有效性预算。为了估算墙钟完成可行性，estimator 使用近期 policy-update interval 的稳健 EWMA，将 slack 映射为近似时间预算。该估计只用于 runtime ordering，不改变 staleness 的真实判定；真实判定仍由现有 version tolerance 执行。

### 5.3 Progress

`actions_completed / max_actions` 只能称为 action-budget consumption，不能称为真实任务完成比例。闭环模型使用：

- actions completed；
- empirical remaining generation time；
- empirical remaining tool time；
- group completion frontier；
- context tokens 和可能的 re-prefill cost。

## 6. Shared Runtime Estimator

### 6.1 Estimator 输出

对每个未完成 group `g`，estimator 输出：

```text
p_valid(g)          group 在过期前成为 trainable 的概率
eta_ready(g)        group 形成 trainable slots 的预计剩余时间
valid_slots(g)      完成时能够贡献的 trainable slots
confidence(g)       当前 bucket 的样本覆盖和置信度
```

对每条 inference-ready trajectory `i` 和候选 worker `w`，输出：

```text
queue_eta(w)
prefill_eta(i, w)
decode_eta(i)
tool_eta(i)
completion_eta(i, w)
estimated_cached_tokens(i, w)
```

初版计算为：

```text
completion_eta(i, w)
  = queue_eta(w)
  + prefill_eta(i, w)
  + remaining_decode_eta(i)
  + remaining_tool_eta(i)
```

其中 `prefill_eta(i, w)` 使用 context length、prefix estimate 和引擎报告的实际 prefill 样本校准。

### 6.2 Predictor 分桶

第一版不引入离线模型，继续使用在线 bucket/EWMA，以控制开销和冷启动风险。建议将当前二维 bucket 逐步扩展为：

```text
group supply bucket:
  version age
  group progress frontier
  readiness composition: ready / inference / tool-wait / unstarted

service-time bucket:
  context-token range
  completed-action range
  environment/tool class
  cache state: affinity hit / prefix hit / full re-prefill
```

为避免状态爆炸，只有样本量足够的 bucket 使用局部 EWMA，否则逐级回退：

```text
fine bucket -> coarse workload bucket -> global EWMA -> conservative default
```

### 6.3 Outcome supervision

不同预测使用不同监督事件：

- `p_valid(g)`：group 是否在 freshness 过期前成为 trainable；
- `eta_ready(g)`：从 snapshot 到 group 成为 trainable 的时间；
- `queue_eta(w)`：Router 排队开始到请求下发的时间；
- `prefill_eta(i,w)`：引擎报告的 prefill tokens/time；
- `decode_eta(i)`：generation wall time 和 generated tokens；
- `tool_eta(i)`：tool-call start/end wall time。

`learner_consumed` 不能作为 group completion predictor 的负样本，因为 completed-but-unconsumed 数据可能只是尚未进入当前 learner batch。

## 7. Unified Decision Logic

### 7.1 Supply forecast

令：

- `B_v`：下一批需要的 trainable trajectory slots；
- `R_v`：已经 completed、freshness-valid 且未消费的 slots；
- `G_v`：当前未完成且仍可挽救的 groups。

已有池的预测供给为：

```text
F_hat_v = sum over g in G_v of p_valid(g) * valid_slots(g)
```

同一 group 只计一次，避免将成员 trajectory 独立累加造成供给高估。

### 7.2 Admission

未覆盖需求为：

```text
U_v = max(0, B_v + reserve_v - R_v - F_hat_v)
```

实际 admission 按 group 对齐：

```text
needed_groups  = ceil(U_v / group_size)
capacity_groups = floor(
    (max_outstanding - current_outstanding)
    / (group_size + redundancy)
)
admitted_groups = min(needed_groups, capacity_groups)
```

`max_outstanding` 是防止无界 producer debt 的安全上限，不是控制器努力维持的 target。

新 task 继续从原 task queue 按原始顺序获得，admission 只决定数量，不选择任务内容。

### 7.3 Scheduling

调度的一级判断是 feasibility，而不是固定权重分数：

1. 优先服务仍可能在 freshness deadline 前完成的 group；
2. 在可行 group 中，优先较小 version slack；
3. 同等 slack 下，优先较短 predicted remaining service；
4. 同等系统成本下，优先已经投入 GPU/tool 成本的 group；
5. 最后使用 FIFO sequence。

如果一个 group 已经被预测为不可挽救，scheduler 不应仅因为它最老而长期占用最高优先级。该 group 仍按现有 staleness semantics 处理，预测器本身不直接修改其有效性。

初版继续保留当前 non-preemptive request-boundary scheduling：priority 在 generation boundary 更新，请求进入 vLLM 后不被本系统中断。

### 7.4 Placement

worker 选择使用与 scheduling 相同的 completion estimate：

```text
w* = argmin_w completion_eta(i, w)
```

它自然同时包含：

- worker queue pressure；
- trajectory current-version affinity；
- prefix locality；
- full re-prefill cost；
- load override。

第一阶段仍保留当前“先在线选择 worker，再进入该 worker priority queue”的 Router 结构。全局 trajectory-worker bipartite matching 不属于第一阶段，因为它会显著扩大控制路径和锁竞争。

### 7.5 Post-refresh KV reconstruction

版本激活后，physical KV 仍由 vLLM 管理。Router：

1. 递增 cache epoch，失效旧 affinity；
2. 从 `p_valid` 较高且已投入 GPU/context 成本的 survivor 中生成 rebuild cohort；
3. 在 bounded coalescing window 内按 freshness feasibility 排序；
4. 对候选 worker 同时考虑 prefix diversity、queue ETA 和后续 locality benefit；
5. 记录真实 scheduler batch、prefill tokens、cache hits 和 reset；
6. 使用这些结果校准下一边界的 `prefill_eta` 与 working-set state。

该过程是 Router-level reconstruction wave，不承诺所有逻辑 wave 请求一定进入同一个 vLLM physical batch。

## 8. Feedback Controller

### 8.1 分离三类误差

必须区分：

```text
forecast error:
  actual existing supply - predicted existing supply

undersupply error:
  learner demand - valid supply available in time

overload signal:
  expired invested work + excessive queue delay
```

Forecast error 主要更新 estimator；undersupply 才推动增加 reserve/top-up；overload 才推动减少 reserve。这样可以消除当前 prediction error 直接控制 reserve 时的方向歧义。

### 8.2 Reserve 更新

第一版沿用带 hysteresis 的 AIMD，但更换输入语义：

```python
if persistent_undersupply and overload_below_budget:
    reserve = min(reserve + additive_step, reserve_max)
elif persistent_overload:
    reserve = max(floor(reserve * decay), reserve_min)
else:
    reserve = reserve
```

其中：

- `persistent_undersupply` 来自 next-batch assembly wait 和 missing valid slots；
- `persistent_overload` 来自 expired invested tokens/actions/tool time 和 queue delay；
- forecast residual 不直接扩缩容，只负责校准 `p_valid` 和 ETA；
- signal patience、deadband 和 cooldown 保留，防止版本间振荡。

### 8.3 Same-version reconciliation

保留当前 bounded top-up 路径：

- 仅当 learner 仍缺数据且等待超过 `reconcile_wait_seconds` 时触发；
- 使用最新 pool snapshot 重新计算 supply；
- 每次最多新增一个 group；
- 每个 policy version 限制 revision 次数；
- revision 只能增加本版本 admission budget，不撤销已经准入的轨迹；
- revision 更新 Router candidate cohorts，但不切换 cache epoch。

## 9. 两个控制时间尺度

### 9.1 Policy-boundary slow loop

```text
finalize outcome of version v-1
  -> update group/service/KV estimators
  -> snapshot valid-ready and unfinished pool
  -> predict F_hat_v
  -> update reserve from undersupply/overload
  -> compute admission budget
  -> construct priority and rebuild cohorts
  -> publish plan(version=v, revision=0)
```

### 9.2 Within-version fast loop

```text
trajectory reaches generation boundary
  -> publish progress/readiness/context
  -> estimate completion_eta per candidate worker
  -> choose worker
  -> enqueue by feasibility/laxity/FIFO order
  -> execute request
  -> record queue/prefill/decode/KV outcome

learner remains undersupplied
  -> reconcile latest pool state
  -> optionally admit one group
  -> publish monotonic plan revision
```

控制器不在 per-token decode path 上运行。

## 10. 数据结构修改

### 10.1 新增 `RuntimeCandidateState`

建议用明确字段替代不断扩展的匿名 tuple：

```python
@dataclass(frozen=True)
class RuntimeCandidateState:
    trajectory_id: str
    group_id: int
    episode_id: int
    env_id: int
    policy_version: int
    current_version: int
    actions_completed: int
    max_actions: int
    readiness: str
    context_tokens: int
    inference_calls: int
    tool_calls: int
    generate_seconds: float
    tool_seconds: float
    kv_owner: int | None
    cache_epoch: int | None
```

`kv_owner` 可以在构造 snapshot 时由 Router shadow directory 合并，不要求 env manager 持有。

### 10.2 新增 `VersionRuntimeForecast`

```python
@dataclass(frozen=True)
class VersionRuntimeForecast:
    version: int
    forecast_id: str
    ready_valid_slots: int
    predicted_inflight_slots: float
    predicted_supply_by_bucket: dict[str, float]
    predicted_group_completion: dict[str, float]
    predicted_group_eta_seconds: dict[str, float]
    estimator_revision: int
```

### 10.3 扩展 `VersionRuntimePlan`

保留现有字段，新增：

```text
plan_id
forecast_id
estimator_revision
ready_valid_slots
predicted_inflight_slots
reserve_before / reserve_after
undersupply_signal
overload_signal
priority_reason/version for diagnostics
```

计划中不需要携带完整 per-worker KV payload 或全部 prompt tokens。Router 使用 plan cohort 加自己的 bounded prefix directory 在线 placement。

### 10.4 新增 `VersionRuntimeOutcome`

```python
@dataclass(frozen=True)
class VersionRuntimeOutcome:
    plan_id: str
    version: int
    final_revision: int
    actual_existing_valid_slots: int
    admitted_trajectories: int
    completed_valid_slots: int
    consumed_valid_slots: int
    learner_wait_seconds: float
    next_batch_latency_seconds: float
    expired_trajectories: int
    expired_actions: int
    expired_tokens: int
    expired_tool_calls: int
    expired_tool_seconds: float
    reprefill_tokens: int
    prefill_seconds: float
    scheduling_wait_seconds: float
```

Outcome 在下一次 version boundary finalization，随后调用 `RuntimeEstimator.observe(outcome)`。

## 11. 代码修改位置

### 11.1 `rollout_scheduler.py`

- 引入具名 candidate/forecast/outcome dataclass；
- 将 `_predict_unfinished_supply()` 封装到 `RuntimeEstimator`；
- 在 `_reset_version_admission()` 开头 finalize 上一版本 outcome；
- 将 forecast residual 与 reserve control signal 拆开；
- 保留 `_admit_version_budget()` 和 `reconcile_version_progress()` 的 group alignment 与 hard cap；
- `VersionAwareRuntimeController.decide()` 接收 forecast，不再自行推断匿名 candidate tuple 的全部语义。

### 11.2 `router.py`

- 继续维护 worker-local pressure、cache epoch 和 bounded prefix directory；
- 增加 queue/prefill/decode EWMA 的 worker feedback snapshot；
- priority key 改为消费 controller/estimator 产生的 feasibility/laxity 信息；
- placement 从固定 `load_slack` 主导逐步迁移到 predicted completion ETA；
- 保留 fixed threshold 作为 estimator 冷启动和 ablation fallback；
- request 结果带回 `plan_id/forecast_id`，保证 outcome attribution。

### 11.3 Env managers

- 继续只在 generation/tool boundary 发布状态；
- 补齐明确的 `readiness`、`context_tokens`、tool start/end 和累计 tool seconds；
- 不进行 token-level RPC；
- 不发送 reward、advantage 或 learner-side value。

### 11.4 vLLM strategy

第一阶段不修改 physical KV policy。复用现有：

- request prompt/cached/prefill tokens；
- engine prefix-cache block counters；
- reset counters；
- scheduler batch ID/size。

只有在缺少可靠 prefill wall time 时，才增加轻量 request-level timing，不修改 block allocator。

## 12. 指标

### 12.1 每个 plan/version

```text
runtime/plan_id
runtime/plan_revision
runtime/ready_valid_slots
runtime/predicted_inflight_slots
runtime/actual_existing_valid_slots
runtime/supply_prediction_error
runtime/supply_prediction_abs_error
runtime/admission_budget
runtime/admission_used
runtime/reserve_before
runtime/reserve_after
runtime/undersupply_slots
runtime/overload_expired_token_fraction
runtime/next_batch_latency_seconds
```

### 12.2 Scheduling/placement

```text
runtime/candidate_completion_probability
runtime/candidate_eta_seconds
runtime/candidate_laxity_seconds
router/scheduling_wait_seconds
router/selected_worker_queue_eta_seconds
router/predicted_prefill_tokens
router/actual_prefill_tokens
router/prefill_prediction_error
router/locality_override_reason
```

### 12.3 Outcome/waste

```text
expired trajectories
expired actions per trajectory
expired generated/prefill tokens
expired tool calls per trajectory
expired tool time
completed-but-unconsumed trajectories
full-context re-prefill tokens per refresh
prefill time per refresh
```

Raw rollout throughput 继续报告，但不用于替代 policy-update interval。

## 13. 正确性约束

必须在代码中断言：

1. admission budget 始终按 `group_size + redundancy` 对齐；
2. trainable contribution 始终按现有 group semantics 计算；
3. outstanding 不超过 hard cap；
4. plan revision 在同一 version 内单调递增；
5. stale 判定只由真实 policy version/tolerance 决定，不由 predictor 决定；
6. estimator 缺失或异常时回退到当前 bucket predictor + FIFO + least-loaded routing；
7. control metadata 不包含 reward、advantage 或 success label；
8. 任何 workload bucket 的冷启动都不能阻止 progress floor；
9. Router engine reset 必须同时失效对应 prefix/affinity shadow state。

## 14. 实施阶段

### Phase A：闭合 admission feedback

目标：先让“预测多少、实际多少、下一轮如何修正”完全可追踪。

1. 增加 `plan_id`、`VersionRuntimeForecast` 和 `VersionRuntimeOutcome`；
2. 统一统计窗口，finalize 上一版本供给和浪费；
3. 将 forecast error 与 undersupply/overload signal 分离；
4. reserve 仅由 undersupply/overload 更新；
5. bucket probability 仅由 completion outcome 更新；
6. 保留 bounded reconciliation；
7. 增加 prediction calibration 和 plan attribution 测试。

完成标准：每个版本都能从日志还原 `forecast -> plan -> actual -> update`，不存在无法归因的 admission change。

### Phase B：统一 remaining-service estimator

1. 增加 readiness、context、generation time 和 tool time buckets；
2. 预测 group completion ETA；
3. scheduler 使用 feasibility/laxity 替换纯 action-budget ordering；
4. 保留现有 priority key 作为 fallback；
5. 验证 near-completion stale trajectories 中 inference-ready queued 的可挽救比例。

完成标准：priority 决策能够解释为“预计何时形成有效 group”，而不只是“版本老、action 多”。

### Phase C：统一 placement 与 KV cost

1. 将 worker queue ETA 和 estimated prefill 纳入 completion ETA；
2. 用真实 vLLM counters 校准 prefill estimate；
3. rebuild cohort 使用同一 group feasibility；
4. working-set routing 比较 locality benefit 与排队代价；
5. 保留无 KV migration 的实现边界。

完成标准：scheduling、placement 和 reconstruction 对同一候选给出一致的完成时间解释。

### Phase D：端到端验证

固定代码后比较：

```text
FIFO + fixed admission
adaptive admission only
admission + version scheduling
full closed-loop runtime
```

至少报告：

- raw rollout throughput；
- mean policy-update interval；
- expired tokens/actions/tool calls/tool time；
- next-batch latency trace；
- supply prediction calibration；
- boundary re-prefill tokens/time；
- controller CPU time 和 scheduling wait；
- 相同训练预算下的 reward/收敛曲线，验证未造成不可接受的训练偏差。

## 15. 测试计划

### 15.1 单元测试

- group supply 不重复计数；
- bucket fallback 和 EWMA 更新；
- forecast residual 不直接改变 reserve；
- undersupply 增加 reserve，overload 减少 reserve；
- simultaneous undersupply/overload 进入 deadband 或保守 hold；
- group alignment、hard cap 和 monotonic revision；
- estimator unavailable fallback；
- engine reset 清理 worker shadow directory。

### 15.2 Trace-driven testbed

对固定 trace 注入：

- trajectory length phase shift；
- tool latency spike；
- worker slowdown；
- KV reset burst；
- learner consumption-rate change。

验证 controller 是否能够在负载变化后恢复到较低 next-batch latency 和较低 expiration，而不是只在一个静态 workload 上找到固定 reserve。

### 15.3 真实 AgenticRL smoke

先运行 4 到 6 step 验证：

- 每个 plan/outcome 可归因；
- top-up 不切换 cache epoch；
- priority queue 实际发生排队和重排；
- vLLM prefill/reset feedback 可用；
- 无 checkpoint、shutdown timeout 和未清理请求。

随后再运行多 workload、多 seed 的正式实验，不在 smoke 阶段调到某个 workload 的最优参数。

## 16. 风险与处理

### 16.1 预测误差导致振荡

使用 bucket fallback、EWMA、signal patience、deadband、cooldown 和每版本 revision 上限。预测器与 actuator 分开更新，避免一个噪声残差同时改变模型和负载。

### 16.2 调度改变完成样本集合

所有策略只使用 runtime state，但执行顺序仍可能改变有限时间内实际完成的 task 集合。必须保留原始 task admission 顺序、FIFO aging，并在 Evaluation 中比较 task/length/version-age 分布和训练收敛。

### 16.3 Controller 开销抵消收益

不在 token path 运行；使用固定数量 buckets、bounded prefix summaries 和 request-boundary 更新。必须单独报告 boundary decision time、per-request routing time、queue lock wait 和 metadata bytes。

### 16.4 ETA 模型与异步 update 相互影响

Policy-update interval 本身受 rollout supply 影响。第一版只使用短期 EWMA 和 bounded reconciliation，不声称精确长期规划。预测的作用是比较候选和控制过量供给，不是提前确定完整训练时间线。

## 17. 最终完成定义

系统达到设计意义上的闭环，需要同时满足：

- 一个共享 estimator 为 admission、scheduling、placement 和 reconstruction 提供供给/成本预测；
- 每次 plan 的预测都能与实际 outcome 对齐；
- outcome 会更新 estimator 和下一轮 reserve；
- same-version reconciliation 修正短期预测失准；
- hard cap、group semantics、staleness semantics 和算法数据接口保持不变；
- 在 workload phase shift 下，controller 能自动改变 admission 和执行顺序，而不依赖人工选择新的固定 `C` 或 reserve；
- 完整系统相对于 fixed/FIFO baseline 降低 policy-update interval 或维持其不退化，同时显著减少 expired invested work 和 refresh re-prefill cost。

这一定义将当前系统从“多个模块共用一份计划”推进为“所有模块围绕有效数据供给，由真实执行结果持续校准的 version-aware runtime”。
