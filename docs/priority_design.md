轨迹优先级与全局调度设计（G2）

本文档将 `resume_aware_runtime_design_v2.md` 中的 **G2：真正的“谁先恢复、恢复到哪”（全局联合调度）** 具体化为可落地实现的优先级定义与调度机制。

目标：

- **把 resume request 当成一等对象**：显式分流、排序、观测与回滚
- **对齐已实现的 G1**：只有 tool-return 触发的请求才标 `request_type=resume`
- **与 placement 联动**：resume 优先走 affinity（last backend / preferred worker url），失败再 fallback

非目标（首版不做）：

- 跨 worker KV 迁移
- 精确的 KV pin/offload/evict（只做 proxy）
- 自动学习权重（先配置化 + 观测闭环）

---

## 1. 核心抽象

### 1.1 请求与轨迹

系统真正排队的是一次次 **GenerateRequest**。本设计首版将“等待/aging/公平性”主要落实在 **请求级（request-level）**，即以“入队到出队的排队时间”作为 WaitingTime 的核心信号；轨迹级状态（TrajectoryState）作为可选增强，用于处理“单轨迹霸占”等极端情况。

- **TrajectoryId**：来自 `meta_info["trajectory_id"]`
- **RequestType**：来自 `meta_info["request_type"]`，取值 `normal|resume`

实现要点：

- 调度的基础对象是请求：每个请求入队时记录 `enqueue_ts`，出队时用 `queue_wait_s = now - enqueue_ts` 做 aging/防饿死
- `trajectory_id` 主要用于 placement（resume affinity）与可观测；轨迹级状态仅在需要时启用（见 1.2）

### 1.2 TrajectoryState（可选：调度侧轨迹状态）

只有在出现以下问题时，才建议引入轨迹级状态：

- 单条轨迹在短时间内产生/重试大量 resume 请求导致“刷屏”
- 需要“轨迹公平”而不仅是“请求公平”（例如每条轨迹都要推进）

可选字段示例（按需启用）：

- `last_enqueue_ts`：最近一次入队时间（用于 queue_wait / waiting）
- `last_dequeue_ts`：最近一次出队服务时间（用于 waiting/aging）
- `last_history_len_tokens`：最近一次上下文长度（cost proxy）
- `last_backend_id`：最近一次后端（affinity / gain proxy）
- `consecutive_resume`：连续 resume 次数（防霸占，可选）

---

## 2. 优先级公式（概念形式 vs 当前实现）

设定优先级的概念公式可以是：

Score(\tau) = \alpha \cdot \text{ResumeGain}(\tau) - \beta \cdot \text{ResumeCost}(\tau) + \gamma \cdot \text{WaitingTime}(\text{request})

其中：

* \tau 代表某个轨迹。
* ResumeGain 是恢复该轨迹的潜在价值，衡量恢复该轨迹后能够提升的系统性能。
* ResumeCost 是恢复轨迹的代价，衡量恢复这个轨迹所需的额外开销，包括上下文重建、缓存恢复等。
* WaitingTime 在首版实现中采用 **请求级 queue waiting**（`queue_wait_s`），用于防止队列饥饿。
* \alpha、\beta、\gamma 是超参数，分别控制 恢复价值、恢复代价和 等待时间 的权重。

实现约束：

- **只对 `request_type=resume` 的请求启用该打分**；normal 请求保持现有策略（FIFO / fair-share）
- 公式中的每一项必须能用可观测/可计算的 proxy 落地，并输出指标用于校准权重

当前代码实现说明：

- scheduler 当前使用的是 `compute_request_priority(...)` + `request_wait_aging_weight * queue_wait_s` 作为实际 priority。
- 文档中的 \(\alpha,\beta,\gamma,\lambda_1,\lambda_2\) 仍保留为“概念/可扩展形式”，若要完全对齐，需要在代码中显式实现该套权重与归一化（见第 6/8 节的增强项）。

---

## 3. 三个因子的可实现定义（proxy）

各个因素的具体定义

1. ResumeGain（恢复价值）

恢复价值表示恢复某条轨迹后，推理任务能够带来的增益。首版推荐使用 **affinity 可行性（proxy）** 作为 gain 的工程近似（可观测、与收益强相关）：

- 如果能命中上次 backend / preferred worker，则更可能：
  - KV 命中、TTFT 更低
  - 上下文重建成本更低
  - 端到端恢复更快

定义：

- `AffinityFeasible`（proxy，不保证命中）：
  - `1`：请求携带 affinity hint（例如 `last_backend_id`，或 gateway 能识别的 preferred worker url）
  - `0`：否则

说明：

- 调度器在出队前通常无法“确认一定命中 KV/worker”，因此 `AffinityFeasible` 只能作为近似条件。
- 真正的命中需要在 dispatch 后观测 `AffinityHit`（见第 4/7 节的指标与动态配额反馈）。

首版 ResumeGain（proxy）：

\[
\text{ResumeGain}(\tau) = \text{AffinityFeasible}
\]

后续增强（非首版必须）：

- CompletionStatus：用 env metrics（例如剩余 steps / 成功概率）近似
- ToolProgress：用 tool success / tool latency 的聚合近似

* 恢复后完成度：轨迹是否接近完成，如果接近完成恢复代价相对较小。
* 恢复后推理进展：恢复后能继续推理的速度与效率。
* 工具调用的时效性：若恢复请求后工具执行返回，能带来何种推理效果。

例如：

\text{ResumeGain}(\tau) = \text{CompletionStatus}(\tau) \times \text{ToolProgress}(\tau)

2. ResumeCost（恢复代价）

恢复代价必须可量化。首版建议两个 proxy：

- `HistoryLength(τ)`：来自 `meta_info["history_len_tokens"]`
- `WorkerLoad(backend)`：scheduler 侧维护的 backend 负载（如 `queue_depth`/`inflight`），没有的话先置 0

定义：

\[
\text{ResumeCost}(\tau) = \lambda_1 \cdot \text{NormHistoryLength}(\tau) + \lambda_2 \cdot \text{NormWorkerLoad}(\tau)
\]

其中 `Norm*` 使用 clip+线性归一化，保证不同量纲可组合。

* 上下文重建代价：若原上下文未能命中缓存，系统需要恢复的历史数据量。
* 恢复时的资源消耗：恢复请求本身的计算成本。
* worker 的负载：如果某个 worker 当前负载过重，恢复该轨迹的代价会增加。

例如：

\text{ResumeCost}(\tau) = \lambda_1 \cdot \text{HistoryLength}(\tau) + \lambda_2 \cdot \text{WorkerLoad}(\tau)

3. WaitingTime（等待时间）

等待时间用于防止队列饥饿。首版按当前实现采用 **请求级（request-level）的调度侧 queue waiting**，而不是轨迹级 waiting，也不是 runtime 的 `pause_age_s`：

- `pause_age_s`：外部 tool wait 的时间（runtime 侧）
- `queue_wait_s`：该请求在 scheduler 队列里“入队到出队”的排队时间（scheduler 侧）

定义（首版）：

\[
\text{WaitingTime}(\text{request}) = now - \text{enqueue\_ts}
\]

说明：

- 该定义天然与调度器的 pending 队列一致，易实现、易观测。
- 若后续需要“轨迹公平”（而非请求公平），再引入可选的 `TrajectoryState` 与 `now - last_dequeue_ts(trajectory)`。

---

## 4. 从公式到可执行：Score 计算与队列结构

### 4.1 Score(τ) 计算时机

- **入队时计算**：每个 resume 请求到达 scheduler 时，基于请求的 `route_meta` 与 backend 健康/负载视图，计算 score 并赋给请求
- **可选重算**：若队列很长，可在出队前按最新负载重算（首版可不做）

### 4.2 队列结构（建议首版采用）

**双队列 + 配额轮询（建议做成“软配额”，吞吐友好）**：

- `resume_queue`：按 `Score(τ)` 降序（高优先级先服务）
- `normal_queue`：现有策略（FIFO 或现有 fair-share）
- 基础出队策略：按配置比例轮询，例如：
  - `resume:normal = 3:1`（目标是每服务 3 个 resume，至少服务 1 个 normal）

收益：

- 把 resume 当成一等对象（显式分流、排序）
- 避免 normal 饥饿（有明确的 fairness 约束）

#### 4.2.1 为什么要“软配额”

硬配额可能带来吞吐损失（强制切换、空转、打断自然聚合）。工程上建议将配额作为 **公平性目标**，但在以下情况下允许打破配额以更“吞吐友好”。

#### 4.2.2 软配额规则（吞吐友好）

- **空队跳过（skip on empty）**：
  - 当某一侧队列为空时，不因配额空转；直接从非空队列出队。
- **超时放行（timeout escape hatch）**：
  - 若 `normal_queue` 头部等待超过 `normal_max_queue_wait_s`，即使当前轮次应当出 `resume`，也允许优先出一个 normal（避免 normal 长尾与饥饿）。
  - 若 `resume_queue` 头部等待超过 `resume_max_queue_wait_s`，即使当前轮次应当出 `normal`，也允许优先出一个 resume（避免 resume 恢复价值被过度稀释）。
- **动态配额（adaptive quota）**：
  - 根据 backlog/负载动态调节 `resume:normal` 目标比例。例如：
    - normal backlog 大（或 normal queue_wait p90 高）→ 提高 normal 配额
    - resume backlog 大且 `affinity_hit`/`affinity_feasible` 高 → 提高 resume 配额
  - 动态配额应当有 **上下界**（例如最小 `1:1`，最大 `10:1`），并且变化需平滑（避免振荡）。

#### 4.2.3 两级策略（先 fairness，再吞吐）

推荐在 `resume_queue` 内部再做一层吞吐友好排序：

- **一级：队列级公平**：通过软配额保证 normal 不被饿死。
- **二级：resume 内排序（基于 proxy 的吞吐友好偏好）**：
  - 由于无法在出队前确定“必定命中”，这里的 `affinity_feasible` 仅作为 **hint 是否存在/是否可分配** 的 proxy，用于“偏好”而非硬保证。
  - 典型实现：
    - tie-breaker：优先出队 `affinity_feasible=1` 的 resume（更可能命中 locality）
    - 同分再按 `queue_wait_s` 更大者优先（继续保证 aging）
  - 当前实现（`EnvAffinityRouter`）：
    - resume 队列排序 key 形如：`(effective_priority, affinity_feasible_proxy, enqueue_seq)`（affinity 仅作为 tie-breaker）

### 4.3 Placement 联动（恢复到哪）

对 `request_type=resume`：

1. 优先尝试 affinity placement：
   - `last_backend_id`（ROLL 内 meta）或 gateway 的 `preferred worker url` 命中且健康
2. 失败则 fallback 到原有 load-balance 策略

并将结果写回可观测字段（区分 proxy 与事后观测）：

- `affinity_feasible`（bool，proxy：hint/可分配性）
- `affinity_hit`（bool，observed：dispatch 后是否命中）
- `fallback_reason`（枚举：not_found/inactive/not_ready/overloaded/disabled）

当前实现说明（调度侧 best-effort）：

- `fallback_reason` 为 best-effort 近似（用于解释“为何没粘住 previous backend”）。
- 路径 A（pull）已实现：scheduler 可定期从 gateway/router 的 `/workers` 拉取 worker 状态（ready/inflight/queue_depth），用于更精细的 reason 判定与 overloaded 判断。
- 已实现的 reason（与设计枚举对齐 + 扩展）：
  - 设计枚举（更细分）：`not_found / inactive / not_ready / overloaded / disabled`
  - 额外扩展：`no_hint / forced_migrate / selected_other`（以及 `hit` 作为对照）

---

## 5. 与 G1 的对齐（严格 resume 边界）

本设计假设 `request_type=resume` 的信号是“干净”的。

G1 已保证：只有“跨 tool wait 边界 + tool-return observation 触发”的请求才标为 resume。

调度侧需要额外监控：

- `resume_mismatch_count`（理应恒为 0）

若出现 mismatch，应当降级为 normal 或记录告警并回滚策略。

---

## 6. 配置化（首版建议配置项）

需要配置化的权重与阈值：

- `alpha, beta, gamma`
- `lambda_1, lambda_2`
- `history_len_clip_min/max`（归一化用）
- `worker_load_clip_min/max`
- `resume_normal_quota`（如 `3:1`）
- `normal_max_queue_wait_s`（soft quota：normal 超时放行阈值）
- `resume_max_queue_wait_s`（soft quota：resume 超时放行阈值）
- `enable_adaptive_quota`（是否启用动态配额）
- `adaptive_quota_min_ratio / adaptive_quota_max_ratio`（动态配额上下界）
- `adaptive_quota_update_interval_s`（动态配额更新周期）
- `enable_resume_priority`（总开关，便于回滚）
- 路径 A（pull gateway 状态）：
  - `gateway_status_url`：gateway/router 基础 URL（会访问 `{gateway_status_url}/workers`）
  - `gateway_status_poll_interval_s`：轮询周期
  - `gateway_status_headers`：轮询时附带的 HTTP headers（例如 API key）
  - `overloaded_inflight_threshold`：inflight 过载阈值（0 表示关闭）
  - `overloaded_queue_depth_threshold`：queue_depth 过载阈值（0 表示关闭）

---

## 7. 可观测性（必须输出的指标）

至少输出这些指标（按 `request_type` 分桶）：

- `queue_wait_s`：scheduler 入队到出队的时间
- `queue_backlog`：队列长度/积压（用于动态配额与诊断）
- `score`：每个 resume 请求的 score 分布
- `history_len_tokens`
- `worker_load`（若可得）
- `affinity_feasible/affinity_hit/fallback_reason`
- `resume_request_count/resume_mismatch_count`（G1 invariant）

当前实现补充（`EnvAffinityRouter.collect_metrics()`）：

- `scheduler/router/queue_wait_bucket_served/{normal|resume}/...`：queue_wait 分桶计数
- `scheduler/router/score_bucket_served/{normal|resume}/...`：effective score 分桶计数
- `scheduler/router/resume_fallback_reason/{reason}`：resume fallback reason 计数
- `scheduler/router/normal_queue_wait_mean_s`：normal 平均 queue_wait

---

## 8. 实施计划（增量落地）

### 阶段 A（最小可用 G2）

- scheduler 支持 `resume_queue`/`normal_queue`
- resume 入队计算 score
- 配额轮询出队
- 输出 queue_wait/score/affinity 指标

### 阶段 B（增强）

- 负载视图更准确（backend inflight/queue depth）
- score 纳入 `affinity_feasible` 的强弱（不只是 0/1）
- 可选：单优先队列 + 更细 fairness

---

## 9. 测试与验证

首版验证建议：

- 单测：给定一组 resume/normal 请求与不同 waiting/history_len，验证出队顺序符合 score 与配额
- 观测：线上/离线对比 `queue_wait_s`、`affinity_hit`、TTFT（若可得）
- 回滚：`enable_resume_priority=false` 应退化为现有调度路径

当前实现补充（默认可跑的纯逻辑单测，不依赖 ray）：

- `tests/distributed/scheduler/test_soft_quota_utils_default.py`
  - 覆盖：ratio 解析、queue_wait/score 分桶、`fallback_reason` 粒度判定（not_found/inactive/not_ready/overloaded/disabled 等）
- `tests/distributed/scheduler/test_soft_quota_choose_queue_default.py`
  - 覆盖：软配额出队选择（空队跳过、超时放行、配额轮询），不依赖 ray

⸻

深度交互

这个公式背后其实有两层逻辑需要深挖：

1. 恢复代价的可量化性

你需要确保恢复代价是系统中可以度量的，不是一个定性的问题。这要求你能够明确：

* 缓存命中率：恢复轨迹时，是否可以直接从缓存中读取之前的数据，或者需要重新计算。
* 推理进度与收益的预估：可以通过评估轨迹是否接近完成来判断恢复后的收益。
* GPU/CPU 负载状况：当前系统资源是否允许该轨迹快速恢复。

这种“代价感知”的调度决策，使得恢复请求不像普通请求那样简单地基于时间或顺序决定，而是更具灵活性和自适应性。

2. 多目标平衡

这个优先级公式本质上涉及到多目标优化：

* 你必须权衡 恢复代价与收益。
* 你也必须确保 等待时间 过长的轨迹不会被长期饿死。

在此基础上，你可能还需要进行 公平性约束，确保长时间没有被恢复的轨迹有机会优先恢复，而不是系统永远偏爱那些快速恢复的轨迹。

在实际系统实现中，应该加入一个“覆盖补偿项”，确保某些轨迹，特别是长时间等待、历史较长的轨迹，在调度时得到更多关注。这样可以防止一些长历史、低恢复代价的轨迹在系统中被遗漏。

3. 动态调整与学习

在实际的运行过程中，这些超参数（\alpha、\beta、\gamma）的权重会根据系统的实时运行情况进行调整。你可以通过以下几种方式进行动态调整：

* 实时反馈机制：根据恢复请求的实际效果调整权重。比如，如果某些请求的恢复代价过高，可以降低 \beta（恢复代价的权重），让系统更倾向于接受一些较低代价的恢复。
* 样本反馈：对系统进行实验并使用A/B 测试来不断优化这些超参数。

通过这些方法，你的调度系统就不仅仅是静态的“公式打分”，而是能根据环境反馈不断自我调整的动态系统。