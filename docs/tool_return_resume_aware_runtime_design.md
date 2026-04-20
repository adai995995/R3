# Tool-return Resume-aware Rollout Runtime（基于 ROLL 现有实现）

## 背景与目标

在 agentic rollout 中，`tool call -> tool wait -> tool return -> continuation` 会把原本连续的推理轨迹切成多段。很多系统把 tool-return 之后的继续推理当作“普通新请求”，带来三类开销：

- **KV/上下文重建**：同一条轨迹的后续 token 继续生成无法命中原 worker 的 KV 或局部缓存。
- **路由失配**：恢复请求被调度到不同 worker，导致 cache miss、延迟增加，且可能形成尾延迟放大。
- **恢复排队成本**：恢复请求与普通请求在队列中竞争时，缺少“恢复代价驱动”的优先级表达。

本文档提出并落地一套 **resume-aware rollout runtime**：将 tool-return 后的 continuation 显式建模为 `resume` 请求，围绕其恢复代价与恢复价值进行路由亲和、请求排序与可观测闭环。

一句话概括：

- 不是对未来 token 做 speculation，而是对未来 resume 做 speculation。

## 非目标（本阶段不做）

为保持与 ROLL 当前架构的兼容和可回滚，本设计 **不在本阶段直接实现**：

- **按 trajectory/rid 的 KV pin/unpin**（需要推理后端/sglang 内核支持细粒度 KV 生命周期与资源隔离）。
- **在 sglang-router 内部实现新的 queue/scheduling policy**（本阶段只利用其 cache_aware + preferred-worker hint）。
- **跨 worker 的 KV 迁移**（复杂且高风险，通常收益不稳定）。

## 现状：ROLL 已具备的关键能力（代码锚点）

ROLL 当前代码中已存在实现 resume-aware 的关键链路，可作为本设计的基础设施。

### 1) 推理请求的路由元信息注入：`_roll_route_meta`

`roll/distributed/scheduler/router.py::RouterClient._preprocess_generate` 会从 `DataProto.meta_info` 中抽取路由相关字段写入 payload 的 `_roll_route_meta`（runtime-only）。

关键字段（当前已支持）：

- `trajectory_id`
- `request_type`（`normal` / `resume`）
- `resume_generation`
- `pause_ts`
- `pause_age_s`
- `history_len_tokens`
- `last_backend_id`
- `tool_type`（预留）
- `fairness_bucket`（预留）

### 2) SGLang 模式：通过 header 实现强亲和

当策略为 `sglang` 且走 sglang-router 时，`SglangProxy` 会将 resume 请求的 `last_backend_id` 映射为 worker URL，并注入：

- `X-ROLL-Request-Type: resume`
- `X-ROLL-Preferred-Worker-Url: <worker-url>`

用于驱动 sglang-router 的 `cache_aware` policy 优先命中目标 worker。

冒烟验证脚本：`scripts/smoke_test_sglang_router_preferred_url.py`

### 3) 非 SGLang/ROLL 内路由：`EnvAffinityRouter` 已具备 resume-aware 打分与队列

`EnvAffinityRouter` 支持（均可配置开关）：

- `enable_resume_aware_routing`：resume 请求在 worker 选择时使用 resume score
- `enable_request_priority_queue`：请求优先级队列，按 effective priority 选择可派发请求
- `force_migrate_age_s`：pause 超过阈值后降低/取消亲和，允许迁移
- `resume_score_weights`、`request_score_weights`：权重配置
- metrics：affinity hit / migration / queue wait / score mean 等

## 核心抽象与数据流

### 请求类型（Request Class）

将推理请求分为两类：

- **Normal request**：正常生成，追求整体吞吐/公平。
- **Resume request**：跨过外部等待边界（tool wait）后的 continuation，目标是最小化恢复代价（KV 命中、减少重放/重建）。

### Resume 的必要字段（最小可用集）

为实现“强亲和 + 可观测 + 逐步精细化”，resume 请求至少需要：

- `request_type="resume"`
- `pause_ts` / `pause_age_s`
- `history_len_tokens`
- `last_backend_id`（上一段生成的实际 backend/worker）

这些字段由 runtime 在 tool-return 边界填入 `DataProto.meta_info`，最终由 `RouterClient` 注入 `_roll_route_meta`。

### 数据流（从 runtime 到 router）

1. Rollout runtime 发现：上一段执行发生了 tool wait，且 tool 已 return。
2. 下一次发起推理前，runtime 将 `request_type=resume`、`pause_age_s`、`history_len_tokens`、`last_backend_id` 写入 `DataProto.meta_info`。
3. `RouterClient._preprocess_generate` 抽取并写入 `_roll_route_meta`。
4. 路由层消费 `_roll_route_meta`：
   - sglang：`SglangProxy` 注入 preferred-worker header
   - 非 sglang：`EnvAffinityRouter` 做 resume-aware worker selection / queue ordering
5. 推理结果回传后，router 将 `selected_backend_id` 写入返回（`RouterClient._postprocess_generate` 已支持）。
6. runtime 读取 `selected_backend_id` 并回写为下一次 resume 的 `last_backend_id`。

## 分阶段设计（可回滚、逐步增强）

### 阶段 L0：强亲和（Preferred-worker affinity）

目标：最大化 resume 的 KV 命中率，几乎不改变现有调度复杂度。

#### L0 行为

- 当 `request_type=resume` 且 `last_backend_id` 存在时，优先路由到上一次命中的 backend。
- preferred backend 不健康/不可用时，自动 fallback 到原逻辑（不影响正确性）。

#### L0 实现点（当前代码已具备）

- runtime：写入 `request_type`/`last_backend_id`/`pause_age_s`/`history_len_tokens`
  - Agentic pipeline 已在 `TrajEnvManager` 实现（见下节“推理引擎改造”建议）。
- router client：注入 `_roll_route_meta`（已实现）
- sglang proxy：注入 `X-ROLL-Preferred-Worker-Url`（已实现）

#### L0 指标（建议）

- `resume_affinity_hit_rate`（ROLL 内路由已有；sglang 侧可通过返回的 `selected_backend_id` 推导）
- `resume_p50/p95`（需在 runtime 侧打点：tool return -> first token）
- `fallback_rate`（preferred 不可用导致 fallback 的比例；建议新增）

#### 回滚

- runtime 不写 `request_type=resume` 即退化为 normal
- 或 router_config 关闭 resume-aware 相关开关

### 阶段 L1：代价驱动的 worker 选择（联合打分）

目标：当 preferred worker 过载、pause 太久、或资源紧张时，允许“有成本的迁移”，同时避免把所有 resume 都硬绑定。

#### L1 行为

- 对候选 worker 计算 resume score，考虑：
  - 亲和：是否为 last backend
  - 负载：worker 当前 in-flight
  - pause_age：越久越倾向迁移（或降低亲和权重）
  - history_len：越长越希望命中 KV（迁移成本更高）
  - fairness_bonus（可选）
- 选择 score 最大者作为目标 worker。

#### L1 实现点（ROLL 已具备，建议做轻量增强）

- 使用 `EnvAffinityRouter.enable_resume_aware_routing`，其 `_compute_resume_score` 已实现上述基本特征与 `force_migrate_age_s`。
- 如需更精细：
  - 增加 `tool_type`、`resume_generation` 等特征
  - 将权重继续放入 `router_config.resume_score_weights`（保持配置驱动）

#### L1 配置建议（示例）

在 pipeline config 的 router 配置中：

- `enable_resume_aware_routing: true`
- `force_migrate_age_s: 30`
- `resume_score_weights`：`aff/load/hist/age` 按业务调参

#### 回滚

- 关闭 `enable_resume_aware_routing`

### 阶段 L2：谁先恢复（Resume-aware request ordering / priority queue）

目标：在高并发下避免 resume 请求被 normal 挤压，同时按“恢复价值 - 恢复代价”排序恢复。

#### L2 行为

- 将请求进入统一 pending 队列，按 effective priority 选择可派发请求：
  - `effective = base_priority + aging * queue_wait`
- `base_priority` 对 resume 特别考虑：
  - `pause_age_s`（等待越久越该被服务）
  - `history_len_tokens`（越长迁移/重建越贵）
  - `hit_prob`（是否存在 last_backend）
  - `rebuild_cost`（可先用 history_len 近似）
  - `fairness_bonus`（按 bucket 做抗饿死）

#### L2 实现点（ROLL 已具备）

`EnvAffinityRouter.enable_request_priority_queue` 已实现 pending 队列、tie-break、以及等待 aging。

`roll/distributed/scheduler/resume_priority.py` 提供了 `compute_request_priority/compute_resume_score` 的权重化形式，方便持续迭代。

#### L2 指标

`EnvAffinityRouter.collect_metrics()` 已包含：

- `scheduler/router/pending_request_count`
- `scheduler/router/resume_queue_wait_mean_s`
- `scheduler/router/resume_score_mean`
- bucket served 计数

#### 回滚

- 关闭 `enable_request_priority_queue`

## 推理引擎（Rollout Runtime）改造建议：基于现有 Agentic 实现做“更准的 resume”

### 现状（Agentic 已部分实现）

`roll/pipeline/agentic/env_manager/traj_env_manager.py::TrajEnvManager` 已做：

- 保存 `_pause_ts`、`_next_request_type`、`_last_backend_id`
- 在下次发起推理时写入 meta：
  - `request_type` / `pause_age_s` / `history_len_tokens` / `last_backend_id`
- 从推理返回读 `selected_backend_id` 回写 `_last_backend_id`

### 建议：仅在跨 tool wait 边界时标注 resume

为了让 “resume==tool-return continuation” 与语义一致，建议将 `request_type=resume` 的触发从“每次 step 后”收敛为：

- **上一轮确实触发了 tool 调用并经历等待**（例如 history 中 `use_tool=True`）
- tool 的 observation 已回填为一条 `role=tool` 消息（`format_messages()` 已能识别）

这样可以避免把每一步普通 env step 都当作 resume，导致：

- resume metrics 失真
- 路由策略对 normal 流量产生不必要扰动

推荐字段补充（可选）：

- `tool_type`：区分不同工具/外部系统（DB/Web/Code sandbox），用于更细的打分与调参。
- `fairness_bucket`：例如 `${tag}/${group_id}`，用于跨 bucket 的公平性。

## 接口与字段定义（ROLL 内部契约）

### `DataProto.meta_info`（runtime -> RouterClient）

| 字段 | 类型 | 含义 |
|------|------|------|
| `trajectory_id` | str | 轨迹标识（建议稳定） |
| `request_type` | str | `"normal"` / `"resume"` |
| `pause_ts` | float | 进入等待的时间戳（秒） |
| `pause_age_s` | float | 进入等待到 resume 的时长（秒） |
| `history_len_tokens` | int | 当前输入上下文长度（token） |
| `last_backend_id` | int\|None | 上一次实际命中的 backend id |
| `resume_generation` | int | 第 N 次 resume（可选） |
| `tool_type` | str\|None | 工具类型（可选） |
| `fairness_bucket` | str\|None | 公平性分桶（可选） |

### `_roll_route_meta`（RouterClient -> Router/Proxy）

`_roll_route_meta` 为 runtime-only 的 payload 字段，router 层应在转发到实际推理后端前 `pop` 掉（`sglang_strategy.generate_request` 已防御性 pop）。

### HTTP Headers（SGLang 模式：Proxy -> sglang-router）

| Header | 含义 |
|--------|------|
| `X-ROLL-Request-Type` | 请求类型（建议值：`resume`） |
| `X-ROLL-Preferred-Worker-Url` | preferred worker URL（强亲和 hint） |

说明：后续如需更精细，可扩展 `pause_age_s/history_len_tokens/tool_type` 等 header；但若 sglang-router 不消费，这些 header 仅用于观测/调试。

## 配置开关与默认建议

本设计坚持“可配置、可回滚”，所有策略建议受 `router_config` 控制。

推荐新增/使用的配置项（已存在于 `EnvAffinityRouter`）：

- `enable_resume_aware_routing`：默认 `false`，验证后逐步开启
- `enable_request_priority_queue`：默认 `false`，吞吐/公平性评估后开启
- `force_migrate_age_s`：默认 `30.0`（可调）
- `max_running_requests_per_worker`：防止过载（可选）
- `resume_score_weights` / `request_score_weights`：逐步调参
- `fairness_enable` / `fairness_boost_max`：需要时开启

## 可观测与验收标准

### 最小验收（L0）

- resume 请求能稳定命中 preferred worker（或在不可用时 fallback）
- `selected_backend_id` 能被 runtime 正确回写到 `last_backend_id`
- resume 延迟有可观测改善（至少 p95 不恶化）

### 推荐指标

- **路由层（已有）**：`scheduler/router/resume_affinity_hit_rate`、`resume_migration_rate`、`resume_queue_wait_mean_s`
- **runtime 层（建议新增）**：
  - `tool_wait_s`（tool call -> tool return）
  - `resume_to_first_token_s`（tool return -> 首 token）
  - `resume_total_latency_s`

## 风险与缓解

- **误标 resume**：会扰动路由策略并污染 metrics。缓解：将 resume 触发绑定到明确的 tool wait 边界，并在日志/metrics 中区分来源。
- **过度亲和导致排队**：preferred worker 过载时，resume 反而更慢。缓解：L1/L2 引入负载项与 queue ordering，且 `force_migrate_age_s` 提供硬降级。
- **一致性风险（预构造 prefix）**：本阶段不做“猜测 observation”；仅在真实 tool return 后发起 resume，确保正确性。

## 实施计划（与仓库改动对齐）

### P0（1-2 天）：闭环可用 + 可观测

- 保持 L0：SGLang preferred-worker affinity + 选 backend 回写（现已具备）
- 将 agentic runtime 的 resume 触发条件改为 “tool-return 边界” （精确化）
- 在 runtime 侧新增基本时延打点（可选）

### P1（3-5 天）：开启 L1/L2 并调参

- 针对非 sglang 或需要 ROLL 内路由的场景，开启 `EnvAffinityRouter` 的 resume-aware routing / priority queue
- 根据真实 workload 调 `resume_score_weights` / `request_score_weights`

### P2（更长期）：后端 KV 分层（需要推理后端配合）

- 设计 KV pin/unpin 协议字段（meta/header）
- 推理后端实现按 trajectory 的 KV 生命周期管理

## 参考与代码入口索引

- `roll/pipeline/agentic/env_manager/traj_env_manager.py`：runtime 写 meta、读 `selected_backend_id`
- `roll/distributed/scheduler/router.py`
  - `RouterClient._preprocess_generate`：注入 `_roll_route_meta`
  - `SglangProxy._build_router_headers`：注入 preferred-worker header
  - `EnvAffinityRouter`：resume-aware routing / priority queue / metrics
- `roll/distributed/scheduler/resume_priority.py`：resume score / request priority 的权重化实现
- `scripts/smoke_test_sglang_router_preferred_url.py`：preferred-worker header 冒烟验证

