# Resume-aware Rollout Runtime（V2：对齐当前 ROLL + SGLang Gateway 实现）

## 0. 文档目的

本设计文档在 `docs/idea.md` 的抽象基础上，**结合当前仓库已经实现的代码**，给出一版“可落地、可验证、可迭代”的 resume-aware rollout runtime 设计（V2）。

V2 的定位是：

- 把 “tool-return → resume request” 显式化并贯通到路由层
- 优先解决 **恢复 locality** 与 **恢复队列/调度** 的工程闭环
- 为后续 **分层上下文管理（GPU/CPU/evict）** 预留协议与指标

## 1. 结论摘要（当前已实现什么、还缺什么）

### 1.1 已实现（可编译、可单测验证）

- **R1：Resume 元信息传递闭环（ROLL 内）**
  - 请求侧：`DataProto.meta_info` 支持 `request_type=resume / pause_age_s / history_len_tokens / last_backend_id` 等字段
  - RouterClient 会将这些字段注入 `_roll_route_meta`（runtime-only）并随 payload 传递

- **R2：Last backend 回写（用于后续 resume affinity）**
  - Router 返回时会携带 `selected_backend_id`（非 sglang router / 以及部分路径）
  - agentic runtime（`TrajEnvManager`）会读取并回写 `_last_backend_id`

- **R3：SGLang gateway（sgl-model-gateway）对 resume affinity 的适配（方案 A 已落地）**
  - 在 gateway 的 `cache_aware` policy 内新增 “preferred worker URL override”
  - 支持 header：
    - `X-ROLL-Preferred-Worker-Url`（主）
    - `X-SMG-Preferred-Worker-Url`（别名）
  - 健康命中则强制路由，否则 fallback 原 `cache_aware`
  - 增加 metrics：`smg_cache_aware_policy_branch_total{branch=...}`
  - 已通过 targeted 单测：`cache_aware_preferred_worker_url_*`

- **R4：G1 严格 resume 判定边界（ROLL agentic runtime）**
  - 只有满足“跨 tool wait 边界 + tool-return observation 触发”的下一次推理请求才标为 `request_type=resume`
  - 覆盖的 EnvManager：
    - `TrajEnvManager`
    - `AgentNativeStepEnvManager`
    - `VLTrajEnvManager`
  - 其他情况下保持 `request_type=normal`，避免误标稀释收益
  - 最小验证：
    - runtime 侧在 `meta_info` 中附带 `resume_expected_tool_return / resume_request_count / resume_mismatch_count`，其中 `resume_mismatch_count` 应恒为 0
    - 纯逻辑校验脚本：`scripts/verify_g1_resume_boundary.py`

### 1.2 还缺（决定你 idea 上限的关键 gap）

- **G2：真正的“谁先恢复、恢复到哪”（全局联合调度）**  
  当前更多是局部 worker 选择/亲和；缺少把 resume request 当成一等对象的全局队列与排序策略（以及与 normal 请求的隔离/融合策略）。

- **G3：分层上下文管理（GPU/CPU/evict）**  
  目前尚无 trajectory/rid 级别的 KV pin/offload/evict 接口；这部分需要推理后端配合，是后续重点。

## 2. 设计目标与非目标

### 2.1 设计目标

- **O1：恢复 locality 优化可落地**：resume 请求尽可能命中上一次 backend（健康则用，否则 fallback）
- **O2：resume 与 normal 分流可控**：至少在 queue/priority 上能表达 “恢复价值 - 恢复代价”
- **O3：可观测**：能拆分观察 `queue_wait / rebuild / locality` 三类代价
- **O4：可回滚**：所有策略必须配置化，关闭后退化为 baseline

### 2.2 非目标（V2 不实现）

- **N1：跨 worker KV 迁移**
- **N2：对未来 tool observation 的 speculation（只做真实 tool-return 后的 resume）**
- **N3：后端精确 KV pin/offload（只做协议与落点设计，等待后端实现）**

## 3. 核心抽象（对齐 `idea.md`）

### 3.1 两类请求

- **NormalRequest**：未跨外部等待边界的连续生成
- **ResumeRequest**：满足：
  1) trajectory 进入过外部等待（tool/env wait）
  2) 现在因为外部 observation 返回，才重新可执行

**注意**：不是“多轮”就叫 resume，必须是“跨外部等待边界后重新进入推理”。

### 3.2 恢复的两类决策

- **Placement（恢复到哪）**：选择 worker/backend（亲和、负载、超时、迁移阈值）
- **Ordering（谁先恢复）**：resume 与 normal 在全局队列里如何排序（aging、fairness、SLO）

V2 已基本解决 placement 的一个强亲和版本（preferred override），ordering 仍需补齐。

## 4. 现有代码链路（V2 的真实落点）

### 4.1 ROLL：meta → `_roll_route_meta`

`roll/distributed/scheduler/router.py::RouterClient._preprocess_generate` 会把下列 meta 注入 `_roll_route_meta`：

- `trajectory_id`
- `request_type`（normal/resume）
- `pause_age_s`
- `history_len_tokens`
- `last_backend_id`
- `tool_type`（预留）
- `fairness_bucket`（预留）

### 4.2 ROLL：SglangProxy 注入 header（affinity hint）

ROLL 侧会在向 router `/generate` 发送请求时注入：

- `X-ROLL-Request-Type`
- `X-ROLL-Preferred-Worker-Url`（由 `last_backend_id` 映射到 worker URL）

> 备注：V2 建议长期以 URL 作为 preferred identity（避免 index 漂移与过滤语义不一致）。

### 4.3 SGLang Gateway：`cache_aware` 支持 preferred URL override（已实现）

在 `xxl_sglang/sgl-model-gateway` 中：

- `routers/header_utils.rs`：新增 `extract_preferred_worker_url()`
- `policies/cache_aware.rs`：select_worker 开头优先尝试 preferred URL 命中
- `observability/metrics.rs`：新增 branch counter
- 单测：`cache_aware_preferred_worker_url_hit/not_found/unhealthy`

## 5. 协议与字段（V2 最小契约）

### 5.1 ROLL 内部：`DataProto.meta_info`

| 字段 | 含义 | 备注 |
|------|------|------|
| `trajectory_id` | 轨迹唯一标识 | 稳定、可复现 |
| `request_type` | normal/resume | 仅 tool-return 触发 resume |
| `pause_ts` / `pause_age_s` | 等待开始/等待时长 | 用于 aging 与迁移阈值 |
| `history_len_tokens` | 上下文长度 | 代价 proxy |
| `last_backend_id` | 上一次命中的 backend | 用于亲和 hint |
| `tool_type` | 工具类别 | 预留给更精细策略 |
| `fairness_bucket` | 公平性桶 | 预留给抗饿死 |

### 5.2 Gateway HTTP headers

| Header | 用途 |
|--------|------|
| `X-ROLL-Preferred-Worker-Url` | resume 强亲和（主） |
| `X-SMG-Preferred-Worker-Url` | alias（可选） |
| `X-ROLL-Request-Type` | 观测/调试（可选） |

## 6. V2 策略设计（分阶段）

### 6.1 L0：强亲和（已落地）

- **行为**：preferred URL 命中健康 worker → 强制路由；否则 fallback
- **目标**：最大化 locality，最小化 resume 的 KV miss（在后端自动保 KV 的前提下）

### 6.2 L1：软亲和（规划）

在 L0 基础上加入轻量迁移条件（仍保持可回滚）：

- preferred 命中但 worker 负载超过阈值 → 允许 fallback（soft override）
- pause_age 超过阈值（例如 `force_migrate_age_s`）→ 降低亲和权重

落点：

- gateway 的 `cache_aware` policy 或 ROLL 的 `EnvAffinityRouter`（二选一；优先在单处做，避免双重策略冲突）

### 6.3 L2：Ordering（谁先恢复，规划）

引入 resume-aware 的全局队列/排序：

- `effective_priority = base_priority + aging * queue_wait`
- base_priority 可近似：`pause_age_s`（价值）与 `history_len_tokens`（代价）等

落点候选：

- ROLL 内：`EnvAffinityRouter.enable_request_priority_queue`（已有雏形）
- 或 gateway 内：独立队列（更接近后端，但改动更大）

V2 建议先在 ROLL 内做 ordering（更贴近 rollout runtime 的语义）。

## 7. 可观测性（V2 必须补齐的指标）

### 7.1 locality（已部分实现）

- gateway：
  - `smg_cache_aware_policy_branch_total{branch=preferred_hit|...}`

### 7.2 queue_wait（需要在 rollout runtime 打点）

- `tool_return_ts → request_enqueued_ts → request_dispatched_ts`
- 输出：
  - resume queue wait p50/p95
  - normal queue wait p50/p95（对照）

### 7.3 rebuild_cost（proxy）

在无法直接读 KV 命中前，先用 proxy：

- `history_len_tokens`
- `resume_generation`（第 N 次 resume 可能更贵）

## 8. 风险与回滚

- **误标 resume**：最大风险。必须把 resume 触发绑定到 tool-return 边界。
- **过度亲和导致排队**：soft override + migrate_age 阈值 + queue ordering 缓解。
- **双重策略冲突**：避免同时在 ROLL 与 gateway 做不一致的重排序；V2 推荐：gateway 做 placement（L0/L1），ROLL 做 ordering（L2）。

回滚开关：

- gateway：关闭 preferred override（回到原 `cache_aware`）
- ROLL：不发 preferred header / 不标 request_type=resume

## 9. 下一步工作清单（建议顺序）

1. **收紧 resume 判定（P0）**
   - 只在 tool-return 边界标注 `request_type=resume`
   - 明确 `tool_type` 的来源（tool wrapper / env wrapper）

2. **在 ROLL 内启用 L2 ordering（P1）**
   - 打开 `EnvAffinityRouter.enable_request_priority_queue`
   - 将 `pause_age_s/history_len_tokens/last_backend_id` 纳入 base_priority

3. **定义分层上下文管理的协议（P2）**
   - 设计 `pin/offload/evict` 的 trajectory/rid 级 API
   - gateway 与 worker 侧对齐语义（暂不实现）

