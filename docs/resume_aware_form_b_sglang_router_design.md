# Resume-aware 形态 B：ROLL Ordering + sgl-model-gateway Placement（设计文档）

本文档描述 **形态 B** 的架构目标、与现有实现的差异、落地约束与分阶段方案。形态 B 对齐 `resume_aware_runtime_design_v2.md` 第 8 节「双重策略冲突」中的推荐分工：**外部 gateway（sgl-model-gateway）负责 placement（含 `cache_aware` 与 preferred-worker hint），ROLL 侧负责轨迹语义的 ordering（谁先恢复）**。

---

## 1. 文档目的与读者

- **目的**：为「同时保留官方 sglang-router 的推理侧调度能力」与「ROLL 内 resume-aware 的队列/优先级语义」提供单一、可实现的架构说明，避免与形态 A（纯 `EnvAffinityRouter`）混为一谈。
- **读者**：修改 `roll/distributed/scheduler/router.py`、配置 `router_args`、或扩展 gateway（preferred URL）的开发者。

---

## 2. 三种形态对照（避免混淆）

| 形态 | Placement（发到哪个 worker） | Ordering（谁先被服务） | 典型 `router_name` |
|------|-----------------------------|----------------------|-------------------|
| **A** | ROLL 内 `EnvAffinityRouter`（sticky / resume score） | ROLL 内双队列、配额、aging（可选） | `EnvAffinityRouter` |
| **B（本文）** | **sgl-model-gateway**（`cache_aware` + preferred URL override） | **ROLL 内全局 ordering**（需在 RouterManager 单点聚合） | 建议新增 `SglangOrderingRouter`（见 §7） |
| **仅原生 router** | sglang-router | sglang-router 内部策略（非轨迹语义） | `SglangRouter` |

形态 B **不是**「`EnvAffinityRouter` 与 `SglangRouter` 简单串联」：若在 ROLL 已选定 dp_rank 再直连 worker，则 sglang-router 不参与 placement，谈不上形态 B。

---

## 3. 设计目标与非目标

### 3.1 目标（O）

- **O1（Placement）**：resume 请求在健康前提下尽可能命中 `last_backend_id` 对应的 worker URL（KV / 上下文局部性），由 **sgl-model-gateway** 利用 `cache_aware` + **`X-ROLL-Preferred-Worker-Url`** 完成。
- **O2（Ordering）**：在高并发下，resume 与 normal 在 **全局** pending 集合上的派发顺序可配置（软配额、`effective_priority = base + aging * queue_wait`、超时逃逸等），语义与 `EnvAffinityRouter.enable_request_priority_queue` 一致，但 **不得**在 gateway 内再实现一套冲突的排序策略。
- **O3（可观测）**：能区分 **ROLL 侧排队等待**（ordering）与 **router 侧选中分支**（placement），便于归因 latency。
- **O4（可回滚）**：关闭 ordering 或关闭 preferred header 或退回形态 A，行为可预期。

### 3.2 非目标（N）

- **N1**：在 sglang-router **内部**实现与 ROLL 重复的 resume/normal 双队列（与本形态分工冲突；除非长期废弃 ROLL ordering）。
- **N2**：跨 worker KV 迁移、rid 级 pin/offload（见 V2 文档 G3）。
- **N3**：保证严格的端到端 FIFO（既不是 router 目标，也不是 resume-aware 目标）。

---

## 4. 核心原则：单一职责、避免双重 placement / 双重 ordering

1. **至多一处决定 placement**：要么 ROLL（形态 A），要么 sglang-router（形态 B）。两处同时「选 worker」会导致策略打架与调试困难。
2. **至多一处决定全局 ordering**：若需要跨 env 的 resume/normal 优先级，ordering 必须在 **汇聚所有生成请求的调度入口** 实现（见 §6）。
3. **协议分层**：轨迹语义仍在 `DataProto.meta_info` → `RouterClient._preprocess_generate` → payload 内 **`_roll_route_meta`**（runtime-only）；进入 HTTP 边界时，placement hint 通过 **Header** 交给 router/gateway（与 `tool_return_resume_aware_runtime_design.md`、`resume_aware_runtime_design_v2.md` 一致）。

---

## 5. 数据流（形态 B）

```
Rollout Runtime（TrajEnvManager 等）
  └─ meta_info: request_type, pause_age_s, history_len_tokens, last_backend_id, ...
RouterClient._preprocess_generate
  └─ payload["_roll_route_meta"] = { ... }
【新增】SglangOrderingRouter（位于 RouterManager 进程）
  ├─ 全局 pending：resume 队列 / normal 队列（与 EnvAffinityRouter 语义对齐）
  ├─ 按配额 + priority + aging 选出「下一个应发往 router 的请求」
  └─ HTTP POST http://{gateway_url}/generate
        Headers: X-ROLL-Request-Type, X-ROLL-Preferred-Worker-Url（resume + 合法 last_backend_id）
        Body: sglang generate JSON（含 rid、sampling_params、input_ids 等）
sgl-model-gateway（cache_aware + preferred override）
  └─ 选择 worker URL → worker 执行推理
响应
  └─ selected_backend_id（若下游返回）→ RouterClient._postprocess_generate → runtime 回写 last_backend_id
```

说明：**不再**由 ROLL 调用 `workers[dp_rank].generate_request.remote(...)` 做 placement（形态 A 路径）；形态 B 下 placement 交给 router。

---

## 6. 关键工程约束：为何 ordering 不能挂在「每个 RouterClient 自带的 SglangProxy」上

当前结构中，`RouterManager.create_client` 在 `router_name == SglangRouter` 时为每个 client 包一层 **`SglangProxy`**，各自向 `http://{router_ip}:{router_port}/generate` 发请求。

- **问题**：若仅在 client 侧做局部节流，无法形成 **跨所有 env/client 的全局 resume/normal 队列**，O2 难以严格满足。
- **形态 B 要求**：在 **RouterManager 持有的单个 Router 实例**（与 `self.router.generate_request` 同源）内实现 ordering，使所有请求在进入 sglang-router 之前经过同一套 pending 逻辑。

---

## 7. 建议新增组件：`SglangOrderingRouter`（概念规格）

> 命名可调整；关键是行为而非名字。

### 7.1 职责

- **初始化**：收集 `worker_urls`（与现有 `grpc_mode` / worker URL 契约一致），并可选将其注册到外部 gateway（POST `/workers`）。
- **generate_request(payload, request_id, uid)**：
  - 读取 `payload` 中的 `_roll_route_meta`，解析 `request_type`（`resume` / `normal`）。
  - 将进入路由器的请求放入全局 pending（条件变量 + 与 `EnvAffinityRouter` 类似的 `_pick_next_dispatchable_request` 语义）。
  - 轮到本 `request_id` 时，构造发往 router 的 HTTP 请求：
    - **Headers**：与现有 `SglangProxy._build_router_headers` 一致（注意：`pop("_roll_route_meta")` 的时机应在发出 HTTP 前，避免把 meta 泄露给 worker JSON；可与当前 `SglangProxy` 保持一致）。
    - **Body**：与当前 `SglangProxy` POST body 一致。
  - 解析响应：`postprocess_generate(...)`，保证与现有 `RouterClient._postprocess_generate` 契约一致。

### 7.2 与 `EnvAffinityRouter` 的代码关系

- **可复用**：`router_config` 下关于 `enable_request_priority_queue`、`resume_normal_quota`、`request_score_weights`、`normal_max_queue_wait_s` / `resume_max_queue_wait_s` 等 **ordering 相关**逻辑。
- **不复用**：`_select_worker_for_request` 中基于 dp_rank 的 placement（形态 A）；形态 B 中 worker 由 sglang-router 决定。
- **metrics**：ordering 相关 counter（pending、queue wait）仍在 ROLL Router 内打点；placement 分支 metric 仍在 gateway/router（如 `smg_cache_aware_policy_branch_total`）。

### 7.3 RouterManager 接入

- `RouterArguments.router_name` 增加枚举值，例如 `SglangOrderingRouter`。
- 与 `SglangRouter` 相同：**仅当 `strategy_name == "sglang"`** 且 worker 策略满足现有 `grpc_mode` 等前置条件时允许选用。
- **`router_meta()`**：对该 Router 返回 `sglang_router: true`（使 `RouterClient` 仍包裹 **`SglangProxy`** 发 HTTP）——但需注意：**若 `SglangOrderingRouter` 已在 Router 内完成 POST，则 RouterClient 不应再 POST 一次**。  

**推荐修正（实现时必须二选一）**：

- **方案 1（推荐）**：`SglangOrderingRouter` 仅在 RouterManager 内使用；`router_meta()` 对该类返回 **`sglang_router: false`**，并实现 **`RouterClient` → InprocProxy → RouterManager.generate_request → SglangOrderingRouter** 全链路，由 **`SglangOrderingRouter` 内部**执行 HTTP POST（即吸收现有 `SglangProxy` 的职责），避免双层 POST。
- **方案 2**：保留 `SglangProxy`，但将 `SglangProxy` 改为「透传」模式（不推荐，易混淆）。

文档建议采用 **方案 1**，并在 `RouterManager.router_meta` 注释中写明：`SglangOrderingRouter` 自带 HTTP 客户端，不再叠加 `SglangProxy`。

---

## 8. 配置契约（示例）

**前置条件**：`actor_infer.strategy_args.strategy_config` 必须**显式**设置 `grpc_mode`（通常为 `false`），以使用 `SglangHttpEngine` 并向子进程 sglang-router 提供真实 HTTP worker URL。若省略 `grpc_mode`，会退化为进程内 `SglangEngine`，仅适配 `EnvAffinityRouter` 直连 worker，**不能**与 `SglangRouter` / `SglangOrderingRouter` 共用。

形态 B 下，`router_args` 建议区分两类字段：

- **Ordering（ROLL）**：`enable_resume_priority`、`enable_request_priority_queue`、`resume_normal_quota`、`normal_max_queue_wait_s`、`resume_max_queue_wait_s`、`request_score_weights`、`request_wait_aging_weight` 等。
- **Placement（gateway）**：不在 ROLL `router_config` 重复实现 `cache_aware` 细节；通过外部 gateway 的参数/启动方式控制 placement 策略。

示例（Hydra 片段，仅作说明）：

```yaml
router_args:
  router_name: SglangOrderingRouter   # 待实现
  router_config:
    enable_resume_priority: true
    enable_request_priority_queue: true
    resume_normal_quota: "3:1"
    normal_max_queue_wait_s: 10.0
    resume_max_queue_wait_s: 5.0
    request_wait_aging_weight: 0.1
    request_score_weights:
      age: 1.0
      hist: 0.001
      hit: 0.5
```

**注意**：形态 B 下应设置 **`enable_resume_aware_routing: false`**（或在 Router 内忽略），避免 ROLL 再次基于 dp_rank 做 placement 打分；resume 的 locality 由 **preferred-worker header + cache_aware** 表达。

---

## 9. 分阶段落地

### 9.1 B0：仅 placement（验证兼容性）

- `router_name: SglangRouter`，不传全局 ordering。
- 验证 `X-ROLL-Preferred-Worker-Url` 与 gateway metrics（preferred_hit / fallback）。

### 9.2 B1：全局 ordering（完整形态 B）

- 实现 `SglangOrderingRouter`（§7）+ RouterManager 接入（§7.3 方案 1）。
- 对比 TensorBoard / 日志：`resume_queue_wait`、`normal_queue_wait`、`resume_affinity_hit_rate`（ROLL）与 gateway branch counters（placement）。

### 9.3 B2：软亲和 / 过载避让（可选）

- 仅在 **gateway `cache_aware`** 或 **router 配置** 侧扩展（例如 preferred 命中但 unhealthy 时 fallback）；ROLL ordering 层不重复实现同名逻辑。

---

## 10. 可观测性

| 层级 | 指标示例 | 含义 |
|------|----------|------|
| ROLL ordering | `scheduler/router/pending_*`、`resume_queue_wait_*` | 全局队列与等待 |
| Gateway/router placement | `smg_cache_aware_policy_branch_total{branch=...}` | preferred 是否命中、是否 fallback |
| Runtime | `resume_ttft_*`（若已打点） | tool-return 到首 token |

归因约定：**queue_wait 上升优先查 ROLL ordering**；**locality 差优先查 preferred hint 与 worker 健康状态**。

---

## 11. 风险与回滚

| 风险 | 缓解 |
|------|------|
| 双层 POST（Router + SglangProxy） | 采用 §7.3 方案 1，单元测试断言单次 `/generate` |
| 全局队列死锁/饥饿 | 保留 `normal_max_queue_wait_s` / `resume_max_queue_wait_s` 逃逸；监控 pending 长度 |
| preferred 过强导致某 worker 堆积 | gateway 侧软亲和（V2 §6.2）；或调低 quota 中 resume 比例 |
| 与形态 A 实验混读 | 实验配置显式标注 `router_name` 与是否启用 ordering |

**回滚**：切回 `router_name: EnvAffinityRouter`（形态 A）或 `SglangRouter`（B0）；关闭 `enable_request_priority_queue`；gateway 关闭 preferred override。

---

## 12. 验收标准（建议）

1. **功能**：并发多个 env 时，resume 与 normal 的全局 pending 行为符合配额配置；无重复 POST。
2. **Placement**：resume 在 healthy + hint 有效时，`selected_backend_id` 与 `last_backend_id` 一致比例接近形态 A 同类场景（允许因 router fallback 略低）。
3. **性能**：对比形态 A，形态 B 在「高并发 + 高 resume 比例」下，`resume` 队列等待或 tail latency 有可归因变化（升降均需解释）。
4. **兼容**：不破坏现有 `SglangRouter`、`EnvAffinityRouter` 配置路径。

---

## 13. 与现有文档的关系

- **契约字段与 Header**：仍以 `tool_return_resume_aware_runtime_design.md`、`resume_aware_runtime_design_v2.md` §4–§5 为准。
- **本文**：只在架构层定义 **形态 B** 的分工与 **RouterManager 单点 ordering** 的工程必要条件；具体代码引用以实现 PR 为准。

---

## 14. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-08 | 初稿：形态 B 定义、`SglangOrderingRouter` 概念规格、RouterManager/`SglangProxy` 约束与分阶段验收 |
| 2026-05-08 | **实现**：`roll/distributed/scheduler/router.py` 中新增 `SglangOrderingRouter(SglangRouter, EnvAffinityRouter)`；`RouterManager` 识别 `router_name: SglangOrderingRouter`（与 `SglangRouter` 相同要求 `strategy_name==sglang` 且 `grpc_mode`）；`RouterArguments` 字面量已扩展。示例配置：`examples/qwen3_agentic_gem/gem_math_hotpotqa_search_ds_sglang_router_form_b.yaml`。`router_meta.sglang_router` 对该类仍为 **False**，由该类自行 `POST /generate`，避免与 `SglangProxy` 双重请求。 |
