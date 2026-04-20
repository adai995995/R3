# ROLL Resume Affinity：SGLang Model Gateway 适配设计（基于当前源码）

## 背景

在 ROLL 的 agentic rollout 中，`tool call -> wait -> tool return -> continuation` 会把一条长轨迹拆成多段推理请求。为了降低 tool-return 之后的恢复开销（KV 命中、减少 prefix 重放、降低尾延迟），ROLL 已把后续请求显式标记为 `resume`，并尝试将恢复请求路由回“上一段实际命中的 worker”。

当前 ROLL 侧已有能力：

- 在推理请求中携带 `request_type=resume`、`pause_age_s`、`history_len_tokens`、`last_backend_id` 等 meta
- 在 SGLang Router（ROLL 侧的 `SglangProxy`）中注入 header（例如 `X-ROLL-Preferred-Worker-Url`）

但 `xxl_sglang` 的 gateway（`sgl-model-gateway/`）目前 **尚未消费这些 header** 来影响 `cache_aware` policy 的 worker 选择，导致 ROLL 的强亲和语义无法在 gateway 侧落地。

本设计文档描述：如何在 **SGLang Model Gateway** 中最小化、可回滚地实现 “resume 强亲和 / 精细化恢复信号” 的适配。

## 当前源码事实（关键锚点）

### 1) Router worker selection 的入口

HTTP regular router：

- `sgl-model-gateway/src/routers/http/router.rs`
  - `select_worker_for_model(model_id, text, headers)`：构造 `available` workers（过滤后数组），并调用 policy
  - policy 接口：`LoadBalancingPolicy::select_worker(workers, info) -> Option<usize>`

**重要语义**：policy 返回的是 **过滤后数组 `available`** 的 index，而不是全局 worker 列表的 index。

### 2) Headers 的解析工具

- `sgl-model-gateway/src/routers/header_utils.rs`
  - 内置支持：
    - `x-smg-target-worker`（目标 worker 下标，字符串）
    - `x-smg-routing-key`（一致性 hash 的路由 key）

### 3) policies 对 header 的使用现状

- `policies/consistent_hashing.rs`：会优先读取 `x-smg-target-worker` 并 direct-route
- `policies/cache_aware.rs`：**不读取 headers**，只用 `request_text + load + radix tree`

因此，当 gateway policy 为 `cache_aware`（默认推荐）时，ROLL 发来的 header 目前不会改变 worker 选择。

## 冲突/风险：为什么直接用 index 很危险

ROLL 的 `last_backend_id` 往往来自其内部 worker/dp_rank 语义；而 gateway policy 的 `x-smg-target-worker` 是 **过滤后数组 index**。当出现以下任一情况时，index 就会漂移：

- 不同 `model_id`：`select_worker_for_model` 会按模型过滤 worker
- 不同 `ConnectionMode`：HTTP/gRPC 过滤
- `is_available()` / circuit breaker 过滤
- worker 动态增删导致顺序变化

因此，**在 gateway 侧使用稳定 worker identity（URL 或 stable worker id）** 是更安全的做法。

## 设计目标

### 功能目标

- **G1：强亲和**：当请求携带 preferred worker（URL/stable id）时，gateway 在 `cache_aware` 下应优先路由到该 worker（健康则用，否则 fallback）。
- **G2：可回滚**：行为应可通过配置/开关关闭，关闭后完全回到原 `cache_aware`。
- **G3：可观测**：暴露命中、fallback、原因等 metrics，便于 A/B 和回归。
- **G4：兼容现有 policy**：不改变 `cache_aware` 的主算法，只在入口增加 header override（或 soft-bias）。

### 非目标（本阶段不做）

- N1：按 trajectory/rid 的 KV pin/unpin（需要 worker/engine 的 KV 生命周期 API）
- N2：跨 worker KV 迁移
- N3：将 ROLL 的复杂 resume score 完整搬入 gateway（先做强亲和与最小信号闭环）

## API / Header 契约（推荐）

为避免 index 漂移，推荐 gateway 支持 **URL 语义** 的 preferred worker header：

- `X-ROLL-Preferred-Worker-Url: <url>`
  - 例如：`http://10.0.0.1:8000`
  - 仅在 `request_type=resume` 时由 ROLL 注入

同时，为与现有 gateway header 体系一致，也可提供别名：

- `X-SMG-Preferred-Worker-Url: <url>`（可选）

可选的恢复信号（后续阶段使用）：

- `X-ROLL-Request-Type: resume|normal`
- `X-ROLL-Pause-Age-S: <float>`
- `X-ROLL-History-Len-Tokens: <int>`

## 方案总览

### 方案 A（推荐）：在 `cache_aware` policy 内实现 preferred-worker override（按 URL）

**思路**：在 `CacheAwarePolicy::select_worker()` 入口处，先检查 headers 中是否指定 preferred worker URL；若存在且该 worker 在 `workers` 数组中且健康，则直接返回对应 index；否则走原逻辑。

优点：

- 与过滤后的 `workers` 数组天然对齐（通过 URL 在 `workers` 中查找）
- 不依赖 worker 数组顺序，不会 index 漂移
- 改动集中在 policy，一处生效（HTTP regular / IGW 等都通过同一 policy）

缺点：

- 多一次 `O(n)` 的 URL scan（n=worker 数通常较小；可优化为构建 map，但先不必）

### 方案 B（可选）：在 router 层（`select_worker_for_model`）做 override

在 `routers/http/router.rs::select_worker_for_model` 中，在调用 policy 前直接选择 preferred worker。

优点：可以做到 endpoint-specific、或根据 request 类型做更细分策略

缺点：需要在 HTTP regular / PD / gRPC / openai router 等多处重复适配，维护成本更高

本设计建议优先使用方案 A。

## 详细设计（方案 A）

### 1) Header 解析扩展

文件：`sgl-model-gateway/src/routers/header_utils.rs`

新增：

- `extract_roll_preferred_worker_url(headers: Option<&HeaderMap>) -> Option<&str>`
  - 读取 `x-roll-preferred-worker-url`（大小写不敏感由 HeaderName 处理）

（可选）同时支持 `x-smg-preferred-worker-url` 作为别名，便于与 gateway 自身命名一致。

### 2) cache_aware policy：preferred override

文件：`sgl-model-gateway/src/policies/cache_aware.rs`

在 `select_worker()` 的开头增加：

1. 从 `info.headers` 取 preferred worker URL（若无则跳过）
2. 在 `workers` 中查找 `w.url() == preferred_url` 且 `w.is_healthy()`
3. 命中则返回该 index（**强亲和**）
4. 否则执行原 `cache_aware` 算法（fallback）

建议支持两种模式（配置开关）：

- **hard**：健康命中则强制路由
- **soft**：健康命中且 `w.load()` 未超过阈值才路由，否则 fallback（避免 preferred 热点导致更慢）

### 3) 可观测指标（metrics）

新增 metrics 分支统计（建议放在现有 policy branch 风格中）：

- `smg_worker_cache_aware_preferred_branch{branch=...}`
  - `preferred_hit`
  - `preferred_miss_not_found`
  - `preferred_miss_unhealthy`
  - `preferred_miss_empty`

并可记录：

- preferred 命中率（hit / (hit + miss)）
- preferred miss 原因分布

### 4) 配置与回滚

在 gateway 配置中新增 cache-aware 相关开关（CLI / config builder）：

- `--cache-aware-preferred-routing {off,hard,soft}`
  - 默认 `off`（确保升级安全）

回滚策略：

- 将开关设为 `off` 即完全恢复现有行为
- 关闭后不改变任何 header 处理或路由结果

## 兼容性与安全性

- **正确性**：preferred 仅影响“选哪个 worker”，不会改变模型输出语义；preferred 不可用时 fallback。
- **多模型/IGW**：preferred URL 只会在 `available` workers 中匹配；若 preferred 指向不同 model 的 worker，会视为 not found 并 fallback。
- **PD 模式**：若 PD 使用不同 policy（prefill/decode），需确认这些 policy 是否也需要 preferred override；本阶段仅保证 regular `cache_aware`。

## 测试计划

### 单元测试（Rust）

1. 在 `policies/cache_aware.rs` 增加 test：
   - 构造两个 worker：w1/w2
   - headers 指定 preferred 为 w2
   - 断言 select_worker 返回 w2 的 index（且 w2 healthy）

2. preferred 指向不存在 URL：
   - 断言走原逻辑（返回值不做强约束，但应记录 miss_not_found）

3. preferred 指向 unhealthy worker：
   - 断言 fallback（并记录 miss_unhealthy）

### 集成测试（最小黑盒）

参考 ROLL 侧已有脚本 `scripts/smoke_test_sglang_router_preferred_url.py`，可以在 `xxl_sglang` 下补一个 e2e：

- 启动 gateway + 2 个 fake worker
- 对 `/generate` 发请求：
  - 不带 header -> baseline
  - 带 `X-ROLL-Preferred-Worker-Url` -> 命中指定 worker

## 迭代路线（与 ROLL 的“精细化 resume”对齐）

### Phase 0：preferred override（本设计）

只解决 “resume 强亲和可落地” 与 “可观测”。

### Phase 1：消费更多 resume 信号（可选）

在 `cache_aware` 中引入轻量偏置：

- `pause_age_s` 越大越允许迁移（降低 preferred 约束）
- `history_len_tokens` 越大越倾向命中 preferred（迁移成本更大）

注意：这属于策略调参域，应保持可配置并有明确指标对比。

### Phase 2：KV pin/unpin（需要 worker/engine 配合）

定义 gateway->worker 的 pin API 与资源隔离策略；本阶段不在本文件范围内。

## 代码改动清单（预期）

- `sgl-model-gateway/src/routers/header_utils.rs`
  - 新增 preferred worker URL 提取函数
- `sgl-model-gateway/src/policies/cache_aware.rs`
  - 在 `select_worker()` 增加 preferred override
  - 增加 policy branch metrics（可选）
  - 增加单元测试
- （可选）`sgl-model-gateway/src/config/types.rs` / `builder.rs`
  - 增加 CLI/config 开关 `cache_aware_preferred_routing`

## 与 ROLL 的对接建议（简述）

- ROLL 侧继续在 resume 请求中注入：
  - `X-ROLL-Preferred-Worker-Url`（优先）
  - （可选）`X-ROLL-Request-Type: resume`
- 避免使用 `X-SMG-Target-Worker` 直接传 index，除非 gateway 明确将 index 语义定义为“过滤后数组 index”（通常不适合跨系统对接）。

