# R3 调度 vs “原生 ROLL” 对比（以推理请求调度为主）

## 背景与范围

本文档聚焦 **推理请求层的调度**（请求如何入队、如何排序、如何选择 backend/worker、如何回写与观测），不展开 Ray 资源层（placement group、角色 actor 部署）细节。

对比对象定义：

- **R3（当前仓库实现）**：支持 `resume` 语义、resume-aware placement、resume/normal 分流与优先级队列、软配额与动态配额、以及指标闭环。
- **“原生 ROLL / baseline”**：不启用上述 resume-aware 能力时的行为（可通过关闭 router_config 开关 + 不携带 resume meta 来复现）。

## 核心差异总览（结论）

- **请求语义**：
  - baseline：生成请求按 normal 处理，缺少 tool-return continuation 的独立语义。
  - R3：将 tool-return continuation 显式标注为 `request_type=resume`，并贯通到路由层（meta → `_roll_route_meta` → router）。

- **Placement（恢复到哪台）**：
  - baseline：更偏 sticky / 简单负载策略，不保证恢复到上一次 backend。
  - R3：resume 具备强/软亲和能力，优先命中上次 backend（或基于打分在过载/过期时迁移）。

- **Ordering（谁先跑）**：
  - baseline：通常 FIFO/简单策略，resume 与 normal 无隔离与优先表达。
  - R3：可启用 **resume/normal 双队列 + priority + aging + 软配额轮询 + 超时放行**，并可用滑动窗口信号做动态配额。

- **观测与回滚**：
  - baseline：缺少 resume 亲和命中、迁移原因、queue wait/score 分桶等细粒度指标。
  - R3：指标齐全，且支持一键回滚（统一开关 `enable_resume_priority`）。

## R3：调度链路（请求如何进入调度器）

### 1) Runtime → RouterClient：注入路由元信息

R3 约定：runtime 将路由相关字段写入 `DataProto.meta_info`，`RouterClient._preprocess_generate()` 抽取字段并注入 payload 的 `_roll_route_meta`（runtime-only）。

典型字段：

- `request_type`: `normal` / `resume`
- `pause_age_s`: 外部等待时间（tool wait）
- `history_len_tokens`: 上下文长度 proxy
- `last_backend_id`: 上次实际命中的 backend id
- `trajectory_id` / `fairness_bucket` 等

### 2) Router：两条 placement 路径

#### A. SGLang 路径（强亲和）

若 strategy 为 `sglang` 并走 sglang-router，`SglangProxy` 会把 resume 的 `last_backend_id` 映射成 `X-ROLL-Preferred-Worker-Url`，提示后端路由优先命中此前 worker。

特点：

- **强亲和**（优先命中 locality），不可用时 fallback。
- ordering 是否发生在 gateway/router 侧取决于其内部 policy；R3 本侧只提供 hint + 指标闭环。

#### B. ROLL 内路由路径（EnvAffinityRouter）

`EnvAffinityRouter` 既能做 baseline sticky（normal），也能在开启开关时做 resume-aware：

- `enable_resume_aware_routing`: resume 选择 worker 时使用 `compute_resume_score(...)`
- `force_migrate_age_s`: pause 超过阈值后降低/取消亲和，允许迁移

## R3：Ordering（队列结构与出队策略）

当 `enable_request_priority_queue=true` 时，R3 启用双队列：

- `pending_resume_requests`
- `pending_normal_requests`

出队机制（软配额 + 吞吐友好规则）：

- `resume_normal_quota`：例如 `3:1`
- **空队跳过**：不因配额空转
- **超时放行**：`normal_max_queue_wait_s`/`resume_max_queue_wait_s` 达到阈值则优先放行该侧，避免饥饿
- **aging**：`effective_priority = base_priority + request_wait_aging_weight * queue_wait_s`

resume 的 `base_priority` 由 `compute_request_priority(...)` 计算，通常包含：

- `pause_age_s`（价值/紧迫度 proxy）
- `history_len_tokens`（代价 proxy）
- `hit_prob`（是否携带 `last_backend_id`）
- `fairness_bonus`（可选）

## “原生 ROLL / baseline” 如何在本仓库复现

### 复现方式 1：关闭统一开关（推荐）

在 `router_args.router_config` 中设置：

- `enable_resume_priority: false`

该开关会统一关闭：

- `enable_resume_aware_routing`
- `enable_request_priority_queue`
- `enable_adaptive_quota`
- 以及软配额超时放行相关阈值

### 复现方式 2：不发送 resume meta（语义回退）

runtime 不写入（或不触发）：

- `request_type=resume`
- `pause_age_s/history_len_tokens/last_backend_id`

则请求会退化为 normal 逻辑，resume-aware 的收益与指标都会消失。

## 行为差异（你应当能观测到什么）

### locality / TTFT（恢复命中）

- baseline：resume continuation 更容易被分配到不同 backend，KV/cache locality 较差。
- R3：resume 优先粘回上次 backend（或在过载/过期条件下迁移），预期 **resume 的尾延迟更稳定**。

### 队列竞争与饥饿

- baseline：resume 与 normal 无隔离与优先表达，高峰期 resume 可能被 normal 挤压。
- R3：软配额与 aging 让 resume 获得更稳定的服务机会，同时通过超时放行避免 normal 饥饿。

### 可解释性（fallback reason）

- baseline：很难解释“为什么没粘回去/为什么迁移”。
- R3：可通过 `resume_fallback_reason/*` 看到 not_found / not_ready / overloaded / forced_migrate 等原因分布。

## 滑动窗口机制（在哪、做什么）

R3 的“调度侧滑动窗口”主要用于 **动态配额（adaptive quota）**：

- 位置：`EnvAffinityRouter` 内部维护两个 `deque(maxlen=adaptive_quota_affinity_window)`：
  - resume affinity hit window
  - resume affinity feasible window
- 用途：在 `_maybe_update_adaptive_quota()` 中结合 backlog 与窗口命中率/可行率，动态调整 `quota_resume_target/quota_normal_target`。

此外，本仓库还有与“滑动窗口”同名但语义不同的用法（不属于调度队列）：

- pipeline 侧 `tps_timer = _Timer(window_size=5)`：用于吞吐/计时的移动窗口统计
- attention/推理侧 `window_size/sliding_window`：用于 flash-attn 或扩散模型模块的滑窗注意力/推理策略

## 关键配置项速查（来自示例/测试配置）

`router_args.router_config` 常用项：

- `enable_resume_priority`
- `enable_request_priority_queue`
- `enable_resume_aware_routing`
- `resume_normal_quota`
- `normal_max_queue_wait_s` / `resume_max_queue_wait_s`
- `enable_adaptive_quota`
- `adaptive_quota_affinity_window`
- `adaptive_quota_min_feasible_rate` / `adaptive_quota_min_hit_rate`
- `force_migrate_age_s`
- `max_running_requests_per_worker`

