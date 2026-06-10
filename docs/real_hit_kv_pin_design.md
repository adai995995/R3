# 真实 Prefix Hit 与 KV Pin 设计（Phase C / Phase D）

本文档在 [system_cost_resume_scheduling_design.md](./system_cost_resume_scheduling_design.md)（形态 A：ROLL 内 system-cost 打分并 **直接选 worker**）之上，定义两阶段工程目标：

| 阶段 | 目标 | 一句话 |
|------|------|--------|
| **Phase C** | **真实 prefix hit** | 调度用的 `p_hit` 来自 SGLang 实测 `cached_tokens` / prefill，而非 belief 或「同 worker」代理 |
| **Phase D** | **真实 KV pin** | tool-wait 期间 RadixCache 按 `lease_score` / TTL 延缓淘汰，而非 gateway 内存表 |

**架构前提（与 Form B 无关）**：

- **Ordering + Placement**：`EnvAffinityRouter` + `enable_system_cost_resume_scheduling` + `enable_resume_aware_routing`
- **Gateway**：可选 **L2 观测**（`GET /kv/resume`）与 **lease 协调**（`POST /kv/lease`），**不**由 gateway 决定发到哪台 worker
- **引擎**：`sglang==0.4.6.post4`（`/mnt/xxl/sglang_r3` 锚定版本）

**相关文档**：

- [system_cost_resume_scheduling_design.md](./system_cost_resume_scheduling_design.md) — 当前已实现的 dispatch / order / lease 公式
- [r3_sglang_remaining_integration_design.md](./r3_sglang_remaining_integration_design.md) — 跨仓库实现状态与分阶段任务索引
- [implementation_status.md](./implementation_status.md) — ROLL 侧开关与联调清单

---

## 1. 当前实现 vs 目标态

### 1.1 已有能力（P0，不必重做）

| 模块 | 能力 |
|------|------|
| R3 `trajectory_value.py` | `compute_system_order_score` / `compute_system_dispatch_score` / `compute_system_lease_ttl` |
| R3 `router.py` | 双队列、`_select_worker_system_cost`、`_enrich_resume_route_meta`、`set_tool_suspend_lease` |
| R3 `sglang_strategy.py` | 从 `meta_info` 读取 `matched_prefix_tokens` 或 **`cached_tokens`** |
| R3 `resume_state.py` | `p_hit_bias` EWMA、`observe_resume_outcome` / `observe_lookup_resume` |
| gateway `kv_lease.rs` | 内存 `KvLeaseStore`：`POST/GET/DELETE /kv/*`（**控制面**） |

### 1.2 缺口（本文要补齐）

| 问题 | 现状 | 目标态 |
|------|------|--------|
| **Hit 是否真实** | `p_hit` 主要来自 belief；`_observe_resume_outcome` 在无 telemetry 时把 **同 worker** 当 `gpu_hit` | 以 **`cached_tokens / history_len`** 为权威；同 worker 但 full prefill 不得记为 hit |
| **Lookup 是否真实** | `GET /kv/resume` 读 gateway 表，`hit_tokens` 常为 POST 时写入的 proxy | C 阶段：lookup 可仍弱依赖；**主路径以 worker 返回为准**；D 阶段：lookup 读 **引擎** 状态 |
| **Pin 是否真实** | `POST /kv/lease` 只更新 gateway `DashMap`，**不**改 RadixCache eviction | D 阶段：worker 持有 lease，evict 时减去 `λ * lease_score` |
| **形态 A 接线** | `EnvAffinityRouter` 未统一设置 `gateway_url`，L3 push 常为 no-op | `gateway_url` ≡ `gateway_status_url` fallback；resume dispatch 可选 push |

### 1.3 非目标

- 不用 scheduler 判断轨迹是否值得学习（与 system-cost 设计一致）。
- 不在 C/D 首版做 **跨 worker KV 迁移**。
- 不要求 C 阶段就完成 D 的物理 pin（可并行规划，分 PR 落地）。
- 不引入 gateway placement（非形态 A 路线）。

---

## 2. 端到端数据流（目标态）

```text
                    ┌─────────────────────────────────────┐
                    │  TrajEnvManager                     │
                    │  tool 前: plan_tool_suspend_lease   │
                    │  tool 后: resume route_meta          │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  EnvAffinityRouter                    │
                    │  C: enrich(meta) ← worker telemetry │
                    │     order_score → 队列               │
                    │     dispatch_score → dp_rank         │
                    │  D: POST lease → worker (+ gateway)   │
                    └──────────────┬──────────────────────┘
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
  ┌────────▼────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
  │ SGLang worker     │   │ sgl-model-gateway │   │ metrics / dump    │
  │ C: meta_info      │   │ D: lease 索引     │   │ matched_prefix,   │
  │   cached_tokens   │   │   (可选)          │   │ lease_remaining   │
  │ D: RadixCache     │   │ GET /kv/resume    │   │ context_class     │
  │   lease+evict     │   │ POST /kv/lease    │   │                   │
  └───────────────────┘   └───────────────────┘   └───────────────────┘
```

---

# Part I — Phase C：真实 Prefix Hit

## C.1 目标与验收

### C.1.1 目标

1. **每次 resume 推理后**，ROLL 获得可解释的 **实测** prefix 复用：`matched_prefix_tokens`（或 `cached_tokens`）。
2. **`p_hit` 用于 system-cost 打分**时，以实测为主、belief 为辅，而非仅用 `last_backend_id` + pause_age。
3. **`p_hit_bias` 反馈**仅对 **真实 full prefill / gpu hit** 更新，禁止用 placement affinity 冒充 KV hit。

### C.1.2 验收标准

| # | 验收项 | 通过条件 |
|---|--------|----------|
| C-A1 | 字段可见 | rollout / checkpoint metrics 中 `matched_prefix_tokens` 在 **同 trajectory 第二次 resume** 上 p50 > 0（有 cache 的场景） |
| C-A2 | 分类正确 | `context_class`：仅当 `matched_prefix / history_len ≥ θ_hit` 时为 `gpu_hit`；同 worker 且 prefill_ratio ≥ 0.9 时为 `full_prefill` |
| C-A3 | 反馈有效 | `enable_belief_feedback=true` 时，连续 full prefill 后 `p_hit_bias` 下降；连续 hit 后上升 |
| C-A4 | 打分区分 | 同 `last_backend`、不同 `cached_tokens` 的两条 meta，`dispatch_score` 可区分 |
| C-A5 | 实验 | A/B：C 全开 vs 仅 belief proxy，resume `resume_prefill_tokens` p95 下降（同负载） |

推荐阈值（可配置）：`θ_hit = 0.3`（matched 占 history 30% 以上视为有效 hit）。

---

## C.2 观测字段与语义

### C.2.1 引擎侧（权威来源）

SGLang 0.4.6 已在 generate 路径产出 **`cached_tokens`**（radix prefix cache 命中）。Phase C **不强制** 新增字段名，但统一映射为：

| 对外字段 | 来源 | 含义 |
|----------|------|------|
| `matched_prefix_tokens` | `meta_info.cached_tokens` 或显式别名 | 本次 prefill 复用的 prefix token 数 |
| `prompt_tokens` | `meta_info` | 总 prompt token |
| `resume_prefill_tokens` | `max(0, prompt_tokens - matched_prefix_tokens)` | 需重算的 prefill 量 |
| `prefill_ratio` | `resume_prefill_tokens / history_len_tokens` | 与 route_meta 对齐的归一化代价 |
| `actual_hit` | `1 if matched_prefix_tokens > 0 else 0` | 二元命中 |

可选增强（P1，非 C 阻塞）：

- `prefill_time_ms` — 用于替换 `history_len` 粗 proxy
- `context_class_gpu_hit` / `full_prefill` — 引擎直接分类（减少 ROLL 推断）

### C.2.2 ROLL 侧（`route_meta` / response）

在 `generate_request` 返回的 `out` 与 `route_meta` 中稳定写入：

```text
matched_prefix_tokens
resume_prefill_tokens
prefill_ratio
actual_hit
context_class          # gpu_hit | cpu_reload | full_prefill
engine_cache_confidence  # clamp(matched_prefix / history_len, 0, 1)
```

**`context_class` 判定规则（必须实现，替换 affinity 冒充）**：

```python
hit_ratio = matched_prefix_tokens / max(history_len_tokens, 1)
if hit_ratio >= theta_hit:
    context_class = "gpu_hit"
elif prefill_ratio >= theta_full:   # e.g. 0.85
    context_class = "full_prefill"
elif affinity_hit and hit_ratio > 0:
    context_class = "cpu_reload"      # 可选：部分命中
else:
    context_class = "full_prefill"
```

禁止：`elif affinity_hit: context_class = "gpu_hit"` 作为 **默认** 分支。

### C.2.3 Gateway lookup（C 阶段从属）

`GET /kv/resume/{tid}` **不能**作为 hit 的权威（当前为控制面表）。C 阶段约定：

- **权威**：worker `meta_info` → ROLL `_observe_resume_outcome`
- **辅助**：lookup 仅提供 `lease_remaining_s`、`worker_url`（D 之前可为 proxy）

---

## C.3 `p_hit` 与 system-cost 打分融合

### C.3.1 测量命中概率

定义：

```text
p_hit_measured = clamp(matched_prefix_tokens / max(history_len_tokens, 1), 0, 1)
```

可选平滑（长上下文）：`p_hit_measured = clamp(matched_prefix_tokens / max(prompt_tokens, 1), 0, 1)`。

### C.3.2 与 belief 融合

在 `compute_system_order_score` / `compute_system_dispatch_score` 入口：

```text
p_hit_belief = apply_p_hit_bias(belief_to_p_hit(level), p_hit_bias)
p_hit_effective = w_m * p_hit_measured + (1 - w_m) * p_hit_belief
```

| 参数 | 建议默认 | 说明 |
|------|----------|------|
| `w_m` | `0.7`（`enable_engine_telemetry=true`） | 实测权重 |
| `w_m` | `0.0`（开关关闭） | 与当前行为兼容 |

当 **`p_hit_measured` 可得**（本次或上一轮 resume 已回填）时强制 `w_m > 0`；首条 resume 无历史测量时退回 belief。

### C.3.3 反馈（`resume_state.py`）

`observe_resume_outcome` 仅使用 **C.2.3 的 context_class**，不再用裸 `affinity_hit` 驱动 `alpha_hit`。

`observe_lookup_resume`（若开启）：仅当 `lookup.hit_tokens > 0` **且** D 阶段引擎回填时融合；C 阶段可忽略 `hit_tokens` 或降权。

---

## C.4 代码改动清单（按仓库）

### C.4.1 R3（必做）

| 文件 | 改动 |
|------|------|
| `roll/distributed/scheduler/router.py` | 重写 `_observe_resume_outcome` 的 `context_class` 逻辑；在 `_compute_request_base_priority` / `_select_worker_system_cost` 前合并 `p_hit_measured` |
| `roll/distributed/scheduler/trajectory_value.py` | 新增 `merge_measured_p_hit(route_meta, belief, weights)`；`compute_system_*` 增加可选参数 `p_hit_override` |
| `roll/distributed/scheduler/resume_state.py` | `observe_resume_outcome` 注释与分支对齐新 `context_class` |
| `roll/distributed/strategy/sglang_strategy.py` | 确认 `cached_tokens` 始终写入 `output_data`；缺省时打 debug metric |
| `roll/pipeline/agentic/env_manager/traj_env_manager.py` | metrics 聚合 `matched_prefix_tokens`、`context_class` 分桶 |
| `examples/toolcall_benchmark/toolcall_benchmark_resume_aware.yaml` | 增加 `enable_belief_feedback: true`、可选 `enable_engine_telemetry: true` |

新增配置（`router_config`）：

```yaml
enable_engine_telemetry: false      # C 主开关：实测 p_hit 融入 score
engine_telemetry_measured_weight: 0.7
engine_telemetry_hit_ratio_threshold: 0.3
engine_telemetry_full_prefill_ratio: 0.85
```

### C.4.2 sglang_r3 Python（可选增强）

| 文件 | 改动 |
|------|------|
| `python/sglang/srt/managers/tokenizer_manager.py` | 保证 `cached_tokens` 在 streaming / non-streaming 一致 |
| `python/sglang/srt/openai_api/adapter.py` | 可选：响应 body 增加 `matched_prefix_tokens` 别名 |
| `python/sglang/srt/metrics/collector.py` | 按 `trajectory_id` 标签（若 rid 可传入）— P1 |

C 阶段 **可不改引擎** 若验证 `cached_tokens` 已足够。

### C.4.3 sgl-model-gateway（C 阶段非阻塞）

无需改 placement。可选：在转发 worker 响应时透传 `x-roll-matched-prefix-tokens`（ROLL 直连 worker 时可跳过）。

---

## C.5 配置示例（形态 A）

```yaml
router_args:
  router_name: EnvAffinityRouter
  router_config:
    enable_resume_priority: true
    enable_resume_aware_routing: true
    enable_request_priority_queue: true
    enable_system_cost_resume_scheduling: true
    enable_trajectory_value_scheduling: false

    enable_belief_feedback: true
    enable_engine_telemetry: true
    engine_telemetry_measured_weight: 0.7

    # 可选 L2；hit 不依赖 lookup
    gateway_status_url: "http://127.0.0.1:30000"
    enable_lookup_resume: false
```

---

## C.6 测试计划（Phase C）

| 类型 | 内容 |
|------|------|
| 单测 | `context_class` 在 `matched=0, affinity=1` → `full_prefill`；`matched/hist=0.5` → `gpu_hit` |
| 单测 | `p_hit_effective` 随 `w_m` 单调 |
| 集成 | 同一 prompt 连续两次 generate，第二次 `cached_tokens > 0` |
| 训练 smoke | resume-aware + `enable_engine_telemetry`，检查 checkpoint JSON 字段 |
| A/B | 对比 prefill p95、affinity hit rate（后者不应再等同 KV hit） |

---

## C.7 风险与回滚

| 风险 | 缓解 |
|------|------|
| `cached_tokens` 恒为 0（cache 未开） | 文档要求 `cache_report` / radix 配置；C 开关默认 false |
| `history_len` 与 `prompt_tokens` 不一致 | 用 `prompt_tokens` 作分母备选 |
| 反馈噪声导致 `p_hit_bias` 震荡 | 保持 EWMA 限幅 ±0.2 |
| 回滚 | `enable_engine_telemetry: false` 恢复纯 belief |

---

# Part II — Phase D：真实 KV Pin

## D.1 目标与验收

### D.1.1 目标

1. Tool suspend 期间，轨迹 KV 在 **目标 worker** 上 **延缓被淘汰**。
2. `POST /kv/lease` 与 **`DELETE /kv/lease`** 驱动引擎状态，gateway 表为 **索引/观测**（可选）。
3. `GET /kv/resume/{tid}` 返回 **引擎真实** `found`、`lease_remaining_s`、`hit_tokens`（若仍 resident）。
4. `compute_system_lease_ttl` 可使用 **真实 `memory_pressure`**，而非常数 `rho_mem=1`。

### D.1.2 验收标准

| # | 验收项 | 通过条件 |
|---|--------|----------|
| D-A1 | Pin 有效 | 有 lease vs 无 lease：同 trajectory tool-wait 后 resume，`matched_prefix_tokens` p50 **显著更高** |
| D-A2 | TTL 过期 | `ttl_s` 到期后 lookup `found=false` 或 `hit_tokens=0`，且 prefill 回升 |
| D-A3 | Score 有效 | 高 `lease_score` 轨迹比低 score 更少 `full_prefill`（控制变量：同 `history_len`） |
| D-A4 | 压力 | 内存压力下低 score lease 先 evict（metric 可观测） |
| D-A5 | R3 闭环 | `set_tool_suspend_lease` → worker/gateway 收到 POST；terminate → DELETE |
| D-A6 | 实验 | resume e2e p95 降，normal queue wait p95 不恶化 > X%（X 由实验定，建议 10%） |

---

## D.2 概念模型

### D.2.1 Lease 记录（引擎权威）

每条 lease 绑定：

```text
trajectory_id   (string, 与 ROLL rid 一致)
worker_id       (local)
expires_at      (monotonic / wall clock)
lease_score     [0, 1]
belief_level    hot | warm | cold  (可选，影响 pin 强度)
kv_handle         (可选：radix 子树根 / session id)
```

**Pin 语义**：在 `expires_at` 之前，与该 trajectory 关联的 radix 节点 **不可被普通 LRU evict**；超时或 `DELETE` 后恢复普通策略。

**非保证**：OOM 紧急路径可打破 lease（必须实现，否则 worker 不稳定）。

### D.2.2 与 system-cost lease 公式关系

ROLL 仍用 [system_cost_resume_scheduling_design.md §6](./system_cost_resume_scheduling_design.md) 计算 `ttl_s` / `lease_score`：

```text
tau* = argmax_tau V_lease(r, tau)
```

D 阶段替换：

```text
rho_mem(w,t)  ← engine memory_pressure (0..1)
KV_bytes(r)   ← resident_bytes 或 history_len * bytes_per_token
P(T_tool <= tau) ← 仍可用 t_tool_ema
```

计算结果通过 `POST /kv/lease` 下发到 **引擎**。

---

## D.3 API 设计

### D.3.1 Worker 内部 API（推荐，形态 A 直连）

在 SGLang HTTP server 增加（路径可前缀 `/internal`）：

```http
POST /internal/kv/lease
Content-Type: application/json

{
  "trajectory_id": "traj-xxx",
  "ttl_s": 30.0,
  "lease_score": 0.82,
  "belief_level": "hot"
}

→ 200 { "ok": true, "expires_at_ms": ..., "resident_blocks": N }

DELETE /internal/kv/lease/{trajectory_id}
→ 200 { "ok": true, "deleted": true }

GET /internal/kv/resume/{trajectory_id}
→ 200 {
  "found": true,
  "lease_remaining_s": 12.5,
  "hit_tokens": 1024,
  "resident_blocks": 16,
  "estimated_prefill_tokens": 128,
  "cache_confidence": 0.91,
  "memory_pressure": 0.4
}
```

**关联方式**（二选一，实现时定夺）：

| 方案 | 说明 | 复杂度 |
|------|------|--------|
| **Rid → Req 映射** | 生成时 `rid=trajectory_id` 已存在；lease 挂在 `Req` 的 radix 叶节点 | 低 |
| **Session 级 radix 子树** | 按 `trajectory_id` 标记子树节点 `lease_score` | 中 |

### D.3.2 Gateway API（索引 + 跨 worker 查询）

保留现有：

```http
POST   /kv/lease
GET    /kv/resume/{trajectory_id}
DELETE /kv/lease/{trajectory_id}
```

D 阶段行为变更：

1. `POST /kv/lease`：除写入 `KvLeaseStore` 外，**异步/同步转发**到 `worker_url` 的 `POST /internal/kv/lease`。
2. `GET /kv/resume/{tid}`：优先 **查询 worker**（由 `worker_url` 定位），合并 gateway 表 TTL；worker 不可达时返回 `found=false`。
3. `DELETE`：转发 worker + 删表。

形态 A 下 ROLL 可 **直连 worker** 调 internal API（绕过 gateway），gateway 仅用于多 worker 聚合查询；配置项 `lease_push_mode: worker | gateway | both`。

### D.3.3 ROLL 调用链（形态 A）

```text
tool suspend:
  TrajEnvManager → RouterManager.set_tool_suspend_lease
    → EnvAffinityRouter._maybe_push_kv_lease
      → POST {worker}/internal/kv/lease  (primary)
      → POST {gateway}/kv/lease          (optional index)

resume 调度前:
  _enrich_resume_route_meta
    → GET {gateway}/kv/resume/{tid}  或 GET {worker}/internal/kv/resume/{tid}
    → 回填 ttl_remaining_s, hit_tokens, memory_pressure

episode end:
  delete_kv_lease → DELETE worker + DELETE gateway
```

**P0 接线（R3）**：

```python
# EnvAffinityRouter.initialize
self.gateway_url = _norm_url(
    router_config.get("gateway_url") or router_config.get("gateway_status_url") or ""
)
```

`_maybe_push_kv_lease` 与 `_delete_kv_lease` 使用同一 fallback。

---

## D.4 RadixCache Eviction 改造

### D.4.1 节点元数据

在 radix 节点或 request 级结构增加：

```text
lease_score: f32      # 0 = 无保护
lease_expires_at: u64
```

### D.4.2 淘汰评分

```text
evict_priority(node) = LRU_key(node) - lambda_lease * node.lease_score
```

- 叶节点继承所属 `trajectory_id` 的有效 lease（取 max lease_score）。
- `lambda_lease` 可配置（如 `1.0`）。

### D.4.3 紧急驱逐

当 `evictable_size < 0` 或 `gpu_free_mem < threshold`：

```text
允许淘汰 lease_score < s_emergency 的节点（如 0.3）
```

并打点 `lease_broken_oom_count`。

### D.4.4 TTL 到期

后台或每次 `GET /internal/kv/resume` 时清理过期 lease 标记，恢复 LRU。

---

## D.5 代码改动清单（按仓库）

### D.5.1 sglang_r3 `python/sglang`（核心）

| 文件 | 改动 |
|------|------|
| `srt/mem_cache/radix_cache.py` | lease 字段、evict 公式、紧急路径 |
| `srt/managers/schedule_batch.py` 或等价 | rid ↔ lease 绑定 |
| `srt/entrypoints/http_server.py` | `/internal/kv/lease`、`/resume`、`DELETE` |
| `srt/managers/tokenizer_manager.py` | `meta_info` 增加 `lease_remaining_s`、`memory_pressure`（可选） |

### D.5.2 sgl-model-gateway

| 文件 | 改动 |
|------|------|
| `src/kv_lease.rs` | `set_lease` 转发 worker；`lookup` 聚合 worker 响应 |
| `src/server.rs` | 路由不变，行为变更 |
| `src/core/worker.rs` | worker HTTP client 调 internal lease API |

### D.5.3 R3

| 文件 | 改动 |
|------|------|
| `roll/distributed/scheduler/router.py` | `gateway_url` fallback；`_maybe_push_kv_lease` 支持 worker 直连；dispatch 后可选 refresh lease |
| `roll/distributed/scheduler/kv_lease_client.py` | `set_kv_lease_worker()`、`lookup_resume_worker()` |
| `roll/distributed/scheduler/trajectory_value.py` | `compute_system_lease_ttl` 使用 `memory_pressure` |
| `roll/distributed/strategy/sglang_strategy.py` | worker URL 暴露给 router（已有 `get_url`） |

新增配置：

```yaml
gateway_url: "http://127.0.0.1:30000"
enable_value_driven_lease: true
enable_gateway_kv_lease_push: true
enable_lookup_resume: true
lease_push_mode: "both"           # worker | gateway | both
lease_worker_path: "/internal/kv/lease"
lease_worker_timeout_s: 2.0
```

---

## D.6 Phase C 与 Phase D 依赖关系

```text
Phase C ──可独立上线──► 调度更准确（实测 hit）
    │
    │ 建议先 C 后 D：D 的 lookup.hit_tokens 在 C 语义下才可解释
    ▼
Phase D ──依赖 lease 接线 + 引擎 evict──► 真实 pin，tool-wait 少 full prefill
```

| 组合 | 行为 |
|------|------|
| 仅 C | 打分准，但 tool-wait 仍可能 evict → prefill 仍可能高 |
| 仅 D（不推荐） | pin 有效，但 `p_hit` 仍猜 → placement 可能次优 |
| C + D | 完整闭环 |

---

## D.7 测试计划（Phase D）

| 类型 | 内容 |
|------|------|
| 单测 | `KvLeaseStore` 转发 mock worker；过期 TTL |
| 引擎单测 | 有 lease 节点不被 LRU 优先 evict |
| curl | POST lease → GET resume `found=true` → sleep(ttl) → `found=false`（仅验证 TTL/接口，不代表真实 tool-wait） |
| 集成 | R3 toolcall + `enable_gateway_kv_lease_push`，gateway/worker 日志见 lease（必须是真实 toolcall 延迟场景） |
| A/B | lease on/off，对比 `matched_prefix_tokens`、`resume_prefill_tokens` p95 |

### D.7.1 强制实验约束：必须使用真实 toolcall 延迟

本项目验证 Phase D 的目标是：**tool suspend 期间（真实外部调用阻塞），引擎侧 KV 能被 lease 延缓淘汰**。
因此集成实验必须满足：

- **必须**：toolcall 的等待来自真实外部依赖（例如：HTTP/RPC、数据库、搜索、文件系统、远端服务、真实执行沙箱等），产生真实 wall-clock 延迟与队列交互。
- **禁止**：使用 `sleep` / `time.sleep` / 人工延迟注入来“构造 tool-wait”。构造延迟只能用于接口/TTL 的最小自测（curl 层），不能作为 D-A1/D-A6 的验收依据。

推荐做法：

- 选择一个会真实调用外部系统的 tool（网络/检索/代码执行/IO），并确保其延迟分布稳定可观测。
- 在实验记录中同时保存：`external_wait_s`（真实 tool 等待）、`matched_prefix_tokens`、`resume_prefill_tokens`、`lookup_resume_found`、`memory_pressure`。

---

## D.8 风险与回滚

| 风险 | 缓解 |
|------|------|
| Lease 导致 OOM | 紧急驱逐 + `memory_pressure` 反馈 + `lease_score` 上限 |
| 多 worker lease 不一致 | gateway 索引 + `worker_url` 必填 |
| 转发延迟拖慢 tool step | 短 timeout、异步 push（P1） |
| 回滚 | `enable_gateway_kv_lease_push: false` + 引擎 lease API 返回 404 时 no-op |

---

## 8. 实施顺序建议

| 顺序 | 里程碑 | 仓库 | 预估 |
|------|--------|------|------|
| 1 | 验证 `cached_tokens` 在 resume 场景有效 | R3 smoke | 0.5d |
| 2 | **Phase C** Router 实测 `p_hit` + `context_class` 修复 | R3 | 2d |
| 3 | **Phase C** 配置与 A/B | R3 | 1d |
| 4 | R3 `gateway_url` fallback + lease push 通路 | R3 | 0.5d |
| 5 | Worker internal lease API + radix evict | sglang | 4–6d |
| 6 | Gateway 转发 + lookup 聚合 | gateway | 2–3d |
| 7 | **Phase D** 集成与 A/B | 全栈 | 2d |

---

## 9. 指标与可观测性

### 9.1 新增 / 强调指标

| 指标 | 阶段 | 含义 |
|------|------|------|
| `env/*/matched_prefix_tokens_*` | C | 实测 hit |
| `env/*/resume_prefill_tokens_*` | C | 重算成本 |
| `scheduler/router/context_class/*_count` | C | gpu_hit / full_prefill 分桶 |
| `scheduler/router/p_hit_measured_mean` | C | 实测 p_hit |
| `scheduler/router/lease_push_success` | D | POST 成功 |
| `scheduler/router/lease_delete_success` | D | DELETE 成功 |
| `lookup_resume_found` | D | 引擎 found |
| `engine/memory_pressure` | D | 租约算 TTL |
| `engine.lease_broken_oom_count` | D | 紧急打破 |

### 9.2 禁止误读

```text
resume_affinity_hit_rate == 1  ⇏  KV gpu hit
lookup_resume_found == 1       ⇏  物理 KV 仍在（D 前仅为控制面）
lease_push_success == 1        ⇏  引擎已 pin（需 D-A1 验证 prefill）
```

---

## 10. 文档修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-31 | 初稿：Phase C（真实 prefix hit）+ Phase D（真实 KV pin），形态 A 专用 |
