# R3 + sglang_r3 System-Cost Resume 剩余开发设计

本文档面向两个仓库：

- `/mnt/xxl/R3`
- `/mnt/xxl/sglang_r3`

目标是在当前 R3 已完成的 system-cost resume scheduling 骨架之上，补齐最终闭环：

```text
tool call suspend
  -> 立即设置 KV lease
  -> tool return 生成 Resume Request
  -> lookup_resume 校准 p_hit / ttl_remaining / prefill cost
  -> router 做 order + placement
  -> gateway / engine 返回真实 telemetry
  -> 更新 p_hit_bias / t_tool_ema / worker_load / lease 状态
```

当前 R3 已具备 `enable_system_cost_resume_scheduling`、`dispatch_score`、`order_score`、system-cost lease proxy、`kv_lease_client.py`；但仍缺少 **形态 A 下 gateway lease 接线完善**、**engine telemetry** 和 **真实 hit/memory/TTL 闭环**。

---

## 1. 当前代码状态

### 1.1 R3 侧已实现

关键文件：

- `roll/pipeline/agentic/env_manager/traj_env_manager.py`
- `roll/distributed/scheduler/router.py`
- `roll/distributed/scheduler/trajectory_value.py`
- `roll/distributed/scheduler/resume_state.py`
- `roll/distributed/scheduler/kv_lease_client.py`
- `roll/pipeline/agentic/context_lifecycle.py`
- `roll/distributed/strategy/sglang_strategy.py`

已完成能力：

- `TrajEnvManager` 能识别 tool-return 边界，并把下一轮请求标为 `request_type=resume`。
- `RouterClient._preprocess_generate()` 会把 resume routing meta 写入 `_roll_route_meta`。
- `EnvAffinityRouter` / `SglangOrderingRouter` 已支持：
  - resume / normal 双队列
  - soft quota
  - `enable_system_cost_resume_scheduling`
  - `compute_system_order_score()`
  - `compute_system_worker_route_score()`
  - `compute_system_lease_ttl()`
- `kv_lease_client.py` 已有：
  - `GET /kv/resume/{trajectory_id}`
  - `POST /kv/lease`
  - `DELETE /kv/lease/{trajectory_id}`
- `SglangOrderingRouter` 已能把 ROLL header 发给 gateway：
  - `X-ROLL-Request-Type`
  - `X-ROLL-Preferred-Worker-Url`
  - `X-ROLL-Pause-Age-S`
  - `X-ROLL-History-Len-Tokens`
  - `X-ROLL-Resume-Lease-Ttl-S`
  - `X-ROLL-Resume-Lease-Score`
  - `X-ROLL-Belief-Level`
  - `X-ROLL-Trajectory-Id`

### 1.2 sglang_r3 侧当前状态

关键文件：

- `docs/r3_sgl_model_gateway.md`
- `sgl-model-gateway/src/server.rs`
- `sgl-model-gateway/src/routers/http/router.rs`
- `sgl-model-gateway/src/routers/header_utils.rs`
- `sgl-model-gateway/src/policies/cache_aware.rs`
- `sgl-model-gateway/src/core/worker.rs`
- `python/sglang/srt/entrypoints/http_server.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/mem_cache/radix_cache.py`

已完成能力：

- `sgl-model-gateway` 已存在于 `sglang_r3` 中，文档明确建议 R3 的 gateway 能力在此演进。
- Rust gateway 的 `/generate` 会把 headers 传给 router。
- `header_utils.rs` 已能解析：
  - `X-ROLL-Preferred-Worker-Url`
  - `X-SMG-Preferred-Worker-Url`
- `cache_aware.rs` 已支持 preferred worker override：
  - 如果 `X-ROLL-Preferred-Worker-Url` 命中健康 worker，则直接选该 worker。
  - 否则回退到 cache-aware / load-aware 选择。
- `/workers` 已存在，可用于 R3 polling worker 健康状态与 load。

未完成能力：

- gateway 还没有 `/kv/resume/{trajectory_id}`。
- gateway 还没有 `/kv/lease` / `/kv/lease/{trajectory_id}`。
- gateway 还没有 per-trajectory lease table。
- gateway 没有把 `lease_score` 用于 eviction / routing 决策。
- Python SGLang worker 没有返回 R3 需要的真实 telemetry：
  - `matched_prefix_tokens`
  - `actual_hit`
  - `estimated_prefill_tokens`
  - `prefill_time_ms`
  - `memory_pressure`
  - `kv_bytes`
- SGLang radix cache 没有按 `trajectory_id` 显式 pin / unpin / lease。

---

## 2. 总体架构

### 2.1 三层责任划分

| 层 | 仓库 | 职责 |
|---|---|---|
| ROLL runtime | `R3` | 识别 tool suspend / tool return，计算 system-cost score，生成 lease intent |
| Gateway control plane | `sglang_r3/sgl-model-gateway` | 维护 `trajectory_id -> lease` 状态，处理 lookup / set / delete，执行 preferred worker 和 telemetry 聚合 |
| SGLang worker engine | `sglang_r3/python/sglang` | 返回 prefix/cache telemetry；后续可选实现真实 KV pin / eviction modifier |

### 2.2 分阶段目标

1. **P0：ROLL + Gateway 控制面闭环**
   - tool suspend 时 R3 立即调用 `POST /kv/lease`。
   - gateway 维护 lease table。
   - `GET /kv/resume/{tid}` 返回 lease 状态和 worker hint。
   - 不要求真实 engine KV pin。

2. **P1：Gateway telemetry 闭环**
   - gateway 在 `/generate` response header 中返回 selected worker。
   - gateway 根据 worker response / headers 汇总 `matched_prefix_tokens`、`prefill_time_ms`。
   - R3 用 lookup + generate telemetry 更新 `p_hit_bias`。

3. **P2：SGLang worker telemetry**
   - Python SGLang worker 在 `meta_info` 或 response header 中返回真实 prefix match 信息。
   - `lookup_resume` 可以更准确地返回 hit / prefill estimate。

4. **P3：真实 KV lease enforcement**
   - SGLang radix cache 支持 lease-aware eviction。
   - gateway 的 `lease_score` 影响真实 KV 保留，而不只是调度 hint。

---

## 3. R3 侧剩余开发

### 3.1 Tool suspend 立即下发 lease

当前问题：

- `TrajEnvManager.step()` 在 LLM 输出可能触发 tool 时，会调用 `_maybe_set_pending_tool_suspend_lease()`。
- 该函数只把 pending lease 写进本进程的 `TrajectorySchedulingState` 和下一轮 meta。
- 真正 `_maybe_push_kv_lease()` 发生在 Router 处理下一次 resume 时，已经错过 tool wait 窗口。

目标：

```text
LLM 输出 tool call
  -> 计算 ttl_s / lease_score
  -> env.step(action) 执行工具前
  -> 立即通知 Router/Gateway 设置 lease
```

建议修改：

1. 在 `RouterManager` 增加 Ray 方法：

```python
async def set_tool_suspend_lease(self, route_meta: Dict[str, Any]) -> Dict[str, Any]:
    return await self.router.set_tool_suspend_lease(route_meta)
```

2. 在 `Router` 基类增加默认 no-op：

```python
async def set_tool_suspend_lease(self, route_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"pushed": False, "reason": "unsupported_router"}
```

3. 在 `EnvAffinityRouter` / `SglangOrderingRouter` 实现：

```python
async def set_tool_suspend_lease(self, route_meta):
    self._sync_scheduling_meta(route_meta)
    self._compute_request_base_priority("resume", route_meta)
    self._merge_resume_lease_ttl_score(route_meta)
    await self._maybe_push_kv_lease(route_meta, phase="tool_suspend")
    return {"pushed": True, "ttl_s": ..., "lease_score": ...}
```

4. 在 `TrajEnvManager._maybe_set_pending_tool_suspend_lease()` 中，在本地 state 写入后，如果 `generate_scheduler` 是 Ray actor，则调用：

```python
ray.get(self.generate_scheduler.set_tool_suspend_lease.remote(route_meta))
```

注意：

- tool suspend lease 失败不能中断 rollout。
- 该调用应短 timeout 或 fire-and-forget；否则工具执行前阻塞会影响实验。
- 如果当前 router 不是 `SglangOrderingRouter` 或没有 gateway_url，应返回 no-op。

### 3.2 Tool terminal / reset 时释放 lease

当前问题：

- `reset()` 会 clear local `TrajectorySchedulingState`，但不会通知 gateway 删除 lease。
- episode terminal 后也没有 `DELETE /kv/lease/{tid}`。

目标：

```text
terminated / max_steps / reset / cancel
  -> delete_kv_lease(trajectory_id)
```

建议修改：

- `RouterManager.delete_kv_lease(trajectory_id)`。
- `EnvAffinityRouter.delete_kv_lease()` 调用 `kv_lease_client.delete_kv_lease()`。
- `TrajEnvManager.reset()` 在切换 trajectory 前，best-effort delete 上一条 trajectory lease。
- `TrajEnvManager.step()` 如果 `terminated or truncated`，best-effort delete 当前 lease。

### 3.3 lookup_resume 结果接入 system-cost 公式

当前问题：

- `_enrich_resume_route_meta()` 已调用 `lookup_resume()`，但 system-cost score 主要仍使用 proxy。
- `lookup_hit_tokens`、`lookup_estimated_prefill_tokens`、`lookup_lease_remaining_s` 尚未系统化进入公式。

建议映射：

```text
lookup_cache_confidence -> p_hit_bias / p_hit override
lookup_hit_tokens       -> matched_prefix_tokens
lookup_estimated_prefill_tokens -> C_hist_prefill
lookup_lease_remaining_s -> ttl_remaining_s
lookup.worker_url       -> preferred worker / last_backend override
```

实现点：

- `roll/distributed/scheduler/trajectory_value.py`
  - `compute_history_prefill_cost()` 优先使用 `lookup_estimated_prefill_tokens`。
  - `compute_system_order_score()` 优先使用 `ttl_remaining_s` / `lookup_lease_remaining_s` 做 delay regret。
- `roll/distributed/scheduler/router.py`
  - `_enrich_resume_route_meta()` 写入 `ttl_remaining_s`。
  - 若 lookup 返回 `worker_url`，映射回 `last_backend_id` 或写入 `preferred_worker_url`。

### 3.4 统一 aging

当前问题：

- system-cost `order_score` 内部已有 `A_age`。
- Router 出队时仍使用 `_effective_request_priority() = base_priority + request_wait_aging_weight * queue_wait_s`。
- 这会造成双重 aging。

建议：

- 当 `enable_system_cost_resume_scheduling=true` 时，`_effective_request_priority()` 不再额外加 `request_wait_aging_weight`，而是出队前重算 `order_score(queue_wait_s=now - enqueue_ts)`。

实现点：

```python
def _effective_request_priority(self, pending):
    if self.enable_system_cost_resume_scheduling and pending.request_type == "resume":
        pending.route_meta["resume_enqueue_ts"] = pending.enqueue_ts
        pending.base_priority = self._compute_request_base_priority("resume", pending.route_meta)
        return pending.base_priority
    return pending.base_priority + self.request_wait_aging_weight * queue_wait
```

### 3.5 R3 metrics / dump

需要确保 rollout dump 和 metrics 中包含：

- `order_score`
- `dispatch_score`
- `system_delay_regret`
- `expected_prefill_saved`
- `ttl_remaining_s`
- `lookup_hit_tokens`
- `lookup_estimated_prefill_tokens`
- `actual_hit`
- `matched_prefix_tokens`
- `prefill_time_ms`
- `kv_bytes_proxy`
- `memory_pressure`
- `lease_push_success`
- `lease_delete_success`

---

## 4. Gateway 侧剩余开发

目标仓库：

```text
/mnt/xxl/sglang_r3/sgl-model-gateway
```

### 4.1 新增 KV lease 数据结构

建议新增文件：

```text
sgl-model-gateway/src/kv_lease.rs
```

核心结构：

```rust
pub struct KvLeaseRecord {
    pub trajectory_id: String,
    pub worker_url: Option<String>,
    pub ttl_s: f64,
    pub lease_score: f64,
    pub belief_level: Option<String>,
    pub created_at: Instant,
    pub updated_at: Instant,
    pub expires_at: Instant,
    pub hit_tokens: Option<u64>,
    pub estimated_prefill_tokens: Option<u64>,
    pub cache_confidence: Option<f64>,
}

pub struct KvLeaseStore {
    records: DashMap<String, KvLeaseRecord>,
}
```

API：

```rust
impl KvLeaseStore {
    pub fn set_lease(&self, req: SetKvLeaseRequest) -> KvLeaseRecord;
    pub fn lookup(&self, trajectory_id: &str) -> Option<KvLeaseRecord>;
    pub fn delete(&self, trajectory_id: &str) -> bool;
    pub fn prune_expired(&self) -> usize;
}
```

### 4.2 新增 HTTP API

在 `server.rs` 增加路由：

```rust
.route("/kv/resume/{trajectory_id}", get(lookup_resume))
.route("/kv/lease", post(set_kv_lease))
.route("/kv/lease/{trajectory_id}", delete(delete_kv_lease))
```

#### `POST /kv/lease`

请求：

```json
{
  "trajectory_id": "gsm8k_0_1_42_0",
  "ttl_s": 8.5,
  "lease_score": 0.73,
  "worker_url": "http://10.0.0.1:23456",
  "belief_level": "hot"
}
```

响应：

```json
{
  "ok": true,
  "trajectory_id": "...",
  "worker_url": "...",
  "ttl_s": 8.5,
  "lease_score": 0.73,
  "lease_remaining_s": 8.5,
  "expires_at_ms": 123456789
}
```

#### `GET /kv/resume/{trajectory_id}`

查询参数：

- `worker_url`：R3 认为的 last worker，可选。

响应：

```json
{
  "found": true,
  "trajectory_id": "...",
  "worker_url": "http://10.0.0.1:23456",
  "lease_remaining_s": 6.2,
  "hit_tokens": 0,
  "resident_blocks": 0,
  "estimated_prefill_tokens": 4096,
  "cache_confidence": 0.65
}
```

首版无真实 engine telemetry 时：

```text
cache_confidence =
  0.85 if lease exists and worker_url matches and not expired
  0.45 if lease exists but worker differs
  0.10 otherwise

estimated_prefill_tokens =
  request history length if known
  else 0
```

#### `DELETE /kv/lease/{trajectory_id}`

响应：

```json
{"ok": true, "deleted": true}
```

### 4.3 gateway AppContext 接入

需要在 `AppContext` 中挂载：

```rust
pub kv_lease_store: Arc<KvLeaseStore>
```

并在 `build_app()` handler 里通过 `state.context.kv_lease_store` 访问。

### 4.4 与 worker routing 结合

当前 `cache_aware.rs` 已支持 `X-ROLL-Preferred-Worker-Url`。还需扩展：

- 如果请求 header 有 `X-ROLL-Trajectory-Id`，先查 lease store。
- 若 lease 未过期且 worker healthy，可用 lease worker 作为 preferred worker。
- 若 header preferred worker 和 lease worker 冲突：
  - HOT / high lease_score：优先 lease worker。
  - worker unhealthy：回退 normal policy。

建议新增 header 解析：

- `X-ROLL-Trajectory-Id`
- `X-ROLL-Resume-Lease-Ttl-S`
- `X-ROLL-Resume-Lease-Score`
- `X-ROLL-Belief-Level`
- `X-ROLL-Request-Type`

文件：

- `sgl-model-gateway/src/routers/header_utils.rs`
- `sgl-model-gateway/src/policies/cache_aware.rs`
- `sgl-model-gateway/src/routers/http/router.rs`

### 4.5 gateway response headers

在 `route_typed_request_once()` 收到 worker response 后，附加 response headers：

- `x-smg-selected-worker-url`
- `x-roll-lease-remaining-s`
- `x-roll-cache-confidence`
- `x-roll-estimated-prefill-tokens`
- `x-roll-matched-prefix-tokens`
- `x-roll-memory-pressure`

当前 R3 已读取 `x-smg-selected-worker-url`；后续应扩展 `SglangOrderingRouter._router_generate()` 读取这些 headers 并写入 `out`。

---

## 5. Python SGLang worker 侧剩余开发

目标仓库：

```text
/mnt/xxl/sglang_r3/python/sglang
```

### 5.1 首阶段不要求真实 pin

`docs/r3_sgl_model_gateway.md` 已建议：R3 的 `/kv/resume`、`POST /kv/lease` 先在 `sgl-model-gateway` 演进，不要在 Python SGLang 引擎里做大分叉。

因此 P0/P1 阶段 Python worker 只需要提供 telemetry，不需要真实 KV pin。

### 5.2 返回 matched prefix telemetry

当前可用信息：

- `schedule_policy.py` 会调用 `tree_cache.match_prefix()`。
- `Req.prefix_indices` 表示命中的 prefix KV indices。
- `TokenizerManager.convert_logprob_style()` / `_handle_batch_output()` 负责把 scheduler output 转为 HTTP `meta_info`。

建议新增字段：

```python
meta_info["matched_prefix_tokens"] = len(req.prefix_indices)
meta_info["estimated_prefill_tokens"] = max(0, len(req.origin_input_ids) - len(req.prefix_indices))
meta_info["prefill_time_ms"] = ...
meta_info["cache_confidence"] = matched_prefix_tokens / max(1, prompt_tokens)
```

实现路径候选：

1. 在 `Req` 上新增 telemetry 字段：
   - `matched_prefix_tokens`
   - `estimated_prefill_tokens`
   - `prefill_start_ts`
   - `prefill_end_ts`
2. 在 `schedule_policy.py` prefix matching 后写入。
3. 在 `scheduler_output_processor_mixin.py` 生成 `BatchTokenIDOut` 时携带。
4. 在 `io_struct.py` 的 `BatchTokenIDOut` / `BatchStrOut` dataclass 增加字段。
5. 在 `tokenizer_manager.py` 构造 `meta_info` 时写入。

### 5.3 HTTP response 传递 telemetry

当前 `/generate` 返回 JSON chunk，字段包含：

```json
{
  "output_ids": [...],
  "meta_info": {...}
}
```

建议把新增 telemetry 放在 `meta_info`，gateway 读取 worker response body 后再透传给 R3。

不建议第一阶段直接依赖 response headers，因为 SGLang worker 的 FastAPI `/generate` 当前主要返回 body。

### 5.4 真实 KV lease enforcement（P3）

真实 pin / eviction 要改：

- `python/sglang/srt/mem_cache/radix_cache.py`
- `python/sglang/srt/mem_cache/base_prefix_cache.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_policy.py`

建议方向：

1. `TreeNode` 增加：

```python
lease_score: float = 0.0
lease_expires_at: Optional[float] = None
trajectory_ids: set[str]
```

2. `RadixCache.evict()` 的 heap ordering 改为：

```text
eviction_score = last_access_time_adjusted - lambda * lease_score
```

或更直接：

```text
未过期 lease 节点不进入 evict candidate，除非 memory emergency
```

3. 新增 worker 内部 API：

```text
POST /kv/lease
GET /kv/resume/{trajectory_id}
DELETE /kv/lease/{trajectory_id}
```

但 P3 前不建议直接做，先完成 gateway 控制面与 telemetry。

---

## 6. R3 与 Gateway API 对接细节

### 6.1 Header 约定

R3 -> gateway `/generate`：

| Header | 含义 |
|---|---|
| `X-ROLL-Request-Type` | `normal` / `resume` |
| `X-ROLL-Trajectory-Id` | lease / lookup 主键 |
| `X-ROLL-Preferred-Worker-Url` | HOT resume 的 worker hint |
| `X-ROLL-Pause-Age-S` | tool-return 后等待 |
| `X-ROLL-History-Len-Tokens` | prefill cost proxy |
| `X-ROLL-Resume-Lease-Ttl-S` | lease TTL |
| `X-ROLL-Resume-Lease-Score` | lease priority |
| `X-ROLL-Belief-Level` | `hot` / `warm` / `cold` |
| `X-ROLL-Lease-Phase` | `tool_suspend` / `resume` |

gateway -> R3 `/generate` response：

| Header / body field | 含义 |
|---|---|
| `x-smg-selected-worker-url` | 实际 worker |
| `x-roll-matched-prefix-tokens` | worker 实测 prefix 命中 |
| `x-roll-estimated-prefill-tokens` | 预计重算 token 数 |
| `x-roll-cache-confidence` | gateway / worker 对 hit 的置信度 |
| `x-roll-lease-remaining-s` | lease 剩余时间 |
| `x-roll-memory-pressure` | worker/gateway 内存压力 proxy |

### 6.2 R3 字段映射

`SglangOrderingRouter._router_generate()` 需要把 response headers 写入 `out`：

```python
out["matched_prefix_tokens"] = ...
out["actual_hit"] = matched_prefix_tokens > 0
out["estimated_prefill_tokens"] = ...
out["prefill_time_ms"] = ...
out["cache_confidence"] = ...
out["ttl_remaining_s"] = ...
out["memory_pressure"] = ...
```

`RouterClient._postprocess_generate()` 需要把这些字段写入 `DataProto.meta_info`。

`TrajEnvManager.make_decision()` 需要把这些字段落到 per-turn metrics。

---

## 7. 实施顺序

### Phase A：ROLL 侧补齐控制面调用

目标：不改 SGLang，也能在 gateway 未实现时 no-op。

任务：

1. `RouterManager.set_tool_suspend_lease()`
2. `Router.set_tool_suspend_lease()` no-op
3. `EnvAffinityRouter.set_tool_suspend_lease()`
4. `SglangOrderingRouter.set_tool_suspend_lease()`
5. `TrajEnvManager._maybe_set_pending_tool_suspend_lease()` 调用 RouterManager
6. `RouterManager.delete_kv_lease()`
7. terminal / reset 时 delete lease
8. metrics：`lease_push_success`、`lease_delete_success`

验收：

- 无 gateway 时训练不报错。
- 有 mock gateway 时 tool call 后、env.step 前能收到 `POST /kv/lease`。
- terminal/reset 后能收到 `DELETE /kv/lease/{tid}`。

### Phase B：Gateway lease store

目标：实现 R3 `kv_lease_client.py` 已约定的 API。

任务：

1. 新增 `kv_lease.rs`
2. `AppContext` 挂载 `KvLeaseStore`
3. `server.rs` 新增 `/kv/resume`、`/kv/lease` 路由
4. `/workers` 输出中保持 `is_healthy`、`load` 字段稳定
5. `cache_aware.rs` 结合 lease worker 做 preferred routing
6. response header 返回 selected worker / lease remaining / confidence

验收：

- `POST /kv/lease` 后立刻 `GET /kv/resume/{tid}` 返回 found。
- TTL 到期后 lookup 返回 found=false 或 confidence 低。
- R3 `enable_lookup_resume=true` 时能看到 `lookup_resume_found`。

### Phase C / Phase D（详细设计见专文）

**完整设计**：[real_hit_kv_pin_design.md](./real_hit_kv_pin_design.md)

| 阶段 | 摘要 |
|------|------|
| **Phase C** | 真实 prefix hit：`cached_tokens` → `p_hit_measured`，修正 `context_class`，融合 system-cost 打分 |
| **Phase D** | 真实 KV pin：worker `RadixCache` lease + eviction，`POST/GET/DELETE` 驱动引擎，gateway 可选索引 |

---

## 8. 测试计划

### 8.1 R3 单测

- `parse_ratio()` 支持 string/list/dict。
- system-cost score 不因 loop/stall 降权。
- `set_tool_suspend_lease()` 在无 gateway 时 no-op。
- `delete_kv_lease()` 在 terminal/reset 时 best-effort。
- `_effective_request_priority()` 不双重 aging。

### 8.2 Gateway Rust 测试

- `KvLeaseStore` set/lookup/delete/expire。
- `/kv/lease` API。
- `/kv/resume/{tid}` API。
- preferred worker unhealthy 时 fallback。
- response header 包含 selected worker 和 lease telemetry。

### 8.3 SGLang worker 测试

- `matched_prefix_tokens` 随重复 prompt 增加。
- `estimated_prefill_tokens = prompt_tokens - matched_prefix_tokens`。
- streaming / non-streaming 都能返回最终 telemetry。

### 8.4 集成 smoke

配置（**形态 A**，`EnvAffinityRouter` 直接选 worker；gateway 仅 L2/L3 控制面）：

```yaml
router_args:
  router_name: EnvAffinityRouter
  router_config:
    enable_resume_priority: true
    enable_resume_aware_routing: true
    enable_request_priority_queue: true
    enable_system_cost_resume_scheduling: true
    gateway_status_url: "http://127.0.0.1:30000"   # 可选：lookup / lease
    enable_lookup_resume: true
    enable_value_driven_lease: true
    enable_gateway_kv_lease_push: true
    resume_normal_quota: "3:1"
```

验收指标：

- `scheduler/router/resume_affinity_hit_rate`
- `scheduler/router/resume_queue_wait_p95_s`
- `env/*/resume_latency_e2e_p95_s`
- `matched_prefix_tokens`
- `cache_confidence`
- `lease_push_success`
- `lease_delete_success`
- `normal_queue_wait_p95_s`

### 8.4.1 实验约束：必须使用真实 toolcall 延迟（禁止 sleep 构造）

本仓库的 resume-aware / Phase D 验证目标是覆盖 **真实 toolcall 阻塞** 下的 tool suspend 行为（KV lease 的意义就在于真实外部等待期间避免 radix cache 被淘汰）。

- **必须**：选择会访问真实外部依赖的 tool（网络/检索/DB/远端执行/文件 IO 等），并确保产生真实 wall-clock 延迟。
- **禁止**：用 `sleep` 人工构造外部等待来作为集成验收依据；`sleep` 仅可用于最小接口连通性或 TTL 自测。

---

## 9. 风险与取舍

### 9.1 Tool suspend 同步调用可能拖慢 env.step

建议：

- 默认短 timeout。
- 失败静默。
- 后续可改 fire-and-forget 或 background queue。

### 9.2 Gateway lease 不是真实 KV pin

P0/P1 阶段只是控制面 belief。文档和指标中要明确：

```text
context_class_gpu_hit / lease_found != physical KV guaranteed hit
```

真实物理 KV pin 属于 Phase D。

### 9.3 Python SGLang 大改风险高

先只加 telemetry，避免直接改 radix eviction。等 gateway 控制面和 A/B 指标稳定后再做真实 lease enforcement。

### 9.4 Header 与 body telemetry 重复

建议：

- worker -> gateway：优先 body `meta_info`。
- gateway -> R3：优先 response headers，必要时保留 body fields。
- R3 内部：统一落到 `DataProto.meta_info`。

---

## 10. 文件级 TODO 总表

### R3

| 文件 | TODO |
|---|---|
| `roll/pipeline/agentic/env_manager/traj_env_manager.py` | tool suspend 前调用 RouterManager 设置 lease；terminal/reset 删除 lease |
| `roll/distributed/scheduler/router.py` | 新增 set/delete lease Ray 方法；lookup telemetry 接入 system-cost；统一 aging |
| `roll/distributed/scheduler/kv_lease_client.py` | 增加 worker_url / phase / history_len_tokens / kv_bytes 字段支持 |
| `roll/distributed/scheduler/trajectory_value.py` | lookup telemetry 优先进入 `C_hist_prefill`、`ttl_remaining_s`、`p_hit` |
| `roll/pipeline/agentic/context_lifecycle.py` | 与 gateway lease 状态对齐，避免只做本地 proxy |
| `roll/distributed/strategy/sglang_strategy.py` | in-process / HTTP worker telemetry 透传 |

### sglang_r3 gateway

| 文件 | TODO |
|---|---|
| `sgl-model-gateway/src/kv_lease.rs` | 新增 lease store |
| `sgl-model-gateway/src/app_context.rs` | 挂载 `Arc<KvLeaseStore>` |
| `sgl-model-gateway/src/server.rs` | 新增 `/kv/resume`、`/kv/lease` 路由 |
| `sgl-model-gateway/src/routers/header_utils.rs` | 解析 ROLL lease headers |
| `sgl-model-gateway/src/routers/http/router.rs` | 选择 worker 后写 selected/telemetry response headers |
| `sgl-model-gateway/src/policies/cache_aware.rs` | lease-aware preferred worker / fallback |
| `sgl-model-gateway/src/core/worker.rs` | 稳定暴露 load / health / model / worker_url |

### sglang_r3 Python worker

| 文件 | TODO |
|---|---|
| `python/sglang/srt/managers/schedule_policy.py` | 记录 prefix match telemetry |
| `python/sglang/srt/managers/schedule_batch.py` | `Req` 增加 telemetry 字段 |
| `python/sglang/srt/managers/io_struct.py` | `BatchTokenIDOut` / `BatchStrOut` 增加 telemetry 字段 |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | batch output 携带 telemetry |
| `python/sglang/srt/managers/tokenizer_manager.py` | `meta_info` 输出 telemetry |
| `python/sglang/srt/mem_cache/radix_cache.py` | Phase D：lease-aware eviction |

---

## 11. 复查补充：易踩坑与遗漏点

### 11.1 R3 侧补充

除第 10 节表格中的主路径文件外，实现时还需要检查：

- `roll/distributed/scheduler/resume_state.py`：`p_hit_bias`、`t_tool_ema`、pending lease 与 gateway outcome 应保持同一份状态语义，避免 Env/Router 各自估计。
- `roll/pipeline/agentic/trajectory_signals.py`：loop / stall / invalid 可继续作为 lifecycle 和观测字段产出，但不得重新进入 system-cost priority。
- `roll/distributed/scheduler/soft_quota_utils.py`：实现已支持多种 `resume_normal_quota` 格式；实验配置仍建议统一写 `"3:1"`。
- `roll/pipeline/agentic/llm_proxy/policy_proxy.py`：新增 `set_tool_suspend_lease()` / `delete_kv_lease()` 时，要补齐本地 client 与 Ray remote client 的转发方法。

关键风险：

- **配置耦合**：`enable_system_cost_resume_scheduling` 须与 `enable_resume_aware_routing: true` 同开，否则 placement 不会走 `_select_worker_system_cost`。
- **形态 A gateway**：`EnvAffinityRouter` 需正确设置 `gateway_url` / `gateway_status_url`，否则 L3 `POST /kv/lease` 为 no-op。
- **L3 双开关**：`POST /kv/lease` 目前依赖 `enable_gateway_kv_lease_push` 和 `enable_value_driven_lease`，只打开 `enable_system_cost_resume_scheduling` 不够。
- **lookup 字段名统一**：避免 `lookup_lease_remaining_s` 写入、`ttl_remaining_s` 读取造成 delay regret 未生效；建议在 `_enrich_resume_route_meta()` 单点归一。
- **lifecycle hard rule 字段**：`model_version_mismatch`、`prefix_hash_mismatch` 已被 score 支持，但 runtime route_meta 仍需稳定写入。
- **覆盖范围**：如果最终实验包含 VL / 多模态 env，需要把 tool-suspend lease 逻辑同步到相应 env manager。

### 11.2 Gateway / SGLang 侧补充

除第 10 节表格中的主路径文件外，实现时还需要检查：

- `sgl-model-gateway/src/routers/http/router.rs` 的 `send_typed_request()` / request header 过滤逻辑：P2/P3 若 Python worker 要读取 lease metadata，必须允许 `X-ROLL-*` 转发到 worker。
- `sgl-model-gateway/src/core/worker_manager.rs`：gateway 侧 `/workers.load` 多数情况下是本地 inflight / guard 语义，不等价于 engine 内部 queue 或 KV memory pressure。
- `sgl-model-gateway/src/observability/metrics.rs`、`sgl-model-gateway/src/observability/inflight_tracker.rs`：需要增加 lease/resume/lookup 相关指标，并区分 gateway inflight 与 engine telemetry。
- `sgl-model-gateway/src/routers/http/pd_router.rs`、`sgl-model-gateway/src/routers/grpc/common/stages/worker_selection.rs`：如果后续走 PD / gRPC，preferred-worker 与 telemetry 路径要单独补齐。
- `python/sglang/srt/managers/detokenizer_manager.py`：如果 telemetry 经 detokenizer 汇总，需要确保字段不丢。
- `python/sglang/srt/metrics/collector.py`：已有 cached-token 指标可作为 P1/P2 的低风险 telemetry 起点。

关键风险：

- **已有 `cached_tokens` 可复用**：如果 worker response 或 metrics 已有 `cached_tokens`，P1 可以先映射为 `matched_prefix_tokens`，不必第一步就改完整 scheduler dataclass 链路。
- **流式响应**：gateway 聚合 telemetry 时要分别覆盖 streaming 与 non-streaming，否则长生成场景会漏指标。
- **DP-aware URL 格式**：`x-smg-selected-worker-url` 必须与 R3 `worker_urls` 使用同一规范，尤其注意 `url@dp_rank` 或类似后缀。
- **旧 `sgl-router/` 混用**：实验以 `sgl-model-gateway` + `EnvAffinityRouter` 为准；旧 `sgl-router/` 没有同等 R3 改动，不要混用。
- **本地 `ContextLifecycleManager` 只是 proxy**：在 engine 未回传真实 hit 前，`gpu_hit` 不能被解释为物理 KV 一定命中。

