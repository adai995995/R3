# Trajectory Value + KV Lease — 实现状态（ROLL 侧）

> 目标：调度侧 `V_traj` / TTL / lease 与推理引擎联调。本文档跟踪 **ROLL 仓库内** 实现；**引擎消费**需在 sgl-model-gateway / SGLang 另行开发。

## 最终目标（三层）

| 层 | 能力 | ROLL 状态 | 引擎状态 |
|---|---|---|---|
| **L1** | `V_traj` ordering/placement、belief、`p_hit_bias`、控制面 lifecycle | **已实现** | 不必须 |
| **L2** | `lookup_resume` 调度前探针、真实 hit 反馈 | **客户端已实现**（404 则静默降级） | **待实现** `GET /kv/resume/{tid}` |
| **L3** | `set_resume_lease`、eviction modifier | **客户端 + header 已实现** | **待实现** `POST /kv/lease`、eviction |

---

## 已实现清单

### 调度与价值

- `roll/distributed/scheduler/trajectory_value.py` — `V_sys`、`V_learn_neg`、`compute_resume_priority`、`compute_lease_ttl`、`plan_tool_suspend_lease`
- `roll/pipeline/agentic/trajectory_signals.py` — invalid/loop/stall/term
- `roll/distributed/scheduler/router.py` — `EnvAffinityRouter` / `SglangOrderingRouter` 集成
- `roll/distributed/scheduler/resume_state.py` — `p_hit_bias`、`t_tool` EMA、`observe_resume_outcome`、`observe_lookup_resume`

### Tool suspend → pending lease

- `TrajEnvManager.step()` 在 `env.step()` 前调用 `_maybe_set_pending_tool_suspend_lease()`
- Router `_attach_lease_headers()` 在 resume 时 `pop_pending_tool_lease`

### L2/L3 网关客户端（ROLL）

- `roll/distributed/scheduler/kv_lease_client.py`
  - `lookup_resume()` → `GET {gateway}/kv/resume/{trajectory_id}`
  - `set_kv_lease()` → `POST {gateway}/kv/lease`
  - `delete_kv_lease()` → `DELETE {gateway}/kv/lease/{trajectory_id}`

### Form B 下行

- HTTP headers：`X-ROLL-Resume-Lease-Ttl-S`、`Lease-Score`、`Belief-Level`、`Request-Type`、`Preferred-Worker-Url`（HOT）
- `enable_gateway_kv_lease_push` 时额外 `POST /kv/lease`（引擎无接口则 debug 日志、不报错）

### 其它

- `enable_refresh_resume_priority_on_dispatch` — 派发前刷新 resume `base_priority`
- `gateway_inflight_load_weight` — placement 时融合 gateway `inflight`（可选）
- 示例 yaml：`gem_math_hotpotqa_search_ds_sglang_router_trajectory_value.yaml`、`..._form_b_trajectory_value.yaml`
- 单测：`test_trajectory_value.py`、`test_resume_state.py`

---

## 配置开关（`router_config`）

| 键 | 默认 | 说明 |
|---|---|---|
| `enable_trajectory_value_scheduling` | false | L1 主开关 |
| `enable_belief_feedback` | false | `p_hit_bias` EWMA |
| `enable_value_driven_lease` | false | 动态 TTL + lease header |
| `enable_lookup_resume` | false | L2 调度前 GET（需 `gateway_url` 或 `gateway_status_url`） |
| `enable_gateway_kv_lease_push` | false | L3 POST lease（Form B） |
| `enable_refresh_resume_priority_on_dispatch` | false | 队列派发前重算 priority |
| `gateway_kv_lookup_path` | `/kv/resume/{trajectory_id}` | |
| `gateway_kv_lease_path` | `/kv/lease` | |
| `gateway_inflight_load_weight` | 0 | >0 时 placement 使用 poll 的 inflight |

`enable_resume_priority: false` 会关闭上述全部 resume 增强。

---

## 仍须在引擎侧实现（联调清单）

1. **`GET /kv/resume/{trajectory_id}`**  
   返回：`hit_tokens`, `estimated_prefill_tokens`, `cache_confidence`, `lease_remaining_s`, `worker_url`

2. **`POST /kv/lease`**  
   Body：`trajectory_id`, `ttl_s`, `lease_score`, `worker_url`, `belief_level`  
   Tool-wait 期间 pin / 延长 TTL；到期 evict

3. **`eviction_score = LRU - λ * lease_score`**（或等价策略）

4. **Generate 响应**（建议增强，替代纯控制面 proxy）  
   `matched_prefix_tokens`, `resume_prefill_tokens`, `context_class_*`

5. **（可选）`DELETE /kv/lease/{tid}`** — episode 结束释放

引擎未实现时：ROLL 行为与 L1 相同；`lookup_resume` / `set_kv_lease` 失败静默，不影响训练。

---

## 建议测试顺序（实现完成后）

| 阶段 | 实验 | 前置 |
|---|---|---|
| 1 | A0 vs A2 | 仅 ROLL，已有初步结果 |
| 2 | A2 vs A2b | `enable_belief_feedback` |
| 3 | A2 vs A2c | `enable_value_driven_lease`（可无引擎） |
| 4 | A3 | 引擎消费 lease + header |
| 5 | A4 | `enable_lookup_resume` + 引擎 GET |

---

## 相关文档

- [trajectory_value_scheduling.md](./trajectory_value_scheduling.md)
- [idea_hierarchical.md](./idea_hierarchical.md)
- [experiment_plan.md](./experiment_plan.md)（建议按 A0–A4 更新 Treatment 配置）
