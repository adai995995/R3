# Trajectory Value Scheduling（轨迹价值调度）

本文档定义 ROLL 侧 **轨迹调度价值** `V_traj` 的计算、与 **Recoverability Belief** 的关系，以及如何与推理引擎（SGLang / sgl-model-gateway）的 **KV TTL / Lease** 协同。与分层路线见 [idea_hierarchical.md](./idea_hierarchical.md)；与 CacheTTL 对照见 [§8](#8-与-cachettl-的对照与扩展)。

> 设计更新：本文保留当前已实现的 `V_traj = V_sys + V_learn_neg` 方案。若要将调度器严格限定为系统成本优化器，并移除 loop / stall / low-quality trajectory 等语义质量信号对 priority 的影响，见替代设计 [system_cost_resume_scheduling_design.md](./system_cost_resume_scheduling_design.md)。

---

## 1. 目标与边界

### 1.1 目标

在 Agentic RL rollout 中，将 tool-return 后的 continuation 显式建模为 **Resume Request**，并用可在线计算的 **轨迹调度价值** 统一指导：

| 决策层 | 当前/计划用途 |
|---|---|
| **Ordering** | resume/normal 谁先进入推理队列 |
| **Placement** | resume 路由到哪台 worker（`EnvAffinityRouter` + system-cost `dispatch_score`） |
| **KV Lease / TTL**（计划） | 将 lease/TTL 经 gateway 控制面或引擎 API 下发，指导 tool-wait 期间 KV 保留 |

### 1.2 非目标（硬边界）

- **不**用 `V_traj` 决定「是否丢弃轨迹 / 是否不进 GRPO batch」——避免改变 group-relative 采样集合。
- **不**用 per-turn **reward 正反馈** 做主排序（稀疏、且常与「压长尾」系统目标错位）。
- **第一层**不要求推理引擎暴露精确 KV page 状态；`p_hit` 是调度侧 **belief**，不是实测命中率。

---

## 2. 价值分解

```text
V_traj = V_sys + V_learn_neg
```

| 项 | 含义 | 角色 |
|---|---|---|
| `V_sys` | 期望恢复净收益 + 收尾紧迫度 + 等待公平 − 负载/重算风险 | **主项**（系统可观测） |
| `V_learn_neg` | invalid / 死循环 / 停滞 / 已终止等 **负反馈** | **只做减法**，表达「不值得再占恢复资源」 |

**收束原则**：轨迹价值 = **「现在优先恢复它」的期望净收益**；学习侧不参与「谁更值得学」，只参与「谁应降权」。

---

## 3. 观测字段（Runtime → Router → 引擎）

### 3.1 Runtime 写入（`TrajEnvManager` → `meta_info` → `_roll_route_meta`）

| 字段 | 含义 |
|---|---|
| `trajectory_id` | 稳定轨迹 ID（lease / 指标闭环） |
| `request_type` | `normal` / `resume` |
| `pause_age_s` | tool-return 后等待时长 |
| `history_len_tokens` | 当前上下文长度 |
| `last_backend_id` | 上次实际 worker |
| `remaining_steps` / `max_steps` / `remaining_steps_ratio` | 收尾紧迫度 |
| `trajectory_invalid` | 非法 action / parse fail（0/1） |
| `trajectory_loop` | 近 k 步重复 response（0/1） |
| `trajectory_stall` | 连续 n 步 reward=0（0/1） |
| `trajectory_terminated` | 已 terminated/truncated（0/1） |

负反馈检测见 `roll/pipeline/agentic/trajectory_signals.py`。

### 3.2 计划透传引擎（tool suspend 时）

| Header / API 字段 | 来源 | 用途 |
|---|---|---|
| `X-ROLL-Resume-Lease-Ttl-S` | `compute_lease_ttl(V_traj, …)` | KV 最长保留秒数 |
| `X-ROLL-Resume-Lease-Score` | `clip(V_traj)` | eviction 优先级 modifier |
| `X-ROLL-Belief-Level` | `hot` / `warm` / `cold` | 是否 pin、是否强亲和 |
| `X-ROLL-Trajectory-Id` | `trajectory_id` | lease 表主键 |

详见 [§7](#7-价值驱动的-kv-ttl--lease)。

---

## 4. Recoverability Belief 与 `p_hit`

### 4.1 `p_hit` 是什么

**`p_hit` = 调度器对「本次 resume 能以低成本复用 last_worker 上下文（≈ KV / prefix 命中）」的主观概率。**

- **不是** SGLang/vLLM 返回的真实 cache hit rate。
- **不是** RL reward 或 learning value。
- **是** 在看不见 KV 内部状态时，用 `pause_age`、负载、轨迹质量等拼出的 **belief**，用于把「恢复划算程度」写进 `V_sys`。

### 4.2 档位与默认映射

| Belief | 默认 `p_hit` | 含义（调度器在说什么） |
|---|---|---|
| **HOT** | 0.85 | tool 刚结束、last_worker 可用 → 很可能仍在 GPU |
| **WARM** | 0.45 | 可能还在，也可能已 evict 或应迁移 |
| **COLD** | 0.10 | 久等 / 无效 / 无 last_backend → 别强亲和 |

### 4.3 分类规则（`classify_belief`）

→ **COLD**：`terminated` / `invalid` / `loop` / `pause_age ≥ cold_pause_age_s` / 无 `last_backend_id`  
→ **HOT**：`pause_age ≤ hot_pause_age_s`、有 `last_backend`、last worker 不过载  
→ **WARM**：其余（含 last worker 过载）

### 4.4 与路由的关系（两层）

1. **离散策略**：HOT 直送 last_worker；COLD least-load；WARM 比较 `route_score`。
2. **连续 `p_hit`**：进入 `V_sys` 的收益项 `+ w_p·p_hit·ñ_h` 与代价项 `- w_c·(1-p_hit)·ñ_h`。

### 4.5 反馈闭环（计划增强）

| 观测（resume 后） | 对 belief / `p_hit` 的影响 |
|---|---|
| `selected_backend_affinity_hit` | 命中 → 下次略增 HOT 倾向 |
| `context_class_full_prefill` / `resume_prefill_tokens` 大 | sticky 但仍重算 → 降 HOT |
| `external_wait_s` 实测 | 更新 per-tool-type `t_tool`，用于 TTL（§7） |

当前实现：aggregate 指标 + 控制面 `ContextLifecycleManager`；**per-trajectory belief 持久更新** 为 P1。

---

## 5. 公式

### 5.1 归一化

```text
ñ_h = log1p(history_len_tokens) / log1p(H_max)     # 默认 H_max=32768
ñ_a = log1p(pause_age_s) / log1p(A_max)            # 默认 A_max=60
ñ_r = remaining_steps / max_steps                  # 越小越「急收尾」
ñ_q = worker_load                                  # placement 时用 in-flight
```

### 5.2 系统价值

```text
V_sys = w_p · p_hit · ñ_h              # 期望恢复收益
      + w_f · (1 - ñ_r)                  # 快结束轨迹优先（压 group barrier，对齐 Heddle 方向）
      + w_a · ñ_a                        # 等待公平 / 长尾
      - w_c · (1 - p_hit) · ñ_h          # 预期全量重算惩罚
      - w_q · ñ_q                        # 目标 worker 负载
```

默认权重：`w_p=1.0, w_f=0.5, w_a=0.3, w_c=0.8, w_q=0.5`（`trajectory_value_weights`）。

### 5.3 学习侧负反馈（仅减法）

```text
V_learn_neg = - c_inv · I_invalid
             - c_loop · I_loop
             - c_stall · I_stall
             - c_term · I_terminated
```

默认：`c_inv=1.5, c_loop=1.0, c_stall=0.5, c_term=10.0`。  
**invalid / loop 同时会把 belief 打成 COLD**，避免为高惩罚轨迹强留 KV。

### 5.4 调度用途

```text
# Ordering（双队列内 resume 排序）
resume_priority = V_traj + η · queue_wait_s

# Placement（EnvAffinityRouter，按 worker 的 p_w 与 load）
route_score(w) = compute_trajectory_value(..., p_hit=p_w(w), worker_load=load(w))

# Lease（计划，§7）
lease_score = clip(V_traj, 0, 1)
ttl_s       = f(t_tool, lease_score, p_hit, ñ_h, ñ_r, penalties)
```

---

## 6. 分层实现路线（对齐 idea_hierarchical）

```text
┌─────────────────────────────────────────────────────────────────┐
│ 第三层：引擎内 Lease Enforcement（eviction = LRU - λ·lease_score）│
│  需 SGLang/gateway 协作 · 参考 CacheTTL pin + TTL               │
└───────────────────────────────▲─────────────────────────────────┘
                                │ X-ROLL-Resume-Lease-* headers / RPC
┌───────────────────────────────┴─────────────────────────────────┐
│ 第二层：Resume Observability（lookup_resume / probe_and_dispatch） │
│  用实测 hit_tokens 校准 p_hit、ttl_s                              │
└───────────────────────────────▲─────────────────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│ 第一层：ROLL 内 V_traj + Belief + Ordering/Placement  【已实现】   │
│  ContextLifecycleManager = 控制面 soft lease 骨架（未驱动物理 KV） │
└─────────────────────────────────────────────────────────────────┘
```

| 层 | 状态 | 说明 |
|---|---|---|
| L1 轨迹价值 + belief 路由 | **已实现** | `enable_trajectory_value_scheduling` |
| L1 控制面 lifecycle | **部分** | 固定 `context_ttl_s`，未接 `V_traj` |
| L2 引擎 query/probe | 未实现 | `lookup_resume(trajectory_id)` |
| L3 引擎 lease enforcement | 未实现 | `set_resume_lease` + eviction modifier |

---

## 7. 价值驱动的 KV TTL / Lease

### 7.1 动机（与 CacheTTL 同构、输入更直接）

[CacheTTL (2511.02230)](https://arxiv.org/abs/2511.02230) 在 **tool wait** 期间用 **TTL** 覆盖 end-of-turn eviction，TTL 由 **tool 耗时分布 + reload 成本 + per-turn queueing** 权衡。

我们在此基础上增加 **ROLL 已算好的 `V_traj` / `p_hit` / 负反馈 / remaining_steps**：

- CacheTTL：主要靠 **工具类型历史** 估「多久回来」。
- 我们：**同一等待窗口内，高价值、长历史、HOT、快收尾** → 更长 TTL、更高 lease_score；**invalid/loop/COLD** → 短 TTL 或不 pin。

### 7.2 Lease 抽象

```text
set_resume_lease(
  trajectory_id,
  worker_url,           # 或 request_id / rid
  ttl_s,                # 秒，到期自动允许 evict
  lease_score,          # ∈ [0,1]，来自 V_traj
  demotion_policy,      # 可选：invalid → immediate_demote
)
```

引擎侧（第三层）不交出 LRU 完全控制权，仅：

```text
eviction_score = LRU_score - λ · lease_score
```

### 7.3 建议 TTL 公式（ROLL 侧计算）

```text
# 基线：预期 tool 等待（在线 EWMA 或 per tool_type 表，对齐 CacheTTL 的 S[f]）
t_tool = E[external_wait_s | tool_type]   # runtime 已在 step 记录 external_wait_s

lease_score = clip(V_traj, 0, 1)

ttl_s = t_tool
      + α · lease_score · ñ_h
      + β · p_hit · t_tool
      + γ · (1 - ñ_r) · t_tool
      - δ · (I_invalid + I_loop) · t_tool

# 硬规则
if belief == COLD or I_terminated:
    ttl_s = min(ttl_s, t_tool_min)   # 或 ttl_s = 0，不 pin
if belief == HOT and lease_score > θ:
    ttl_s = max(ttl_s, t_tool)       # 至少覆盖典型 tool 等待
```

**语义**：

| 因子 | 对 TTL 的影响 |
|---|---|
| `t_tool` | 对齐 CacheTTL：覆盖「工具大概多久回来」 |
| `lease_score` · `ñ_h` | 高价值 + 长上下文 → 多留（重算贵） |
| `p_hit` | 相信 KV 还在 → 值得占 GPU |
| `(1-ñ_r)` | 快收尾 → 多留，利于 program 连续性 |
| invalid/loop | 缩短，防止死循环占显存（CacheTTL 长尾 tool 问题） |

### 7.4 透传协议（计划）

**时机**：LLM decode 结束、**进入 tool suspend** 时（等价 CacheTTL 的 tool-call handler on_leave）。

**HTTP（gateway 控制面，可选）**：

```http
X-ROLL-Trajectory-Id: <trajectory_id>
X-ROLL-Resume-Lease-Ttl-S: <ttl_s>
X-ROLL-Resume-Lease-Score: <lease_score>
X-ROLL-Belief-Level: hot|warm|cold
X-ROLL-Request-Type: resume   # 下一轮
```

**RPC（第二层以后）**：

```text
POST /kv/lease  { rid, worker_url, ttl_s, lease_score }
DELETE /kv/lease/{rid}
GET  /kv/lease/{rid}  -> { remaining_s, lease_score }
```

### 7.5 与 ROLL 控制面的关系

当前 `ContextLifecycleManager`（可选，与 gateway 联调时）：

- `pin_context` / `retain_context` / `offload_context` / `context_ttl_s`
- **不保证** 真实 SGLang KV 被保留（见 `resume_aware_context_fix_changes.md`）

**演进**：`ttl_s` / `lease_score` 由 `compute_lease_from_trajectory_value()` 生成（与 `V_traj` 同源），替换固定 `context_ttl_s=300`；引擎接入后，控制面状态与物理 lease **对账**（`context_class_*` 指标）。

---

## 8. 与 CacheTTL 的对照与扩展

| 维度 | CacheTTL | 本设计（Value-driven TTL） |
|---|---|---|
| 解决的问题 | end-of-turn eviction 在 tool wait 中浪费 KV | 同左 |
| TTL 主输入 | tool 历史分布、reload、queueing 模型 | **`V_traj`、`p_hit`、`t_tool`、负反馈、`ñ_r`** |
| 排序 | program-level FCFS | resume 双队列 + **`V_traj`**（已实现） |
| 鲁棒性 | TTL 到期强制 evict | 同左 + **invalid/loop 缩短 TTL** |
| 实现位置 | vLLM 内 pin | ROLL 算 + **SGLang/gateway enforce**（计划） |
| 优势 | 成熟 cost/benefit 闭式 | **调度与 KV 共用同一价值标量**，无需在引擎内重复猜「重要性」 |

**论文叙事建议**：CacheTTL 优化 **agent serving 的 KV retention**；我们优化 **RL rollout 中 resume 的全栈调度**，并把 **轨迹价值** 作为 TTL/ordering/placement 的统一参数。

---

## 9. Belief 反馈与 Lease 统一（实现说明）

### 9.1 Belief 反馈

**模块**：`roll/distributed/scheduler/resume_state.py`

- 按 `trajectory_id` 维护 `p_hit_bias`（EWMA，限幅 ±0.2）。
- **写入**：每次 tool-return 后 `update_tool_wait(trajectory_id, external_wait_s)`（`TrajEnvManager.step`）。
- **更新**：每次 resume 推理返回后 `observe_resume_outcome`（命中 gpu_hit ↑，full_prefill / 高 prefill_ratio ↓）。
- **读取**：下一次 `compute_resume_priority(..., p_hit_bias=...)` → `apply_p_hit_bias`。

配置：`enable_belief_feedback: true`

### 9.2 Lease 与 V_traj 统一

- `compute_lease_ttl(route_meta, p_hit, v_traj, t_tool_s, belief_level)` 与 `V_traj` 同源。
- `t_tool_s` 来自 `TrajectorySchedulingState` 的 `external_wait_s` EWMA。
- `ContextLifecycleManager.retain/pin` 使用动态 `ttl_s`，不再固定 `context_ttl_s`（当 `enable_value_driven_lease=true`）。
- 可选经 gateway 下发 header：`X-ROLL-Resume-Lease-Ttl-S`、`X-ROLL-Resume-Lease-Score`、`X-ROLL-Belief-Level`（引擎消费为 P1/P2）。

配置：`enable_value_driven_lease: true`

---

## 10. 配置与实现状态

### 10.1 已实现（`router_config`）

| 键 | 默认 | 说明 |
|---|---|---|
| `enable_trajectory_value_scheduling` | `false` | 开启后 resume priority / route 用 `V_traj` |
| `trajectory_value_weights` | 见代码 | `w_p, w_f, w_a, w_c, w_q, h_max, a_max` |
| `learning_penalty_weights` | 见代码 | `c_inv, c_loop, c_stall, c_term` |
| `belief_config` | 见代码 | `p_hot/warm/cold`, `hot/cold_pause_age_s` |
| `enable_belief_feedback` | `false` | `p_hit` EWMA 反馈 |
| `enable_value_driven_lease` | `false` | 动态 TTL + lease header |
| `default_t_tool_s` | `5.0` | tool 等待先验（秒） |
| `tool_wait_ema_alpha` | `0.2` | `external_wait_s` EWMA |

示例 yaml：

- `examples/toolcall_benchmark/toolcall_benchmark_resume_aware.yaml`
- `examples/qwen3_agentic_gem/gem_math_hotpotqa_search_ds_sglang_router_trajectory_value.yaml`

### 10.2 计划配置（引擎侧）

| 键 | 说明 |
|---|---|
| `enable_value_driven_lease` | 是否计算并下发 ttl/lease_score |
| `lease_ttl_weights` | `α, β, γ, δ` |
| `lease_t_tool_ewma_alpha` | 在线更新 `t_tool` |

### 10.3 代码入口

| 模块 | 路径 |
|---|---|
| 价值与 belief | `roll/distributed/scheduler/trajectory_value.py` |
| Runtime 负反馈 | `roll/pipeline/agentic/trajectory_signals.py` |
| Env manager | `roll/pipeline/agentic/env_manager/traj_env_manager.py` |
| Router 集成 | `roll/distributed/scheduler/router.py` |
| 控制面 lifecycle | `roll/pipeline/agentic/context_lifecycle.py` |
| Belief / tool-wait 状态 | `roll/distributed/scheduler/resume_state.py` |

### 10.4 指标

**ROLL 调度**：

- `scheduler/router/trajectory_value_mean`
- `scheduler/router/belief_state/{hot,warm,cold}_count`
- `scheduler/router/penalty/trajectory_*_count`

**Resume 效果**（见 `key_indicator.md`）：

- `resume_latency_e2e_s`、`resume_prefill_tokens`
- `scheduler/router/resume_affinity_hit_rate`（ROLL 选 worker 闭环）

**Lease（计划）**：

- `scheduler/router/lease_ttl_mean_s`
- `scheduler/router/lease_score_mean`
- `context_class_gpu_hit` / `cpu_reload` / `full_prefill` 与 `lease_score` 分桶相关性

---

## 11. 实验与消融建议

| 实验组 | 内容 |
|---|---|
| A0 | baseline，无 resume-aware |
| A1 | L0/L2 only（无 `V_traj`） |
| A2 | **`enable_trajectory_value_scheduling`**（L1 完整） |
| A3 | A2 + **value-driven TTL header**（需 gateway 消费） |
| A4 | A3 + **lookup_resume 反馈**（L2） |

关注：长尾（多 turn、长 history）bucket 的 P95 `resume_latency`、吞吐、`full_prefill` 比例下降。

---

## 12. 回滚

- 调度：`enable_trajectory_value_scheduling: false` → 回退 `resume_priority.py`。
- Lease：不发 `X-ROLL-Resume-Lease-*` → 引擎行为与未改前一致。
- 控制面：关闭 `enable_value_driven_lease` 后仍可用固定 `context_ttl_s`。

---

## 13. 相关文档

- [idea_hierarchical.md](./idea_hierarchical.md) — 三层分工总览
- [idea_value.md](./idea_value.md) — 价值 × 恢复代价原始公式
- [tool_return_resume_aware_runtime_design.md](./tool_return_resume_aware_runtime_design.md) — L0/L1/L2 与 G1 resume 边界
- [resume_aware_context_fix_changes.md](./resume_aware_context_fix_changes.md) — gateway header 与 ContextLifecycle
- [key_indicator.md](./key_indicator.md) — 验收指标
