# 分层设计：Belief 调度 → 引擎可观测 → Value-driven KV Lease

本文档是 **总览**；公式、字段、配置与实现状态以 **[trajectory_value_scheduling.md](./trajectory_value_scheduling.md)** 为准。

---

## 三层分工

| 层 | 名称 | 做什么 | 依赖后端 | 状态 |
|---|---|---|---|---|
| **L1** | ROLL 内 soft scheduling | Resume 语义、`V_traj`、`p_hit` belief、ordering/placement、suspend pending lease | 否 | **已实现** |
| **L2** | Resume observability | `lookup_resume` / `probe_and_dispatch`，用实测 hit 校准 belief | 轻量 API | ROLL 客户端已实现；**引擎 GET 待联调** |
| **L3** | Lease enforcement | `set_resume_lease` + `eviction_score = LRU - λ·lease_score` | 引擎协作 | ROLL header/POST 已实现；**引擎 enforce 待联调** |

```text
tool-return
    → Resume Request（非 Fresh）
    → classify_belief → HOT / WARM / COLD → p_hit
    → V_traj = V_sys + V_learn_neg
    → ordering / placement / (计划) TTL·lease_score → SGLang
```

---

## 第一层（L1）：Belief-based Resume Scheduling

**问题**：调度器看不见 KV，不能把「刚 tool-return」等同于「高价值」。

**做法**：

1. **Resume Request**：仅在 tool-return 边界标注 `request_type=resume`（G1）。
2. **Belief**：`p_hit` 是「低成本恢复的主观概率」，不是后端实测值（见主文档 §4）。
3. **路由三分支**：
   - **HOT**：直送 `last_worker`（`EnvAffinityRouter` + system-cost placement）。
   - **WARM**：比较 `route_score(last)` vs 其它 worker。
   - **COLD**：放弃强亲和，load-aware。
4. **轨迹价值**：`V_sys`（恢复收益−重算−负载）+ `V_learn_neg`（invalid/loop/stall/term 只减不加）。
5. **Soft lease（控制面）**：`ContextLifecycleManager` 记录 HOT/WARM/EVICTED；**待**与 `V_traj` 算出的 `ttl_s` 对齐。

**反馈（计划加强）**：affinity hit、full_prefill → 更新 per-trajectory belief；当前以 aggregate 指标为主。

---

## 第二层（L2）：后端可观测

**问题**：`restore_cost` 若永远靠猜，`p_hit` 与 TTL 会漂。

**最小 API**（二选一）：

```text
lookup_resume(prefix_hash, trajectory_id)
  -> hit_tokens, resident_blocks, estimated_prefill_tokens, cache_confidence

probe_and_dispatch(request, candidate_worker)
  -> hit 则 resume，miss 则 prefill（可与 tool 后处理并行）
```

**用途**：校准 `p_hit`、验证 belief 分桶、为 §7 TTL 提供 `cache_confidence`。

---

## 第三层（L3）：Value-driven KV Lease（对齐 CacheTTL，参数更直接）

**问题**：end-of-turn eviction 在 agent tool wait 中浪费 KV；固定 TTL 无法区分高/低价值轨迹。

**参考**：[CacheTTL (2511.02230)](https://arxiv.org/abs/2511.02230) — tool 期间 pin KV + TTL，权衡 reload、queueing、tool 耗时分布。

**我们的扩展**：ROLL 已具备 **`V_traj`、`p_hit`、负反馈、`remaining_steps`**，可直接驱动：

```text
lease_score = clip(V_traj, 0, 1)
ttl_s = f(t_tool, lease_score, p_hit, ñ_h, ñ_r, I_invalid, I_loop)
```

透传（计划）：

```text
set_resume_lease(trajectory_id, prefix_hash?, ttl, lease_score, demotion_policy)
```

引擎 eviction：

```text
eviction_score = LRU_score - λ × lease_score
```

**原则**：

- 高 `V_traj` + HOT → 更长 TTL，更高 anti-eviction 权重。
- invalid / loop / COLD → **缩短或取消** lease，避免死循环占显存（CacheTTL 长尾 tool 同理）。
- **不**用 reward 正反馈决定 lease；与 GRPO 解耦。

协议草案见 [trajectory_value_scheduling.md §7](./trajectory_value_scheduling.md#7-价值驱动的-kv-ttl--lease)。

---

## 与「轨迹价值」文档的关系

- 概念与模块拆分：[idea_value.md](./idea_value.md)
- **可落地公式、字段、CacheTTL 对照、实现状态**：[trajectory_value_scheduling.md](./trajectory_value_scheduling.md)

---

## 实施顺序建议

1. **P0（完成）**：L1 — `enable_trajectory_value_scheduling` + 单测 + 消融 yaml。
2. **P1**：`compute_lease_ttl(V_traj)` + HTTP header 下发；gateway 分桶消费 TTL。
3. **P2**：L2 `lookup_resume`；belief 反馈闭环。
4. **P3**：L3 引擎 `set_resume_lease` + eviction modifier。
