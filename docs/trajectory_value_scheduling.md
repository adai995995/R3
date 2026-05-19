# Trajectory Value Scheduling（轨迹价值调度）

## 1. 目标

在 Agentic RL rollout 中，将 tool-return 后的 continuation 显式建模为 **Resume Request**，并用可在线计算的 **轨迹调度价值** 统一指导：

- **Ordering**：resume/normal 谁先进入推理队列
- **Placement**：resume 请求路由到哪台 worker（`EnvAffinityRouter`）
- **Gateway hint**：是否向 sgl-model-gateway 发送 `X-ROLL-Preferred-Worker-Url`（Form B）

**不用于**：丢弃轨迹、改变 GRPO 进 batch 的样本集合。

## 2. 价值分解

```text
V_traj = V_sys + V_learn_neg
```

| 项 | 含义 | 默认 |
|---|---|---|
| `V_sys` | 期望恢复净收益 + 收尾紧迫度 + 等待公平 − 负载/重算风险 | 主项 |
| `V_learn_neg` | 无效/死循环/停滞等 **负反馈**（只做减法） | 默认启用 |

不使用 per-turn **reward 正反馈** 作为主排序键（稀疏、且与系统目标易错位）。

## 3. 观测字段（Runtime → Router）

`TrajEnvManager.make_decision()` 写入 `DataProto.meta_info`，由 `RouterClient` 注入 `_roll_route_meta`：

| 字段 | 类型 | 含义 |
|---|---|---|
| `trajectory_id` | str | 轨迹 ID |
| `request_type` | str | `normal` / `resume` |
| `pause_age_s` | float | tool-return 后等待时长 |
| `history_len_tokens` | int | 当前上下文 token 数 |
| `last_backend_id` | int? | 上次命中 worker |
| `remaining_steps` | int | `max_steps - step` |
| `max_steps` | int | 环境最大步数 |
| `trajectory_invalid` | float | 0/1，非法 action / parse fail |
| `trajectory_loop` | float | 0/1，近 k 步重复 action |
| `trajectory_stall` | float | 0/1，连续 n 步无 reward 进展 |
| `trajectory_terminated` | float | 0/1，已终止 |

## 4. Recoverability Belief（HOT / WARM / COLD）

调度器根据 **rollout 可见信息** 估计 KV 可恢复概率 `p_hit`，而非后端精确状态：

| 状态 | `p_hit` 默认 | 路由策略 |
|---|---|---|
| HOT | 0.85 | 优先 `last_backend`；Form B 发 preferred header |
| WARM | 0.45 | `route_score` 比较 last vs 其它 worker |
| COLD | 0.10 | 放弃强亲和，load-aware |

分类规则（见 `trajectory_value.classify_belief`）：

- `pause_age_s` 短、有 `last_backend`、worker 未过载、未 `force_migrate` → HOT
- `pause_age_s` 很长或 `trajectory_invalid/loop` 高 → COLD
- 其余 → WARM

## 5. 公式

归一化：

```text
ñ_h = log1p(h) / log1p(H_max)
ñ_a = log1p(pause_age_s) / log1p(A_max)
ñ_r = remaining_steps / max_steps        # 越小越急
```

系统价值：

```text
V_sys = w_p · p_hit · ñ_h
      + w_f · (1 - ñ_r)
      + w_a · ñ_a
      - w_c · (1 - p_hit) · ñ_h
```

学习侧负反馈：

```text
V_learn_neg = - c_inv · I_invalid
             - c_loop · I_loop
             - c_stall · I_stall
             - c_term · I_terminated
```

调度：

```text
resume_priority = V_sys + V_learn_neg + η · queue_wait_s     # Ordering
route_score(w)  = V_sys + w_p·p_w·ñ_h - w_q·load(w) - w_c·(1-p_w)·ñ_h   # Placement
```

## 6. 配置（`router_config`）

| 键 | 默认 | 说明 |
|---|---|---|
| `enable_trajectory_value_scheduling` | `false` | 开启后替代 legacy `compute_request_priority` / 增强 `resume_score` |
| `trajectory_value_weights` | 见代码默认值 | `w_p, w_f, w_a, w_c, w_q, ...` |
| `belief_p_hot` / `belief_p_warm` / `belief_p_cold` | 0.85 / 0.45 / 0.10 | belief 映射到 p_hit |
| `belief_hot_pause_age_s` | 5.0 | 低于此倾向 HOT |
| `learning_penalty_weights` | 见代码 | `c_inv, c_loop, c_stall, c_term` |

与现有开关关系：

- `enable_trajectory_value_scheduling=true` 时，resume 的 **base_priority** 使用 `V_traj`
- `enable_resume_aware_routing=true` 时，worker 选择使用 `route_score`（含 belief 分支）
- Form B：`enable_resume_aware_routing=false`，仅 ordering + conditional preferred header

## 7. 指标

- `scheduler/router/trajectory_value_mean`
- `scheduler/router/belief_state/{hot,warm,cold}_count`
- `scheduler/router/penalty_invalid_count` 等

Runtime 侧继续上报 `resume_latency_*`、`selected_backend_affinity_hit` 等（见 `key_indicator.md`）。

## 8. 代码入口

- `roll/distributed/scheduler/trajectory_value.py` — 价值与 belief 计算
- `roll/pipeline/agentic/trajectory_signals.py` — runtime 负反馈信号
- `roll/pipeline/agentic/env_manager/traj_env_manager.py` — 写入 meta
- `roll/distributed/scheduler/router.py` — `EnvAffinityRouter` / `SglangOrderingRouter` 集成

## 9. 回滚

设置 `enable_trajectory_value_scheduling: false` 即回退到 `resume_priority.py` 原有公式。
