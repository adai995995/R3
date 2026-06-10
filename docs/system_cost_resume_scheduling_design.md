# System-Cost Resume Scheduling Design

本文档定义一版替代当前 `V_traj = V_sys + V_learn_neg` 的 resume-aware 调度设计。核心变化是：**调度器只优化系统成本，不判断轨迹学习价值**。

当前实现中 `trajectory_invalid` / `trajectory_loop` / `trajectory_stall` / `trajectory_terminated` 会进入 `V_learn_neg`，并可能影响 priority、belief 与 placement。新设计建议把这些信号拆成两类：

- **语义质量信号**：不进入核心 priority，例如 loop、stall、low reward、bad reasoning、wrong answer。
- **系统生命周期信号**：作为硬状态机规则，例如 terminal、max step、cancel、model version mismatch、prefix hash mismatch。

目标是在保持 rollout 语义不变的前提下，最大化 KV 复用，减少重复 prefill，降低 tool-return 后恢复成本，并避免改变 RL 数据分布。

---

## 1. 设计边界

### 1.1 调度器应该优化什么

Resume-centric 调度只看系统变量：

- `last_backend_id`：上一次实际落到的 worker。
- `history_len_tokens`：历史上下文长度，作为 prefill 重算成本 proxy。
- `p_hit` / `belief`：在某个 worker 上恢复时 KV / prefix 命中的概率估计。
- `worker_load` / `queue_delay`：worker 当前负载和排队成本。
- `queue_wait_s`：请求在 router 队列中的等待时间，用于 aging / fairness。
- `t_tool_ema_s`：工具等待时间估计，用于 TTL。
- `ttl_remaining_s`：已有 lease 剩余时间。
- `KV_bytes` / `memory_pressure`：KV pin 的 byte-second 机会成本。
- `matched_prefix_tokens` / `actual_hit` / `prefill_time`：后端 telemetry，用于反馈校准。

### 1.2 调度器不应该优化什么

以下信号不进入核心 priority / placement / lease score：

- `loop`
- `stall`
- `low reward`
- `bad reasoning`
- `wrong answer`
- `trajectory seems unpromising`

理由：

1. 它们表达的是学习语义或轨迹质量，不是 resume recovery cost。
2. 一条 loop / stall 轨迹仍然可能有很长 history，KV miss 后仍然要 full prefill。
3. 在 async rollout、timeout、partial rollout、资源抢占等实现中，调度降权可能改变样本进入训练的时机，甚至改变有效数据分布。
4. 轨迹是否继续应由 rollout policy 决定，例如 `max_steps`、`max_tokens`、tool limit、env terminal。

---

## 2. 信号分类

### 2.1 不进入核心 priority 的语义信号

| 信号 | 处理建议 |
|---|---|
| `loop` | 不进入 priority；由 rollout 的 `max_steps` / `max_tokens` 结束 |
| `stall` | 不进入 priority；可作为 debug/analysis 指标 |
| `low reward` | 不使用 |
| `bad reasoning` | 不使用 |
| `trajectory_invalid` | 若只是语义/格式质量，不使用；若 env terminal，则走生命周期规则 |
| `tool failed but env returns observation` | 仍然是正常 resume，不降权 |

### 2.2 可保留的系统生命周期信号

| 信号 | 处理建议 |
|---|---|
| `terminated = true` | 不再生成 resume，释放 KV / lease |
| `max_steps reached` | rollout 结束，释放 KV / lease |
| `max_tokens reached` | rollout 结束，释放 KV / lease |
| `request canceled` | 删除 pending request，释放 KV / lease |
| `model_version mismatch` | KV 不可复用，标记 COLD 或 lease invalid |
| `prefix_hash mismatch` | KV 不可复用，标记 COLD 或 lease invalid |
| `worker inactive / not ready` | 不作为候选恢复 worker |

这些信号是系统有效性或生命周期，不是“轨迹是否值得学习”的判断。

---

## 3. 三个分离的 Score

不要用一个万能 `V_traj` 同时处理 ordering、placement、lease。新设计拆成三类 score，共享同一个基础收益项。

### 3.1 基础收益：期望节省的 history prefill

```text
B_reuse(r, w, t) =
    p_hit(r, w, t) * C_hist_prefill(r)
```

含义：

- `r`：当前 resume request。
- `w`：候选 worker。
- `t`：当前时间。
- `p_hit(r,w,t)`：在 worker `w` 上恢复时 KV / prefix 命中的概率。
- `C_hist_prefill(r)`：KV miss 时需要重算 history context 的 prefill 成本。

首版可以用 `history_len_tokens` 近似：

```text
C_hist_prefill(r) = normalize(history_len_tokens)
```

后续接入 engine telemetry 后，可以升级为：

```text
C_hist_prefill(r) =
    prefill_time_ms(history) 或 estimated_prefill_tokens * cost_per_token
```

---

## 4. Placement / Dispatch Score

Placement 解决“恢复到哪里”的问题。

```text
dispatch_score(r, w, t) =
    p_hit(r, w, t) * C_hist_prefill(r)
    - lambda_q * Q_w(t)
    - lambda_load * Load_w(t)
```

其中：

- `Q_w(t)`：worker `w` 的预计 queue delay。
- `Load_w(t)`：worker 当前 inflight / queue depth / local running requests。
- `lambda_q`、`lambda_load`：把排队和负载转换为统一成本的权重。

决策：

```text
w* = argmax_w dispatch_score(r, w, t)
```

直觉：

- 回 `last_backend`：`p_hit` 高，但可能排队更久。
- 走 least-load：排队短，但可能需要 full prefill。
- 只有当粘回 last backend 节省的 history prefill 大于额外排队/负载成本时，才坚持 affinity。

二选一时可以写成：

```text
p_hit(r, w_last, t) * C_hist_prefill(r)
>
lambda_q * (Q_w_last(t) - Q_w_least(t))
```

---

## 5. Ordering Score

Ordering 解决“谁先恢复”的问题。它不应该只按 `B_reuse` 排序，因为高 reuse benefit 不一定马上损失；真正紧急的是继续等待会损失 KV reuse opportunity 的请求。

### 5.1 Delay Regret

```text
R_delay(r, Delta) =
    [p_hit*(r, t) - p_hit*(r, t + Delta)]
    * C_hist_prefill(r)
```

其中：

```text
p_hit*(r, t) = p_hit(r, w*, t)
```

即按当前最优 dispatch worker 估算命中概率。

首版如果没有可微/可查的 `p_hit(t + Delta)`，可以用 TTL / pause age 的分段 proxy：

```text
R_delay ~= decay_rate(pause_age_s, ttl_remaining_s, belief) * C_hist_prefill
```

### 5.2 Order Score

```text
order_score(r, t) =
    lambda_1 * R_delay(r, Delta)
    + lambda_2 * max(0, dispatch_score(r, w*, t))
    + lambda_3 * A_age(r, t)
```

其中：

```text
A_age(r, t) =
    min(queue_wait_s(r) / T_age, A_max)
```

解释：

- `R_delay`：继续等会损失多少 KV reuse opportunity。
- `max(0, dispatch_score)`：现在恢复是否有正的系统收益。
- `A_age`：纯系统 fairness，防止 starvation。

注意：`A_age` 使用 router 侧 `queue_wait_s`，不是 runtime 侧 `pause_age_s`。`pause_age_s` 描述 tool-return 后到当前的外部等待；`queue_wait_s` 描述请求进入 scheduler 后的排队时间。

---

## 6. Lease / TTL Score

Lease 解决 tool wait 阶段“KV pin 多久、是否值得 pin”的问题。

```text
V_lease(r, tau) =
    P(T_tool <= tau) * C_hist_prefill(r)
    - lambda_mem * KV_bytes(r) * tau * rho_mem(w, t)
```

其中：

- `P(T_tool <= tau)`：工具在 TTL `tau` 内返回的概率。
- `C_hist_prefill`：保留 KV 后未来能省下的 prefill 成本。
- `KV_bytes * tau`：pin KV 的 byte-second 成本。
- `rho_mem(w,t)`：当前 worker / gateway memory pressure。
- `lambda_mem`：内存机会成本权重。

决策：

```text
tau* = argmax_tau V_lease(r, tau)
lease_score = normalize(max_tau V_lease(r, tau))
```

首版 proxy：

```text
P(T_tool <= tau) ~= clamp(tau / max(t_tool_ema_s, eps), 0, 1)
KV_bytes(r) ~= history_len_tokens * bytes_per_token_proxy
rho_mem(w,t) ~= 1.0  # 无 telemetry 时先固定
```

后续接入 gateway / engine telemetry 后替换为真实 `kv_bytes`、memory pressure 和 matched prefix。

---

## 7. Belief 与反馈闭环

### 7.1 第一阶段：调度侧 belief proxy

在没有真实 KV telemetry 时，`p_hit` 仍然是 belief：

- 有 `last_backend_id` 且 pause age 短：HOT。
- 有 `last_backend_id` 但等待较久或 worker 忙：WARM。
- 无 `last_backend_id`、worker inactive、model/prefix mismatch：COLD。

语义质量信号不再把 belief 打成 COLD。COLD 只表达 **KV 不可复用或恢复不划算的系统判断**。

### 7.2 第二阶段：后端 telemetry 校准

详细工程方案见 [real_hit_kv_pin_design.md](./real_hit_kv_pin_design.md)（Phase C：真实 hit；Phase D：真实 pin）。

当 L2/L3 可用后，用真实观测更新：

- `actual_hit`
- `matched_prefix_tokens`
- `estimated_prefill_tokens`
- `prefill_time_ms`
- `lease_remaining_s`
- `worker_url`
- `memory_pressure`

更新对象：

- `p_hit_bias`
- `worker_load_ema`
- `t_tool_ema_s`
- `last_backend_id`
- `bytes_per_token_proxy` / `KV_bytes`

---

## 8. 与当前实现的迁移关系

当前实现：

```text
V_traj = V_sys + V_learn_neg
```

其中 `V_learn_neg` 包括 invalid / loop / stall / terminated 等负反馈。

目标实现：

```text
dispatch_score = expected_prefill_saved - queue/load_cost
order_score = delay_regret + positive_dispatch_value + aging
lease_score / ttl = expected_tool_return_within_ttl * prefill_saved - memory_byte_second_cost
```

建议迁移步骤：

1. 保留 `request_type=resume`、`last_backend_id`、`history_len_tokens`、`pause_age_s`、`queue_wait_s` 等系统字段。
2. 从核心 priority / placement / lease score 中移除 `trajectory_loop`、`trajectory_stall`、low reward、bad reasoning。
3. 将 `trajectory_terminated` 从 score penalty 改为 lifecycle hard rule。
4. 将 `trajectory_invalid` 拆分：语义 invalid 不影响调度；系统 invalid / terminal 走 lifecycle。
5. 将 `compute_resume_priority()` 迁移为 `compute_order_score()`。
6. 将 `compute_worker_route_score()` 迁移为 `compute_dispatch_score()`。
7. 将 `compute_lease_ttl()` 的输入改为 `C_hist_prefill`、`t_tool_ema_s`、`KV_bytes`、`memory_pressure`、`p_hit`。

---

## 9. 配置注意事项

### 9.1 `resume_normal_quota` 必须统一格式

当前 `parse_ratio()` 只解析字符串形式：

```yaml
resume_normal_quota: "3:1"
```

部分旧配置写成：

```yaml
resume_normal_quota: [3, 1]
```

这会在代码中被 `str(...)` 转成 `"[3, 1]"`，解析失败后回退为 `1:1`，导致实验中实际配额与注释不一致。

设计上建议：

1. 短期：所有 YAML 统一写 `"3:1"`。
2. 中期：让 `parse_ratio()` 同时支持 `"3:1"`、`[3, 1]`、`{"resume": 3, "normal": 1}`。
3. 指标中输出 `quota_resume_target` / `quota_normal_target`，实验报告以实际值为准。

### 9.2 新开关建议

为避免和旧 `enable_trajectory_value_scheduling` 混淆，建议新增开关：

```yaml
router_args:
  router_config:
    enable_system_cost_resume_scheduling: true
    enable_trajectory_value_scheduling: false
```

兼容策略：

- `enable_system_cost_resume_scheduling=true` 时，禁用 `V_learn_neg`。
- 若两个开关同时为 true，优先使用 system-cost 设计并打印 warning。

---

## 10. 推荐实现闭环

```text
1. generate 落到某个 backend
   -> 记录 last_backend_id、history_len_tokens、KV_bytes(proxy)

2. 触发 tool call，trajectory suspend
   -> 根据 history_len、t_tool_ema、memory_pressure(proxy) 计算 ttl_s / lease_score
   -> gateway/engine 尝试 pin KV

3. tool return，生成 Resume Request
   -> 根据 last_backend_id、ttl_remaining、p_hit_bias、worker_load 估计 p_hit 和 dispatch_score

4. router 做 ordering 和 placement
   -> 优先恢复 delay regret 高、reuse benefit 高、且等待已久的 resume

5. engine 执行后返回 telemetry
   -> actual_hit / matched_prefix_tokens / prefill_time / queue_wait
   -> 更新 p_hit_bias、worker_load_ema、t_tool_ema、last_backend_id
```

---

## 11. 非目标

- 不用 scheduler 判断样本是否值得学习。
- 不用 reward / correctness / semantic quality 做正负优先级。
- 不因为 loop / stall 提前丢弃 trajectory。
- 不在首版要求 KV migration。
- 真实 prefix hit / KV pin 的落地步骤见 [real_hit_kv_pin_design.md](./real_hit_kv_pin_design.md)（Phase C/D）；P0 可先用 belief proxy。
