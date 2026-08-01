# Motivation Observation 数据清单与实验矩阵

## 目标

实验只回答以下三个问题，不比较 learner token 吞吐或 updates/hour：

1. 增大最大在途轨迹数 `C` 后，rollout 输出 token 吞吐是否继续提高，但一次 policy update 的端到端间隔不再下降。
2. 这种分离是否来自 bounded staleness 将已经投入资源的轨迹变成不可训练数据。
3. AppWorld 与 τ-bench 的最佳 `C` 是否不同，从而说明固定并发上限不足以适配不同 AgenticRL workload。

## 实验环境

- Workload：AppWorld、τ-bench retail。
- 模型：Qwen3-4B-Instruct-2507。
- GPU：单机 8 卡，4 卡 actor/learner，4 卡 rollout。
- 算法：REINFORCE，`group_size=1`，避免 GRPO group 语义影响结果。
- 每次 policy update 消费 4 条轨迹。
- 每条轨迹最多 16 次 action，每次最多生成 256 tokens，context 上限 32768。
- FIFO trajectory scheduling，固定 outstanding watermark admission。
- 关闭 version-aware priority、动态 admission、KV rebuilding 等新系统策略。
- 关闭所有 checkpoint。

## 最小实验矩阵

| 目的 | Workload | Staleness tolerance | 最大在途轨迹 `C` | Steps |
|---|---|---:|---|---:|
| Observation 1、3 | AppWorld | 2 | 4、8、16、32 | 20 |
| Observation 1、3 | τ-bench retail | 2 | 4、8、16、32 | 20 |
| Observation 2 | AppWorld | 1000（近似关闭） | 16、32 | 20 |

第一轮使用一个固定 seed 验证现象与指标。正式论文结果再对关键点补 3 seeds，不在原型阶段搜索最优参数。

## 必须输出的数据

### 每个 run

- `raw_rollout_output_tokens_per_second`：policy rollout 实际生成的 output tokens / rollout 墙钟时间。
- `policy_update_interval_mean_seconds`：去掉 warmup 后，`time/step_total` 的均值；这是核心训练推进指标。
- `stale_trajectory_fraction_of_admitted`：因版本过期淘汰的轨迹数 / admission 的轨迹总数。
- `learner_data_wait_seconds_total` 与 `learner_data_wait_fraction`。
- 至少跨过一次版本更新的轨迹数和比例。
- 每条跨版本轨迹平均跨过多少个边界。
- survivor 在新版本第一次请求产生的 re-prefill tokens、TTFT 和恢复完成时间。

### 每条被版本淘汰的 trajectory

- trajectory ID、起始/结束版本、淘汰原因和版本年龄。
- 已完成 actions、最大 action budget、剩余 action budget、action budget progress。
- 已生成 output tokens、累计 logical inference tokens。
- tool calls、精确 runner-side tool wall time。
- 环境侧总墙钟时间。
- rollout inference service time。
- vLLM V1 engine-step 归因时间及其中的 prefill/decode 部分。
- vLLM V0 request-attributed model execute/forward time（兼容性辅助项）。
- engine-reported prefill tokens 和 cached prompt tokens。
- admission、完成、淘汰时间戳。

### 每次 version boundary

- 边界版本、当时 in-flight/cross-version/survivor/expired 数量。
- 已投入 actions、tokens、tool wall time 和 engine-step 归因时间。
- 每条 survivor 第一次请求的 dispatch、first-token、finish 相对边界 `t=0` 的时间。
- survivor 第一次请求的 full re-prefill tokens、TTFT、decode time、decode throughput。
- 随恢复请求完成而下降的 unresolved survivor 数量。

## 归一化方式

淘汰成本同时报告：

- 占 admitted trajectories 的比例；
- 每次 policy update 的 actions/tokens/tool time/engine-step 归因时间；
- 占全部 policy-rollout engine-step 归因时间的比例。

实验总时长内的绝对淘汰数量只作为辅助信息，不能作为跨配置主比较。

## 时间指标的准确含义

- `generate_seconds` 是请求经过 rollout 推理服务的墙钟时间，包含排队与执行，不等于纯 GPU 时间。
- `tool_wall_seconds` 是 AppWorld/τ-bench runner 对真实工具执行区间的精确墙钟计时。
- vLLM V1 不会在 `RequestOutput` 暴露旧版 `RequestMetrics`。当前在每次 `EngineCore.step()` 外测量墙钟时间，再按该批次每个请求实际调度的 token 份额分摊；所有请求归因值之和等于引擎批次时间，不会因共 batch 被重复累计。
- `engine_step_seconds_attributed` 是全局去重的 **GPU 引擎工作时间代理**，仍包含调度与引擎开销，并不等于 CUDA kernel profiler 得到的纯 GPU 时间。
- prefill/decode 拆分根据每个调度块中 prompt token 与 decode token 的比例继续分摊，因此支持 chunked prefill。论文中应称为 **prefill/decode engine-step attributed time**。
- `model_execute_seconds` 只在 vLLM V0 路径保留为兼容性辅助项，不作为 V1 正式实验的核心计算浪费指标。

## 输出文件

统一分析器 `scripts/analyze_motivation_observation.py` 生成：

- `run_summary.csv`：Figure 1、Figure 2 和跨 workload 最佳 `C` 比较。
- `primary_metrics.csv`：只保留 rollout 吞吐、平均 step 时间、stale 轨迹浪费和跨版本恢复等论文主指标。
- `stale_trajectories.csv`：淘汰轨迹进度 CDF 与浪费成本。
- `boundary_recovery.csv`：以 version refresh 为 `t=0` 的 KV recovery 时间线。
- `manifest.json`：原始 report 路径、行数和指标语义说明。

实验入口：

```bash
bash scripts/run_motivation_workload_sweep.sh appworld baseline
bash scripts/run_motivation_workload_sweep.sh tau baseline
CAPS="16 32" bash scripts/run_motivation_workload_sweep.sh appworld relaxed
```
