# Full Runtime 50-Step Matched Pilot

更新时间：2026-08-01

## 1. 目的

本实验用于判断完整 version-aware runtime 是否在真实 AgenticRL workload
中产生方向正确的端到端效果。它不是模块消融，也不用于报告最终论文结果。

## 2. 实验设置

| 设置 | 数值 |
| --- | --- |
| Workload | tau-bench Retail、AppWorld |
| 模型 | Qwen3-4B-Instruct-2507 |
| 训练算法语义 | independent trajectory，REINFORCE，`group_size=1` |
| GPU | 单机 8 卡：4 learner + 4 rollout |
| Learner updates | 50 |
| Seed | 91 |
| Rollout batch size | 4 |
| Environment groups | 16 |
| Staleness tolerance | 2 |
| Maximum in-flight trajectories | 32 |
| Router request capacity | 128 |
| Checkpoint | disabled |

两组策略的 workload、task seed、模型采样 seed、GPU、batch、版本容忍度、
环境并发和 Router 请求容量完全相同。

- **FIFO**：`outstanding_watermark` admission，FIFO scheduling，关闭
  version priority、KV reconstruction 和 locality routing；
- **Full Runtime**：feedback-driven admission、version-aware scheduling、
  completion-ETA placement、post-refresh KV reconstruction 和
  closed-loop AIMD reserve。

统计口径：

- Policy-update interval 使用相邻 version-boundary timestamp 的差值，
  去掉前两个 warmup interval 后取均值；
- raw rollout throughput 使用终止报告中的 whole-run response tokens/s；
- stale waste 使用 `async_waste` 和 `rollout/stale_*` 终止累计值；
- 两种策略都完成 50 个 learner update，并消费 200 条有效轨迹。

## 3. 主要结果

### 3.1 tau-bench Retail

| 指标 | FIFO | Full Runtime | 变化 |
| --- | ---: | ---: | ---: |
| Policy-update interval (s/update) | 15.797 | 15.611 | -1.18% |
| Raw rollout response throughput (token/s) | 516.45 | 203.30 | -60.64% |
| Admitted trajectories | 616 | 243 | -60.55% |
| Expired trajectories | 355 | 33 | -90.70% |
| Expired logical inference tokens | 18,855,497 | 2,130,037 | -88.70% |
| Expired-token fraction | 66.10% | 18.65% | -71.79% |
| Expired actions | 3,491 | 368 | -89.46% |
| Expired tool calls | 1,561 | 168 | -89.24% |
| Expired tool time (s) | 1.667 | 0.535 | -67.93% |

Full Runtime 将 raw rollout activity 降低约 61%，但没有减慢 learner：
平均 update interval 小幅缩短 1.18%。同时，过期轨迹和过期 token 分别减少
90.70% 和 88.70%。该 workload 支持“更多 rollout 吞吐并不等于更快训练”
这一核心判断。

剩余过期轨迹更深：每条过期轨迹的平均 actions 从 9.83 增至 11.15，
平均 logical tokens 从 53,114 增至 64,547。这里应优先报告总浪费的下降；
该现象说明 runtime 消除了大量额外 admission，但尚未保证每条残余深轨迹
都能被挽救。

### 3.2 AppWorld

| 指标 | FIFO | Full Runtime | 变化 |
| --- | ---: | ---: | ---: |
| Policy-update interval (s/update) | 15.709 | 13.785 | -12.25% |
| Raw rollout response throughput (token/s) | 479.16 | 176.55 | -63.15% |
| Admitted trajectories | 612 | 211 | -65.52% |
| Expired trajectories | 380 | 3 | -99.21% |
| Expired logical inference tokens | 19,562,677 | 156,494 | -99.20% |
| Expired-token fraction | 64.61% | 1.40% | -97.83% |
| Expired actions | 4,738 | 31 | -99.35% |
| Expired tool calls | 4,237 | 28 | -99.34% |
| Expired tool time (s) | 125.804 | 0.861 | -99.32% |

AppWorld 中，Full Runtime 不仅减少约 63% 的 raw rollout activity，还将
平均 update interval 缩短 12.25%。过期轨迹、tokens、actions、tool calls
和 tool time 均减少约 99%。这说明在长工具交互 workload 中，减少无效
producer debt 可以直接改善 learner 的端到端推进速度。

每条剩余过期轨迹平均消耗 52,165 logical tokens、10.33 个 action 和
9.33 次工具调用；FIFO 分别为 51,481、12.47 和 11.15。两种策略下被淘汰
轨迹都已经投入较深，区别主要来自 Full Runtime 将过期轨迹总数从 380
压缩到 3。

## 4. 机制是否真实执行

| Runtime signal | tau-bench Retail | AppWorld |
| --- | ---: | ---: |
| Router scheduling decisions | 3,888 | 2,695 |
| Priority-queued requests | 1,399 | 151 |
| Priority-reordered requests | 279 | 12 |
| Completion-ETA selected requests | 3,665 | 2,577 |
| Rebuild-selected requests | 215 | 110 |
| Engine KV feedback coverage | 94.55% | 95.99% |
| Mean routing decision time | 3.70 ms | 1.36 ms |
| Mean completion-ETA absolute error | 5.22 s | 2.08 s |

这些数据确认结果不是仅来自配置中打开开关：trajectory priority、ETA
placement、KV reconstruction 和真实引擎 KV feedback 都进入了请求路径。

tau-bench 的 ETA 绝对误差仍偏高，是后续 estimator calibration 的明确
改进点。两组 Full Runtime 的动态 reserve 最终都达到 8；tau-bench 发生
多次增减振荡，而 AppWorld 在早期增大后基本保持稳定。正式实验前不应再
针对单个 workload 手工调 reserve，但应通过消融检查动态 reserve 是否是
主要收益来源。

## 5. 当前判断

该单 seed pilot 足以回答“完整方法是否值得继续”：**值得**。

1. 两个 workload 中，Full Runtime 都以显著更少的 raw rollout 工作完成
   相同数量的 learner update；
2. tau-bench Retail 的训练推进速度基本不变并小幅改善；
3. AppWorld 的平均 update interval 改善 12.25%；
4. 两个 workload 的过期资源浪费均大幅下降；
5. 控制决策开销处于毫秒级，没有抵消端到端收益。

但该结果仍不能作为最终论文结论：只有一个 seed，且 Full Runtime 同时
打开了 admission、scheduling、placement 和 reconstruction，无法归因各
模块贡献，也尚未验证训练收敛分布。

下一步应冻结当前配置，运行以下消融，并补充至少三个 seed：

```text
FIFO + fixed admission
adaptive admission only
admission + version-aware scheduling/placement
full runtime
```

结果文件：

```text
output/tau_bench_retail_qwen3_4b_fifo_c32_seed91_50step/terminal_waste.step_50.json
output/tau_bench_retail_qwen3_4b_runtime_c32_seed91_50step/terminal_waste.step_50.json
output/appworld_qwen3_4b_fifo_c32_seed91_50step/terminal_waste.step_50.json
output/appworld_qwen3_4b_runtime_c32_seed91_50step/terminal_waste.step_50.json
```

