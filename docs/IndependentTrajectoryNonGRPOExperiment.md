# 非 GRPO 独立轨迹实验

## 实验目的

这组实验用于排除 GRPO 的 group completion 语义，验证系统收益是否只是由
`group_size > 1` 的队头等待造成。每条 WebShop 轨迹在本实验中都是独立的训练
样本，不需要等待同组其他轨迹完成。

## 实验设置

- 场景：真实 WebShop 交互，单条轨迹最多 10 个 action。
- 模型：Qwen3-4B-Instruct-2507。
- 硬件：单机 8 卡；GPU 0-3 训练，GPU 4-7 rollout。
- 算法：trajectory-level REINFORCE；`group_size=1`，不使用 GRPO group 归一化。
- 训练：每次更新消费 4 条轨迹，共 6 次更新；每个 run 消费 24 条轨迹。
- 异步设置：8 个独立环境，最多 24 条在途轨迹，版本容忍度为 2。
- 生成：每个 action 最多 96 tokens，单条轨迹最多 10 个 action。
- 随机种子：49、50、51。
- Checkpoint：关闭中间和最终 checkpoint。

匹配对照只改变 runtime 策略：

1. `FIFO`：固定 outstanding watermark、FIFO 轨迹调度、不做 KV working-set rebuild。
2. `Version Runtime`：version-adaptive admission、version priority、progress floor、
   跨版本 KV working-set rebuild、working-set routing 和 soft locality。

## 三个 Seed 的结果

| Seed | 策略 | 原始轨迹 | 训练消费 | 过期轨迹 | 原始 logical tokens | 有效 logical tokens | 原始 response tok/s | 有效 response tok/s | Rollout 时间 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 49 | FIFO | 47 | 24 | 18 | 816,314 | 433,155 | 266.42 | 134.84 | 69.91 s |
| 49 | Version Runtime | 25 | 24 | 0 | 434,994 | 431,738 | 140.91 | 138.79 | 71.76 s |
| 50 | FIFO | 55 | 24 | 14 | 723,117 | 339,421 | 252.07 | 113.56 | 76.51 s |
| 50 | Version Runtime | 25 | 24 | 1 | 453,247 | 413,560 | 145.35 | 135.10 | 79.28 s |
| 51 | FIFO | 55 | 24 | 16 | 1,087,837 | 569,750 | 294.19 | 143.80 | 80.11 s |
| 51 | Version Runtime | 29 | 24 | 0 | 526,424 | 485,494 | 151.96 | 137.35 | 78.96 s |

## 汇总

| 指标 | FIFO | Version Runtime | 变化 |
|---|---:|---:|---:|
| 训练消费轨迹 | 72 | 72 | 相同 |
| 原始轨迹 | 157 | 79 | -49.7% |
| 过期轨迹 | 48 | 1 | -97.9% |
| 过期轨迹已执行 action | 311 | 10 | -96.8% |
| 过期 logical tokens | 833,364 | 39,687 | -95.2% |
| 原始 logical tokens | 2,627,268 | 1,414,665 | -46.2% |
| 有效 logical tokens | 1,342,326 | 1,330,792 | -0.9% |
| 计算转化率（有效 / 原始 logical tokens） | 51.1% | 94.1% | +43.0 pp |
| 平均原始 response tok/s | 270.89 | 146.08 | -46.1% |
| 平均有效 response tok/s | 130.74 | 137.08 | +4.9% |
| 平均 rollout 时间 | 75.51 s | 76.67 s | +1.5% |
| 平均 learner wait fraction | 27.28% | 28.93% | +1.65 pp |

所有 6 个 run 都完成 6 次真实训练更新，均没有 shutdown timeout，也没有写入模型
checkpoint。

## 结论

该现象不依赖 GRPO。即使每条轨迹完全独立，FIFO 的 rollout 侧仍表现得更忙，
原始 response throughput 高 85.4%，但没有多推进一次训练更新，也没有产生更多被
learner 消费的样本。Version Runtime 用少 46.2% 的推理计算交付了几乎相同的有效
logical tokens，并将计算转化率从 51.1% 提高到 94.1%。

本组实验支持核心 observation：raw rollout throughput 不能代表训练推进速度；
超量生产会把 GPU 计算投入到最终因版本过期而不可训练的轨迹中。

这组短实验还不能证明训练 wall-clock 显著加速。Version Runtime 的平均 rollout
时间慢 1.5%，learner wait fraction 高 1.65 个百分点；当前固定 reserve=4 略偏保守。
下一步应使用相同的 `group_size=1` 设置做 admission-only、priority-only 和
KV-only 消融，再单独调节 reserve，而不是把收益全部归因于完整系统。

## 产物

- 配置：`examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_independent_*_6step.yaml`
- Seed 49：`output/webshop_qwen3_4b_independent_{fifo,runtime}_6step/terminal_waste.step_6.json`
- Seed 50：`output/webshop_qwen3_4b_independent_{fifo,runtime}_6step_seed50/terminal_waste.step_6.json`
- Seed 51：`output/webshop_qwen3_4b_independent_{fifo,runtime}_6step_seed51/terminal_waste.step_6.json`

`step_reinforce` 的预检不计入结果：当前 WebShop pipeline 每条完整轨迹只产生一条
训练记录，而 `step_reinforce` 需要每条记录包含 `step` 字段。本实验因此采用与现有
轨迹数据结构匹配的 trajectory-level `reinforce`。

## 高并发压力实验

为了验证低压力下 step 时间变化不大的原因，后续匹配实验将独立 WebShop 环境数从
8 增加到 16，将最大在途轨迹数从 24 增加到 48。Runtime 的 reserve 从 4 增加到
12，并将每个推理 worker 的 priority service slot 设为 2，使同时到达的请求真正
进入 Runtime 可控制的等待队列。该预实验使用 seed 52，其他训练条件保持不变。

| 指标 | FIFO | Version Runtime | 变化 |
|---|---:|---:|---:|
| 完成 6 次更新的 rollout 时间 | 79.90 s | 74.47 s | -6.8% |
| 稳态平均 step 时间 | 11.90 s | 10.61 s | -10.8% |
| 稳态 step rollout 等待 | 2.66 s | 1.36 s | -49.0% |
| Learner wait fraction | 25.25% | 19.47% | -5.78 pp |
| 训练消费轨迹 | 24 | 24 | 相同 |
| 原始轨迹 | 90 | 39 | -56.7% |
| 过期轨迹 | 40 | 6 | -85.0% |
| 原始 logical tokens | 1,728,321 | 676,751 | -60.8% |
| 过期 logical-token fraction | 48.70% | 23.22% | -25.48 pp |
| 有效 response tok/s | 119.59 | 124.67 | +4.2% |
| Priority queued requests | 0 | 167 | 已触发 |
| Priority reordered requests | 0 | 30 | 已触发 |

高并发下，FIFO 的额外工作开始与形成下一批训练数据争抢推理资源。Version Runtime
不仅减少无效计算，还通过真实排队和重排缩短了有效 batch 的形成时间，因此节省的
计算开始转化为 step 时间和 learner wait 的下降。

Runtime 仍有 6 条轨迹过期，因为该压力实验故意使用较大的 reserve=12 来维持 16
路并发。这说明负载不能无限增大，admission 与 priority 必须共同控制。该结果目前
只有一个 seed，应补两个 seed 后再报告显著性；它也只证明系统时间收益，不证明调度
后的训练数据分布和最终收敛完全一致。

压力实验产物：

- FIFO：`output/webshop_qwen3_4b_independent_pressure_fifo_seed52_6step/terminal_waste.step_6.json`
- Runtime：`output/webshop_qwen3_4b_independent_pressure_runtime_seed52_6step/terminal_waste.step_6.json`
- 配置：`examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_independent_pressure_*_6step.yaml`

## Sokoban 跨场景压力实验

为了检查 WebShop 上的现象是否依赖单一环境，使用相同的独立轨迹和高并发设置，
将环境替换为 16 个 `SimpleSokoban` 实例。模型仍为 Qwen3-4B-Instruct-2507，
使用 4 张训练 GPU、4 张 rollout GPU，训练 batch 为 4，最大在途轨迹数为 48，
完成 6 次真实 REINFORCE 更新。该预实验使用 seed 53，FIFO 与 Version Runtime
只改变 runtime 策略，均不保存 checkpoint。

| 指标 | FIFO | Version Runtime | 变化 |
|---|---:|---:|---:|
| 完成 6 次更新的 rollout 时间 | 61.34 s | 62.49 s | +1.9% |
| 稳态平均 step 时间 | 8.76 s | 8.72 s | -0.4% |
| 稳态 step rollout 等待 | 0.275 s | 0.328 s | +0.053 s |
| Learner wait fraction | 10.32% | 11.48% | +1.16 pp |
| 训练消费轨迹 | 24 | 24 | 相同 |
| 原始轨迹 | 112 | 36 | -67.9% |
| 过期轨迹 | 42 | 4 | -90.5% |
| 原始 logical tokens | 646,833 | 222,055 | -65.7% |
| 过期 logical-token fraction | 39.17% | 12.46% | -26.72 pp |
| 计算转化率（有效 / 原始 logical tokens） | 19.97% | 62.79% | 3.14 倍 |
| 有效 response tok/s | 25.30 | 26.37 | +4.2% |
| Priority queued requests | 0 | 149 | 已触发 |
| Priority reordered requests | 0 | 18 | 已触发 |
| KV rebuild selections | 0 | 0 | 未触发 |

这组实验复现了计算浪费问题：在 learner 同样消费 24 条轨迹的情况下，FIFO 多生成
了 76 条轨迹，并产生了 42 条过期轨迹。Version Runtime 将原始推理计算减少
65.7%，将过期轨迹减少 90.5%，有效 response throughput 仍提高 4.2%。这说明
“rollout 更忙、原始吞吐更高”并不等于更多可训练数据，且该现象不只出现在 WebShop。

但 Sokoban 没有复现 WebShop 的 step 时间收益。FIFO 的 learner wait 只有 10.32%，
稳态每步 rollout 等待仅 0.275 秒，训练关键路径主要由 actor 更新等固定开销构成；
因此即使 Runtime 大幅减少后台无效计算，稳态 step 时间仍基本不变。这是一个合理的
负结果：系统的 wall-clock 收益依赖 rollout 是否处在训练 batch 形成的关键路径。

此外，本次轨迹都在单个版本内快速结束，跨版本 partial trajectory 和 KV rebuild
候选均为 0。Runtime 的真实优先级队列已经生效，但该环境主要验证 admission 对
过量生产的抑制，不能验证跨版本 urgency 和 KV working-set rebuilding。Sokoban
模型行为质量也较弱，因此本实验只作为系统 observation，不用于证明最终训练收敛。

Sokoban 压力实验产物：

- FIFO：`output/sokoban_qwen3_4b_independent_pressure_fifo_seed53_6step/terminal_waste.step_6.json`
- Runtime：`output/sokoban_qwen3_4b_independent_pressure_runtime_seed53_6step/terminal_waste.step_6.json`
- 配置：`examples/qwen2.5-0.5B-agentic/agent_val_sokoban_qwen3_4b_independent_pressure_*_6step.yaml`
