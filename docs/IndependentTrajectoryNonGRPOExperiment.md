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
