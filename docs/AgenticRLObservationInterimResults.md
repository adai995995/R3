# AgenticRL 超量采样 Observation 阶段性结果

## 1. 当前证据状态

截至 2026-07-22：

- Tower of Hanoi 正式并发扫描已经完成：30 learner steps、3 个 seeds、6 个并发点，共 18 次独占 GPU 运行。
- WebShop 已完成 rollout-bound calibration：4 steps、1 个 seed、2 个并发点；只用于验证实验设置，不作为正式统计结论。
- HotpotQA 正式实验尚未开始。此前启动在 Hydra 参数解析阶段失败，没有产生有效数据，也没有模型进程残留。
- 所有有效运行均关闭 checkpoint；Tower 正式运行期间 8 张 GPU 无其他任务干扰。

因此，当前可以形成 Tower 单 workload 的正式 observation，但还不能声称完成跨场景复现。

## 2. 研究问题

固定 learner 每个 step 消费的轨迹数，只提高 rollout 最大在途轨迹数 `C`，观察：

1. 适量并发是否能减少 learner 等待并加速训练；
2. rollout 原始吞吐能否等比例转化为可训练数据和 learner 更新；
3. 额外 admission 是否造成版本过期，并浪费已经执行的 actions 和 tokens。

主要指标是端到端 `learner updates/hour`。Raw tokens/s 描述 rollout 侧完成了多少工作；它不是训练进度。

## 3. Tower of Hanoi 正式设置

- 场景：4 盘 Tower of Hanoi，最多 20 actions。
- 模型：Qwen3-4B-Instruct-2507。
- 算法：REINFORCE，`group_size=1`，不包含 GRPO group 队头阻塞语义。
- 资源：4 张 actor GPU + 4 张 rollout GPU，全分离运行。
- learner batch：每 step 固定消费 4 条完整轨迹。
- 每轮生成上限：512 tokens。
- 策略：原生 FIFO + 固定 outstanding watermark；关闭 version-aware admission、priority 和 KV rebuild。
- 版本容忍度：2。
- 并发：`C={4,8,12,16,24,32}`。
- 每个点：30 learner steps，seeds `57/58/59`，去除前 3 个 warmup steps。
- checkpoint：关闭。

原始汇总：`output/tower_hanoi_fifo_load_formal30_summary.csv`。

## 4. Tower 正式结果

下表为 3 seeds 的均值加减标准差：

| C | step 时间 (s) | 更新/小时 | raw tokens/s | trainable tokens/s | stale token 比例 |
|---:|---:|---:|---:|---:|---:|
| 4 | 34.05 +/- 2.84 | 106.23 +/- 9.27 | 162.54 +/- 7.30 | 153.51 +/- 0.11 | 11.4% +/- 10.5% |
| 8 | 24.49 +/- 0.89 | 147.15 +/- 5.36 | 321.74 +/- 17.41 | 198.47 +/- 5.73 | 61.7% +/- 3.0% |
| 12 | 20.73 +/- 0.41 | 173.70 +/- 3.45 | 417.32 +/- 14.09 | 213.65 +/- 3.58 | 72.5% +/- 3.0% |
| 16 | 19.83 +/- 2.85 | 184.00 +/- 26.10 | 500.58 +/- 19.45 | 207.41 +/- 15.80 | 79.4% +/- 2.6% |
| 24 | 20.17 +/- 4.08 | 183.07 +/- 34.07 | 642.26 +/- 38.78 | 188.65 +/- 28.50 | 87.8% +/- 2.4% |
| 32 | 17.60 +/- 0.73 | 204.80 +/- 8.38 | 777.86 +/- 40.94 | 216.45 +/- 6.22 | 87.8% +/- 0.5% |

轨迹层面的转化与浪费：

| C | raw 轨迹/运行 | learner 消费 | 转化率 | stale 轨迹/运行 | stale 前已执行 actions |
|---:|---:|---:|---:|---:|---:|
| 4 | 121.3 | 120 | 98.9% | 1.3 | 23.3 |
| 8 | 140.7 | 120 | 85.3% | 20.7 | 321.3 |
| 12 | 157.3 | 120 | 76.3% | 37.3 | 457.7 |
| 16 | 174.7 | 120 | 68.8% | 54.7 | 614.7 |
| 24 | 205.3 | 120 | 58.6% | 85.3 | 852.7 |
| 32 | 237.0 | 120 | 50.8% | 117.0 | 1018.7 |

## 5. Tower 支持的结论

1. 适量增加并发能够加速训练。`C=4 -> 8` 时平均 step 时间下降约 28%，更新速度提高约 39%。
2. Raw rollout 工作与有效训练进度明显分离。`C=4 -> 32` 时 raw token throughput 提高约 4.8 倍，但 learner 更新速度只提高约 1.9 倍，trainable token throughput 只提高约 1.4 倍。
3. 额外轨迹的转化效率持续下降。并发从 4 提高到 32 后，轨迹转化率从 98.9% 降到 50.8%，平均浪费 actions 从 23.3 增加到 1018.7。
4. 固定高并发不是免费性能。即使端到端训练仍可能变快，大量 rollout GPU 与环境工作没有进入 learner batch。

当前数据不支持以下强结论：

- 不能声称端到端 step 时间存在稳定 U 型曲线；正式数据中 C=32 仍然最快。
- 不能声称高并发一定让 learner 变慢；C=24 相比 C=16 的轻微退化落在较大的跨 seed 波动内。
- 不能从 Tower 单场景推出该现象在所有 AgenticRL workload 中普遍成立。

一个重要实验边界是当前只有 32 个 env groups，C=32 已经触及 producer 数量上限。扫描没有覆盖 `C > num_env_groups` 的更高压力区间。同时 actor 与 rollout 使用完全分离的 GPU，rollout 浪费不会直接争抢 learner GPU。

## 6. WebShop 校准结果

设置：Qwen3-4B、REINFORCE/group_size=1、batch=4、4 actor + 4 rollout GPUs、最多 20 actions、每 action 最多 192 tokens、16K context、checkpoint 关闭。

| C | step 时间 | 更新/小时 | rollout 时间 | train 时间 | learner wait | raw tokens/s | trainable tokens/s | stale token 比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 23.72s | 151.80 | 13.70s | 5.10s | 63.3% | 126.15 | 126.15 | 0% |
| 8 | 14.14s | 254.62 | 3.35s | 5.37s | 32.3% | 157.34 | 104.26 | 63.0% |

这个 calibration 通过 rollout-bound 准入门槛：C=4 时 learner 等待明显，C=4 到 C=8 的 step 时间缩短约 40%，训练与模型更新时间基本稳定。同时 raw throughput 上升约 25%，trainable throughput 却下降约 17%。

但当前只有 4 steps、1 seed，且两个点每 step 的训练 token 数差异较大。它只能说明正式 WebShop 扫描值得做，不能单独用于论文结论。

原始汇总：`output/webshop_rollout_bound_calibration_summary.csv`。

## 7. 历史跨场景 Pilot

在独立的原生 Observation worktree 中，之前已经完成过一组 `8 updates x 1 seed` 的固定 admission pilot：

- learner batch `N=8`；
- 每个版本固定新增 `K={8,10,12,14,16}` 条完整轨迹；
- 横轴为 `K/N={1.0,1.25,1.5,1.75,2.0}`；
- Qwen3-4B-Instruct-2507、REINFORCE/group_size=1、4 actor + 4 rollout GPUs；
- 原生 FIFO，checkpoint 关闭。

各场景均以自己的 `K/N=1` 为 1.0，不比较不同 workload 的绝对吞吐。比较 `K/N=2` 与基线：

| Workload | Raw throughput | Trainable goodput | Updates/hour | Stale token fraction |
|---|---:|---:|---:|---:|
| WebShop | 1.185x | 0.621x | 0.609x | 41.4% |
| SimpleSokoban | 2.112x | 1.057x | 0.942x | 39.6% |
| HotpotQA-query + local retrieval | 1.619x | 0.859x | 0.844x | 40.1% |

三个场景都出现了相同方向：raw rollout throughput 增长，但 trainable goodput 和 learner updates 没有等比例增长；高负载点的 stale work 接近 40%。WebShop 和 Hotpot 的训练推进明显下降；Sokoban 的 raw throughput 增长约 111%，更新速度仍下降约 6%。

现有图：

- `../../../native_observation/results/cross_workload_analysis_internal/within_workload_rates.pdf`
- `../../../native_observation/results/cross_workload_analysis_internal/within_workload_stale_waste.pdf`
- `../../../native_observation/results/cross_workload_analysis_internal/cross_workload.csv`

这组结果可以作为跨机制的一致性 pilot，但证据强度低于 Tower 正式实验：

1. 每个点只有 8 updates、1 个 seed，没有误差线；
2. 本地归档只保留了跨场景汇总和 Hotpot 单场景明细，WebShop/Sokoban 原始 run records 没有完整复制到当前目录，无法仅凭本地副本重新审计每次运行的 GPU 干扰状态；
3. Hotpot 使用 HotpotQA 问题与轻量本地检索语料，不是真实 Wikipedia 检索，也不应称为标准 HotpotQA benchmark；
4. 它采用“每个版本固定新增 K 条轨迹”，与 Tower 的 outstanding cap `C` 是两种不同负载控制方式，不能把横轴直接合并。

因此，历史 pilot 可以支持“该现象值得跨场景正式验证”，不能替代多 seed 正式实验。

## 8. 当前可引用的 Motivation

> Increasing rollout concurrency can accelerate asynchronous AgenticRL training, but raw rollout work grows much faster than effective training progress. As concurrency increases, a growing fraction of admitted trajectories expires after consuming model and environment work. Runtime load should therefore be controlled by trainable conversion and version state, rather than raw rollout throughput alone.

中文表述：

> 增加 rollout 并发可以加速全异步 AgenticRL 训练，但原始 rollout 工作量的增长远快于有效训练进度。随着并发提高，越来越多已经消耗模型推理和环境资源的轨迹在进入 learner 前过期。因此 runtime 不应只追求原始 rollout 吞吐，而应根据可训练样本转化与版本状态控制负载。

## 9. 后续缺口

实验保持暂停时，当前报告可作为阶段性证据。完成跨场景结论仍需要：

1. WebShop 至少 30 steps x 3 seeds 的正式并发扫描；
2. 修正 HotpotQA 的 Hydra seed 参数，并使用真实、可复现的检索语料重新校准；
3. 每个场景分别验证 rollout-bound，不能跨场景比较绝对吞吐；
4. 最终跨场景图使用归一化并发 `C/B` 与场景内部归一化指标，明确区分正式实验和 calibration。

恢复实验时使用 `CrossWorkloadFormalRunbook.md`。WebShop 与 Hotpot 的配置解析已在远端 `xxl_test` 容器通过；配置验证过程没有启动 GPU 进程，也没有生成 checkpoint。

当前跨场景阶段性图位于 `output/pdf/cross_workload_observation_current.pdf`。它明确标记 Tower 为 formal-ready、WebShop 为 calibration、HotpotQA 为 pending，不能作为最终三场景论文图。
