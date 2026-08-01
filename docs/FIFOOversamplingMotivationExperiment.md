# 固定并发超量采样的 Motivation 实验

> **正式结果更新（2026-07-22）：** 本文第 3-6 节记录的是早期 6-step、单 seed pilot，用于定位实验设置。后续 `30 steps x 3 seeds` 正式复现实验没有稳定复现 U 型曲线，不能再用本 pilot 声称“高并发必然拖慢训练”。正式结果与可引用结论见 `AgenticRLObservationInterimResults.md`；pilot 文件仅保留用于追溯。

> **数据版本说明：** 2026-07-21 的首次 pilot 与另一个 8-GPU 任务重叠，已单独保留并标记为 contaminated。本文下表采用 2026-07-22 在 8 张 GPU 完全空闲、所有 GPU 进程均属于 `xxl_test` 时完成的 clean rerun。

## 1. 要回答的问题

这组实验不比较新系统，也不调优 GPU 利用率，只回答三个基础问题：

1. 增加 rollout 并发能否先缩短端到端训练 step？
2. rollout 原始吞吐的增长能否等比例转化为可训练数据？
3. 并发超过平衡点后，是否会因为排队和版本过期反而拖慢训练？

实验固定 learner 每个 step 消费 4 条轨迹，只改变最大在途轨迹数 `C`。因此，`time/step_total` 和每小时训练更新数是主要指标；GPU 利用率不是本实验的目标指标。

## 2. 为什么不把当前 WebShop 配置作为主实验

WebShop 的低并发基线已经接近训练侧瓶颈：

- `C=4` 时，稳态总 step 时间约为 `14.08s`；
- 其中 rollout 等待约为 `4.39s`；
- raw 和 trainable response throughput 都约为 `107 tokens/s`；
- 没有产生 stale 轨迹。

在这个资源配比下，rollout 很快就能凑齐 4 条样本。继续增加并发很难缩短 learner 自身的训练计算，因此只能观察到平台或浪费，无法展示“适量超量采样先带来收益”的左半段。

问题不是 WebShop 不能用于 AgenticRL，而是当前任务长度、batch 大小和 4 张 rollout GPU 的组合让它过早进入 learner-bound 区间。

## 3. Tower of Hanoi Pilot 设置

- 场景：Tower of Hanoi，4 个盘子；最优解至少需要 15 次 action；每条轨迹最多 20 次 action。
- 模型：Qwen3-4B-Instruct-2507。
- 算法：REINFORCE，`group_size=1`，没有 GRPO group 完成语义。
- 资源：4 张 Actor GPU + 4 张 rollout GPU，全分离运行。
- 训练 batch：每个 learner step 固定消费 4 条轨迹。
- 上下文：每轮最多生成 512 tokens。
- 调度：原生 FIFO + outstanding watermark；新系统的 version priority、自适应 admission 和 KV rebuild 均关闭。
- 版本容忍度：`trajectory_staleness_tolerance=2`。
- 并发扫描：`C={4,8,12,16,24,32}`。
- 训练长度：6 个更新，seed=57；统计时去掉第 0 个启动 step。
- checkpoint：完全关闭。

运行入口：

```bash
RUN_LABEL=clean bash scripts/run_tower_hanoi_fifo_load_pilot.sh
python3 scripts/analyze_fifo_load_sweep.py \
  --pattern "tower_hanoi_qwen3_4b_independent_fifo_clean_c*_seed57_6step/terminal_waste.step_6.json" \
  --warmup 1 \
  --output output/tower_hanoi_fifo_load_clean_summary.csv
```

## 4. Clean Pilot 结果

| 最大在途轨迹 C | 平均 step 时间 (s) | 更新/小时 | rollout 转化率 | stale 轨迹 | 浪费 actions |
|---:|---:|---:|---:|---:|---:|
| 4 | 48.82 | 73.74 | 96% | 1 | 17 |
| 8 | **17.86** | **201.59** | 92% | 2 | 27 |
| 12 | 21.00 | 171.46 | 63% | 7 | 58 |
| 16 | 21.72 | 165.73 | 56% | 9 | 111 |
| 24 | 32.28 | 111.51 | 55% | 20 | 101 |
| 32 | 47.78 | 75.35 | 42% | 30 | 176 |

rollout 转化率定义为 `learner consumed trajectories / raw trajectories`。

原始汇总：`output/tower_hanoi_fifo_load_clean/tower_hanoi_fifo_load_clean_summary.csv`

观察图：`output/tower_hanoi_fifo_load_clean/tower_hanoi_fifo_load_clean_observation.pdf`

观察图按“结果、原因、代价”组织：图 (a) 展示端到端训练 step 时间，图 (b) 展示 rollout 转化为 learner 实际消费轨迹的比例，图 (c) 展示最终被丢弃轨迹已经执行的 actions，并在柱顶标出 stale 轨迹数。

受干扰的旧数据仍保存在 `output/tower_hanoi_fifo_load_pilot_summary.csv`，对应 PDF 带有醒目的 shared-GPU 警告，不与 clean 数据混用。

## 5. Clean Pilot 支持的现象

1. **适量并发确实加速训练。** `C=4 -> 8` 时，平均 step 从 `48.82s` 降到 `17.86s`，每小时更新数提高约 2.73 倍。
2. **raw rollout throughput 与训练进展发生分离。** `C=8 -> 16` 时 raw throughput 增长约 86%，但每小时更新数下降约 18%，可训练轨迹 goodput 也下降。
3. **更高并发持续增加 stale 浪费。** stale 轨迹从 `C=8` 的 2 条增加到 `C=32` 的 30 条，被丢弃轨迹已经执行的 actions 从 27 增加到 176；C=32 的更新速度已接近供给不足的 C=4。
4. **trainable token/s 不能单独代表训练进展。** C=32 每个 batch 平均约有 12,988 个训练 tokens，而 C=8 约为 4,796 个；长轨迹会抬高 token/s，但不会让固定 4 条轨迹的 learner batch 更快形成。
5. 当前单 seed 下的平衡点在 `C=8` 附近。它不是应被硬编码的最佳参数，而是证明 runtime 需要根据实际供需和版本浪费动态调节负载。

## 6. Pilot 的边界

这是一组独占 GPU、单 seed、6-step 的定位实验，能够作为 clean pilot，但还不是最终论文统计：

- Tower 轨迹长度方差较大，单 step 的训练 token 数和训练耗时会变化；
- stale token 比例可能被少数超长轨迹放大，因此最终应同时报告轨迹数、actions 和 tokens；
- 各并发点分别重启进程，启动时间未计入稳态 step，但正式实验仍需统一 warmup；
- 当前结果证明该 workload 中固定并发存在明显平衡区间，但尚未证明新系统优于最佳静态参数。

## 7. 下一组正式实验

clean pilot 已完成。下一步扩大统计量：

- `C={4,8,12,16,32}`，必要时补 C=24；
- 50 learner steps；
- 3 个 seeds；
- 每个点报告均值、标准差和 95% 置信区间；
- 同时保留 `time/step_total`、raw/trainable throughput、stale trajectories/actions/tokens 和版本边界状态。

随后再选择 WebShop 和另一个不同长度分布的多轮场景复现同一趋势。只有完成多 seed、跨场景验证后，才进入新系统与最佳静态基线的对比。

## 8. 跨场景实验的瓶颈准入标准

WebShop 和 HotpotQA 先各自运行 `C=B`、`C=2B` 的短 calibration，满足以下条件后才展开正式并发扫描：

1. `C=B` 时，`time/step_rollout` 至少占端到端关键路径时间的 50%，或 learner wait fraction 不低于 30%；
2. `C=B -> 2B` 时，端到端平均 step 时间至少缩短 10%，证明适量并发确实能够缓解 rollout 供给不足；
3. 两个点的 `time/step_train` 和 `time/step_model_update` 不应发生同量级变化，避免把训练侧波动误判为 rollout 收益；
4. calibration 不通过时，优先增加真实轨迹 action 数、生成长度或检索上下文，不能通过人为 sleep 制造结果；
5. 每个正式场景内部按 `C/B` 比较，不跨场景比较绝对 step 时间或吞吐。

当前候选设置：

- WebShop：`B=4`，最多 20 actions，每次最多 192 tokens，16K context；
- HotpotQA with search：`B=8`，最多 8 actions、6 次搜索，每次最多 512 tokens，24K context，retrieval top-k=12；
- 两个场景都使用 `group_size=1`、REINFORCE、4 actor + 4 rollout GPUs、FIFO + outstanding watermark，并关闭 checkpoint。
