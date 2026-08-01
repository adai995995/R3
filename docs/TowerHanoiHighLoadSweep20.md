# Tower of Hanoi 高并发粗扫（20 Steps）

## 实验目的

继续增大最大在途轨迹数，寻找 rollout 吞吐继续提高、但端到端训练速度不再改善的负载区间。

核心训练进度指标是端到端 learner step 时间，不使用 learner 消费 token 速率判断训练是否加速。

## 固定条件

- Workload：Tower of Hanoi，4 个盘，最多 20 actions
- 模型：Qwen3-4B-Instruct-2507
- 算法：REINFORCE，`group_size=1`
- 每次 learner 更新：4 条完整轨迹
- 版本过期容忍度：2
- GPU：4 张 actor training + 4 张 rollout，完全分离
- 调度：FIFO + outstanding watermark
- 环境组：128
- Seed：60
- 每个配置：20 learner steps
- 稳态统计：排除 step 0-4，统计 step 5-19
- Checkpoint：关闭

扫描 `C={32,48,64,96,128}`，其中 C 是最大在途轨迹数。

## 结果

| C | 稳态平均 step (s) | 中位数 (s) | P95 (s) | 20 steps 累计时间 (s) | Updates/hour | Raw rollout tokens/s | Stale 轨迹 | 浪费 actions | Stale token 比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16.87 | 18.67 | 22.79 | 350.14 | 213.4 | 720.6 | 96 | 649 | 87.8% |
| 48 | 16.41 | 18.71 | 21.25 | 356.01 | 219.4 | 1010.2 | 172 | 1065 | 93.0% |
| 64 | **14.64** | 18.02 | 19.97 | **329.93** | **245.9** | 1245.7 | 189 | 1203 | 93.4% |
| 96 | 15.90 | 19.06 | 19.84 | 350.87 | 226.4 | 1414.0 | 270 | 1505 | 94.3% |
| 128 | 14.84 | 10.99 | 20.45 | 338.13 | 242.7 | 2004.2 | 391 | 2141 | 96.9% |

## 当前 Observation

1. 从 C=32 增大到 C=64，rollout 吞吐提高 72.9%，平均 learner step 缩短 13.2%。适度提高并发确实能够加快训练。
2. C=64 是本轮平均 step 和 20-step 累计时间的最佳点。
3. 从 C=64 增大到 C=96，raw rollout 吞吐继续提高 13.5%，但平均 learner step 反而增加 8.6%。
4. 从 C=64 增大到 C=128，raw rollout 吞吐提高 60.9%，但平均 learner step 慢 1.3%，20-step 累计时间增加 2.5%。
5. 同一过程中，stale 轨迹从 189 增至 391，浪费 actions 从 1203 增至 2141。额外 rollout 工作没有转化为更快的训练更新。

因此，这轮数据第一次直接观察到了：

> rollout 侧继续变忙、原始生成吞吐继续升高，但端到端训练速度在中等负载后饱和并出现反复；继续超量采样主要增加版本过期和无效计算。

## 边界

这是单 seed、20-step 的粗扫，用于定位候选拐点，不是论文最终统计。C=96 与 C=128 不完全单调，说明高负载区间存在较大 step 方差。正式确认应只复测 `C={64,96,128}`，使用更长运行和多个 seed。

本轮没有保存 GPU preflight、持续 GPU monitor 和 resolved config 文件，因此自动审计字段 `validity_evidence_complete=0`。运行期间人工确认启动前后 GPU 空闲；五组均完成 20 steps、runner status 为 0、checkpoint 文件数为 0、shutdown timeout 为 0。

## 数据

- 汇总：`output/tower_hanoi_highload20_summary.csv`
- 原始目录：`output/tower_hanoi_qwen3_4b_independent_fifo_highload20_c{32,48,64,96,128}_seed60_20step/`

