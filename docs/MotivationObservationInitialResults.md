# Motivation Observation 初步结果

## 实验状态

- 日期：2026-07-23。
- 分支：`version_driven`。
- 容器：`xxl_test`，单机 8 卡。
- Workload：AppWorld、τ-bench retail。
- 模型：Qwen3-4B-Instruct-2507。
- 算法：REINFORCE，`group_size=1`。
- GPU 划分：4 卡 actor/learner，4 卡 rollout。
- 每个 policy update 消费 4 条轨迹。
- 每条轨迹最多 16 actions，context 上限 32768。
- Baseline：FIFO、固定 outstanding watermark、staleness tolerance=2。
- 并发矩阵：`C={4,8,16,32}`，每组 20 steps，前 2 steps 作为 warmup。
- 因果对照：AppWorld `C={16,32}`，staleness tolerance=1000。
- 全部 10 个正式 run 成功退出，未保存 checkpoint。

## Baseline 主结果

### AppWorld

| C | Rollout output tok/s | 平均 step (s) | Stale / admitted | Stale output tokens / traj | Stale actions / traj | Stale tool calls / traj | Stale 引擎工作占比 | Survivor re-prefill tokens / update |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 135.26 | 18.71 | 0.00% | 0.00 | 0.00 | 0.00 | 0.00% | 10,402 |
| 8 | 193.55 | **15.27** | 11.11% | 632.73 | 11.45 | 10.27 | 10.73% | 17,417 |
| 16 | 295.22 | 16.01 | 37.66% | 694.36 | 13.03 | 11.78 | 40.35% | 23,810 |
| 32 | **439.12** | 17.84 | 57.74% | 615.42 | 12.08 | 10.92 | 59.65% | 27,831 |

### τ-bench retail

| C | Rollout output tok/s | 平均 step (s) | Stale / admitted | Stale output tokens / traj | Stale actions / traj | Stale tool calls / traj | Stale 引擎工作占比 | Survivor re-prefill tokens / update |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 118.74 | 22.86 | 0.00% | 0.00 | 0.00 | 0.00 | 0.00% | 13,124 |
| 8 | 180.04 | 15.78 | 9.28% | 1,101.89 | 14.67 | 5.00 | 15.05% | 14,741 |
| 16 | 295.55 | **15.47** | 35.57% | 770.83 | 11.13 | 4.40 | 44.67% | 21,872 |
| 32 | **444.30** | 18.81 | 57.41% | 672.64 | 10.64 | 3.99 | 61.59% | 27,240 |

## Relaxed-staleness 对照

| C | 设置 | Rollout output tok/s | 平均 step (s) | Stale / admitted |
|---:|---|---:|---:|---:|
| 16 | tolerance=2 | 295.22 | 16.01 | 37.66% |
| 16 | tolerance=1000 | 202.02 | **13.88** | 0.00% |
| 32 | tolerance=2 | 439.12 | 17.84 | 57.74% |
| 32 | tolerance=1000 | 226.92 | **13.61** | 0.00% |

该对照只用于拆分系统成本。放宽 off-policy 容忍度会改变 learner 可接受的数据范围，不是最终系统方案。

## 初步 Observation

1. 小幅超量采样是有益的。AppWorld 从 `C=4` 增至 `C=8`、τ-bench 从 `C=4` 增至 `C=8/16` 时，平均 step 明显下降。
2. Rollout 吞吐与训练推进在高并发下分离。两个 workload 的最高 raw throughput 都出现在 `C=32`，但平均 step 均明显慢于各自最佳点。
3. 最佳固定并发与 workload 有关。当前单 seed 中 AppWorld 最佳为 `C=8`，τ-bench 最佳为 `C=16`。
4. 被淘汰的不是浅轨迹。出现 stale 后，平均已完成约 10.6 至 14.7 个 actions，并产生数百至上千 output tokens。
5. Stale 计算占比随并发急剧上升。`C=32` 时 AppWorld 为 59.65%，τ-bench 为 61.59%。
6. 跨版本轨迹是常态。baseline 各组有约 86.9% 至 92.0% 的 admitted trajectories 至少跨过一次 policy update。
7. 版本更新后的恢复工作随并发增大。两个 workload 的 survivor re-prefill exposure 均从每次 update 约 1 万 tokens 增长到约 2.7 万 tokens。
8. Relaxed-staleness 对照表明，baseline 的高 raw throughput 中包含大量“淘汰后继续补入”的无效 churn：raw throughput 降低时，训练 step 反而更快。

## 数据位置

统一目录：

`output/motivation_v2_combined_20step_analysis/`

- `primary_metrics.csv`：论文主指标。
- `run_summary.csv`：完整 run 级指标。
- `stale_trajectories.csv`：435 条 stale trajectory 的逐条进度与成本。
- `boundary_recovery.csv`：700 条 survivor 首请求恢复记录。
- `manifest.json`：原始 report 列表与指标语义。

每个 run 的原始配置、日志、GPU 监控和 terminal report 位于：

`output/motivation_v2_<workload>_<scenario>_tol<tolerance>_c<C>_seed<seed>_20step/`

## 限制

- 当前是单 seed、20 steps 的初步 observation，足以验证趋势，不足以作为最终论文置信区间。
- `engine_step_seconds_attributed` 是按实际 scheduled-token share 去重后的 vLLM V1 引擎墙钟时间代理，包含调度开销，不等于 CUDA kernel profiler 的纯 GPU 时间。
- τ-bench 的本地工具函数执行很快，因此 tool wall time 很小；其环境与用户模拟成本主要体现在 environment wall time。
