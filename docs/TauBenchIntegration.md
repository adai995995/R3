# τ-bench 接入说明

## 接入范围

当前接入使用原始 τ-bench 的 retail 和 airline 环境，包括任务、数据库、工具、状态转移和最终奖励。策略模型通过 R3 的代理接口生成客服动作，工具调用仍由 τ-bench 环境执行。

τ-bench 的隐藏用户也通过同一组 rollout 引擎生成回复，但请求会标记为 `track_trajectory=false`。因此用户模拟器的 token 和消息不会进入策略模型的训练轨迹；相关开销会单独记录在 `traj_environment_model_*` 指标中。

每个策略请求都会携带稳定的 trajectory ID、策略版本、当前版本、已完成 action 数和 trajectory priority。这样该 workload 可以直接进入 version-aware admission、priority 和 KV placement 的统一运行时闭环。

## 固定依赖版本

当前验证使用：

- 仓库：`sierra-research/tau-bench`
- commit：`59a200c6d575d595120f1cb70fea53cef0632f6b`
- 安装位置：`/ufs_500T/xxl/agentic/version_driven/third_party/tau-bench`

容器内安装命令：

```bash
pip install 'litellm==1.41.0'
pip install -e /ufs_500T/xxl/agentic/version_driven/third_party/tau-bench --no-deps
```

## 运行 smoke test

在 `xxl_test` 容器内执行：

```bash
cd /ufs_500T/xxl/agentic/version_driven/R3
python3 examples/start_agentic_pipeline.py \
  --config_path qwen2.5-0.5B-agentic \
  --config_name agent_val_tau_bench_qwen3_4b_smoke
```

默认配置使用 Qwen3-4B、4 张训练卡和 4 张 rollout 卡，执行 2 个训练 step，不保存 checkpoint。可通过 Hydra override 修改实验名、输出目录和训练步数。

Airline 使用独立配置：

```bash
python3 examples/start_agentic_pipeline.py \
  --config_path qwen2.5-0.5B-agentic \
  --config_name agent_val_tau_bench_airline_qwen3_4b_smoke
```

固定依赖版本中的 Airline 仅提供 `test` split。这里把它作为系统 workload 使用，不用该实验比较任务泛化能力或模型准确率。

## 已验证结果

真实两步实验 `tau_bench_qwen3_4b_smoke_real2` 已完成：

- 完成 2 次 learner 更新，进程退出码为 0；
- learner 消费 4 条有效轨迹，其中 2 条的 version age 为 1；
- 记录到 1 次版本边界和 4 条边界时已启动的轨迹；
- 策略轨迹共包含 31 次 action；
- 环境用户调用、策略推理、工具调用和 token 开销被分别统计；
- shutdown timeout 和零进度异常均为 0。

该小负载下，版本边界时的 4 条轨迹已经完成，因此没有未完成 survivor，也没有触发 first-wave KV rebuild。要验证 survivor priority 和 KV working-set rebuilding，需要提高在途上限或延长单条任务，使版本更新发生在轨迹执行过程中。

结果文件：

```text
output/tau_bench_qwen3_4b_smoke_real2/terminal_waste.step_2.json
output/logs/tau_bench_qwen3_4b_smoke_real2/log_rank_DRIVER_0_1.log
```

`terminal_waste` 表示固定训练步数结束时仍未被 learner 消费的异步工作，不应当直接解释为 stale discard。版本过期损失应使用 `async_waste/version_stale_*` 指标。

## FIFO 20-step observation（Retail）

在原生 FIFO、固定在途上限下完成了单 seed、20 learner steps 的 observation。每个 learner step 需要 4 条轨迹，固定 16 个可用环境 worker，只改变最大在途轨迹数 `C`。前 2 个 warmup steps 不计入均值，表中统计后 18 个 steps：

| C | 平均 step 时间 | 中位 step 时间 | P95 step 时间 | raw rollout tokens/s | stale 轨迹 | stale actions |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 21.11 s | 21.16 s | 26.35 s | 120.72 | 0 | 0 |
| 6 | 15.91 s | 15.01 s | 21.21 s | 159.34 | 1 | 11 |
| 8 | 13.38 s | 14.01 s | 15.63 s | 190.21 | 8 | 96 |
| 10 | 12.85 s | 12.71 s | 16.89 s | 242.82 | 19 | 221 |
| 12 | 12.44 s | 11.20 s | 18.31 s | 252.50 | 30 | 325 |
| 16 | 13.09 s | 11.61 s | 17.94 s | 352.67 | 52 | 544 |
| 24 | 12.22 s | 10.22 s | 18.96 s | 443.60 | 95 | 949 |
| 32 | 15.10 s | 10.39 s | 32.74 s | 464.26 | 149 | 1390 |

从 `C=4` 增加到 `C=12`，平均 learner step 时间缩短 41.1%，说明适度提高并发能明显加快凑 batch。`C=10` 到 `C=24` 形成宽平台区：平均 step 时间只在 12.22--13.09 秒之间变化，但 raw rollout 吞吐和 stale waste 持续增长。仅从 `C=12` 增加到 `C=24`，raw throughput 提高 75.7%，平均 step 时间却只缩短 1.8%，stale actions 从 325 增至 949。

`C=32` 出现右侧回退：相对 `C=24`，raw throughput 只再提高 4.7%，平均 step 时间却增加 23.6%，P95 从 18.96 秒升至 32.74 秒；stale actions 增至 1390。完整曲线支持“并发不足时增加负载有收益，随后训练推进进入平台，极端 backlog 最终造成长尾和回退”。

需要注意，系统只有 16 个环境 worker。`C<=16` 主要表示实际运行负载；`C>16` 主要表示 admission 后形成的 queued backlog，而不是 24/32 条轨迹同时执行。这一区间用于观测过度 admission 和版本浪费压力。

八个负载点均完成 20/20 steps，runner 正常退出，未生成 checkpoint，且实验启动前没有其他 GPU 计算进程。目前仍只有 1 个 seed；正式论文结论还需要对代表点做多 seed 复现。

汇总与图片：

```text
output/tau_bench_fifo_observation_seed73_20step_summary.csv
output/pdf/tau_bench_fifo_observation_seed73_20step_full_sweep.pdf
output/pdf/tau_bench_fifo_observation_seed73_20step_full_sweep.png
```

## FIFO 20-step observation（Airline）

Airline 使用同一套原生 FIFO 基线和硬件配置：Qwen3-4B-Instruct-2507、REINFORCE、`group_size=1`、每个 learner step 消费 4 条轨迹、4 张 actor GPU、4 张 rollout GPU、16 个环境 worker，版本容忍度为 2。实验只改变最大在途轨迹数 `C`，运行 20 个 learner steps，剔除前 2 个 warmup steps，seed 为 74，不保存 checkpoint。

12-action 质量短测中有 75% 的轨迹撞到最大轮数。Airline 正式实验因此使用 20-action 上限；短测中每条轨迹平均执行 12--16 次推理和 5.25--7.25 次工具调用，训练序列仍在 12,288 token 上限内。它比 Retail 更能代表真实的长程 Agent workload。

| C | 平均 step 时间 | 中位 step 时间 | P95 step 时间 | raw rollout tokens/s | stale 轨迹 | stale actions |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 33.57 s | 33.92 s | 45.60 s | 131.15 | 0 | 0 |
| 6 | 25.78 s | 25.59 s | 33.62 s | 187.65 | 6 | 115 |
| 8 | 20.81 s | 21.09 s | 29.23 s | 262.41 | 15 | 271 |
| 10 | 19.21 s | 19.11 s | 28.38 s | 312.06 | 20 | 331 |
| 12 | 17.71 s | 17.13 s | 27.12 s | 347.08 | 34 | 567 |
| 16 | 15.91 s | 15.22 s | 23.17 s | 441.91 | 50 | 799 |
| 24 | 18.87 s | 17.70 s | 32.47 s | 510.28 | 87 | 1309 |
| 32 | 21.52 s | 19.65 s | 31.35 s | 583.13 | 129 | 1787 |

`C=4` 到 `C=16` 是合理的左侧下降区间：平均 learner step 时间缩短 52.6%，说明适度增加在途轨迹确实能减少等待长轨迹尾部的时间。但 raw throughput 和 stale work 增长得更快，版本浪费从 0 增至 50 条轨迹、799 个已经执行的 action。

超过 16 个环境 worker 后出现明确反噬。`C=16` 增至 `C=24` 时，raw rollout throughput 又提高 15.5%，平均 learner step 时间却增加 18.6%，P95 增加 40.1%；stale actions 从 799 增至 1309。`C=32` 的 raw throughput 比 `C=16` 高 31.9%，但 learner step 反而慢 35.2%，stale actions 达到 1787。

从 `C=8` 到 `C=32`，raw rollout throughput 提高 122.2%，平均 learner step 时间却只变化 3.4%，同时 stale actions 从 271 增至 1787。这直接表明 rollout 侧做了更多工作，并不代表训练进程得到等比例加速。

八个负载点均完成 20/20 steps，runner 正常退出，配置有效性检查通过；checkpoint、重复终态记录和 shutdown timeout 均为 0。

汇总与图片：

```text
output/tau_bench_airline_fifo_observation_seed74_20step_summary.csv
output/pdf/tau_bench_airline_fifo_observation_seed74_20step_full_sweep.pdf
output/pdf/tau_bench_airline_fifo_observation_seed74_20step_full_sweep.png
```

## 跨 workload 结论

Retail 和 Airline 都显示相同的总体规律：低并发时增加在途轨迹会缩短 learner step；继续增加后，raw rollout throughput 和 stale work 仍持续上升，但训练推进进入平台，极端 backlog 最终让平均时间和长尾回退。

两个 workload 的最佳区域并不相同：Retail 是较宽的 `C=10--24` 平台，Airline 在 `C=16` 左右达到最低平均 step 时间。这说明系统不应该硬编码一个固定的超量采样比例，而应根据当前轨迹长度、版本年龄、完成速率和队列压力动态控制 admission。
