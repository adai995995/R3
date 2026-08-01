# AppWorld 接入与 Observation 实验

## 接入方式

AppWorld 运行在独立的 Python 3.11 sidecar 中，ROLL 继续使用原来的
Python 3.10 环境。每个 ROLL `env_id` 对应一个独立的 AppWorld HTTP
服务端口：

```text
env_id=0  -> 127.0.0.1:18200
env_id=1  -> 127.0.0.1:18201
...
env_id=15 -> 127.0.0.1:18215
```

`AppWorldRunner` 向模型暴露一个 `execute_appworld_code` 工具。每次模型
交互只允许执行一个 AppWorld API，因此一条任务轨迹会自然包含多轮推理
和工具调用。所有 AppWorld API 参数必须按名字传入。

启动环境服务：

```bash
scripts/start_appworld_servers.sh
```

运行单元测试：

```bash
python3 -m pytest -q \
  tests/pipeline/agentic/agent_runner/test_appworld_runner.py
```

## 实验设置

- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- 设备：4 张 actor 训练 GPU + 4 张 rollout GPU
- 任务：AppWorld train split 中 difficulty=3 的 18 个多步骤任务
- AppWorld 环境实例：16
- 每条轨迹最多 16 次推理/工具执行
- 上下文上限：32768 token
- 每次训练需要 4 条轨迹
- staleness 容忍度：2 个版本
- 调度：FIFO
- admission：固定最大在途轨迹数
- 测试点：`C = {4, 8, 16, 32}`
- 每组：6 个 learner step；前 2 个 step 不计入平均耗时
- checkpoint：关闭
- seed：83

运行命令：

```bash
CAPS="4 8 16 32" \
STEPS=6 \
WARMUP=2 \
WORKLOAD=appworld_fifo_observation_32k \
scripts/run_appworld_fifo_observation_pilot.sh
```

## 指标口径

这里只保留三个问题对应的指标：

1. rollout 吞吐：所有 rollout 实际输出 token / rollout 墙钟时间。
2. 端到端平均 step 耗时：`time/step_total`，包含等待 batch、actor
   训练和参数同步。
3. 版本淘汰浪费：只统计 `discard_reason` 以 `version_` 开头的轨迹，
   分别计算已输出 token 和已执行工具调用的总量与每条平均值。

实验结束时仍在运行的轨迹以及 shutdown abort 不计入版本淘汰浪费。

## Pilot 结果

| 最大在途 C | rollout token/s | 平均 step 秒 | 版本淘汰轨迹 | 浪费 token 总量 | 每条浪费 token | 浪费工具调用总量 | 每条工具调用 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 137.50 | 18.25 | 0 | 0 | 0.00 | 0 | 0.00 |
| 8 | 170.29 | 15.01 | 4 | 2911 | 727.75 | 48 | 12.00 |
| 16 | 244.65 | 13.86 | 9 | 6536 | 726.22 | 109 | 12.11 |
| 32 | 392.77 | 17.38 | 27 | 17380 | 643.70 | 313 | 11.59 |

这个 pilot 出现了预期的 sweet spot：

- 从 `C=4` 增加到 `C=16` 时，适度超量采样缩短了平均 step 时间。
- 从 `C=16` 增加到 `C=32` 时，rollout 吞吐继续明显提高，但平均
  step 时间反而增加。
- 被版本淘汰的轨迹平均已经执行约 12 次工具调用，丢掉的不是刚启动
  的浅轨迹，而是已经投入较多计算和环境资源的深轨迹。

这组结果可作为 AppWorld 上的初步 observation；论文结论仍需增加 seed
和更长运行时间。

结果文件：

```text
output/appworld_fifo_observation_32k_seed83_6step_summary.csv
```
