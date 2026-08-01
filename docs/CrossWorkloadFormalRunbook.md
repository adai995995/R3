# 跨场景 Observation 正式实验 Runbook

## 1. 当前状态

新 GPU 实验暂时暂停。以下准备已经完成：

- WebShop 与 Hotpot runner 支持配置解析模式，不启动 Ray、模型或 GPU；
- 两个 runner 启动前都会等待 8 张 GPU 完全空闲，并保存 `gpu_preflight.txt`；
- 配置验证器强制检查 FIFO、outstanding watermark、REINFORCE、`group_size=1`、4+4 GPU 分离和 checkpoint 全关闭；
- 每次运行结束必须存在非空 terminal report，checkpoint 目录不得包含文件；
- runner 正常结束或异常退出时都会停止 Ray；
- Hotpot 的 Hydra `rollout_seed` 覆盖问题已修正；
- Hotpot 已改为最多 6 actions、4 次搜索、retrieval top-k=4，避免一次超长检索占满上下文。

配置解析验证已在远端 `xxl_test` 容器通过：

```bash
VALIDATE_ONLY=1 STEPS=1 SEEDS=61 CAPS=4 RUN_LABEL=configcheck_v3 \
  bash scripts/run_webshop_rollout_bound_sweep.sh

VALIDATE_ONLY=1 STEPS=1 SEEDS=71 CAPS=8 RUN_LABEL=configcheck_v3 \
  bash scripts/run_hotpot_rollout_bound_sweep.sh
```

该验证不产生 checkpoint，也不启动 GPU 进程。

## 2. 恢复实验后的执行顺序

### 2.1 WebShop 正式扫描

WebShop calibration 已经通过 rollout-bound 门槛。先运行：

```bash
STEPS=30 \
SEEDS="61 62 63" \
CAPS="4 8 16 32" \
RUN_LABEL=formal30 \
RUN_TIMEOUT_SECONDS=10800 \
bash scripts/run_webshop_rollout_bound_sweep.sh
```

四个点分别覆盖无超量、适量、中度和高度在途负载。若 C=32 仍未进入收益平台，不直接声称存在 U 型，而是根据 raw/trainable 分离和 stale waste 决定是否补 C=24 或提高 env group 上限后增加更高负载点。

### 2.2 Hotpot calibration

Hotpot 在正式运行前必须满足两个额外条件：

1. `http://127.0.0.1:8001/health` 可用；
2. 明确语料来源。若继续使用轻量本地语料，结果名称必须写成 `HotpotQA-query + local retrieval`，不能称为标准 HotpotQA benchmark。

先运行短 calibration：

```bash
STEPS=4 \
SEEDS=71 \
CAPS="8 16" \
RUN_LABEL=calibration_topk4 \
RUN_TIMEOUT_SECONDS=10800 \
bash scripts/run_hotpot_rollout_bound_sweep.sh
```

只有满足以下门槛才展开正式矩阵：

- C=8 的 learner wait fraction 不低于 30%，或 rollout 时间占 step 关键路径至少 50%；
- C=8 到 C=16 的 step 时间至少缩短 10%；
- train 和 model-update 时间没有同量级变化；
- 轨迹包含真实多轮搜索，而不是普遍一次 search 结束。

通过后再根据 calibration 选择四个并发点并运行 30 steps x 3 seeds。不要在看到 calibration 之前预设最佳 C。

## 3. 每次运行的有效性证据

每个 output 目录至少应包含：

- `driver.log`；
- `runner.status`，成功运行必须为 0；
- `gpu_preflight.txt`，启动前 compute process 列表为空；
- `gpu_process_monitor.csv`，每 30 秒记录一次整个 run 中的 GPU compute process；
- `resolved_config.yaml`，由与真实入口相同的 Hydra compose 和 dataclass 转换生成；
- `terminal_waste.step_N.json`；
- resolved config 或足以恢复所有命令行 overrides 的 runner 记录；
- 空的或不存在的 `checkpoints/` 目录。

分析时必须同时检查：

- 完成 learner updates 数等于配置值；
- learner 确实执行反向传播和参数更新；
- raw/trainable token throughput、updates/hour、learner wait fraction；
- stale trajectories、stale tokens、stale 前 actions 和 tool calls；
- shutdown timeout、terminal tail 和工具错误；
- 各场景是否满足 rollout-bound 准入条件。

`gpu_preflight.txt` 用于证明启动时独占，`gpu_process_monitor.csv` 用于检查运行期间是否出现额外进程。分析器要求正式 run 至少包含两个 monitor 时间点；发现无法归属到本实验的进程时，该 run 必须作废并重跑。

## 4. 跨场景绘图规则

- 每个 workload 在自己的 C/B=1 点归一化，不比较场景间绝对 token rate；
- 正式图使用 3 seeds 的均值与 95% 置信区间；
- 主图展示 learner updates/hour、raw 与 trainable throughput 的分离、stale work；
- Tower 使用 outstanding cap C，旧原生 pilot 使用 fixed admission K。两种横轴语义不同，除非重新跑成同一控制方式，否则不能画在同一条曲线上；
- 单 seed pilot 与正式实验必须使用不同标题和图注。

统一聚合入口：

```bash
python3 scripts/aggregate_cross_workload_observation.py \
  --workload Tower-of-Hanoi:4:formal:output/tower_hanoi_fifo_load_formal30_summary.csv \
  --legacy-evidence Tower-of-Hanoi \
  --workload WebShop:4:calibration:output/webshop_rollout_bound_calibration_summary.csv \
  --pending HotpotQA \
  --min-seeds 3 \
  --min-steps 30 \
  --output output/cross_workload_observation_current.csv \
  --audit output/cross_workload_observation_current.audit.json
```

当前 Tower 正式实验早于自动 preflight/monitor 门禁，因此通过 `--legacy-evidence` 显式标记为同期人工 GPU 独占审计，不能用于任何新 run。WebShop 正式数据生成后将 `calibration` CSV 替换为正式汇总且状态改为 `formal`；Hotpot 完成后删除 `--pending` 并增加对应 workload。聚合器会按 seed 配对归一化，并自动检查：

- C=B 与 C=2B 是否通过 rollout-bound gate；
- seed 数和每个并发点的 seed 集合是否一致；
- 配置 steps 是否全部完成；
- shutdown timeout 是否为 0；
- 新 run 的 GPU/checkpoint/config 有效性证据是否完整。

只有 `formal_ready=1` 的 workload 才能进入最终跨场景正式图。

当前阶段性 evidence map：

```bash
python3 scripts/plot_cross_workload_observation.py \
  output/cross_workload_observation_current.csv \
  output/cross_workload_observation_current.audit.json \
  output/pdf/cross_workload_observation_current.pdf
```

该图会保留 calibration 与 pending 行，但带有 `PROVISIONAL EVIDENCE MAP` 标记。论文最终图必须增加严格门禁：

```bash
python3 scripts/plot_cross_workload_observation.py \
  output/cross_workload_observation_formal.csv \
  output/cross_workload_observation_formal.audit.json \
  output/pdf/cross_workload_observation_formal.pdf \
  --require-all-formal
```

只要任一 workload 仍为 calibration、pilot、pending，或者没有通过 formal readiness 审计，命令就会拒绝生成最终图。
