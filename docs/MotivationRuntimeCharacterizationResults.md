# AgenticRL Runtime Characterization Results

## 1. 实验目的

本轮实验补充三个基础 observation：

1. 固定最大在途轨迹数时，不同 policy update 的下一批有效样本形成时间是否仍明显变化。
2. 被版本淘汰的未完成轨迹中，是否存在已经接近完成、理论上可能被挽救的轨迹。
3. 每次 policy refresh 是否产生重复的 survivor full re-prefill 和 rollout 恢复过程。

本轮只运行原生 FIFO baseline，不启用 version-aware admission、trajectory priority 或 KV rebuild 优化。

## 2. 实验设置

- Workload：τ-bench Airline
- Model：Qwen3-4B
- Hardware：单机 8 GPU，4 learner GPU + 4 rollout GPU
- Staleness tolerance：2
- Max actions：20
- Checkpoint：关闭

两组运行：

- Queue/progress profile：C=32，50 policy steps，seed 74/75/76
- KV refresh profile：C=16，30 policy steps，seed 74/75/76

## 3. 固定 C 下的动态变化

C=32 的 3 个 seed 共得到 144 个有效 update：

- 下一批有效训练数据形成时间：平均 10.21 s，中位数 7.69 s，P95 25.65 s。
- 最小值 0.018 s，最大值 67.07 s。
- ready trajectory 数与 next-batch latency 的 Spearman 相关系数为 -0.359。

结论：即使 workload 和 C 都固定，不同 update 的未来有效供给仍有很大波动。固定 outstanding count 不能完整描述 pool 的可训练供给能力。

## 4. 接近完成但过期的轨迹

C=32、50-step 数据中，共有 94 条轨迹在未完成状态下过期：

- 平均完成进度 70.1%，中位数 70%。
- 56.4% 已完成至少 70%。
- 17.0% 已完成至少 90%。
- 33.0% 只剩不超过 4 个 action。

C=16、30-step 数据中，共有 65 条轨迹在未完成状态下过期：

- 平均完成进度 71.9%。
- 58.5% 已完成至少 70%。
- 20.0% 已完成至少 90%。
- 36.9% 只剩不超过 4 个 action。
- 过期时 78.5% 处于 policy inference，21.5% 处于 environment-model 阶段。

但当前运行中的实际推理排队很轻：

- vLLM request queue 平均约 0.03 ms。
- Router scheduling wait 平均约 0.008-0.009 ms。

因此当前数据只证明“存在接近完成却过期的轨迹”，尚不能证明这些轨迹是因为 FIFO inference queue 排队而过期，也不能直接证明 priority scheduler 可以挽救它们。需要额外的受控 inference-pressure 实验。

## 5. Policy Refresh 的 KV 恢复成本

KV profile 共覆盖 87 次真实 policy refresh：

- 每次 refresh 平均有 9.64 条 survivor 发出该 epoch 的第一次请求。
- 每次平均有 3.93 条 survivor 发生 full re-prefill。
- 每次平均重新 prefill 40,148 tokens，P90 为 50,097 tokens。
- 每次 survivor prefill 的 attributed engine-step time 平均为 0.839 s。
- survivor 首次完成请求的 P50 平均为 2.72 s，P90 平均为 4.32 s。
- refresh 后恢复到 refresh 前 90% decode throughput 的估计时间平均为 1.51 s。
- 下一批有效样本形成前，平均又有 2.20 条轨迹过期，浪费约 2,976 output tokens。

这些计时来自 request/engine-step wall-time attribution，不是 CUDA kernel trace。归一化 throughput 也会受 refresh 附近请求稀疏和 burst 影响，适合描述反复出现的恢复现象，不应解释成精确 GPU kernel 开销。

## 6. Prefix Working Set

当前原生 FIFO 在每个 worker 的第一个 engine batch 中平均只有约 1.14 个请求，因此“首批覆盖多少 prefix cluster”几乎退化为单请求问题：

- 128-2048 token 深度处，survivor 基本共享同一个公共前缀。
- 4096 token 深度处，每 worker 平均 2.45 个 survivor、2.31 个 prefix cluster。
- 但实际 first batch 太小，当前数据没有显示出显著的 FIFO-vs-oracle cluster coverage gap。

这说明 refresh 后确实存在 KV working-set 重建成本，但当前 native dispatch 没有形成一个足够大的、可直接比较 prefix 选择策略的 first wave。后续需要显式定义 first-wave window 或受控 coalescing batch，再比较 FIFO 与 offline oracle。

## 7. 当前可以写入 Motivation 的结论

可以写：

- 同一 workload、同一固定 C 下，不同 update 的有效样本形成延迟仍有数量级波动。
- 一部分过期轨迹已经完成 70%-90%，不是全部都在浅层被淘汰。
- 每次 policy refresh 都会反复产生约 40K survivor re-prefill tokens，并伴随可见的 rollout 恢复区间。

暂时不能写：

- FIFO inference queue 是这些近完成轨迹过期的主要原因。
- 当前 priority scheduler 已经能够挽救这些轨迹。
- 当前 first-wave prefix 策略相对 FIFO 有明显 cluster coverage 优势。

## 8. 受控 Inference Queue Pressure Pilot

为了判断 scheduler 是否存在潜在挽救空间，额外运行一组明确的 synthetic pressure：

- τ-bench Airline，C=32，32 个 active env group。
- FIFO，staleness tolerance=2，15 steps，seed 78。
- 每个 vLLM worker 的 `max_num_seqs` 限制为 2。

结果：

- 未完成即过期轨迹 58 条，平均完成进度 64.1%。
- 50.0% 已完成至少 70%，29.3% 只剩不超过 4 个 action。
- 每条过期轨迹累计 vLLM request queue time 平均 15.67 s，中位数 8.95 s。
- 已完成至少 70% 的 29 条轨迹，累计 queue time 平均 24.52 s。
- 只剩不超过 4 个 action 的 17 条轨迹，累计 queue time 平均 25.93 s。
- 已完成至少 90% 的 7 条轨迹，累计 queue time 平均 32.20 s。
- next-batch latency 平均 19.78 s，P95 52.64 s。

这组结果证明：当 inference queue 确实成为瓶颈时，一部分接近完成但最终过期的轨迹曾花费大量时间排队，执行顺序存在潜在优化空间。它仍不是 priority scheduler 的效果证明；“其中多少条真的能被挽救”需要 trace replay 或 oracle ordering。

该结果不能直接推广为默认 Airline workload 的行为，因为 `max_num_seqs=2` 是人为施加的受控压力。

## 9. 下一步最小实验

1. 使用同一条 pressure trace 离线重放 FIFO 与 urgency ordering，计算“本可完成但实际过期”的 oracle salvage fraction。
2. 在不改变 admission 数量的前提下运行 FIFO 与 version urgency ordering，验证 `priority_queued_requests > 0` 并比较实际 salvage fraction。
3. 将 first wave 明确定义为 refresh 后每 worker 的前 K 个 survivor request，或引入固定的短 coalescing window，再比较 FIFO 与 prefix-diverse oracle。
4. 在第二个长轨迹 workload 上重复受控 queue pressure，检查现象是否依赖 Airline。

## 10. 数据位置

- Queue/progress 原始报告：
  `output/motivation_v4_queue_profile_tau_airline_baseline_tol2_c32_seed{74,75,76}_50step/`
- Queue/progress 汇总：
  `output/motivation_v4_runtime_queue_profile_analysis/`
- KV 原始报告：
  `output/motivation_v4_kv_profile_tau_airline_baseline_tol2_c16_seed{74,75,76}_30step/`
- KV 汇总：
  `output/motivation_v4_kv_refresh_profile_analysis/`
- 修正 phase 后的 runtime characterization：
  `output/motivation_v4_kv_runtime_characterization/`
- 受控 engine queue pressure：
  `output/motivation_v5_engine_queue_pressure_tau_airline_baseline_tol2_c32_seed78_15step/`
- 受控 pressure 汇总：
  `output/motivation_v5_engine_queue_pressure_analysis/`
- 论文风格 Observation 图（4 张独立 PDF/PNG + 1 张总览）：
  `output/motivation_observation_figures_v4/`
