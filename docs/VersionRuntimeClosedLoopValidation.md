# Version-aware Runtime 闭环验证记录

更新时间：2026-07-19

## 1. 本轮闭环的定义

当前闭环不是指已经找到最优策略，而是指系统已经能够重复执行以下流程：

1. rollout runtime 收集轨迹版本、执行进度、GPU/context 投入和 learner 等待状态；
2. 统一控制器生成 admission、轨迹优先级和 KV rebuilding 决策；
3. Router 在线接收同一版本内的计划修订并执行优先调度、worker placement 和更新后首批请求重组；
4. 执行结果回流到下一次控制决策；
5. 最终报告区分有效样本、占位样本、过期浪费、边界 re-prefill 和 learner 等待。

## 2. 已完成的系统能力

- 自适应 admission reserve，并支持 learner 等待时的小步在线补量；
- 接近完成、接近过期、已有 GPU 投入的轨迹优先调度；
- 同一版本内在线修订 Router 计划，不触发额外 cache epoch；
- 参数更新后的 KV working-set rebuilding 和首波请求 coalescing；
- rebuild 请求使用独立的有界突发并发，并通过 request ID 在 vLLM EngineCore
  空闲入口执行 5ms 的首批收集；
- vLLM scheduler 返回真实 batch ID/size，Router 不再用逻辑 wave 冒充引擎 batch；
- 引擎 KV reset 和 prefix-cache 计数反馈会反向清除 Router 中失效的 placement 状态；
- admission、priority、KV placement 共用同一份 version runtime plan；
- stale token、valid goodput、placeholder、re-prefill、learner wait 和 Router KV 指标；
- trace testbed 直接复用生产控制函数，减少模拟器与线上实现偏离。

## 3. 自动化验证

最终定向回归结果：`75 passed, 1 deselected`。

被跳过的旧测试会启动并长期保留 Ray actor，不属于本轮控制逻辑。新增测试覆盖：

- 在线 admission 补量与 plan revision；
- Router 在线更新计划且不切换 cache epoch；
- priority coalescing；
- 实际排队、coalescing 和请求重排三类指标的分离；
- KV rebuild wave coalescing；
- 普通 priority 并发与 rebuild 突发并发的隔离；
- vLLM EngineCore request-ID marker 和真实 scheduler batch 元数据；
- 引擎 reset 后 Router KV shadow state 的失效处理；
- GPU-invested working set；
- valid/placeholder goodput 与异常检测；
- stale token 增量记账，防止迟到事件重复累计。

## 4. Trace 验证

产物：`output/version_runtime_closed_loop_trace_seed48.json`

在同一条 synthetic trace 上，FIFO 与统一 runtime 的主要结果如下：

| 指标 | FIFO | Unified runtime |
| --- | ---: | ---: |
| admitted trajectories | 329 | 217 |
| learner-consumed trajectories | 52 | 126 |
| learner shortfall | 188 | 114 |
| stale trajectories | 245 | 75 |
| stale inference tokens | 649,056 | 98,304 |
| saved prefill ratio | 28.37% | 75.93% |

这组结果只用于验证控制路径和预期方向，不作为论文效果结论。

## 5. 真实 AgenticRL 闭环验证

配置：`examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_version_runtime_closed_loop_pressure_6step.yaml`

场景为 WebShop，模型为 Qwen3-4B-Instruct-2507，单机 8 卡中 4 卡训练、4 卡 rollout，rollout batch size 为 4，完成 6 个真实训练 step；checkpoint 保存关闭。

主要结果：

- learner 消费 24 条有效轨迹，占位轨迹 0；
- 同一版本内发生 1 次在线 plan revision，追加 2 条轨迹；
- Router 完成 173 次 version-aware 调度，其中 72 次真实排队、16 次请求重排；
- 166 个请求带有引擎 KV 反馈，覆盖 95.95% 的 Router 决策；
- 观测到 26 次引擎 KV reset，并触发 35 次 Router shadow-state 失效；
- raw response throughput 为 114.39 token/s，trainable response goodput 为 100.93 token/s；
- learner 等待 39.01 秒，占 rollout 阶段 39.59%；
- shutdown timeout 和 rollout cancel 均为 0；
- 未生成 checkpoint 文件。

报告：`output/webshop_qwen3_4b_version_runtime_closed_loop_pressure_6step/terminal_waste.step_6.json`

## 6. 真实 vLLM 首批组 batch 验证

配置：`examples/qwen2.5-0.5B-agentic/agent_val_webshop_qwen3_4b_version_runtime_engine_batch_probe_2step.yaml`

该实验保持相同模型和 4+4 GPU 布局，将 rollout batch size 提高到 8，使跨版本 survivor 数量能够超过 rollout engine 数量。它只验证组 batch 路径，不用于比较最终训练收益。

主要结果：

- 完成 2 个真实训练 step，learner 消费 16 条有效轨迹；
- 发生 3 次在线 top-up，共追加 6 条轨迹；
- 18 条 rebuild 请求全部完成 EngineCore request-ID 登记；
- 5 条 survivor 首请求被引擎报告为 4 个 scheduler batch；
- 其中 1 个 batch 包含 2 条 survivor，真实最大 batch size 为 2；
- Router 发生 124 次真实排队和 15 次请求重排；
- 188 个请求带有引擎 KV 反馈，观测到 8 次 KV reset；
- shutdown timeout、rollout cancel 和 checkpoint 文件均为 0。

报告：`output/webshop_qwen3_4b_version_runtime_engine_batch_probe_2step/terminal_waste.step_2.json`

## 7. 当前结论边界

这两次实验说明 admission、priority、KV rebuilding、真实 EngineCore 首批收集和 KV 状态反馈已经形成端到端执行闭环。但它们仍是短程单 seed 功能验证，不能据此宣称吞吐或训练效率稳定优于基线。

下一阶段应固定代码，运行 FIFO、仅 admission、admission+priority、完整系统四组消融，并在至少三个 AgenticRL workload、多个 oversampling 档位和多个 seed 上比较 trainable token goodput、learner update interval、stale invested-token waste 和 boundary re-prefill cost。
