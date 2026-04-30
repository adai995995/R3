实验设计文档：评估 EnvAffinityRouter 相对原生 ROLL 的性能收益（单机 8*A800，全解耦，全异步）

1. 背景与目标
背景：你在 R3 中实现了 EnvAffinityRouter（resume/normal 分队列、软配额、resume-aware affinity、aging、可选 gateway 状态等），希望在“角色全解耦、完全异步”的 Agentic RL 工作负载下验证其调度收益。
目标：在不做人为“注入延迟”的前提下，通过选择天然存在异步/长尾/多轮 resume 的任务，对比：
端到端吞吐是否提升（steps/s、trajectories/s、tokens/s）
端到端延迟是否更稳（尤其 P95/P99 step latency）
resume 亲和命中/迁移是否改善（命中率、迁移率、forced migrate 率）
排队与公平是否改善（resume vs normal queue wait、backlog、bucket 分布）

2. 假设与可证伪点（Hypotheses）
H1（resume 命中）：EnvAffinityRouter 在 tool-use/多轮场景中提高 resume_affinity_hit_rate，降低 resume_migration_rate。
H2（长尾抗性）：在天然长尾（工具执行波动、token 长度差异、外部服务抖动）下，EnvAffinityRouter 降低 P99 step latency，并降低 resume 被饿死的概率（resume_queue_wait_mean_s 降）。
H3（吞吐）：在 infer 成为瓶颈且存在 queue 的情况下，EnvAffinityRouter 提升或保持 tokens/s，同时提升有效完成轨迹数（trajectory 完成率/速度）。
H4（成本）：EnvAffinityRouter 不会显著降低 normal 请求体验（normal_queue_wait_mean_s 不大幅上升，或在可接受范围）。
任何假设不成立都可定位原因：resume 流量不足、无队列竞争、backend 差异不明显、forced-migrate 阈值过低、软配额配置不合理等。

3. 实验范围与约束
硬件：单机 8*A800（或 8 GPU 同等规格）。
运行方式：Ray 多角色分布式，全解耦（infer/train/env 不 colocate），全异步（env 产生请求异步进入 scheduler）。
约束：不通过人工 sleep/注入延迟来制造长尾；长尾来自真实工具链/系统/任务本身。
4. 工作负载选择（推荐两条主线）
为保证“不注入延迟也能有差异”，至少选择一种天然产生 tool-call 与长尾的场景：

4.1 OpenReward 工具链（优先推荐）
特点：模型输出驱动 tool-call（解析→执行→回写→继续生成），天然多轮与 resume。
适合验证：resume-aware affinity、软配额/优先级队列对 resume 饿死的改善。
产出信号强：hit/migration/queue_wait 指标直接可用。
4.2 Rock/Sandbox（env/rock/sandbox_manager_v2.py 类）
特点：环境侧调用外部 CLI/沙箱/子进程/I/O，天然系统级长尾。
适合验证：在“系统长尾 + 高并发”下 P99 是否改善、队列是否更稳。
注意：如果任务本身 resume 强度不高，则对 “resume affinity” 的收益可能不如 OpenReward 显著。
若只能选一个：先选 OpenReward（更贴合你的调度设计初衷）。

5. 系统配置：角色全解耦与资源切分（8 卡）
建议至少做两档资源切分，确保 infer 成为瓶颈，从而调度可见：

配置 S1（偏真实训练）：Infer 6 GPU + Train 2 GPU + Env CPU
配置 S2（偏调度验证/放大差异）：Infer 8 GPU + Train 极轻/关闭（只 rollout/val 或将训练步数极少）
两档都可跑，但优先 S2用于快速证明调度收益，再回到 S1 验证真实训练形态。

关键要求：

env actor 不占 GPU
actor_infer 独占 GPU 集
actor_train 不抢占 infer GPU（或最小化）
其它参数（模型、seed、batch、max_steps、env 数）在对照组之间保持一致
6. 对照组设计（A/B/C）
必须做到“只差调度策略，不差别的”。推荐三组：

A：原生/基线（Baseline）

目标：尽量接近原生 ROLL 的调度行为
方法：使用同一 Router 类或同一框架，但关闭你的关键策略开关（例如关闭 resume 优先级、请求优先队列、软配额等），保留最朴素 sticky/least-load 路由。
B：你的策略（Treatment）

开启：resume-aware + 请求优先队列（resume/normal）+ 软配额 + aging 等你要验证的完整组合。
C：拆分贡献（可选但强烈建议）

C1：只开 resume-aware affinity（不开优先队列/软配额）
C2：只开优先队列/软配额（不做 resume-aware 评分）
目的：把收益拆成“亲和命中贡献”和“队列/配额贡献”，避免结果解释不清。
7. 变量与控制（确保实验可信）
7.1 固定项（必须固定）
模型、权重路径、dtype、tp/pp 设置
角色资源切分（GPU mapping）
env 类型与数量、seed、max_steps/总步数
生成参数（max_new_tokens、temperature、top_p 等）
batch/并发（rollout_batch_size、infer_batch_size、max_env_num_per_worker、max_concurrency 等）
7.2 自变量（只改这些）
router 策略开关（A/B/C）
可选：软配额比例、force_migrate_age_s 等（做消融/敏感性分析）
8. 观测指标（Metrics）
8.1 端到端指标（必须）
吞吐：steps/s、trajectories/s、tokens/s（任选其一但要一致）
延迟分位：step latency P50/P90/P95/P99（全异步最看 P99）
完成率：单位时间完成 trajectory 数、abort/失败率
8.2 调度侧证据指标（你的优势证明）
直接使用你 EnvAffinityRouter.collect_metrics() 输出的：

scheduler/router/resume_affinity_hit_rate
scheduler/router/resume_migration_rate
scheduler/router/resume_forced_migration_rate
scheduler/router/resume_queue_wait_mean_s
scheduler/router/normal_queue_wait_mean_s
backlog、bucket served、fallback reason 分布等（用于定位问题：饿死、forced migrate、hint 不可用、过载等）
8.3 资源利用（可选但建议）
sglang：inflight、queue_depth、GPU 显存占用、水位
Ray：actor 任务排队、CPU 占用（env/tool 可能成为瓶颈）
9. 实验流程（每个点怎么跑）
对每个系统配置（S1/S2）与每个对照组（A/B/C）：

冷启动预热：运行固定预热步数（例如 100 step）不计入统计，避免 JIT/cuda graph/缓存影响。
正式采样窗口：再运行固定步数（例如 1000～5000 step，视场景稳定性），开始采集指标。
重复次数：每个实验点至少 3 次，报告均值与方差/置信区间。
记录工件：保留每次 run 的配置快照、日志、指标导出文件。
10. 分析方法与判定标准
10.1 主结论指标（建议以它们为“是否有收益”的判定）
P99 step latency：B 是否显著低于 A（或更稳定）
吞吐：B 是否 ≥ A（或在不牺牲吞吐前提下降低 P99）
resume 命中率：B 是否显著高于 A/C1
resume queue wait：B 是否显著低于 A（尤其在 backlog 存在时）
10.2 解释链（如何证明是“调度”带来的）
若 B 的 hit_rate↑ 且 migration_rate↓，同时 P99↓，则强证据指向 resume-aware affinity 的价值。
若 B 的 queue_backlog_resume↓、resume_queue_wait↓，同时 normal 不严重退化，则证据指向软配额/优先级队列。
若吞吐变化不大但 P99 明显改善：说明调度主要改善 tail，而非平均吞吐（这在异步系统很常见）。
11. 风险与排障预案
看不出差异：大概率是“resume 不够 / 无队列竞争 / 工具链不够长尾”。解决：
换 OpenReward tool-use 场景
提高并发（env 数、infer_batch_size、max_env_num_per_worker）
选择更长输出/更复杂任务（token 长度差异更大）
结果波动大：增加采样步数、增加重复次数；确保预热剔除；固定 seed 与资源独占。
normal 退化过多：调整 resume_normal_quota、aging 权重、max_queue_wait_s、force_migrate_age_s 等。
12. 最终交付物（你跑完应该产出什么）
一张总表：S1/S2 × A/B/C 的 吞吐、P50/P95/P99、hit_rate、migration_rate、resume/normal queue wait。
2～3 张关键曲线：
step latency 分布（尤其 tail）
resume hit/migration 随时间或随并发的变化
backlog 与 queue wait 的变化
结论段：在什么负载下收益最大、瓶颈是什么、下一步参数怎么调。

---

13. 仓库内最适合的 OpenReward tool-use 示例（入口 / YAML / 说明）

- 入口脚本：`examples/agentic_demo/run_openreward_endless_terminals.sh`
  - 调用方式：
    - `python examples/start_agentic_pipeline.py --config_path agentic_demo --config_name openreward_endless_terminals_reinforce_qwen35_2b`
  - 依赖：
    - 需要安装 `openreward`（脚本里写了 `pip install openreward`）
    - 需要设置 `OPENREWARD_API_KEY`

- 示例 YAML：`examples/agentic_demo/openreward_endless_terminals_reinforce_qwen35_2b.yaml`
  - 关键点：
    - `custom_envs.*.env_type: "openreward_env"`（在 `roll/pipeline/agentic/env/__init__.py` 里注册）
    - `async_generation_ratio: 1`（全异步生成）
    - tool-call 终止符：`actor_infer.generating_args.stop_strings: ["</tool_call>"]`
    - `parse_tool_call_parameter_to_dict: true`（工具参数解析）

- 对应环境实现：
  - `roll/pipeline/agentic/env/openreward/openreward_env.py`（解析 `<tool_call>`，调用 OpenReward session tool，再回写 `<tool_response>`）

> 建议：如果你想用本地模型（例如 Qwen2.5-0.5B-Instruct）做快速实验，把该 YAML 里的 `pretrain/reward_pretrain` 换成你的本地路径，并把 `actor_infer.strategy_args.strategy_name` 换成你现在跑通的 `sglang`。

---

14. A/B/C 严格对照：只改 `router_args.router_config` 的 overrides 模板

说明：下面只改路由开关，其他（模型/并发/seed/角色映射）保持不变，用于“严格对照”。
用法示例（Hydra override）：

Baseline A（近似原生 ROLL / 关闭所有 resume 优先级特性）
- `+router_args.router_name=EnvAffinityRouter`
- `+router_args.router_config.enable_resume_priority=false`

Treatment B（开启你的完整调度策略：resume-aware + 队列 + 软配额）
- `+router_args.router_name=EnvAffinityRouter`
- `+router_args.router_config.enable_resume_priority=true`
- `+router_args.router_config.enable_resume_aware_routing=true`
- `+router_args.router_config.enable_request_priority_queue=true`
- `+router_args.router_config.resume_normal_quota=3:1`
- `+router_args.router_config.request_wait_aging_weight=0.1`
- `+router_args.router_config.force_migrate_age_s=30.0`
- （可选）`+router_args.router_config.normal_max_queue_wait_s=2.0`
- （可选）`+router_args.router_config.resume_max_queue_wait_s=2.0`

Ablation C1（只测 resume-aware affinity：不开优先级队列/软配额）
- `+router_args.router_name=EnvAffinityRouter`
- `+router_args.router_config.enable_resume_priority=true`
- `+router_args.router_config.enable_resume_aware_routing=true`
- `+router_args.router_config.enable_request_priority_queue=false`

Ablation C2（只测优先级队列/软配额：不做 resume-aware 评分）
- `+router_args.router_name=EnvAffinityRouter`
- `+router_args.router_config.enable_resume_priority=true`
- `+router_args.router_config.enable_resume_aware_routing=false`
- `+router_args.router_config.enable_request_priority_queue=true`
- `+router_args.router_config.resume_normal_quota=3:1`

注意：
- `enable_resume_priority=false` 会在代码里统一回滚一组 resume 相关行为（最适合作为“原生对照”）。
- 若你要对比“原生 ROLL 的 SglangRouter”，可以额外做一组：`+router_args.router_name=SglangRouter`（不属于“只改 router_config”范畴，但可作为补充对照）。

---

15. S1/S2 两档 8 卡角色映射建议（Infer/Train/Env 解耦）

目标：actor_infer / actor_train / env 全解耦，并且 infer 成为瓶颈（便于观察调度收益）。

S1（偏真实训练）
- **GPU 0-5：actor_infer（6 卡）**
- **GPU 6-7：actor_train（2 卡）**
- **env（train_env/val_env）：CPU-only**
- 适用：你希望在接近真实训练资源分配下评估收益。

S2（偏调度验证 / 放大差异）
- **GPU 0-7：actor_infer（8 卡）**
- **actor_train：极轻（1 卡）或临时不跑训练，仅 rollout/val**
- **env：CPU-only，尽量提高并发让 infer 排队**
- 适用：快速验证调度策略能否改善 tail（P99）和 resume 命中率。

解耦检查清单（跑起来后从日志确认）：
- actor_infer 有独立的 sglang server，并通过 health check（`health_generate_original 200`）
- RouterManager 打印 `EnvAffinityRouter resume-aware config`（说明路由策略生效）
- env worker 不占 GPU（Ray 资源请求里 `num_gpus` 接近 0 或未声明）
