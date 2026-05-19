核心是：

在 Agentic RL rollout 中，把 tool-return 后继续执行的请求显式建模为 Resume Request，并根据“轨迹价值 × 恢复收益 − 恢复代价 − 资源机会成本 + 公平性补偿”来联合决定：谁先进入推理引擎、路由到哪里、哪些 KV cache 值得保留/迁移/offload/淘汰。

你现在的说法里有一个小问题：“高价值，如上一轮是工具调用”。上一轮是工具调用本身不一定代表高价值。它更准确地表示：这个请求是 resume request，可能有较高恢复收益，因为它很可能能复用上一轮的上下文/KV。高价值应该来自 RL 语义和系统收益的组合，而不是仅仅来自“刚 tool-return”。

可以把你的系统拆成三层。

第一层是轨迹调度侧，也就是 rollout runtime / router。
它维护每条 trajectory 的状态：
trajectory_id
prompt_id / group_id
step_id
last_worker
last_kv_location
last_action_type: llm / tool / env
estimated_learning_value
estimated_resume_cost
waiting_time
expected_remaining_steps
expected_tool_return_time

当 tool call 返回后，这个请求不再被当作普通 fresh request，而是被标记成：
Resume Request

然后调度器计算：
resume_priority =
    learning_value × expected_resume_benefit
  - resume_cost
  - interference_cost
  + fairness_aging

第二层是亲和性路由。

如果上一轮的 KV cache 还在原 worker 上，那么优先路由回原 worker。这个是 cache-affinity。

但它不能是硬规则，而应该是代价规则：
route_score(worker) =
    resume_priority
  - restore_cost(worker)
  - queue_delay(worker)
  - interference_cost(worker)

也就是说，原 worker 有 KV 命中优势，但如果它队列很长，或者显存压力很高，可能不如迁移到其他 worker 或者重新 prefill。你的系统应该选择净收益最高的 worker，而不是无条件回原 worker。

第三层是 KV cache 管理侧。

KV cache manager 接收 trajectory value 和 resume metadata，但不要直接“高价值轨迹所有 KV 都保留”。更稳的是把轨迹价值投影到 segment/page：
trajectory value
→ segment value
→ KV page score

cache manager 用 page score 决定：
GPU retain
CPU offload
remote migration
prefetch
eviction / recompute

一个简化的 page score 可以是：
page_score =
    learning_value
  × resume_probability
  × avoided_prefill_cost
  ÷ kv_memory_size
  - transfer_cost
  + aging



  -------------
还需要避免三个概念混淆。

第一个混淆：Resume Request 不等于高价值请求。

Resume Request 的价值来自两个部分：
学习价值：这条轨迹对 RL 训练是否重要
系统价值：恢复它是否能节省大量 prefill / migration / waiting cost

“上一轮是工具调用”只能说明它是 resume candidate，不能说明它一定高价值。比如一个 trajectory 调用了工具但已经明显走偏，继续执行可能训练价值低；反过来，一个 fresh request 如果属于稀有 prompt 或高不确定任务，也可能很有训练价值。

所以你可以说：

tool-return 是触发 resume decision 的事件，不是 value 的定义。

第二个混淆：亲和性调度不应覆盖价值调度。

你说“优先返回上一轮的推理 worker”是对的，但这只是为了降低恢复成本。它应该作为 cost model 的一部分，而不是固定策略。

否则你会退化成 Verl 这类 cache-affinity placement。Heddle 也指出，静态 cache-affinity 会导致 load imbalance；least-load 又会带来 cache recomputation 和干扰。你的差异应该是：亲和性不是绝对规则，而是被纳入净价值函数。

更好的表述是：
优先回到 last_worker，当且仅当 KV hit benefit > extra queueing/interference cost。

第三个混淆：KV cache 调度不应该只按轨迹价值。

这是你系统里最容易被攻击的点。因为 paged KV 的资源消耗和收益不是按 trajectory 均匀分布的。一个 trajectory 的不同 page 价值不同：

shared prompt prefix：通常价值高，因为可复用、重算成本高。
tool observation 相关 page：对 resume 关键，价值较高。
很长的中间 reasoning：可能占空间大，但边际价值低。
刚生成的 tail：取决于后续是否马上 resume。

所以更强的说法不是“根据轨迹价值调度 KV cache”，而是：

根据轨迹价值、page 重算成本、复用概率和 page 大小计算 value-density，再指导 KV cache。

这可以直接变成你的核心贡献。

最终建议你把整个 idea 定义成四个模块：
1. Resume Request Abstraction
   Fresh request 和 Resume request 显式区分。

2. Trajectory Value Estimator
   估计 trajectory 的 learning value，而不是用长度当价值。

3. Cost-aware Affinity Router
   综合 last_worker KV hit、queue delay、migration cost、interference，决定路由。

4. Value-density KV Manager
   把 trajectory value 投影到 KV pages，决定保留、offload、迁移、淘汰。


---------------
kv 调度视图：
不要要求调度器完整感知每个 KV page 是否还在，而是建立一个轻量的 Resume State Directory。
这个 Directory 不需要暴露所有 KV block 的细节，只维护对调度足够的信息：
trajectory_id
last_worker
prefix_hash / block_hash range
last_seen_step
kv_state: GPU_HOT / CPU_OFFLOADED / REMOTE / UNKNOWN / EVICTED
resident_pages_estimate
reusable_tokens_estimate
last_touch_time
eviction_risk
restore_cost_estimate

它的作用不是替代 vLLM/SGLang 的 prefix cache，而是在调度侧提供一个“恢复状态的近似视图”。worker 只需要在几个事件上回报：
on_prefill_commit: 哪些 prefix/block 被 materialized
on_tool_suspend: 这个 trajectory 当前 KV 状态
on_eviction: 哪些 trajectory/block 被驱逐或降级
on_offload: 哪些 KV 被移到 CPU/remote
on_hit/miss: resume 时实际命中多少

这样你的贡献就从“依赖推理引擎自己的 prefix matching”变成：

给 rollout scheduler 增加一层跨 worker 的 resume-state observability。

你不需要一开始就做到精确 page-level。可以先做三档状态：
HOT: 大概率还在 last_worker GPU 上
WARM: 在 CPU/remote，可恢复但有 reload/migration 成本
COLD: 不确定或已淘汰，需要 recompute

调度器用这个状态估计恢复代价：
resume_cost =
  if HOT: queue_delay(last_worker) + small_validation_cost
  if WARM: queue_delay(target_worker) + reload_or_migration_cost
  if COLD: queue_delay(best_worker) + prefill_recompute_cost

然后再和轨迹价值结合：
resume_priority =
  learning_value × avoided_recompute_benefit
  - resume_cost
  - memory_opportunity_cost
  + fairness_aging

两阶段 speculative resume routing。
调度侧不强行假设 KV 一定在，而是先向候选 worker 发一个轻量 probe：
probe(trajectory_id, prefix_hash) -> hit_tokens, resident_pages, estimated_restore_time

worker 返回：
worker_id
gpu_hit_tokens
cpu_hit_tokens
queue_delay
memory_pressure
estimated_prefill_needed

调度器再选择 worker：
route_score(worker) =
  value × hit_tokens_saved
  - queue_delay
  - restore_time
  - interference_cost
  这个方式的好处是：你不需要把所有 KV 状态持续同步给调度器，只在 resume 发生时做低成本查询。坏处是会增加一次调度 RPC，但 tool-return 场景下通常可以接受，而且可以和 tool result post-processing 并行

---------------
## 轨迹价值调度（已实现）

见 **[trajectory_value_scheduling.md](./trajectory_value_scheduling.md)**：

- `V_traj = V_sys + V_learn_neg`（系统净收益 + 负反馈惩罚）
- Belief HOT/WARM/COLD 驱动 placement / conditional preferred header
- 配置开关：`enable_trajectory_value_scheduling`