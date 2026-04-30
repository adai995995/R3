一，resume latency / tool-return 到重新进入推理的时间。
这是你这个 idea 最直接命中的指标。因为你的核心就是把 tool-return 后的继续执行从普通 request 中分离出来，围绕“恢复”做调度和上下文管理。Continuum 已明确指出，多轮 tool-call workload 的关键额外开销之一就是每一轮返回后的 queueing delay；即使有 CPU offloading，这个 per-turn queueing delay 仍会累积。 
如果你的系统有效，这个指标应该最先下降。

第二，resume request 的 prefill / reload 开销，例如恢复时的 prefill tokens、prefill 时间、KV reload 时间。
Continuum 的主收益之一就是减少 end-of-turn eviction 带来的 repetitive prefill/reload。 
你如果做了恢复请求识别、GPU 保留 / CPU offload / 淘汰的分层管理，这一项非常有希望明显改善。
这是你最容易做出“机制—指标”强对应关系的一项。

第三，resume hit rate / 恢复命中率。
也就是 tool-return 后，恢复请求有多大比例能够：

* 回到原 backend，或
* 命中可复用上下文，或
* 避免完整重建。
    这个指标虽然不是最终 KPI，但它是最关键的中间证据。因为你的方法如果真有效，必须先证明“恢复不是当普通新请求重跑”。这和 Continuum 的 TTL retainment 思路、以及 AgentMath 里 prefix-aware load balancing 的收益逻辑是一致的。 

第四，end-to-end rollout throughput。
这是最容易被论文接受的总指标。Heddle、ROLL Flash、AgentMath 都把 rollout throughput 作为主结果之一：Heddle报告最高 2.5× 提升，ROLL Flash 在 agentic tasks 上最高 2.72×，AgentMath 报告训练吞吐 4–5× 提升。 
你的系统如果主要优化恢复阶段，通常也会反馈到 rollout throughput 上，但我预计它更像是中等幅度提升，而不是像大规模全局 orchestration 那样特别夸张。

第五，GPU 利用率 / trainer starvation time。
如果恢复更快，rollout worker 更少卡在恢复阶段，trainer 更少等数据，GPU 利用率会上升。ROLL Flash 的核心卖点之一就是提升资源利用率与 rollout-train decoupling 效果。 
不过这类指标通常是“二阶收益”，要在前面的恢复指标改善后才会体现出来。

第六，长尾样本的完成时间 / 尾延迟（P90/P95）。
你的系统如果真在恢复阶段有效，收益往往会更多体现在长历史、多 tool、恢复频繁的长尾轨迹上，而不是均值。Heddle直接把问题定义成长尾 trajectory 的 queueing、interference 和 per-token time；Continuum 也专门展示了 P90/P95 latency 的改善。 
所以你很应该看：

* 长工具链样本
* 多 turn 样本
* 长 history 样本
    这些 bucket 的 tail latency。

如果只保留一个最小指标集，我建议你盯这 4 个：

* resume latency
* resume prefill/reload cost
* resume hit rate
* rollout throughput