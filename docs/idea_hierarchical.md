第一层：不改后端的 resume-aware soft scheduling。这是最小可跑版本。

包括 Resume Request、HOT/WARM/COLD recoverability belief、confidence-aware routing、soft lease、反馈闭环。这一层解决“调度器看不到 KV 状态”的问题，但它本身偏轻。

第二层：后端协作的 resume observability。这是系统贡献的关键。

你不一定要做完整 page-level KV eviction，但至少要给推理后端加一个轻量 API，让 scheduler 能知道“这个 resume request 去某个 worker 大概能命中多少”。否则你的 restore_cost 永远是猜测。

最小 API 可以不是 pin/evict，而是 query：
lookup_resume(prefix_hash, trajectory_id) 
  -> hit_tokens, resident_blocks, estimated_prefill_tokens, cache_confidence

  或者更保守：
  probe_and_dispatch(request, candidate_worker)
  -> 如果命中，直接 resume；如果未命中，直接 prefill，不额外往返


第三层：后端协作的 soft lease enforcement。这是硬度最高、也最能和 Heddle 拉开差距的部分。

不要一开始做 page-level value-density eviction，可以先做 trajectory/segment-level lease：
set_resume_lease(
  trajectory_id,
  prefix_hash,
  ttl,
  lease_score,
  demotion_policy
)
worker 的 KV manager 不需要完全交出控制权，只需要在 eviction 时把 lease_score 作为 tie-breaker 或 priority modifier。
例如：
eviction_score = LRU_score - λ × lease_score

--------------------------------------------
第一层实现 ：
tool-return 请求来了
↓
它是 Resume Request，不是 Fresh Request
↓
我判断它的 KV 可能处于 HOT / WARM / COLD 哪种状态
↓
根据状态选择路由策略
这三个状态可以这样理解：
HOT:
  我很有把握它上一轮的 KV 还在 last_worker 上。
  策略：直接送回 last_worker，不做 probe。

WARM:
  KV 可能还在，但不确定。
  策略：可以 probe last_worker，或者比较 last_worker 和低负载 worker。

COLD:
  KV 大概率没了，或者恢复收益很低。
  策略：别强行亲和，按普通 load-aware routing 走。

重点是：HOT/WARM/COLD 不是后端告诉你的精确事实，而是你维护的一个 belief，也就是概率判断。

比如：
pause 时间很短 → 更可能 HOT
history 很长 → 如果命中，收益很大
last_worker 内存压力高 → 更可能已经被 evict
上次类似请求命中过 → 提高 HOT 概率
上次 sticky 回去但还是重算了 → 降低 HOT 概率

系统不是在说：
我知道 KV page 一定在，所以我调度它。
而是在说：
我根据 rollout 层可见的信息，估计这个 resume request 低成本恢复的概率，并按置信度选择路由策略。
------------
这就是 belief-based resume scheduling。