在 Agentic RL rollout 中，一条轨迹通常不是一次性完成，而是反复经历：

LLM 推理 → tool/env 调用 → 等待返回 → 继续推理

现有异步 rollout 系统已经解决了一个问题：

tool 等待时不要占着 GPU。

但它们通常没有解决下一个问题：

tool-return 之后，如何低代价地把同一条被打断的轨迹重新接回推理。

也就是说，现有系统主要优化的是 wait hiding，
而你要优化的是 resume efficiency。

这个问题之所以成立，是因为 tool-return 后的继续执行并不便宜。
它会带来至少三类额外代价：

1. 恢复排队代价：tool 返回了，但不一定能立刻重新进入推理
2. 恢复重建代价：原有 KV / 上下文没了，需要重新 prefill 或 reload
3. 恢复 locality 代价：没有回到合适的 backend，上下文复用效果变差

所以你的问题不再是“tool 会让系统慢”，而是：

tool-return 会触发一种特殊的恢复事件，而现有系统仍把它当普通新请求处理。

⸻

2. 你的核心洞察是什么

你的核心洞察其实只有一句：

tool-return 后的继续执行，不是普通新请求，而是恢复请求（resume request）。

这句话非常关键。

因为一旦接受这个抽象，很多系统决策就都要改：

* 调度不该只看谁先 ready，而该看谁现在恢复更值
* 路由不该只看普通 cache-aware affinity，而该看哪里恢复代价更低
* cache 不该只按普通 LRU 淘汰，而该对即将恢复的上下文做差异化保留
* 上下文管理不该只有“留在 GPU / 被淘汰”，而应允许 GPU 保留 / CPU offload / 淘汰 三层状态

所以你的 idea 不是“做一个更好的 cache policy”，也不是“再做一个异步框架”，而是：

做一个 resume-aware rollout runtime。

⸻

3. 你的系统抽象是什么

你这套系统最核心的抽象，是把请求分成两类。

3.1 普通请求

定义：

未跨越外部等待边界，由模型内部连续推进的生成请求。

例如：

* 一次普通连续文本生成
* 一次尚未触发 tool 的内部 decode

3.2 恢复请求

定义：

同一条 trajectory 在外部等待后，由新的 tool/env observation 触发的继续生成请求。

也就是说，恢复请求必须满足两个条件：

1. 之前这条轨迹进入过外部等待
2. 现在是因为外部 observation 返回，才重新可执行

这一定义的关键点在于：

不是“多轮”就叫恢复，只有“跨越外部等待边界后重新进入推理”才叫恢复。

⸻

4. 你的核心方法是什么

你的方法可以分成三层。

⸻

4.1 第一层：恢复请求识别

系统在 runtime 里不再只维护一个普通 ready queue，
而是显式识别：

* FreshRequest
* ResumeRequest

并为恢复请求维护最小元信息，例如：

* trajectory_id
* resume_flag
* last_backend_id
* pause_ts
* history_len
* tool_type

这一步的意义不是工程细节，而是：

把“恢复”从普通 request 流里剥离出来，变成一种一等运行时事件。

⸻

4.2 第二层：恢复代价驱动的联合调度

这是你最核心的方法点。

你不是简单说：

* 恢复请求优先
* tool-return 后直接插队

这都太粗糙。

你真正的方法应该是：

对恢复请求和 worker 做联合打分，根据恢复价值与恢复代价决定“谁先恢复、恢复到哪”。

更准确地说，调度对象不是单独的 request，也不是单独的 worker，
而是一个二元组：

(\tau, w)

其中：

* \tau：一条待恢复的 trajectory
* w：一个候选推理 worker / backend

然后定义一个联合分数：

Score(\tau,w)=Value(\tau)-Cost(\tau,w)

其中：

恢复价值 Value(τ)

表示这条轨迹现在恢复值不值。
可由这些因素近似：

* 已经等了多久
* 是否接近完成
* 是否属于关键长尾轨迹
* 是否长期未被服务

恢复代价 Cost(τ,w)

表示把这条轨迹恢复到某个 worker 上，要付出多大代价。
最小版本可以包含：

* 是否回原 backend
* 距离上次执行过去多久
* history 有多长
* 当前 worker 负载多大

例如：

Cost(\tau,w)=
\alpha \cdot \mathbf{1}[w \neq last\_backend]
+\beta \cdot age(\tau)
+\gamma \cdot history\_len(\tau)
+\delta \cdot load(w)

这样，你的调度器做的就不是简单优先级队列，而是：

在所有待恢复轨迹与所有可用 worker 中，选择当前恢复收益最高的匹配。

这比“普通 affinity routing”更强，也比“恢复请求固定高优先级”更合理。

⸻

4.3 第三层：面向恢复请求的分层上下文管理

这是你的系统机制层。

如果只有恢复调度，没有恢复相关的上下文管理，系统收益会很有限。
因为恢复代价很多时候来自：

* KV 已经没了
* 长历史要重建
* GPU memory 不够，导致上下文过早淘汰

所以你要做的不是“改一改 LRU”，而是：

为恢复请求关联的上下文设计一个分层管理机制。

最自然的是三层：

A. GPU 热缓存层

适合：

* 很快就会 resume
* 上下文保留收益高
* GPU 上继续留着更值

B. CPU 温存层

适合：

* 可能很快回来，但不值得长期占 GPU
* 上下文较长，直接丢掉重建太贵
* 可以接受后续 reload

C. 冷淘汰层

适合：

* 恢复概率低
* 恢复收益小
* 当前系统压力大

所以这部分的本质不是“KV cache policy”，而是：

resume-aware memory hierarchy

它服务的目标不是一般 prefix reuse，而是：

future resume