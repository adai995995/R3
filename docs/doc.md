把 tool-return 后的恢复，视为一种可预测的 continuation 事件；系统在工具等待期间，提前为最可能恢复的轨迹保留 KV、维持 worker affinity，并预准备恢复前缀，从而降低 resume cost，而不是把它当作普通新请求重新调度。

再压缩一点：

不是对未来 token 做 speculation，而是对未来 resume 做 speculation。

核心三点：
	1.	问题：tool call 把连续推理切断，现有系统通常把返回后的恢复当新请求处理，导致重复支付 KV 重建、路由失配和恢复排队成本。
	2.	方法：对等待中的轨迹估计恢复价值，只对高价值轨迹做提前准备。
	3.	动作：保 KV、保 worker 亲和、预构造 observation-conditioned resume prefix。

一句更像论文标题的话：

Tool-return-aware continuation scheduling for interrupted agent trajectories.


