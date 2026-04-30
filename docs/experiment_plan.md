# Resume-aware 调度性能对比实验方案

## 实验目标

对比你魔改的 **Resume-aware ROLL** 和 **原生 ROLL（baseline）** 在 AgenticRL tool-call 场景下的性能差异。

## 核心对比维度

根据你的设计文档，主要对比三个维度：

1. **恢复 locality**：resume 请求命中上次 backend 的比例
2. **恢复延迟**：tool-return 后到首 token 的时间（TTFT）
3. **队列公平性**：resume 与 normal 请求的排队时间分布

## 实验设计

### 对照组设置

**Baseline（原生 ROLL）**：
- 关闭所有 resume-aware 功能
- `enable_resume_priority: false`
- 不发送 resume meta 信息

**Treatment（Resume-aware ROLL）**：
- 开启 resume-aware 功能
- `enable_resume_priority: true`
- `enable_resume_aware_routing: true`
- `enable_request_priority_queue: true`

### 测试场景

使用 **GSM8K + Python tool-call** 场景：
- 数学问题求解需要调用 Python 代码执行器
- 每个问题平均触发 2-3 次 tool-call
- 能充分测试 resume 调度的效果

### 关键配置参数

```yaml
# 中等负载，能体现调度差异
rollout_batch_size: 64
max_steps: 50  # 足够获得稳定统计

# Resume-aware 配置（Treatment 组）
router_args:
  router_config:
    enable_resume_priority: true
    enable_resume_aware_routing: true
    enable_request_priority_queue: true
    resume_normal_quota: [3, 1]  # resume:normal = 3:1
    force_migrate_age_s: 30.0
    resume_score_weights:
      age: 1.0
      hist: 0.001
      aff: 0.5
      load: 0.5
    request_score_weights:
      age: 1.0
      hist: 0.001
      hit: 0.5
```

## 关键指标

### 1. Locality 指标（最重要）

- `resume_affinity_hit_rate`：resume 请求命中上次 backend 的比例
  - **预期**：Treatment 组 > 80%，Baseline 组 < 50%
- `resume_migration_rate`：resume 请求被迁移的比例
  - **预期**：Treatment 组更低

### 2. 延迟指标

- `resume_ttft_p50/p95`：tool-return 后到首 token 的延迟
  - **预期**：Treatment 组 p95 降低 20-40%
- `resume_queue_wait_mean_s`：resume 请求的排队时间
  - **预期**：Treatment 组更稳定

### 3. 吞吐指标

- `samples_per_sec`：整体吞吐量
  - **预期**：Treatment 组提升 10-20%（因为减少了重建开销）

### 4. 公平性指标

- `normal_queue_wait_mean_s`：normal 请求的排队时间
  - **预期**：Treatment 组不应显著恶化（软配额保护）

## 实验步骤

### 步骤 1：运行 Baseline

```bash
cd /export/xxl/R3
python examples/start_agentic_pipeline.py \
    --config-path examples/toolcall_benchmark \
    --config-name toolcall_benchmark_baseline
```

### 步骤 2：运行 Treatment

```bash
cd /export/xxl/R3
python examples/start_agentic_pipeline.py \
    --config-path examples/toolcall_benchmark \
    --config-name toolcall_benchmark_resume_aware
```

### 步骤 3：对比分析

查看日志目录：
- Baseline: `./output/toolcall_benchmark/baseline/logs/`
- Treatment: `./output/toolcall_benchmark/resume_aware/logs/`

提取关键指标并对比。

## 预期结果

如果你的 resume-aware 调度有效，应该看到：

1. **Locality 提升**：
   - Resume affinity hit rate: 50% → 80%+
   - 减少 KV cache miss 和重建开销

2. **延迟改善**：
   - Resume TTFT p95: 降低 20-40%
   - Resume queue wait: 更稳定

3. **吞吐提升**：
   - 整体 samples/sec: 提升 10-20%

4. **公平性保持**：
   - Normal 请求不被饿死
   - 软配额机制生效

## 可能的问题和调试

### 问题 1：Affinity hit rate 没有提升

**原因**：
- Resume meta 信息没有正确传递
- SGLang gateway 的 preferred-worker 机制没生效

**调试**：
```bash
# 检查 meta 信息是否正确
grep "request_type.*resume" logs/*.log

# 检查 preferred-worker header 是否发送
grep "X-ROLL-Preferred-Worker-Url" logs/*.log
```

### 问题 2：Resume 延迟反而更高

**原因**：
- 过度亲和导致某些 worker 过载
- 需要调整 `force_migrate_age_s` 或 `load` 权重

**调试**：
```bash
# 查看 worker 负载分布
grep "worker_load" logs/*.log

# 查看 migration 原因
grep "resume_fallback_reason" logs/*.log
```

### 问题 3：Normal 请求被饿死

**原因**：
- Resume 配额设置过高
- 需要调整 `resume_normal_quota`

**调试**：
```bash
# 查看 normal 请求的排队时间
grep "normal_queue_wait" logs/*.log
```

## 进阶实验

### 实验 A：不同负载下的表现

测试不同 `rollout_batch_size`：
- 低负载：16
- 中负载：64
- 高负载：256

观察 resume-aware 调度在不同负载下的收益。

### 实验 B：不同 tool-call 频率

测试不同任务：
- 低频 tool-call：简单数学题（1-2 次/轨迹）
- 高频 tool-call：复杂推理题（5-10 次/轨迹）

观察 resume-aware 调度在不同场景下的收益。

### 实验 C：权重调优

调整 `resume_score_weights` 和 `request_score_weights`：
- 增大 `age` 权重：更激进地服务等待久的请求
- 增大 `aff` 权重：更强的 locality 偏好
- 增大 `load` 权重：更均衡的负载分布

找到最优权重配置。

## 论文/报告建议

如果要写论文或技术报告，建议包含：

1. **问题定义**：
   - Tool-call 场景下的三类恢复代价
   - 现有系统的不足

2. **核心洞察**：
   - Resume request 是一等运行时事件
   - 需要 resume-aware 的调度和上下文管理

3. **方法**：
   - L0: 强亲和（preferred-worker）
   - L1: 软亲和（联合打分）
   - L2: Resume-aware ordering（优先级队列）

4. **实验结果**：
   - Locality 提升
   - 延迟改善
   - 吞吐提升
   - 公平性保持

5. **消融实验**：
   - 只开 L0 vs L0+L1 vs L0+L1+L2
   - 不同权重配置的影响
