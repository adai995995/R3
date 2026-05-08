# Tool-call Resume-aware 调度性能对比实验

## 快速开始

在 roll-r3-dev 容器内运行：

```bash
cd /export/xxl/R3
bash examples/toolcall_benchmark/run_experiment.sh
```

这个脚本会自动运行两组实验并对比结果。

## 实验配置

### Baseline（原生 ROLL）
- 配置文件：`toolcall_benchmark_baseline.yaml`
- 关闭所有 resume-aware 功能
- 输出目录：`./output/toolcall_benchmark/baseline/`

### Resume-aware ROLL
- 配置文件：`toolcall_benchmark_resume_aware.yaml`
- 开启所有 resume-aware 优化
- 输出目录：`./output/toolcall_benchmark/resume_aware/`

## 关键差异

| 配置项 | Baseline | Resume-aware |
|--------|----------|--------------|
| `enable_resume_priority` | false | true |
| `enable_resume_aware_routing` | false | true |
| `enable_request_priority_queue` | false | true |
| `resume_normal_quota` | - | [3, 1] |
| `force_migrate_age_s` | - | 30.0 |

## 关键指标

实验完成后，从日志中提取以下指标：

### 1. Locality（最重要）
```bash
grep -r 'resume_affinity_hit_rate' ./output/toolcall_benchmark/*/logs/
```
**预期**：Resume-aware > 80%，Baseline < 50%

### 2. 延迟
```bash
grep -r 'resume_ttft_p95' ./output/toolcall_benchmark/*/logs/
```
**预期**：Resume-aware 降低 20-40%

### 3. 吞吐
```bash
grep -r 'samples_per_sec' ./output/toolcall_benchmark/*/logs/
```
**预期**：Resume-aware 提升 10-20%

### 4. 公平性
```bash
grep -r 'normal_queue_wait' ./output/toolcall_benchmark/*/logs/
```
**预期**：Resume-aware 不显著恶化

## 详细实验方案

完整的实验设计、预期结果、调试方法等，请参考：
- [实验方案文档](../../docs/experiment_plan.md)

## 调整参数

如果效果不明显，可以尝试：

1. **增加负载**：调大 `rollout_batch_size`（64 → 128）
2. **调整权重**：修改 `resume_score_weights` 和 `request_score_weights`
3. **增加训练步数**：调大 `max_steps`（50 → 100）

## 故障排查

### 问题：Affinity hit rate 没有提升

检查 resume meta 是否正确传递：
```bash
grep "request_type.*resume" ./output/toolcall_benchmark/resume_aware/logs/*.log
```

### 问题：Resume 延迟反而更高

查看 worker 负载分布和 migration 原因：
```bash
grep "worker_load\|resume_fallback_reason" ./output/toolcall_benchmark/resume_aware/logs/*.log
```

### 问题：Normal 请求被饿死

查看 normal 请求的排队时间：
```bash
grep "normal_queue_wait" ./output/toolcall_benchmark/resume_aware/logs/*.log
```

如果过高，调整 `resume_normal_quota` 或 `normal_max_queue_wait_s`。
