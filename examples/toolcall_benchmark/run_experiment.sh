#!/bin/bash
# Resume-aware 调度性能对比实验脚本

set -e

echo "=========================================="
echo "Resume-aware 调度性能对比实验"
echo "=========================================="
echo ""
echo "实验目标："
echo "  对比 Resume-aware ROLL vs 原生 ROLL（baseline）"
echo "  在 AgenticRL tool-call 场景下的性能差异"
echo ""
echo "关键指标："
echo "  1. Resume affinity hit rate（locality）"
echo "  2. Resume TTFT p50/p95（延迟）"
echo "  3. Samples per second（吞吐）"
echo "  4. Normal queue wait（公平性）"
echo "=========================================="

cd /export/xxl/R3

# 实验 1: Baseline（原生 ROLL）
echo ""
echo "实验 1: Baseline（关闭 resume-aware 功能）"
echo "------------------------------------------"
echo "配置："
echo "  - enable_resume_priority: false"
echo "  - enable_resume_aware_routing: false"
echo "  - enable_request_priority_queue: false"
echo ""
echo "开始训练..."

python examples/start_agentic_pipeline.py \
    --config_path toolcall_benchmark \
    --config_name toolcall_benchmark_baseline

echo ""
echo "实验 1 完成！"
echo "日志目录: ./output/toolcall_benchmark/baseline/logs/"

# 实验 2: Resume-aware ROLL
echo ""
echo "实验 2: Resume-aware ROLL（开启所有优化）"
echo "------------------------------------------"
echo "配置："
echo "  - enable_resume_priority: true"
echo "  - enable_resume_aware_routing: true"
echo "  - enable_request_priority_queue: true"
echo "  - resume_normal_quota: [3, 1]"
echo "  - force_migrate_age_s: 30.0"
echo ""
echo "开始训练..."

python examples/start_agentic_pipeline.py \
    --config_path toolcall_benchmark \
    --config_name toolcall_benchmark_resume_aware

echo ""
echo "实验 2 完成！"
echo "日志目录: ./output/toolcall_benchmark/resume_aware/logs/"

# 对比分析
echo ""
echo "=========================================="
echo "性能对比分析"
echo "=========================================="

python - <<'PYTHON'
import os
import re
from pathlib import Path

baseline_log = Path("./output/toolcall_benchmark/baseline/logs")
resume_log = Path("./output/toolcall_benchmark/resume_aware/logs")

print("\n请查看以下日志文件进行详细对比：")
print(f"  Baseline:      {baseline_log}")
print(f"  Resume-aware:  {resume_log}")
print()

print("关键指标对比（需要从日志中提取）：")
print()
print("1. Locality 指标")
print("   - resume_affinity_hit_rate")
print("   - resume_migration_rate")
print()
print("2. 延迟指标")
print("   - resume_ttft_p50/p95")
print("   - resume_queue_wait_mean_s")
print()
print("3. 吞吐指标")
print("   - samples_per_sec")
print()
print("4. 公平性指标")
print("   - normal_queue_wait_mean_s")
print()

print("预期结果：")
print("  如果 resume-aware 调度有效，应该看到：")
print("  ✓ Resume affinity hit rate: 50% → 80%+")
print("  ✓ Resume TTFT p95: 降低 20-40%")
print("  ✓ Samples/sec: 提升 10-20%")
print("  ✓ Normal queue wait: 不显著恶化")
print()

# 尝试提取一些基本指标（如果日志格式支持）
def extract_metric(log_dir, pattern):
    """从日志中提取指标"""
    try:
        for log_file in log_dir.glob("*.log"):
            with open(log_file) as f:
                for line in f:
                    match = re.search(pattern, line)
                    if match:
                        return match.group(1)
    except Exception:
        pass
    return "N/A"

print("尝试自动提取指标...")
print()

# 这里可以根据实际日志格式添加提取逻辑
# baseline_hit_rate = extract_metric(baseline_log, r"resume_affinity_hit_rate[:\s]+([0-9.]+)")
# resume_hit_rate = extract_metric(resume_log, r"resume_affinity_hit_rate[:\s]+([0-9.]+)")

print("提示：使用以下命令手动提取指标：")
print()
print("  # 提取 affinity hit rate")
print("  grep -r 'resume_affinity_hit_rate' ./output/toolcall_benchmark/*/logs/")
print()
print("  # 提取延迟指标")
print("  grep -r 'resume_ttft_p95' ./output/toolcall_benchmark/*/logs/")
print()
print("  # 提取吞吐指标")
print("  grep -r 'samples_per_sec' ./output/toolcall_benchmark/*/logs/")
print()

PYTHON

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 查看日志文件，提取关键指标"
echo "  2. 对比 Baseline vs Resume-aware 的性能差异"
echo "  3. 如果效果不明显，调整权重配置或增加负载"
echo ""
echo "详细实验方案请参考: docs/experiment_plan.md"
