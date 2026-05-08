#!/bin/bash
# Tool-call 调度性能对比测试脚本 - 端到端 AgenticRL 训练

set -e

echo "=========================================="
echo "Tool-call 调度性能基准测试"
echo "端到端 AgenticRL 训练场景"
echo "=========================================="

cd /export/xxl/R3

# 测试 1: 原生 SGLang
echo ""
echo "测试 1: 原生 SGLang (/export/xxl/sglang)"
echo "------------------------------------------"
echo "安装原生 SGLang..."
cd /export/xxl/sglang
pip install -e . --no-build-isolation > /dev/null 2>&1
python -c "import sglang; print(f'SGLang 版本: {sglang.__version__}')"
python -c "import sglang; print(f'SGLang 路径: {sglang.__file__}')"

echo "运行 AgenticRL 训练..."
cd /export/xxl/R3
python examples/start_agentic_pipeline.py \
    --config-path examples/toolcall_benchmark \
    --config-name toolcall_benchmark_native

echo "测试 1 完成！"

# 测试 2: 魔改版 SGLang
echo ""
echo "测试 2: 魔改版 SGLang (/export/xxl/xxl_sglang)"
echo "------------------------------------------"
echo "安装魔改版 SGLang..."
cd /export/xxl/xxl_sglang
pip install -e . --no-build-isolation > /dev/null 2>&1
python -c "import sglang; print(f'SGLang 版本: {sglang.__version__}')"
python -c "import sglang; print(f'SGLang 路径: {sglang.__file__}')"

echo "运行 AgenticRL 训练..."
cd /export/xxl/R3
python examples/start_agentic_pipeline.py \
    --config-path examples/toolcall_benchmark \
    --config-name toolcall_benchmark_custom

echo "测试 2 完成！"

# 对比结果
echo ""
echo "=========================================="
echo "性能对比分析"
echo "=========================================="
python - <<'PYTHON'
import json
import os
from pathlib import Path

native_log = Path("./output/toolcall_benchmark/native/logs")
custom_log = Path("./output/toolcall_benchmark/custom/logs")

print("请查看以下日志文件进行详细对比：")
print(f"  原生版本: {native_log}")
print(f"  魔改版本: {custom_log}")
print()
print("关键指标对比：")
print("  1. 吞吐量 (samples/sec)")
print("  2. 平均延迟 (ms)")
print("  3. P95/P99 延迟")
print("  4. GPU 利用率")
print("  5. 内存使用")
PYTHON

echo ""
echo "测试完成！"
