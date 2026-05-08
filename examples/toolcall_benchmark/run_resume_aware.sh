#!/bin/bash
set -e

echo "=========================================="
echo "实验 2: Resume-aware（开启所有优化）"
echo "=========================================="

cd /export/xxl/R3

python examples/start_agentic_pipeline.py \
    --config_path toolcall_benchmark \
    --config_name toolcall_resume_aware

echo ""
echo "Resume-aware 实验完成！"
echo "日志: ./output/toolcall_benchmark/resume_aware/logs/"
