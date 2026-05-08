#!/bin/bash
set -e

echo "=========================================="
echo "实验 1: Baseline（关闭 resume-aware）"
echo "=========================================="

cd /export/xxl/R3

python examples/start_agentic_pipeline.py \
    --config_path toolcall_benchmark \
    --config_name toolcall_baseline

echo ""
echo "Baseline 实验完成！"
echo "日志: ./output/toolcall_benchmark/baseline/logs/"
