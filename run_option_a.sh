#!/bin/bash
# 选项A执行脚本：应用最优参数并验证EMG效果
# 最后更新：2025-12-15

set -e  # 遇到错误立即退出

echo "=========================================="
echo "执行选项A：应用最优参数并验证EMG效果"
echo "=========================================="
echo ""

# 步骤1：重新生成q0（使用最优参数）
echo "步骤1/5: 重新生成q0（使用最优参数）..."
echo "----------------------------------------"
python scripts/q0_builder.py --datasets train dev test
echo "✓ q0重新生成完成"
echo ""

# 步骤2：评估q0质量（验证优化效果）
echo "步骤2/5: 评估q0质量（验证优化效果）..."
echo "----------------------------------------"
python scripts/eval_q0.py \
    --q0-file data/q0_dev.jsonl \
    --baseline-file output/dev_with_uncertainty.jsonl
echo "✓ q0质量评估完成"
echo ""

# 步骤3：重新运行Day7（α*搜索）
echo "步骤3/5: 重新运行Day7（α*搜索）..."
echo "----------------------------------------"
python scripts/emg_bucket_search.py \
    --dev-file output/dev_with_uncertainty.jsonl \
    --q0-file data/q0_dev.jsonl \
    --uncertainty-file output/dev_uncertainty_buckets.csv \
    --output-file output/bucket_alpha_star.csv
echo "✓ Day7 α*搜索完成"
echo ""

# 步骤4：重新运行Day8（α(u)拟合）
echo "步骤4/5: 重新运行Day8（α(u)拟合）..."
echo "----------------------------------------"
python scripts/emg_fit_alpha_u.py \
    --input-file output/bucket_alpha_star.csv \
    --output-dir output
echo "✓ Day8 α(u)拟合完成"
echo ""

# 步骤5：重新运行Day9（EMG评估）
echo "步骤5/5: 重新运行Day9（EMG评估）..."
echo "----------------------------------------"
python scripts/eval_emg.py \
    --baseline-file output/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --use-consistency-gating \
    --output-dir output
echo "✓ Day9 EMG评估完成"
echo ""

echo "=========================================="
echo "✓ 选项A执行完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  - q0质量: 查看步骤2的输出"
echo "  - Day7结果: output/bucket_alpha_star.csv"
echo "  - Day8结果: output/alpha_u_lut.json, output/alpha_u_curve.png"
echo "  - Day9结果: output/metrics_emg_q0.json, output/emg_comparison_charts_q0.png"
echo ""

