#!/bin/bash
# 重跑Day7和Day8脚本
# 最后更新：2025-12-15

set -e  # 遇到错误立即退出

echo "=========================================="
echo "重跑 Day7 和 Day8"
echo "=========================================="
echo ""

# 检查前置文件
echo "检查前置文件..."
echo "----------------------------------------"

# 必需的输入文件
DEV_UNCERTAINTY_FILE="output/dev_with_uncertainty.jsonl"
Q0_DEV_FILE="data/q0_dev.jsonl"
UNCERTAINTY_BUCKETS_FILE="output/dev_uncertainty_buckets.csv"

# 检查文件是否存在
missing_files=()

if [ ! -f "$DEV_UNCERTAINTY_FILE" ]; then
    echo "⚠ $DEV_UNCERTAINTY_FILE 不存在"
    missing_files+=("$DEV_UNCERTAINTY_FILE")
else
    echo "✓ $DEV_UNCERTAINTY_FILE 存在"
fi

if [ ! -f "$Q0_DEV_FILE" ]; then
    echo "⚠ $Q0_DEV_FILE 不存在（需要使用优化后的q0）"
    missing_files+=("$Q0_DEV_FILE")
else
    echo "✓ $Q0_DEV_FILE 存在"
fi

if [ ! -f "$UNCERTAINTY_BUCKETS_FILE" ]; then
    echo "⚠ $UNCERTAINTY_BUCKETS_FILE 不存在（脚本会自动生成）"
else
    echo "✓ $UNCERTAINTY_BUCKETS_FILE 存在"
fi

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "❌ 缺少必需文件，请先准备："
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "提示："
    echo "  - dev_with_uncertainty.jsonl: 运行 python scripts/uncertainty_analysis.py"
    echo "  - q0_dev.jsonl: 运行 python scripts/q0_builder.py --datasets dev"
    exit 1
fi

echo ""
echo "所有前置文件已就绪"
echo ""

# 步骤1：重新运行Day7（α*搜索）
echo "步骤1/2: 重新运行Day7（α*搜索）..."
echo "----------------------------------------"
python scripts/emg_bucket_search.py \
    --dev-file "$DEV_UNCERTAINTY_FILE" \
    --q0-file "$Q0_DEV_FILE" \
    --uncertainty-file "$UNCERTAINTY_BUCKETS_FILE" \
    --output-file output/bucket_alpha_star.csv
echo "✓ Day7 α*搜索完成"
echo ""

# 步骤2：重新运行Day8（α(u)拟合）
echo "步骤2/2: 重新运行Day8（α(u)拟合）..."
echo "----------------------------------------"
python scripts/emg_fit_alpha_u.py \
    --input-file output/bucket_alpha_star.csv \
    --output-dir output
echo "✓ Day8 α(u)拟合完成"
echo ""

echo "=========================================="
echo "✓ Day7 和 Day8 重跑完成！"
echo "=========================================="
echo ""
echo "输出文件："
echo "  - Day7结果: output/bucket_alpha_star.csv"
echo "  - Day8结果: output/alpha_u_lut.json"
echo "  - Day8图表: output/alpha_u_curve.png"
echo ""
echo "下一步："
echo "  运行 python scripts/eval_emg.py 验证EMG效果"
echo ""

