#!/bin/bash
# Day7和Day8一键运行脚本

set -e  # 遇到错误立即退出

cd /mnt/workspace/EMG-PKRI

echo "=========================================="
echo "运行 Day7 和 Day8"
echo "=========================================="

# 1. 拉取最新代码
echo ""
echo "1. 拉取最新代码..."
git pull origin main

# 2. 激活环境
echo ""
echo "2. 激活Python环境..."
source venv/bin/activate

# 3. 检查前置文件
echo ""
echo "3. 检查前置文件..."

# 检查dev_with_uncertainty.jsonl
if [ ! -f "output/dev_with_uncertainty.jsonl" ]; then
    echo "⚠ dev_with_uncertainty.jsonl 不存在"
    echo "   提示：如果确定文件不存在，可以运行以下命令生成："
    echo "   python scripts/uncertainty_analysis.py --dev-file dev.jsonl --output-dir output --base-model /mnt/workspace/models/qwen/Qwen3-1___7B"
    read -p "是否现在运行uncertainty_analysis.py生成该文件? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/uncertainty_analysis.py --dev-file dev.jsonl --output-dir output --base-model /mnt/workspace/models/qwen/Qwen3-1___7B
    else
        echo "退出。请先确保前置文件存在。"
        exit 1
    fi
else
    count=$(wc -l < output/dev_with_uncertainty.jsonl)
    echo "   ✓ dev_with_uncertainty.jsonl 存在，共 $count 行"
fi

# 检查q0_dev.jsonl
if [ ! -f "data/q0_dev.jsonl" ]; then
    echo "⚠ q0_dev.jsonl 不存在"
    echo "   提示：如果确定文件不存在，可以运行以下命令生成："
    echo "   python scripts/q0_builder.py --datasets dev"
    read -p "是否现在运行q0_builder.py生成该文件? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/q0_builder.py --datasets dev
    else
        echo "退出。请先确保前置文件存在。"
        exit 1
    fi
else
    count=$(wc -l < data/q0_dev.jsonl)
    echo "   ✓ q0_dev.jsonl 存在，共 $count 行"
fi

# 检查uncertainty_buckets.csv
if [ ! -f "output/uncertainty_buckets.csv" ]; then
    echo "⚠ uncertainty_buckets.csv 不存在"
    echo "   提示：通常运行uncertainty_analysis.py会自动生成该文件"
    echo "   如果dev_with_uncertainty.jsonl已存在，但该文件不存在，可能需要重新运行uncertainty_analysis.py"
    read -p "是否现在运行uncertainty_analysis.py生成该文件? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/uncertainty_analysis.py --dev-file dev.jsonl --test-file test.jsonl --output-dir output --base-model /mnt/workspace/models/qwen/Qwen3-1___7B
    else
        echo "退出。请先确保前置文件存在。"
        exit 1
    fi
else
    count=$(wc -l < output/uncertainty_buckets.csv)
    echo "   ✓ uncertainty_buckets.csv 存在，共 $count 行"
fi

# 4. 运行Day7
echo ""
echo "=========================================="
echo "4. 运行Day7（EMG α搜索）..."
echo "=========================================="
python scripts/emg_bucket_search.py \
    --dev-file output/dev_with_uncertainty.jsonl \
    --q0-file data/q0_dev.jsonl \
    --uncertainty-file output/uncertainty_buckets.csv \
    --output-file output/bucket_alpha_star.csv

# 验证Day7输出
if [ ! -f "output/bucket_alpha_star.csv" ]; then
    echo "✗ Day7运行失败：未生成bucket_alpha_star.csv"
    exit 1
fi

echo "✓ Day7运行成功"

# 5. 运行Day8
echo ""
echo "=========================================="
echo "5. 运行Day8（PAV拟合）..."
echo "=========================================="
python scripts/emg_fit_alpha_u.py \
    --input-file output/bucket_alpha_star.csv \
    --output-dir output

# 验证Day8输出
if [ ! -f "output/alpha_u_lut.json" ]; then
    echo "✗ Day8运行失败：未生成alpha_u_lut.json"
    exit 1
fi

echo "✓ Day8运行成功"

# 6. 验证输出
echo ""
echo "=========================================="
echo "6. 验证输出..."
echo "=========================================="
echo "Day7输出:"
ls -lh output/bucket_alpha_star.csv
echo ""
echo "Day8输出:"
ls -lh output/alpha_u_lut.json 2>/dev/null || echo "alpha_u_lut.json 不存在"
ls -lh output/alpha_u_curve.png 2>/dev/null || echo "alpha_u_curve.png 不存在"

# 简要验证
echo ""
echo "简要验证："
python3 << 'EOF'
import pandas as pd
import json

# 验证Day7输出
try:
    df = pd.read_csv('output/bucket_alpha_star.csv')
    print(f"✓ bucket_alpha_star.csv: {len(df)} 个bucket")
    print(f"  前3个bucket的alpha_star:")
    print(df[['bucket_id', 'u_mean', 'alpha_star']].head(3).to_string(index=False))
except Exception as e:
    print(f"✗ bucket_alpha_star.csv 验证失败: {e}")

# 验证Day8输出
try:
    with open('output/alpha_u_lut.json', 'r') as f:
        lut = json.load(f)
    print(f"\n✓ alpha_u_lut.json: {len(lut['u'])} 个查表点")
    print(f"  u范围: [{min(lut['u']):.4f}, {max(lut['u']):.4f}]")
    print(f"  alpha范围: [{min(lut['alpha']):.4f}, {max(lut['alpha']):.4f}]")
    
    # 检查单调性
    is_decreasing = all(lut['alpha'][i] >= lut['alpha'][i+1] for i in range(len(lut['alpha'])-1))
    print(f"  单调递减性: {'✓ 通过' if is_decreasing else '✗ 失败'}")
except Exception as e:
    print(f"✗ alpha_u_lut.json 验证失败: {e}")
EOF

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

