#!/bin/bash
# 检查 Day7 和 Day8 是否完成

echo "=========================================="
echo "检查 Day7 和 Day8 完成状态"
echo "=========================================="
echo ""

# 检查 Day7 输出
echo "【Day7 输出检查】"
if [ -f "output/bucket_alpha_star.csv" ]; then
    echo "✓ bucket_alpha_star.csv 存在"
    ls -lh output/bucket_alpha_star.csv
    echo ""
    echo "内容预览："
    python3 << 'EOF'
import pandas as pd
try:
    df = pd.read_csv('output/bucket_alpha_star.csv')
    print(f"  - Bucket数量: {len(df)}")
    print(f"  - 必需字段: {list(df.columns[:7])}")
    print(f"\n  前3个bucket的alpha_star:")
    if 'u_mean' in df.columns and 'alpha_star' in df.columns:
        print(df[['bucket_id', 'u_mean', 'alpha_star', 'f1_at_alpha_star']].head(3).to_string(index=False))
    else:
        print(df.head(3).to_string(index=False))
except Exception as e:
    print(f"  ✗ 读取失败: {e}")
EOF
else
    echo "✗ bucket_alpha_star.csv 不存在 - Day7 未完成"
fi

echo ""
echo "【Day8 输出检查】"
if [ -f "output/alpha_u_lut.json" ]; then
    echo "✓ alpha_u_lut.json 存在"
    ls -lh output/alpha_u_lut.json
    echo ""
    echo "内容预览："
    python3 << 'EOF'
import json
try:
    with open('output/alpha_u_lut.json', 'r') as f:
        lut = json.load(f)
    print(f"  - 查表点数: {len(lut['u'])}")
    print(f"  - u范围: [{min(lut['u']):.4f}, {max(lut['u']):.4f}]")
    print(f"  - alpha范围: [{min(lut['alpha']):.4f}, {max(lut['alpha']):.4f}]")
    
    # 检查单调性
    is_decreasing = all(lut['alpha'][i] >= lut['alpha'][i+1] for i in range(len(lut['alpha'])-1))
    print(f"  - 单调递减性: {'✓ 通过' if is_decreasing else '✗ 失败'}")
    
    print(f"\n  前3个点:")
    for i in range(min(3, len(lut['u']))):
        print(f"    u={lut['u'][i]:.4f} -> alpha={lut['alpha'][i]:.4f}")
except Exception as e:
    print(f"  ✗ 读取失败: {e}")
EOF
else
    echo "✗ alpha_u_lut.json 不存在 - Day8 未完成"
fi

if [ -f "output/alpha_u_curve.png" ]; then
    echo "✓ alpha_u_curve.png 存在"
    ls -lh output/alpha_u_curve.png
else
    echo "✗ alpha_u_curve.png 不存在"
fi

echo ""
echo "=========================================="
echo "完成状态总结"
echo "=========================================="

if [ -f "output/bucket_alpha_star.csv" ] && [ -f "output/alpha_u_lut.json" ] && [ -f "output/alpha_u_curve.png" ]; then
    echo "✓ Day7 和 Day8 已完成！"
    exit 0
elif [ -f "output/bucket_alpha_star.csv" ] && [ ! -f "output/alpha_u_lut.json" ]; then
    echo "⚠ Day7 已完成，但 Day8 未完成"
    echo "  运行命令: python scripts/emg_fit_alpha_u.py --input-file output/bucket_alpha_star.csv --output-dir output"
    exit 1
elif [ ! -f "output/bucket_alpha_star.csv" ]; then
    echo "✗ Day7 未完成"
    echo "  运行命令: bash run_day7_day8.sh"
    exit 1
else
    echo "⚠ 部分文件缺失，请检查输出目录"
    exit 1
fi

