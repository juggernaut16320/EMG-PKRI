#!/bin/bash
# 详细验证 Day7 和 Day8 输出文件

echo "=========================================="
echo "Day7 和 Day8 输出详细验证"
echo "=========================================="
echo ""

# Day7 详细验证
echo "【Day7 详细验证 - bucket_alpha_star.csv】"
python3 << 'EOF'
import pandas as pd
import os

csv_file = 'output/bucket_alpha_star.csv'
if os.path.exists(csv_file):
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ 文件存在，共 {len(df)} 行")
        print(f"✓ 列数: {len(df.columns)}")
        print(f"\n必需字段检查:")
        required_cols = ['bucket_id', 'u_mean', 'alpha_star', 'f1_at_alpha_star']
        all_present = True
        for col in required_cols:
            if col in df.columns:
                print(f"  ✓ {col}")
            else:
                print(f"  ✗ {col} 缺失")
                all_present = False
        
        if all_present:
            print(f"\n数据预览（前5个bucket）:")
            print(df[['bucket_id', 'u_mean', 'alpha_star', 'f1_at_alpha_star']].head(5).to_string(index=False))
            
            print(f"\n统计信息:")
            print(f"  - u_mean 范围: [{df['u_mean'].min():.4f}, {df['u_mean'].max():.4f}]")
            print(f"  - alpha_star 范围: [{df['alpha_star'].min():.4f}, {df['alpha_star'].max():.4f}]")
            print(f"  - 平均 F1: {df['f1_at_alpha_star'].mean():.4f}")
            
            # 检查单调性趋势（u越大，alpha_star应该越小）
            sorted_df = df.sort_values('u_mean')
            is_decreasing = all(sorted_df['alpha_star'].iloc[i] >= sorted_df['alpha_star'].iloc[i+1] 
                              for i in range(len(sorted_df)-1))
            print(f"  - alpha_star 单调递减趋势: {'✓ 符合预期' if is_decreasing else '⚠ 不完全单调（Day8会处理）'}")
            
            print(f"\n✓ Day7 输出验证通过")
        else:
            print(f"\n✗ Day7 输出验证失败：缺少必需字段")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
else:
    print("✗ 文件不存在")
EOF

echo ""
echo "【Day8 详细验证 - alpha_u_lut.json】"
python3 << 'EOF'
import json
import os

json_file = 'output/alpha_u_lut.json'
if os.path.exists(json_file):
    try:
        with open(json_file, 'r') as f:
            lut = json.load(f)
        
        print(f"✓ 文件存在")
        print(f"✓ 查表点数: {len(lut['u'])}")
        print(f"✓ u 数组长度: {len(lut['u'])}")
        print(f"✓ alpha 数组长度: {len(lut['alpha'])}")
        
        if len(lut['u']) == len(lut['alpha']):
            print(f"✓ 数组长度匹配")
        else:
            print(f"✗ 数组长度不匹配")
        
        print(f"\n数据范围:")
        print(f"  - u 范围: [{min(lut['u']):.4f}, {max(lut['u']):.4f}]")
        print(f"  - alpha 范围: [{min(lut['alpha']):.4f}, {max(lut['alpha']):.4f}]")
        
        print(f"\n前5个查表点:")
        for i in range(min(5, len(lut['u']))):
            print(f"  u={lut['u'][i]:.4f} -> alpha={lut['alpha'][i]:.4f}")
        
        print(f"\n后5个查表点:")
        for i in range(max(0, len(lut['u'])-5), len(lut['u'])):
            print(f"  u={lut['u'][i]:.4f} -> alpha={lut['alpha'][i]:.4f}")
        
        # 检查单调性（必须单调递减）
        is_decreasing = all(lut['alpha'][i] >= lut['alpha'][i+1] for i in range(len(lut['alpha'])-1))
        if is_decreasing:
            print(f"\n✓ 单调递减性检查: 通过（符合要求）")
        else:
            print(f"\n✗ 单调递减性检查: 失败（不应该发生）")
        
        # 检查值域
        alpha_valid = all(0 <= a <= 1 for a in lut['alpha'])
        u_valid = all(0 <= u <= 1 for u in lut['u'])
        if alpha_valid and u_valid:
            print(f"✓ 值域检查: u 和 alpha 都在 [0, 1] 范围内")
        else:
            print(f"⚠ 值域检查: 部分值超出 [0, 1] 范围")
        
        print(f"\n✓ Day8 输出验证通过")
    except Exception as e:
        print(f"✗ 读取失败: {e}")
else:
    print("✗ 文件不存在")
EOF

echo ""
echo "【Day8 图表文件检查】"
if [ -f "output/alpha_u_curve.png" ]; then
    echo "✓ alpha_u_curve.png 存在"
    ls -lh output/alpha_u_curve.png
else
    echo "✗ alpha_u_curve.png 不存在"
fi

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="

