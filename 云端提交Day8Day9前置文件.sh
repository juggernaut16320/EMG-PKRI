#!/bin/bash
# 云端提交Day8和Day9前置文件到Git
# 在云端执行此脚本

echo "=========================================="
echo "提交Day8和Day9前置文件到Git"
echo "=========================================="
echo ""

cd /mnt/workspace/EMG-PKRI

# 配置Git用户信息（如果没有配置）
if [ -z "$(git config user.email)" ]; then
    echo "配置Git用户信息..."
    git config user.email "dsw@cloud.com"
    git config user.name "DSW User"
fi

# 先拉取远程更新（避免冲突）
echo "拉取远程更新..."
git pull origin main --no-edit || echo "拉取失败，继续..."

# 检查文件是否存在
echo "检查文件..."
echo "----------------------------------------"

files_to_commit=()
missing_files=()

# Day8输入文件
if [ -f "output/bucket_alpha_star.csv" ]; then
    echo "✓ output/bucket_alpha_star.csv 存在"
    files_to_commit+=("output/bucket_alpha_star.csv")
else
    echo "✗ output/bucket_alpha_star.csv 不存在"
    missing_files+=("output/bucket_alpha_star.csv")
fi

# Day9输入文件
if [ -f "output/test_with_uncertainty.jsonl" ]; then
    echo "✓ output/test_with_uncertainty.jsonl 存在"
    files_to_commit+=("output/test_with_uncertainty.jsonl")
else
    echo "✗ output/test_with_uncertainty.jsonl 不存在"
    missing_files+=("output/test_with_uncertainty.jsonl")
fi

if [ -f "data/q0_test.jsonl" ]; then
    echo "✓ data/q0_test.jsonl 存在"
    files_to_commit+=("data/q0_test.jsonl")
else
    echo "✗ data/q0_test.jsonl 不存在"
    missing_files+=("data/q0_test.jsonl")
fi

echo ""

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "⚠ 以下文件缺失，将被跳过："
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
fi

if [ ${#files_to_commit[@]} -eq 0 ]; then
    echo "❌ 没有文件需要提交"
    exit 1
fi

# 添加文件到Git
echo "添加文件到Git..."
echo "----------------------------------------"
for file in "${files_to_commit[@]}"; do
    git add "$file"
    echo "✓ 已添加: $file"
done

echo ""
echo "提交文件..."
git commit -m "添加Day8和Day9前置文件（供本地运行）"

echo ""
echo "推送到远程..."
git push origin main

echo ""
echo "=========================================="
echo "✓ 提交完成！"
echo "=========================================="
echo ""
echo "已提交的文件："
for file in "${files_to_commit[@]}"; do
    echo "  - $file"
done
echo ""
echo "本地拉取后运行："
echo "  1. git pull origin main"
echo "  2. python scripts/emg_fit_alpha_u.py --input-file output/bucket_alpha_star.csv --output-dir output"
echo "  3. python scripts/eval_emg.py --baseline-file output/test_with_uncertainty.jsonl --q0-file data/q0_test.jsonl --alpha-lut-file output/alpha_u_lut.json --use-consistency-gating --output-dir output"
echo ""

