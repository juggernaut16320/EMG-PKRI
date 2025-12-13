# 云端运行 Day7 和 Day8 指南

## ✅ 准备情况检查

### 代码状态
- ✅ **Day7 代码**：`scripts/emg_bucket_search.py` - 已实现并提交
- ✅ **Day8 代码**：`scripts/emg_fit_alpha_u.py` - 已实现并提交
- ✅ **单元测试**：已通过所有测试（27个Day7测试 + 11个Day8测试）

### 前置文件要求

**Day7 需要的前置文件：**
1. ✅ `output/dev_with_uncertainty.jsonl` - Day4输出（包含baseline预测结果和uncertainty）
2. ✅ `data/q0_dev.jsonl` - Day6输出（已在云端生成）
3. ✅ `output/uncertainty_buckets.csv` - Day4输出（不确定性分桶结果）

**Day8 需要的前置文件：**
1. ✅ `output/bucket_alpha_star.csv` - Day7输出

---

## 📋 前置条件检查脚本

在云端运行以下命令检查前置文件：

```bash
#!/bin/bash
# 前置条件检查

cd /mnt/workspace/EMG-PKRI

echo "=========================================="
echo "前置条件检查"
echo "=========================================="

# Day7 前置文件
echo ""
echo "Day7 前置文件检查："
echo "1. dev_with_uncertainty.jsonl:"
if [ -f "output/dev_with_uncertainty.jsonl" ]; then
    count=$(wc -l < output/dev_with_uncertainty.jsonl)
    echo "   ✓ 存在，共 $count 行"
else
    echo "   ✗ 不存在，需要先运行 uncertainty_analysis.py"
fi

echo "2. q0_dev.jsonl:"
if [ -f "data/q0_dev.jsonl" ]; then
    count=$(wc -l < data/q0_dev.jsonl)
    echo "   ✓ 存在，共 $count 行"
else
    echo "   ✗ 不存在，需要先运行 q0_builder.py"
fi

echo "3. uncertainty_buckets.csv:"
if [ -f "output/uncertainty_buckets.csv" ]; then
    count=$(wc -l < output/uncertainty_buckets.csv)
    echo "   ✓ 存在，共 $count 行"
else
    echo "   ✗ 不存在，需要先运行 uncertainty_analysis.py"
fi

echo ""
echo "=========================================="
```

---

## 🚀 云端运行步骤

### 步骤1：拉取最新代码

```bash
cd /mnt/workspace/EMG-PKRI

# 拉取最新代码
git pull origin main

# 验证Day7和Day8脚本存在
ls -lh scripts/emg_bucket_search.py scripts/emg_fit_alpha_u.py
```

### 步骤2：激活Python环境

```bash
# 激活venv环境
source venv/bin/activate

# 验证环境
which python
python --version

# 检查依赖（如果需要）
pip list | grep -E "numpy|pandas|sklearn|matplotlib"
```

### 步骤3：检查前置文件

```bash
# 检查Day7前置文件
echo "检查Day7前置文件："
ls -lh output/dev_with_uncertainty.jsonl 2>/dev/null || echo "⚠ dev_with_uncertainty.jsonl 不存在"
ls -lh data/q0_dev.jsonl 2>/dev/null || echo "⚠ q0_dev.jsonl 不存在"
ls -lh output/uncertainty_buckets.csv 2>/dev/null || echo "⚠ uncertainty_buckets.csv 不存在"
```

**如果 dev_with_uncertainty.jsonl 不存在**，需要先运行：

```bash
# 运行Day4的不确定性分析（如果还没运行）
python scripts/uncertainty_analysis.py \
    --dev-file data/dev.jsonl \
    --test-file data/test.jsonl \
    --output-dir output
```

### 步骤4：运行Day7（EMG α搜索）

```bash
# 基本运行（使用默认参数）
python scripts/emg_bucket_search.py

# 或者指定文件路径（推荐）
python scripts/emg_bucket_search.py \
    --dev-file output/dev_with_uncertainty.jsonl \
    --q0-file data/q0_dev.jsonl \
    --uncertainty-file output/uncertainty_buckets.csv \
    --output-file output/bucket_alpha_star.csv

# 运行完成后检查输出
ls -lh output/bucket_alpha_star.csv
head -5 output/bucket_alpha_star.csv
```

**预期运行时间**：约 1-2 小时（取决于dev集大小和计算资源）

### 步骤5：运行Day8（PAV拟合）

```bash
# 运行Day8（使用Day7的输出作为输入）
python scripts/emg_fit_alpha_u.py \
    --input-file output/bucket_alpha_star.csv \
    --output-dir output

# 运行完成后检查输出
ls -lh output/alpha_u_lut.json output/alpha_u_curve.png
```

**预期运行时间**：< 1 分钟

---

## 🔍 验证输出

### Day7 输出验证

```bash
# 检查bucket_alpha_star.csv
echo "Day7输出验证："
head -3 output/bucket_alpha_star.csv
echo ""
echo "检查必需字段："
python3 << EOF
import pandas as pd
df = pd.read_csv('output/bucket_alpha_star.csv')
print(f"Bucket数量: {len(df)}")
print(f"必需字段检查:")
required_cols = ['bucket_id', 'u_mean', 'alpha_star', 'f1_at_alpha_star']
for col in required_cols:
    if col in df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ {col} 缺失")
print(f"\n前3个bucket的alpha_star:")
print(df[['bucket_id', 'u_mean', 'alpha_star']].head(3))
EOF
```

**预期输出特征**：
- 包含多个bucket（通常6-10个）
- alpha_star 值在 [0, 1] 范围内
- 应该呈现单调递减趋势（u越大，alpha_star越小）

### Day8 输出验证

```bash
# 检查alpha_u_lut.json
echo "Day8输出验证："
python3 << EOF
import json
with open('output/alpha_u_lut.json', 'r') as f:
    lut = json.load(f)
print(f"查表点数: {len(lut['u'])}")
print(f"u范围: [{min(lut['u']):.4f}, {max(lut['u']):.4f}]")
print(f"alpha范围: [{min(lut['alpha']):.4f}, {max(lut['alpha']):.4f}]")
print(f"\n前3个点:")
for i in range(3):
    print(f"  u={lut['u'][i]:.4f}, alpha={lut['alpha'][i]:.4f}")
print(f"\n后3个点:")
for i in range(len(lut['u'])-3, len(lut['u'])):
    print(f"  u={lut['u'][i]:.4f}, alpha={lut['alpha'][i]:.4f}")

# 检查单调性（应该是单调递减的）
is_decreasing = all(lut['alpha'][i] >= lut['alpha'][i+1] for i in range(len(lut['alpha'])-1))
print(f"\n单调递减性: {'✓ 通过' if is_decreasing else '✗ 失败'}")
EOF

# 检查图表文件
ls -lh output/alpha_u_curve.png
```

---

## 🐛 常见问题排查

### 问题1：dev_with_uncertainty.jsonl 不存在

**解决方法**：
```bash
# 运行uncertainty_analysis.py生成
python scripts/uncertainty_analysis.py \
    --dev-file data/dev.jsonl \
    --output-dir output
```

### 问题2：q0_dev.jsonl 不存在

**解决方法**：
```bash
# 运行q0_builder.py生成（如果还没运行）
python scripts/q0_builder.py --datasets dev
```

### 问题3：uncertainty_buckets.csv 不存在

**解决方法**：
```bash
# 运行uncertainty_analysis.py生成
python scripts/uncertainty_analysis.py \
    --dev-file data/dev.jsonl \
    --test-file data/test.jsonl \
    --output-dir output
```

### 问题4：Day7运行报错"缺少pred_probs或uncertainty"

**原因**：dev.jsonl中缺少必需字段

**解决方法**：
- 确保使用 `output/dev_with_uncertainty.jsonl`（Day4的输出）
- 或者先运行 `uncertainty_analysis.py` 生成完整结果

### 问题5：Day8运行报错"缺少alpha_star字段"

**原因**：Day7的输出文件格式不正确

**解决方法**：
- 检查Day7是否成功运行
- 验证 `output/bucket_alpha_star.csv` 文件格式

---

## 📝 一键运行脚本

创建 `run_day7_day8.sh`：

```bash
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
if [ ! -f "output/dev_with_uncertainty.jsonl" ]; then
    echo "⚠ dev_with_uncertainty.jsonl 不存在，正在运行uncertainty_analysis.py..."
    python scripts/uncertainty_analysis.py --dev-file data/dev.jsonl --output-dir output
fi

if [ ! -f "data/q0_dev.jsonl" ]; then
    echo "⚠ q0_dev.jsonl 不存在，正在运行q0_builder.py..."
    python scripts/q0_builder.py --datasets dev
fi

if [ ! -f "output/uncertainty_buckets.csv" ]; then
    echo "⚠ uncertainty_buckets.csv 不存在，正在运行uncertainty_analysis.py..."
    python scripts/uncertainty_analysis.py --dev-file data/dev.jsonl --output-dir output
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

# 5. 运行Day8
echo ""
echo "=========================================="
echo "5. 运行Day8（PAV拟合）..."
echo "=========================================="
python scripts/emg_fit_alpha_u.py \
    --input-file output/bucket_alpha_star.csv \
    --output-dir output

# 6. 验证输出
echo ""
echo "=========================================="
echo "6. 验证输出..."
echo "=========================================="
echo "Day7输出:"
ls -lh output/bucket_alpha_star.csv
echo ""
echo "Day8输出:"
ls -lh output/alpha_u_lut.json output/alpha_u_curve.png

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
```

**使用方法**：
```bash
chmod +x run_day7_day8.sh
./run_day7_day8.sh
```

---

## ⏱️ 预期运行时间

- **Day7（EMG α搜索）**：约 1-2 小时
  - 取决于dev集大小（4,948条）
  - 每个bucket需要计算多个α值的F1/NLL
  
- **Day8（PAV拟合）**：< 1 分钟
  - 数据量小，主要是计算和绘图

---

## ✅ 运行成功标志

### Day7成功标志：
- ✅ 生成了 `output/bucket_alpha_star.csv`
- ✅ 文件包含多个bucket的alpha_star值
- ✅ 日志显示"搜索完成"

### Day8成功标志：
- ✅ 生成了 `output/alpha_u_lut.json`
- ✅ 生成了 `output/alpha_u_curve.png`
- ✅ 查表包含100个点（默认）
- ✅ 查表是单调递减的

---

**最后更新**：2025-12-14  
**状态**：✅ 代码已就绪，可以云端运行

