# 云端运行Day11命令

## Day11：EMG + PKRI 联合融合试验

### 前置条件检查

确保以下文件已存在（来自Day4, Day6, Day8, Day10）：

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate

# 检查文件
ls -lh data/test_with_uncertainty.jsonl
ls -lh data/q0_test.jsonl
ls -lh data/qpkri_test.jsonl
ls -lh output/alpha_u_lut.json
ls -lh output/knowledge_threshold.json  # 可选
```

---

## 运行步骤

### 步骤1：拉取最新代码

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate
git pull origin main
```

### 步骤2：使用q₀运行EMG评估（基线对比）

```bash
python scripts/eval_emg.py \
    --baseline-file data/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --knowledge-source q0 \
    --use-consistency-gating \
    --knowledge-threshold-file output/knowledge_threshold.json \
    --output-dir output
```

**输出文件**：
- `output/metrics_emg_q0.json`
- `output/emg_comparison_table_q0.csv`
- `output/emg_comparison_charts_q0.png`

### 步骤3：使用q_PKRI运行EMG评估

```bash
python scripts/eval_emg.py \
    --baseline-file data/test_with_uncertainty.jsonl \
    --q0-file data/qpkri_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --knowledge-source qpkri \
    --use-consistency-gating \
    --knowledge-threshold-file output/knowledge_threshold.json \
    --output-dir output
```

**输出文件**：
- `output/metrics_emg_qpkri.json`
- `output/emg_comparison_table_qpkri.csv`
- `output/emg_comparison_charts_qpkri.png`

---

## 一键运行脚本

创建运行脚本：

```bash
cat > run_day11.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

cd /mnt/workspace/EMG-PKRI
source venv/bin/activate

echo "=========================================="
echo "Day11: EMG + PKRI 联合融合试验"
echo "=========================================="

git pull origin main

echo ""
echo "步骤1: 使用q₀运行EMG评估..."
python scripts/eval_emg.py \
    --baseline-file data/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --knowledge-source q0 \
    --use-consistency-gating \
    --knowledge-threshold-file output/knowledge_threshold.json \
    --output-dir output

echo ""
echo "步骤2: 使用q_PKRI运行EMG评估..."
python scripts/eval_emg.py \
    --baseline-file data/test_with_uncertainty.jsonl \
    --q0-file data/qpkri_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --knowledge-source qpkri \
    --use-consistency-gating \
    --knowledge-threshold-file output/knowledge_threshold.json \
    --output-dir output

echo ""
echo "✓ Day11 评估完成！"
echo ""
echo "输出文件："
echo "  - output/metrics_emg_q0.json"
echo "  - output/metrics_emg_qpkri.json"
echo "  - output/emg_comparison_table_q0.csv"
echo "  - output/emg_comparison_table_qpkri.csv"
SCRIPT_EOF

chmod +x run_day11.sh
./run_day11.sh
```

---

## 验证结果

### 检查输出文件

```bash
# 检查文件是否存在
ls -lh output/metrics_emg_q0.json
ls -lh output/metrics_emg_qpkri.json
ls -lh output/emg_comparison_table_q0.csv
ls -lh output/emg_comparison_table_qpkri.csv

# 查看q₀结果的关键指标
python << 'EOF'
import json
with open('output/metrics_emg_q0.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    if 'test_set' in data and 'emg' in data['test_set']:
        emg = data['test_set']['emg']
        print("q₀ EMG结果:")
        print(f"  F1: {emg['f1']:.4f}")
        print(f"  NLL: {emg['nll']:.4f}")
        print(f"  ECE: {emg['ece']:.4f}")
EOF

# 查看q_PKRI结果的关键指标
python << 'EOF'
import json
with open('output/metrics_emg_qpkri.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    if 'test_set' in data and 'emg' in data['test_set']:
        emg = data['test_set']['emg']
        print("q_PKRI EMG结果:")
        print(f"  F1: {emg['f1']:.4f}")
        print(f"  NLL: {emg['nll']:.4f}")
        print(f"  ECE: {emg['ece']:.4f}")
EOF
```

### 对比分析

```bash
# 使用pandas对比两个结果
python << 'EOF'
import json
import pandas as pd

# 加载两个结果
with open('output/metrics_emg_q0.json', 'r', encoding='utf-8') as f:
    q0_data = json.load(f)
with open('output/metrics_emg_qpkri.json', 'r', encoding='utf-8') as f:
    qpkri_data = json.load(f)

# 提取EMG结果
q0_emg = q0_data['test_set']['emg']
qpkri_emg = qpkri_data['test_set']['emg']

# 对比
print("=" * 60)
print("q₀ vs q_PKRI 对比")
print("=" * 60)
print(f"{'指标':<15} {'q₀':<15} {'q_PKRI':<15} {'变化':<15}")
print("-" * 60)
for metric in ['f1', 'nll', 'ece']:
    q0_val = q0_emg[metric]
    qpkri_val = qpkri_emg[metric]
    if metric == 'f1':
        change = qpkri_val - q0_val
        change_pct = (change / q0_val) * 100
        print(f"{metric.upper():<15} {q0_val:<15.4f} {qpkri_val:<15.4f} {change:+.4f} ({change_pct:+.2f}%)")
    else:
        change = qpkri_val - q0_val
        change_pct = (change / q0_val) * 100
        print(f"{metric.upper():<15} {q0_val:<15.4f} {qpkri_val:<15.4f} {change:+.4f} ({change_pct:+.2f}%)")
EOF
```

---

## 注意事项

1. **α(u)查表**：两个评估使用相同的α(u)查表（基于q₀训练的），确保公平对比
2. **门控参数**：两个评估使用相同的知识阈值和一致性门控设置
3. **数据对齐**：确保q₀和q_PKRI的样本ID完全一致
4. **输出文件名**：自动包含知识源标识，避免覆盖

---

## 预期结果

根据Day10的PKRI模型性能：
- q_PKRI的验证集指标：F1=0.7864, Accuracy=0.6479
- 预期在EMG融合中可能带来：
  - F1提升（主要目标）
  - NLL降低（概率质量改善）
  - 高不确定性切片可能有更明显改善

