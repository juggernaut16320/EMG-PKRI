# 云端运行 Day9 改进

**更新时间**：2025-12-14

---

## 📋 前置文件检查

在运行之前，确保以下文件存在：

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate

# 检查前置文件
python3 << 'EOF'
import os

print("检查 Day9 改进前置文件...")
print("=" * 60)

files_to_check = [
    ("output/dev_with_uncertainty.jsonl", "Dev集Baseline预测（Day4输出）"),
    ("data/q0_dev.jsonl", "Dev集q₀后验（Day6输出）"),
    ("output/alpha_u_lut.json", "α(u)查表（Day8输出）"),
    ("output/test_with_uncertainty.jsonl", "Test集Baseline预测（Day4输出）"),
    ("data/q0_test.jsonl", "Test集q₀后验（Day6输出）")
]

all_exist = True
for file_path, description in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r') as f:
                count = sum(1 for _ in f)
            print(f"✓ {file_path}")
            print(f"  描述: {description}")
            print(f"  大小: {size/1024:.2f} KB, 行数: {count}")
        else:
            print(f"✓ {file_path}")
            print(f"  描述: {description}")
            print(f"  大小: {size/1024:.2f} KB")
    else:
        print(f"✗ {file_path} 不存在")
        print(f"  描述: {description}")
        all_exist = False
    print()

print("=" * 60)
if all_exist:
    print("✓ 所有前置文件都已就绪，可以运行 Day9 改进")
else:
    print("✗ 部分前置文件缺失，请先运行前置任务")
EOF
```

---

## 🚀 步骤1：搜索知识阈值

### 运行命令

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate
git pull origin main

# 搜索最优知识阈值（在dev集上）
python scripts/search_knowledge_threshold.py \
    --dev-file output/dev_with_uncertainty.jsonl \
    --q0-file data/q0_dev.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --output-dir output
```

### 预期输出

1. **控制台输出**：显示每个候选阈值的F1/NLL/ECE，以及最优阈值
2. **输出文件**：`output/knowledge_threshold.json`

**示例输出**：
```
搜索最优知识阈值（指标: f1，候选阈值: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]）...
评估阈值 0.40...
  阈值 0.40: F1=0.8845, NLL=0.3902, ECE=0.0651
评估阈值 0.50...
  阈值 0.50: F1=0.8867, NLL=0.3889, ECE=0.0645
评估阈值 0.60...
  阈值 0.60: F1=0.8873, NLL=0.3878, ECE=0.0642 ✅
...
✓ 最优阈值 = 0.6200 (f1=0.8873)
```

---

## 🚀 步骤2：使用门控优化重新评估

### 2.1 仅使用知识阈值门控

```bash
python scripts/eval_emg.py \
    --baseline-file output/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --knowledge-threshold-file output/knowledge_threshold.json \
    --output-dir output
```

### 2.2 仅使用一致性门控

```bash
python scripts/eval_emg.py \
    --baseline-file output/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --use-consistency-gating \
    --output-dir output
```

### 2.3 同时使用两种门控（推荐）

```bash
python scripts/eval_emg.py \
    --baseline-file output/test_with_uncertainty.jsonl \
    --q0-file data/q0_test.jsonl \
    --alpha-lut-file output/alpha_u_lut.json \
    --knowledge-threshold-file output/knowledge_threshold.json \
    --use-consistency-gating \
    --output-dir output
```

---

## 📊 步骤3：对比分析结果

### 对比以下方法

1. **Baseline**（原始EMG，无门控）
2. **EMG + 知识阈值门控**
3. **EMG + 一致性门控**
4. **EMG + 两种门控**

### 查看结果

```bash
# 查看指标文件
python3 << 'EOF'
import json

# 读取指标文件（如果存在）
try:
    with open('output/metrics_emg.json', 'r') as f:
        metrics = json.load(f)
    
    if 'test_set' in metrics:
        print("Test Set 指标对比:")
        print("=" * 60)
        for method, m in metrics['test_set'].items():
            print(f"{method}:")
            print(f"  F1: {m['f1']:.4f} ({m['f1']*100:.2f}%)")
            print(f"  NLL: {m['nll']:.4f}")
            print(f"  ECE: {m['ece']:.4f}")
            print()
        
        # 对比分析
        if 'baseline' in metrics['test_set'] and 'emg' in metrics['test_set']:
            baseline = metrics['test_set']['baseline']
            emg = metrics['test_set']['emg']
            print("EMG vs Baseline:")
            print(f"  F1 变化: {(emg['f1'] - baseline['f1'])*100:+.2f}%")
            print(f"  NLL 变化: {(baseline['nll'] - emg['nll'])/baseline['nll']*100:+.2f}%")
            print(f"  ECE 变化: {(baseline['ece'] - emg['ece'])/baseline['ece']*100:+.2f}%")
except FileNotFoundError:
    print("指标文件不存在，请先运行评估脚本")
EOF
```

---

## ✅ 验证要点

### 1. 知识阈值搜索结果
- ✓ 最优阈值是否合理？（通常应该在0.5-0.7之间）
- ✓ 是否找到了明显的最优值？（F1/NLL有明显峰值）

### 2. 门控效果
- ✓ F1是否提升？（理想：提升或至少不下降）
- ✓ NLL是否降低？（理想：进一步降低）
- ✓ ECE是否降低？（理想：进一步降低）
- ✓ 高u切片表现是否改善？（关键验证点）

### 3. 方法对比
- ✓ 哪种门控效果最好？
- ✓ 两种门控结合是否更好？

---

## 🔧 如果遇到问题

### 问题1：文件不存在
```bash
# 如果 dev_with_uncertainty.jsonl 不存在
python scripts/uncertainty_analysis.py \
    --dev-file dev.jsonl \
    --output-dir output \
    --base-model /mnt/workspace/models/qwen/Qwen3-1___7B
```

### 问题2：知识阈值搜索失败
- 检查dev集和q₀文件是否匹配（ID对应）
- 检查alpha_lut文件是否正确加载

### 问题3：门控效果不明显
- 可以尝试不同的优化指标（f1 vs nll）
- 可以尝试更细的阈值网格

---

## 📝 下一步

根据运行结果：
1. **如果效果显著**：更新文档，记录改进效果
2. **如果效果不明显**：分析原因，考虑进一步优化
3. **如果需要调整**：修改阈值网格或门控逻辑

