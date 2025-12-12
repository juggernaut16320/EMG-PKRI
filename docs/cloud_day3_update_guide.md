# 云上 Day3 增量更新和数据验证指南

> **目标**：在云服务器上从 git 增量更新代码，并完成 Day3 的数据验证

---

## 📋 前置条件

- ✅ 云服务器上已 clone Day2 的代码
- ✅ 已激活 conda 环境：`conda activate emgpkri`
- ✅ 已安装所有依赖：`pip install -r requirements.txt`
- ✅ Day2 的 baseline 模型已训练完成（`checkpoints/baseline-lora/`）
- ✅ 数据文件已准备好（`data/test.jsonl`，可选：`data/hard_eval_set.jsonl`）

---

## 🔄 步骤 1：从 Git 增量更新代码

### 1.1 进入项目目录

```bash
cd /path/to/your/project  # 替换为你的项目路径
```

### 1.2 检查当前分支和状态

```bash
git status
git branch
```

### 1.3 拉取最新代码

```bash
# 拉取远程最新代码
git fetch origin

# 查看远程更新内容
git log HEAD..origin/main --oneline

# 合并远程更新（如果本地有未提交的更改，先 stash）
git pull origin main
```

**如果遇到冲突：**

```bash
# 如果有本地未提交的更改，先暂存
git stash

# 拉取更新
git pull origin main

# 恢复本地更改（如果需要）
git stash pop
```

### 1.4 验证更新成功

```bash
# 检查新文件是否存在
ls scripts/eval_baseline.py
ls scripts/hardset_maker.py
ls tests/test_eval_baseline*.py

# 查看更新内容
git log --oneline -5
```

---

## ✅ 步骤 2：验证代码更新

### 2.1 检查文件完整性

```bash
# 检查关键文件是否存在
test -f scripts/eval_baseline.py && echo "✓ eval_baseline.py 存在" || echo "✗ eval_baseline.py 不存在"
test -f scripts/hardset_maker.py && echo "✓ hardset_maker.py 存在" || echo "✗ hardset_maker.py 不存在"
test -f tests/test_eval_baseline_logic.py && echo "✓ 测试文件存在" || echo "✗ 测试文件不存在"
```

### 2.2 运行单元测试（可选，验证代码逻辑）

```bash
# 运行核心逻辑测试（不依赖 torch）
python -m pytest tests/test_eval_baseline_logic.py -v

# 如果安装了所有依赖，可以运行完整测试
python -m pytest tests/test_eval_baseline.py -v
```

---

## 🚀 步骤 3：准备 Day3 评估

### 3.1 检查依赖文件

```bash
# 检查 baseline 模型是否存在
ls -lh checkpoints/baseline-lora/

# 应该看到以下文件：
# - adapter_model.safetensors (或 adapter_model.bin)
# - adapter_config.json
# - tokenizer.json
# 等

# 检查测试数据是否存在
ls -lh data/test.jsonl

# 检查困难集（如果存在）
ls -lh data/hard_eval_set.jsonl 2>/dev/null || echo "hard_eval_set.jsonl 不存在（可选）"
```

### 3.2 检查配置文件

```bash
# 查看配置文件
cat configs/config.yaml | grep -A 5 "training:"
cat configs/config.yaml | grep -A 5 "model:"
cat configs/config.yaml | grep -A 5 "hardset:"
```

**确认配置项：**
- `training.output_dir`: 应该指向 `checkpoints/baseline-lora`
- `model.name_or_path`: 基础模型路径（如 `Qwen/Qwen3-1.7B`）
- `hardset.confidence_threshold`: 高置信度阈值（默认 0.8）

### 3.3 创建输出目录

```bash
# 确保输出目录存在
mkdir -p output

# 检查输出目录权限
ls -ld output
```

---

## 📊 步骤 4：运行 Day3 评估

### 4.1 基本评估（使用默认配置）

```bash
# 激活环境（如果还没激活）
conda activate emgpkri

# 运行评估
python scripts/eval_baseline.py
```

### 4.2 指定参数运行（如果需要）

```bash
# 指定 checkpoint 路径
python scripts/eval_baseline.py \
    --checkpoint checkpoints/baseline-lora \
    --base-model Qwen/Qwen3-1.7B

# 指定数据文件
python scripts/eval_baseline.py \
    --test-file test.jsonl \
    --hard-file hard_eval_set.jsonl

# 自定义置信度阈值
python scripts/eval_baseline.py \
    --confidence-threshold 0.9

# 指定输出目录
python scripts/eval_baseline.py \
    --output-dir output
```

### 4.3 后台运行（如果评估时间较长）

```bash
# 使用 nohup 后台运行
nohup python scripts/eval_baseline.py > eval_baseline.log 2>&1 &

# 查看进程
ps aux | grep eval_baseline

# 查看实时日志
tail -f eval_baseline.log

# 查看输出（完成后）
cat eval_baseline.log
```

---

## ✅ 步骤 5：验证评估结果

### 5.1 检查输出文件

```bash
# 检查评估指标文件
ls -lh output/metrics_baseline.json

# 检查高置信错误样本文件
ls -lh output/high_conf_error_samples.jsonl

# 查看文件内容（前几行）
head -20 output/metrics_baseline.json
head -5 output/high_conf_error_samples.jsonl
```

### 5.2 验证评估指标

```bash
# 使用 Python 查看评估结果
python << EOF
import json

# 读取评估指标
with open('output/metrics_baseline.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

print("=" * 60)
print("测试集评估结果:")
print("=" * 60)
test_metrics = metrics['test_set']
print(f"Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall: {test_metrics['recall']:.4f}")
print(f"F1: {test_metrics['f1']:.4f}")
print(f"总样本数: {test_metrics['total_samples']}")

if metrics.get('hard_set'):
    print("\n" + "=" * 60)
    print("困难集评估结果:")
    print("=" * 60)
    hard_metrics = metrics['hard_set']
    print(f"Accuracy: {hard_metrics['accuracy']:.4f}")
    print(f"F1: {hard_metrics['f1']:.4f}")
    print(f"总样本数: {hard_metrics['total_samples']}")

print("\n" + "=" * 60)
print(f"高置信错误样本数: {metrics['high_conf_error_count']}")
print(f"置信度阈值: {metrics['confidence_threshold']}")
print("=" * 60)
EOF
```

### 5.3 检查高置信错误样本

```bash
# 统计高置信错误样本数量
wc -l output/high_conf_error_samples.jsonl

# 查看前几个错误样本
head -3 output/high_conf_error_samples.jsonl | python -m json.tool

# 分析错误样本的标签分布
python << EOF
import json

with open('output/high_conf_error_samples.jsonl', 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f]

print(f"总错误样本数: {len(samples)}")
print(f"平均置信度: {sum(s['pred_prob'] for s in samples) / len(samples):.4f}")

# 统计真实标签分布
true_label_dist = {}
for s in samples:
    label = s['true_label']
    true_label_dist[label] = true_label_dist.get(label, 0) + 1

print("\n真实标签分布:")
for label, count in sorted(true_label_dist.items()):
    print(f"  标签 {label}: {count} 个")
EOF
```

---

## 🔍 步骤 6：问题排查

### 6.1 常见问题

**问题 1：模型加载失败**

```bash
# 检查 checkpoint 路径是否正确
ls -la checkpoints/baseline-lora/

# 检查基础模型路径
# 如果是本地模型，确认路径存在
# 如果是 HuggingFace 模型，确认网络连接正常
```

**问题 2：CUDA 内存不足**

```bash
# 减小 batch_size
python scripts/eval_baseline.py --batch-size 8

# 或使用 CPU（较慢）
python scripts/eval_baseline.py --device cpu
```

**问题 3：数据文件不存在**

```bash
# 检查数据文件
ls -lh data/test.jsonl
ls -lh data/hard_eval_set.jsonl

# 如果 hard_eval_set.jsonl 不存在，可以先运行 hardset_maker.py
python scripts/hardset_maker.py
```

**问题 4：配置文件错误**

```bash
# 验证配置文件格式
python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"
```

### 6.2 调试模式

```bash
# 使用 Python 交互式调试
python << EOF
import sys
sys.path.insert(0, 'scripts')
from eval_baseline import load_config, load_baseline_model

# 测试配置加载
config = load_config()
print("配置加载成功")
print(f"数据目录: {config.get('data_dir')}")
print(f"输出目录: {config.get('output_dir')}")

# 测试模型加载（如果 GPU 可用）
# model, tokenizer = load_baseline_model(
#     'checkpoints/baseline-lora',
#     'Qwen/Qwen3-1.7B'
# )
# print("模型加载成功")
EOF
```

---

## 📝 步骤 7：结果记录

### 7.1 保存评估结果

```bash
# 备份评估结果
mkdir -p results/day3
cp output/metrics_baseline.json results/day3/
cp output/high_conf_error_samples.jsonl results/day3/

# 记录评估时间
echo "$(date): Day3 评估完成" >> results/day3/evaluation_log.txt
```

### 7.2 生成简要报告

```bash
python << EOF
import json
from datetime import datetime

with open('output/metrics_baseline.json', 'r', encoding='utf-8') as f:
    metrics = json.load(f)

report = f"""
Day3 基线评估报告
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

测试集结果:
- Accuracy: {metrics['test_set']['accuracy']:.4f}
- F1: {metrics['test_set']['f1']:.4f}
- Precision: {metrics['test_set']['precision']:.4f}
- Recall: {metrics['test_set']['recall']:.4f}
- 样本数: {metrics['test_set']['total_samples']}
"""

if metrics.get('hard_set'):
    report += f"""
困难集结果:
- Accuracy: {metrics['hard_set']['accuracy']:.4f}
- F1: {metrics['hard_set']['f1']:.4f}
- 样本数: {metrics['hard_set']['total_samples']}
"""

report += f"""
高置信错误样本:
- 数量: {metrics['high_conf_error_count']}
- 阈值: {metrics['confidence_threshold']}
"""

print(report)
with open('results/day3/report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
EOF

cat results/day3/report.txt
```

---

## 🎯 完成检查清单

- [ ] ✅ 从 git 成功拉取最新代码
- [ ] ✅ 验证新文件存在
- [ ] ✅ 检查 baseline 模型存在
- [ ] ✅ 检查测试数据存在
- [ ] ✅ 运行 eval_baseline.py
- [ ] ✅ 验证输出文件生成
- [ ] ✅ 检查评估指标合理
- [ ] ✅ 检查高置信错误样本
- [ ] ✅ 保存评估结果

---

## 📚 下一步

完成 Day3 评估后，可以：

1. **分析评估结果**：查看 baseline 在测试集和困难集上的表现差异
2. **分析高置信错误**：研究为什么模型对这些样本高置信但预测错误
3. **准备 Day4**：开始不确定性分析（uncertainty_analysis.py）

---

## 💡 提示

- 如果评估时间较长，建议使用 `screen` 或 `tmux` 保持会话
- 定期检查磁盘空间：`df -h`
- 如果遇到问题，查看日志文件：`tail -f eval_baseline.log`
- 可以并行运行多个评估任务（使用不同的输出目录）

---

**最后更新**：2025-12-12  
**适用版本**：Day3 评估脚本

