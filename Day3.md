### Day3 要完成的代码（基线评估 + 高置信错误分析）

> **环境准备**：开始前请确保已激活 conda 环境：`conda activate emgpkri`（或激活 venv：`source venv/bin/activate`）

```text
project/
│
├── data/
│   ├── test.jsonl
│   ├── hard_eval_set.jsonl
│
├── checkpoints/
│   └── baseline-lora/          # Day2 训练好的模型
│
├── output/
│   ├── metrics_baseline.json
│   ├── high_conf_error_samples.jsonl
│
├── configs/
│   └── config.yaml
│
├── scripts/
│   └── eval_baseline.py
│
├── tests/
│   ├── test_eval_baseline.py
│   └── test_eval_baseline_logic.py
│
└── requirements.txt
```

### `eval_baseline.py`（基线评估 + 高置信错误分析工具）

**功能：**
- 在 `test.jsonl` 上评估 baseline：Accuracy / F1 / confusion matrix
- 在 `hard_eval_set.jsonl` 上单独评估，比较 hard vs 普通样本表现
- 选出一批高置信错误样本（如预测概率>0.8但错了），存成样本库

**关键特性：**

- **模型加载**：支持加载 PEFT/LoRA 模型进行推理
- **批量预测**：支持批量处理，提高推理效率
- **完整指标**：计算 Accuracy、F1、Precision、Recall 和 Confusion Matrix
- **分类别指标**：提供每个类别的详细指标（class_0 和 class_1）
- **高置信错误识别**：自动识别高置信但预测错误的样本
- **结果保存**：保存评估指标和高置信错误样本

**使用方式：**

```bash
# 基本使用（从 config.yaml 读取所有参数）
python scripts/eval_baseline.py

# 指定 checkpoint 和模型路径
python scripts/eval_baseline.py \
    --checkpoint checkpoints/baseline-lora \
    --base-model /mnt/workspace/models/qwen/Qwen3-1___7B

# 指定数据文件
python scripts/eval_baseline.py \
    --test-file test.jsonl \
    --hard-file hard_eval_set.jsonl

# 自定义置信度阈值
python scripts/eval_baseline.py --confidence-threshold 0.9

# 自定义批处理大小（如果 GPU 内存不足）
python scripts/eval_baseline.py --batch-size 8

# 使用 CPU（如果 GPU 不可用）
python scripts/eval_baseline.py --device cpu
```

**输入格式：**

**位置：** `data/test.jsonl`, `data/hard_eval_set.jsonl`（可选）

**格式：** JSONL（每行一个 JSON 对象）

**必需字段：**
- `id`: 样本 ID（如 `s0`, `s1`）
- `text`: 文本内容
- `coarse_label`: 标签（0=非敏感, 1=敏感）

**示例：**
```json
{"id": "s0", "text": "这是一条测试文本", "coarse_label": 0}
{"id": "s1", "text": "这是另一条测试文本", "coarse_label": 1}
```

**输出格式：**

**位置：** `output/metrics_baseline.json`, `output/high_conf_error_samples.jsonl`

**metrics_baseline.json 结构：**
```json
{
  "test_set": {
    "accuracy": 0.8595,
    "precision": 0.8942,
    "recall": 0.8883,
    "f1": 0.8913,
    "confusion_matrix": [[1405, 337], [358, 2848]],
    "per_class": {
      "class_0": {
        "precision": 0.7969,
        "recall": 0.8064,
        "f1": 0.8016,
        "support": 1742
      },
      "class_1": {
        "precision": 0.8942,
        "recall": 0.8883,
        "f1": 0.8913,
        "support": 3206
      }
    },
    "total_samples": 4948
  },
  "hard_set": {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "confusion_matrix": [],
    "total_samples": 0
  },
  "high_conf_error_count": 444,
  "confidence_threshold": 0.8
}
```

**high_conf_error_samples.jsonl 结构：**
```json
{"id": "s123", "text": "错误样本文本", "true_label": 1, "pred_label": 0, "pred_prob": 0.95, "pred_probs": [0.95, 0.05]}
```

**配置参数（config.yaml）：**

```yaml
# 模型配置
model:
  name_or_path: "Qwen/Qwen3-1.7B"  # 或本地路径
  max_length: 512

# 训练配置（用于读取 checkpoint 路径）
training:
  output_dir: "checkpoints/baseline-lora"

# 困难子集配置（用于高置信错误阈值）
hardset:
  confidence_threshold: 0.8  # 高置信度阈值
```

**增量评估说明：**

- **新 baseline 模型训练后：**
  - 只需重新运行 `eval_baseline.py` 即可获得最新评估结果
  - 无需修改代码

- **新数据加入后：**
  - 重新生成 `test.jsonl` 和 `hard_eval_set.jsonl`
  - 重新运行 `eval_baseline.py` 即可

**工程/容器化约定：**

- 所有路径从 `config.yaml` 或 CLI 参数读取：
  - `data_dir`（默认 `./data`）
  - `output_dir`（默认 `./output`）
  - `checkpoint_path`（从 `training.output_dir` 读取）

- 模型路径：
  - 支持 HuggingFace 模型 ID（如 `Qwen/Qwen3-1.7B`）
  - 支持本地路径（如 `/mnt/workspace/models/qwen/Qwen3-1___7B`）

- 输出目录：
  - 统一使用 `output/` 目录
  - 方便云端和本地统一使用

**评估监控：**

- 评估过程中会输出：
  - 每 100 个批次的处理进度
  - 测试集和困难集的评估指标
  - 高置信错误样本数量

- 评估完成后会输出：
  - 完整的评估指标
  - Confusion Matrix
  - 输出文件路径

**注意事项：**

- **只评估二分类**：只关注敏感/非敏感二分类，不涉及子标签
- **困难集可选**：如果 `hard_eval_set.jsonl` 不存在或为空，评估仍可正常进行
- **内存优化**：如果显存不足，可以：
  - 减小 `--batch-size`（默认 16）
  - 使用 `--device cpu`（较慢但不需要 GPU）

---

## Day3 评估成果 ✅

### 评估完成情况

**评估时间：** 2025-12-12 11:08:34 ~ 11:12:20（约 3.8 分钟）

**评估配置：**
- **基础模型**：Qwen3-1.7B（本地路径：`/mnt/workspace/models/qwen/Qwen3-1___7B`）
- **Checkpoint**：`checkpoints/baseline-lora/`
- **测试集**：4,948 条样本（test.jsonl）
- **困难集**：0 条样本（hard_eval_set.jsonl 为空，可选）
- **批处理大小**：16
- **设备**：CUDA

### 测试集评估指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **Accuracy** | **85.95%** | 准确率 |
| **F1 Score** | **89.13%** | F1 分数（主要指标） |
| **Precision** | **89.42%** | 精确率 |
| **Recall** | **88.83%** | 召回率 |
| **总样本数** | **4,948** | 测试集样本数 |

### Confusion Matrix

```
预测\真实    非敏感(0)  敏感(1)
非敏感(0)     1405       358
敏感(1)        337      2848
```

**分析：**
- **真阴性（TN）**：1,405（正确预测为非敏感）
- **假阳性（FP）**：337（错误预测为敏感）
- **假阴性（FN）**：358（错误预测为非敏感）
- **真阳性（TP）**：2,848（正确预测为敏感）

### 分类别指标

**Class 0（非敏感）：**
- Precision: 79.69%
- Recall: 80.64%
- F1: 80.16%
- Support: 1,742 个样本

**Class 1（敏感）：**
- Precision: 89.42%
- Recall: 88.83%
- F1: 89.13%
- Support: 3,206 个样本

### 高置信错误样本

- **数量**：444 个
- **置信度阈值**：≥ 0.8
- **说明**：这些样本模型预测置信度很高（≥0.8），但预测错误，是需要重点关注的难例

### 模型保存位置

**输出文件：**
- `output/metrics_baseline.json` - 完整评估指标
- `output/high_conf_error_samples.jsonl` - 高置信错误样本列表

### 评估统计

- **总评估时间**：约 3.8 分钟
- **模型加载时间**：约 21 秒
- **测试集预测时间**：约 3.5 分钟（4,948 个样本，310 个批次）
- **指标计算时间**：< 1 秒

### 与 Day2 训练时验证集指标对比

| 指标 | Day2 验证集 | Day3 测试集 | 差异 |
|------|------------|------------|------|
| **Accuracy** | 84.6% | 85.95% | +1.35% |
| **F1 Score** | 88.3% | 89.13% | +0.83% |
| **Precision** | 87.1% | 89.42% | +2.32% |
| **Recall** | 89.6% | 88.83% | -0.77% |

**分析：**
- 测试集表现略好于验证集，说明模型泛化能力良好
- F1 分数提升 0.83 个百分点，模型性能稳定
- Precision 提升明显，说明模型在预测敏感内容时更加精确

### 结论

✅ **Day3 任务已完成**

- 基线模型评估成功完成
- 模型在测试集上表现良好（F1: 89.13%）
- 成功识别 444 个高置信错误样本
- 所有评估结果已保存到 `output/` 目录
- 评估结果可用于后续 Day4-Day12 的分析

**下一步：** 可以开始 Day4 的任务，进行不确定性分析（uncertainty_analysis.py）。

---

## 使用建议

### 1. 分析高置信错误样本

```bash
# 查看高置信错误样本
head -10 output/high_conf_error_samples.jsonl | python -m json.tool

# 统计错误类型
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

### 2. 对比不同置信度阈值

```bash
# 使用不同阈值重新识别错误样本
python scripts/eval_baseline.py --confidence-threshold 0.9
python scripts/eval_baseline.py --confidence-threshold 0.85
```

### 3. 生成困难集（可选）

如果需要评估困难集，可以先运行：

```bash
python scripts/hardset_maker.py
```

然后重新运行评估：

```bash
python scripts/eval_baseline.py
```

---

**最后更新**：2025-12-12  
**状态**：✅ 已完成

