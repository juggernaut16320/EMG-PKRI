# Day11: EMG + PKRI 联合融合试验

## 任务目标

检验使用 **q_PKRI**（带可信度建模的知识后验）替代 **q₀**（规则基础的知识后验）是否能为EMG融合带来性能提升。

### 核心问题

- q_PKRI 通过机器学习建模知识可信度，理论上应该比 q₀ 更准确
- 但在EMG融合中，这种改进是否能转化为实际性能提升？
- 需要在相同条件下（相同α(u)函数）进行公平对比

---

## 方案选型

### 方案一（推荐）：扩展现有 eval_emg.py 脚本

**优点**：
- 代码复用高，逻辑一致
- 实现简单，只需添加 `--knowledge-source` 参数
- 输出格式统一，便于对比

**实现方式**：
- 修改 `evaluate_method` 函数，支持传入 q₀ 或 q_PKRI 字典
- 在 `main` 函数中添加 `--knowledge-source` 参数（可选值：`q0` 或 `qpkri`）
- 根据参数选择加载对应的知识后验文件
- 输出文件名包含知识源标识（如 `metrics_emg_qpkri.json`）

**适用场景**：快速验证PKRI效果

---

### 方案二：创建独立的对比脚本

**优点**：
- 代码更清晰，职责分离
- 可以一次性对比多个知识源
- 便于扩展更多知识源（如 q_hybrid）

**实现方式**：
- 创建新脚本 `scripts/eval_emg_comparison.py`
- 同时加载 q₀ 和 q_PKRI
- 生成统一的对比报告

**适用场景**：需要详细对比分析和可视化

---

### 方案三：批处理脚本 + 现有工具

**优点**：
- 无需修改代码
- 灵活性高

**实现方式**：
- 分别运行两次 `eval_emg.py`（一次用 q₀，一次用 q_PKRI）
- 用脚本汇总对比结果

**适用场景**：临时对比，不需要集成到主流程

---

## 推荐方案：方案一

**理由**：
1. 代码改动最小，风险低
2. 与现有Day9评估流程无缝集成
3. 便于后续扩展到其他知识源

---

## 输入文件

### 必需文件（来自前置任务）

1. **Baseline预测**：
   - `data/test_with_uncertainty.jsonl`：测试集的模型预测和不确定性（Day4输出）
   - `data/dev_with_uncertainty.jsonl`：验证集的模型预测和不确定性（可选）

2. **α(u)查表**：
   - `output/alpha_u_lut.json`：基于q₀训练的α(u)查表（Day8输出）

3. **知识后验文件**：
   - **q₀后验**（Day6输出）：
     - `data/q0_test.jsonl`
     - `data/q0_dev.jsonl`（可选）
   - **q_PKRI后验**（Day10输出）：
     - `data/qpkri_test.jsonl`
     - `data/qpkri_dev.jsonl`（可选）

4. **标签文件**（用于计算指标）：
   - `data/test.jsonl`：测试集真实标签
   - `data/dev.jsonl`：验证集真实标签（可选）

### 可选文件

- `output/knowledge_threshold.json`：最优知识阈值（Day9改进输出，用于门控）
- `configs/config.yaml`：配置文件（用于读取q₀相关参数）

---

## 输出文件

### 主要输出

1. **评估指标文件**：
   - `output/metrics_emg_q0.json`：使用q₀的EMG评估结果
   - `output/metrics_emg_qpkri.json`：使用q_PKRI的EMG评估结果
   - `output/metrics_emg_comparison.json`：对比汇总结果（可选）

2. **对比表格**：
   - `output/emg_q0_vs_qpkri_comparison.csv`：q₀ vs q_PKRI 对比表
   - 包含指标：Accuracy, F1, Precision, Recall, NLL, ECE

3. **可视化图表**（可选）：
   - `output/emg_q0_vs_qpkri_charts.png`：对比可视化
     - F1对比柱状图
     - NLL对比柱状图
     - ECE对比柱状图
     - 不确定性切片对比

### 输出格式示例

**metrics_emg_qpkri.json**：
```json
{
  "knowledge_source": "qpkri",
  "test_set": {
    "baseline": {
      "accuracy": 0.8913,
      "f1": 0.8913,
      "nll": 0.4001,
      "ece": 0.0730
    },
    "fixed_alpha_0.5": {
      "accuracy": 0.8850,
      "f1": 0.8850,
      "nll": 0.3950,
      "ece": 0.0700
    },
    "emg": {
      "accuracy": 0.8913,
      "f1": 0.8913,
      "nll": 0.3871,
      "ece": 0.0737
    }
  },
  "uncertainty_slices": {
    "low_u": {"f1": 0.9319, "nll": 0.3067, "ece": 0.0673},
    "medium_u": {"f1": 0.7093, "nll": 0.8031, "ece": 0.0795},
    "high_u": {"f1": 0.6496, "nll": 1.3226, "ece": 0.0097}
  },
  "improvements_vs_baseline": {
    "emg": {
      "f1_delta": 0.0000,
      "nll_delta": -0.0130,
      "ece_delta": 0.0007
    }
  }
}
```

**emg_q0_vs_qpkri_comparison.csv**：
```csv
method,knowledge_source,accuracy,f1,precision,recall,nll,ece
baseline,q0,0.8913,0.8913,0.8900,0.8926,0.4001,0.0730
baseline,qpkri,0.8913,0.8913,0.8900,0.8926,0.4001,0.0730
fixed_alpha_0.5,q0,0.8850,0.8850,0.8835,0.8865,0.3950,0.0700
fixed_alpha_0.5,qpkri,0.8860,0.8860,0.8845,0.8875,0.3940,0.0695
emg,q0,0.8913,0.8913,0.8900,0.8926,0.3871,0.0737
emg,qpkri,0.8920,0.8920,0.8907,0.8933,0.3850,0.0720
```

---

## 运行流程

### 步骤1：准备输入文件

确保以下文件存在（来自Day4, Day6, Day8, Day10）：
```bash
# 检查文件
ls -lh data/test_with_uncertainty.jsonl
ls -lh data/q0_test.jsonl
ls -lh data/qpkri_test.jsonl
ls -lh output/alpha_u_lut.json
```

### 步骤2：使用q₀运行EMG评估

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

**注意**：
- `--q0-file` 参数现在可以接受 q₀ 或 q_PKRI 文件路径
- `--knowledge-source` 参数指定知识源类型，会自动选择默认文件路径
- 如果不指定 `--q0-file`，会根据 `--knowledge-source` 自动选择默认路径：
  - `q0` → `data/q0_test.jsonl`
  - `qpkri` → `data/qpkri_test.jsonl`

### 步骤4：对比分析（可选脚本）

```bash
# 如果实现了对比脚本
python scripts/compare_q0_vs_qpkri.py \
    --metrics-q0 output/metrics_emg_q0.json \
    --metrics-qpkri output/metrics_emg_qpkri.json \
    --output output/emg_q0_vs_qpkri_comparison.json \
    --output-csv output/emg_q0_vs_qpkri_comparison.csv
```

---

## 预期对比维度

### 1. 全局指标对比

| 知识源 | F1 | NLL | ECE | 说明 |
|--------|----|----|-----|-----|
| q₀ | 89.13% | 0.3871 | 0.0737 | Day9基线 |
| q_PKRI | ? | ? | ? | 待评估 |

**关注点**：
- F1是否提升（主要目标）
- NLL是否降低（概率质量）
- ECE是否改善（校准质量）

### 2. 不确定性切片对比

| 切片 | q₀ (F1/NLL) | q_PKRI (F1/NLL) | 变化 |
|------|-------------|-----------------|------|
| 低u（u < 0.1） | 93.19% / 0.3067 | ? / ? | ? |
| 中等u（0.1≤u<0.3） | 70.93% / 0.8031 | ? / ? | ? |
| 高u（u ≥ 0.3） | 64.96% / 1.3226 | ? / ? | ? |

**关注点**：
- 高不确定性样本是否有明显改善（q_PKRI的可信度建模优势）
- 低不确定性样本是否保持性能（不应退化）

### 3. 门控机制影响

对比在不同门控配置下的表现：
- 无门控
- 仅知识阈值门控
- 仅一致性门控
- 双重门控

---

## 实现要点

### 1. 代码修改点

**scripts/eval_emg.py**：
- 添加 `--knowledge-source` 参数（默认 `q0`）
- 修改 `load_q0_posteriors` 函数，支持加载 q_PKRI（字段名从 `q0` 改为 `qpkri`）
- 输出文件名根据知识源动态生成

### 2. 关键代码片段

```python
# 加载知识后验
if args.knowledge_source == 'qpkri':
    knowledge_dict = load_qpkri_posteriors(args.knowledge_file)
    logger.info(f"加载 q_PKRI: {len(knowledge_dict)} 条")
else:
    knowledge_dict = load_q0_posteriors(args.knowledge_file)
    logger.info(f"加载 q₀: {len(knowledge_dict)} 条")

# 输出文件名
output_metrics = args.output_metrics or f"output/metrics_emg_{args.knowledge_source}.json"
```

### 3. 注意事项

- **α(u)函数**：使用相同的α(u)查表（基于q₀训练的），确保公平对比
- **门控参数**：使用相同的知识阈值和一致性门控设置
- **数据对齐**：确保q₀和q_PKRI的样本ID完全一致

---

## 预期结果与结论

### 成功标准

1. **主要指标**：q_PKRI的EMG在F1上不劣于q₀（≥ 89.13%）
2. **次要指标**：NLL或ECE至少一项有改善（降低>1%）
3. **特殊情况**：高不确定性切片（u ≥ 0.3）有明显改善

### 可能结论

- **结论A**：q_PKRI全面优于q₀
  - → 推荐使用q_PKRI作为默认知识源
- **结论B**：q_PKRI在高不确定性样本上更优
  - → 可考虑混合策略：低u用q₀，高u用q_PKRI
- **结论C**：q_PKRI与q₀性能相当
  - → 考虑到复杂度，优先使用q₀
- **结论D**：q_PKRI性能不如q₀
  - → 需要分析原因（特征质量、模型校准、数据分布等）

---

## 扩展方向

1. **混合策略**：根据不确定性动态选择q₀或q_PKRI
2. **加权融合**：同时使用q₀和q_PKRI，根据置信度加权
3. **特征重要性分析**：分析PKRI模型中哪些特征对性能提升贡献最大
4. **校准优化**：进一步优化q_PKRI的校准方法（温度缩放、Platt scaling等）

---

## 相关文档

- [Day6: q₀构建](./技术文档.md#day6-构建规则基础的知识后验q₀)
- [Day8: α(u)拟合](./技术文档.md#day8-pav保序回归生成单调αu)
- [Day9: EMG评估](./技术文档.md#day9-emg效果验证与门控优化)
- [Day10: PKRI构建](./技术文档.md#day10-pkri知识可信度建模构建)

