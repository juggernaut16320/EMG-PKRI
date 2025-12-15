# Day1-Day12 调参任务总结

**最后更新**：2025-12-15

---

## 📊 调参任务概览

从Day1到Day12中，以下任务需要调参和寻找最佳参数：

### ✅ **需要调参的任务（5个）**

| Day | 任务 | 调参内容 | 调参方式 | 优先级 | 状态 |
|-----|------|---------|---------|--------|------|
| **Day2** | Baseline训练 | 训练超参数 | 手工调整或网格搜索 | P2（可选） | ✅ 已完成（使用默认参数） |
| **Day6** | q₀构建 | q₀参数 | 网格搜索（脚本已创建） | **P0（快速模式已完成）** | ✅ 快速模式完成 |
| **Day7** | EMG α搜索 | α网格粒度 | 手工调整（可选） | P2（可选） | ✅ 已完成（使用默认网格） |
| **Day9** | 知识阈值门控 | 知识阈值 | 自动搜索（脚本已创建） | P1（已完成） | ✅ 已完成 |
| **Day10** | PKRI构建 | PKRI模型参数、构建方法 | 手工调整或实验对比 | P1（部分完成） | ⏳ 部分完成 |

---

## 📋 详细说明

### Day2: Baseline训练 ⭐⭐⭐

**调参内容**：
- **训练超参数**（`configs/config.yaml`）：
  - `learning_rate`: 2e-4（学习率）
  - `per_device_train_batch_size`: 8（训练批次大小）
  - `num_epochs`: 3（训练轮数）
  - `warmup_steps`: 100（Warmup步数）
- **LoRA超参数**：
  - `lora_r`: 8（LoRA rank）
  - `lora_alpha`: 16（LoRA alpha）
  - `lora_dropout`: 0.1（LoRA dropout）
  - `target_modules`: ["q_proj", "v_proj"]（目标模块）

**调参方式**：
- 目前使用默认参数，效果良好（验证集F1: 88.3%）
- 如果需要进一步优化，可以：
  - 尝试不同的learning rate（1e-4, 2e-4, 5e-4）
  - 调整batch size（根据显存）
  - 尝试不同的LoRA rank（4, 8, 16）

**当前状态**：✅ 已完成，使用默认参数效果良好

**是否需要调参**：❌ **可选**（当前性能已满足需求）

---

### Day6: q₀构建 ⭐⭐⭐⭐⭐ **（当前优先级最高）**

**调参内容**（`configs/config.yaml` 和 `scripts/q0_builder.py`）：
- `base_sensitive_prob`: 0.1（基础敏感概率）
- `max_sensitive_prob`: 0.75（最大敏感概率）
- `min_matches_for_sensitive`: 1（触发敏感的最小匹配数）
- `politics_weight`: 0.6（涉政权重）
- `abuse_weight`: 0.5（辱骂权重）
- `porn_weight`: 1.0（色情权重，基准）
- `tanh缩放因子`: 5（概率映射缩放因子）

**调参方式**：
- **脚本**：`scripts/q0_hyperparameter_search.py`
- **快速模式**（推荐）：
  - 搜索空间：12种组合
  - 搜索参数：`min_matches`, `max_sensitive_prob`, `base_sensitive_prob`
  - 预计耗时：1-2小时
- **完整模式**：
  - 搜索空间：384种组合
  - 搜索所有关键参数
  - 预计耗时：6-12小时

**调优目标**：
- **方案A**（推荐）：优化q0自身质量（最大化q0的F1分数）
- **方案B**：优化EMG最终效果（需要运行完整Day7流程，耗时）

**当前状态**：✅ **快速模式已完成**（2025-12-15）

**是否需要调参**：✅ **已完成快速模式**（可考虑完整模式或直接应用结果）

**调参结果**（快速模式，12种组合）：
- **最优参数**：
  - `min_matches_for_sensitive`: 1
  - `max_sensitive_prob`: 0.85
  - `base_sensitive_prob`: 0.15
  - `porn_weight`: 1.0
  - `politics_weight`: 0.6
  - `abuse_weight`: 0.5
- **最优效果**：
  - Precision: **86.13%**（保持稳定）
  - Recall: **37.96%**（+7.5%，目标>40%，接近）
  - F1: **52.70%**（+5.1%，目标>55%，接近但未达）
- **Top 3组合**：
  1. F1=52.70%: min_matches=1, max_prob=0.85, base_prob=0.15
  2. F1=52.55%: min_matches=1, max_prob=0.85, base_prob=0.1
  3. F1=51.78%: min_matches=1, max_prob=0.75, base_prob=0.15

**分析**：
- ✅ F1从50.14%提升到52.70%（+5.1%），改善明显
- ⚠️ 未达到目标55%，但这是rule-based方法的固有局限
- ✅ Precision保持86%以上，质量可靠
- 📊 相比初始状态（30.44%）提升73%，改善显著

**下一步**：
1. **应用最优参数**：更新config.yaml并重新生成q0
2. **验证EMG效果**：重新运行Day7-Day9，验证最终效果
3. **可选**：运行完整模式调参（384种组合，6-12小时，可能提升有限）

---

### Day7: EMG α搜索 ⭐⭐

**调参内容**（`configs/config.yaml`）：
- `alpha_grid`: [0, 0.25, 0.5, 0.75, 1.0]（α网格）
- `metric`: "f1"（优化指标：f1或nll）
- `bucket_type`: "equal_width"（分桶类型）

**调参方式**：
- 目前使用默认5点网格，已找到最优α*
- 如果需要更精细的搜索，可以：
  - 使用更细粒度网格（0.1步长）：[0, 0.1, 0.2, ..., 1.0]
  - 尝试不同分桶策略（等宽 vs 等频）

**当前状态**：✅ 已完成，结果符合理论预期

**是否需要调参**：❌ **可选**（当前网格已足够，结果良好）

---

### Day9: 知识阈值门控 ⭐⭐⭐

**调参内容**：
- `knowledge_threshold`: 知识阈值（用于门控判断）

**调参方式**：
- **脚本**：`scripts/search_knowledge_threshold.py`
- **方法**：在验证集上自动搜索最优阈值
- **搜索范围**：默认网格（如[0.4, 0.5, 0.6, 0.7, 0.8, 0.9]）
- **优化指标**：F1或NLL

**当前状态**：✅ 已完成（已实施，阈值=0.9，效果不明显但已实施）

**是否需要调参**：❌ **已完成**（已通过脚本自动搜索）

**注意**：虽然已实施，但效果不明显，可以考虑进一步优化搜索策略

---

### Day10: PKRI构建 ⭐⭐⭐⭐

**调参内容**（`scripts/pkri_train.py`）：

#### 1. PKRI模型参数
- `model_type`: "lr"（模型类型，目前使用逻辑回归）
- LR模型的超参数（C值等，由sklearn自动处理）

#### 2. 校准方法参数
- `calibration_method`: "platt"（Platt scaling）
  - 可选：`platt`, `temperature`（温度缩放）

#### 3. 可信度映射参数（已实施 ✅）
- `confidence_mapping`: "tanh"（Tanh拉伸映射）
  - 可选：`tanh`, `piecewise`, `raw`
  - Tanh参数：`confidence = 0.3 + 0.6 * np.tanh((proba - 0.5) * 4)`

#### 4. q_PKRI构建方法参数（已实施 ✅）
- `build_method`: "improved_weighted"（改进加权融合）
  - 可选：`original`, `improved_weighted`, `q0_based`, `offset`
  - `improved_weighted`公式：`p_sensitive = base_prob + (max_prob - base_prob) * (match_strength * 0.7 + confidence * 0.3)`

**调参方式**：
- **当前**：已通过实验选择`tanh`映射和`improved_weighted`构建方法
- **可选优化**：
  - 尝试不同的置信度映射参数（tanh的系数）
  - 尝试不同的构建方法权重（0.7和0.3的权重）
  - 尝试不同的校准方法

**当前状态**：⏳ **部分完成**（已选择较优方案，但可进一步微调）

**是否需要调参**：⚠️ **可选**（当前方案已较优，进一步调参收益可能有限）

---

## ❌ **不需要调参的任务**

| Day | 任务 | 原因 |
|-----|------|------|
| **Day1** | 数据准备 | 数据处理流程，无参数需要调优 |
| **Day3** | Baseline评估 | 评估脚本，无参数 |
| **Day4** | 不确定性分析 | 计算不确定性u，无参数 |
| **Day5** | 校准分析 | 计算ECE/MCE，无参数 |
| **Day8** | PAV拟合 | 算法本身，无需调参 |
| **Day11** | EMG+PKRI对比 | 对比实验，无参数 |
| **Day12** | 可视化总结 | 报告生成，无参数 |

---

## 🎯 调参优先级和建议

### **P0（当前必须）**
1. **Day6: q₀参数调优** ⭐⭐⭐⭐⭐
   - 脚本已创建：`q0_hyperparameter_search.py`
   - 当前状态：✅ **快速模式已完成**（2025-12-15）
   - 调优结果：F1从50.14%提升到52.70%（+5.1%）
   - 最优参数已确定，待应用并验证EMG效果
   - **下一步**：应用最优参数，重新运行Day6-Day9验证最终效果
   - **可选**：运行完整模式（384种组合，6-12小时，可能提升有限）

### **P1（重要但非紧急）**
2. **Day10: PKRI参数微调** ⭐⭐⭐⭐
   - 当前已有较优方案
   - 可选的进一步优化：置信度映射参数、构建方法权重
   - 预期收益：可能进一步提升q_PKRI质量

### **P2（可选优化）**
3. **Day2: Baseline训练超参数** ⭐⭐⭐
   - 当前性能已满足需求（F1: 89.13%）
   - 如需进一步优化，可尝试不同的学习率、batch size等
   - 预期收益：可能提升1-2%的F1

4. **Day7: α网格粒度** ⭐⭐
   - 当前网格已足够（5点）
   - 如需更精细，可使用0.1步长网格
   - 预期收益：可能略微改善α*精度

5. **Day9: 知识阈值搜索策略** ⭐⭐⭐
   - 已实施自动搜索
   - 可进一步优化搜索范围或策略
   - 预期收益：可能进一步提升门控效果

---

## 📝 调参工作流建议

### 当前推荐顺序

1. **第一步（P0）**：Day6 q₀参数调优 ✅ **已完成**
   - 快速模式已完成（2025-12-15）
   - 最优参数：min_matches=1, max_prob=0.85, base_prob=0.15
   - 最优F1: 52.70%（+5.1%）

2. **第二步（当前进行）**：应用最优参数并重新运行Day6-Day9
   - ✅ 更新`configs/config.yaml`中的q0参数（待执行）
   - ⏳ 重新生成q0（使用最优参数）
   - ⏳ 重新运行Day7（α*搜索）
   - ⏳ 重新运行Day8（α(u)拟合）
   - ⏳ 重新运行Day9（EMG评估）

3. **第三步（可选）**：如果效果不理想，考虑完整模式调参
   ```bash
   python scripts/q0_hyperparameter_search.py \
       --dataset-file data/dev.jsonl \
       --baseline-file output/dev_with_uncertainty.jsonl \
       --lexicon-dir configs/lexicons \
       --mode full \
       --output-dir output
   ```
   - 搜索空间：384种组合
   - 预计耗时：6-12小时
   - 预期收益：可能提升有限（rule-based方法固有局限）

3. **第三步（可选P1）**：Day10 PKRI参数微调
   - 如果q_PKRI效果不理想，可尝试不同的置信度映射或构建方法

4. **第四步（可选P2）**：其他可选优化
   - Day2训练超参数（如果需要进一步提升baseline）
   - Day7 α网格细化（如果需要更精细的α*）

---

## 🔍 参数范围参考

### Day6 q₀参数（快速模式搜索范围）
- `min_matches_for_sensitive`: [1, 2]
- `max_sensitive_prob`: [0.65, 0.75, 0.85]
- `base_sensitive_prob`: [0.1, 0.15]

### Day6 q₀参数（完整模式搜索范围）
- `min_matches_for_sensitive`: [1, 2]
- `max_sensitive_prob`: [0.65, 0.75, 0.85, 0.95]
- `base_sensitive_prob`: [0.05, 0.1, 0.15, 0.2]
- `politics_weight`: [0.5, 0.6, 0.7, 0.8]
- `abuse_weight`: [0.4, 0.5, 0.6, 0.7]

### Day2 Baseline训练参数（可选优化范围）
- `learning_rate`: [1e-4, 2e-4, 5e-4]
- `per_device_train_batch_size`: [8, 16, 32]（根据显存）
- `lora_r`: [4, 8, 16]
- `lora_alpha`: [8, 16, 32]

---

## ✅ 总结

**需要调参的任务**：5个（Day2, Day6, Day7, Day9, Day10）

**当前优先级最高**：**Day6 q₀参数调优**（P0，必须执行）

**已完成调参**：Day9知识阈值门控（已自动搜索）

**可选调参**：Day2, Day7, Day10的进一步优化

