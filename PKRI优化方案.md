# PKRI优化方案

**创建时间**：2025-12-15  
**基于分析**：Day11 q₀ vs q_PKRI 深度分析结果

---

## 📊 问题诊断

基于Day11分析结果，发现以下关键问题：

1. **PKRI可信度分布过窄**：
   - 平均可信度：0.6480
   - 范围：0.5496 - 0.6717
   - 标准差：仅0.0195（几乎无法区分）
   - >0.6的比例：98.69%（几乎都在0.6以上）

2. **q_PKRI过于保守**：
   - q_PKRI平均max概率：0.5744
   - q₀平均max概率：0.8006
   - 平均差异：-0.2262（q_PKRI远低于q₀）
   - q_PKRI < q₀ 的比例：85.27%

3. **准确率对比**：
   - Baseline：85.95%
   - q₀：69.38%
   - q_PKRI：62.79%（最差）

4. **预测一致性差**：
   - q₀ vs q_PKRI 预测不一致：30.32%
   - 相关系数：0.4124（中等相关性）

---

## 🎯 优化方案（按优先级）

### P0-优化1：PKRI可信度映射改进 ⭐⭐⭐

**问题**：
- 当前可信度范围过窄(0.55-0.67)，标准差仅0.0195
- 无法有效区分高可信度/低可信度样本
- 代码位置：`scripts/pkri_train.py:226-247`的`predict_confidence`函数

**当前代码**：
```python
def predict_confidence(features, model):
    proba = model.predict_proba(features)[:, 1]
    confidence = proba  # 直接使用模型概率
    return confidence
```

**优化方案**：

**方案1a：Tanh拉伸映射（推荐）**
```python
def predict_confidence(features, model):
    proba = model.predict_proba(features)[:, 1]
    # 将[0.5, 1.0]映射到[0.3, 0.9]，让分布更分散
    confidence = 0.3 + 0.6 * np.tanh((proba - 0.5) * 4)
    return confidence
```

**方案1b：分段线性映射**
```python
def predict_confidence(features, model):
    proba = model.predict_proba(features)[:, 1]
    # 低概率区间扩展，高概率区间保持
    confidence = np.where(
        proba < 0.6,
        0.4 + (proba - 0.5) * 2,      # [0.5,0.6] -> [0.4,0.6]
        0.6 + (proba - 0.6) * 0.75    # [0.6,1.0] -> [0.6,0.9]
    )
    return confidence
```

**方案1c：温度缩放（更复杂但更科学）**
```python
def predict_confidence(features, model):
    # 需要先在验证集上学习温度参数T
    # T<1会让分布更分散
    logits = model.decision_function(features)  # 需要访问logits
    temperature = 0.5  # 可学习参数
    confidence = 1.0 / (1.0 + np.exp(-logits / temperature))
    return confidence
```

**预期效果**：
- 可信度范围扩展至[0.3, 0.9]或更宽
- 标准差提升至0.1以上
- 能更好区分高/低可信度样本

**实施步骤**：
1. 修改`scripts/pkri_train.py`的`predict_confidence`函数
2. 重新训练PKRI模型
3. 重新生成q_PKRI文件
4. 运行`analyze_q0_vs_qpkri.py`验证改进效果

---

### P0-优化2：q_PKRI构建公式改进 ⭐⭐⭐

**问题**：
- 当前公式：`p_sensitive = base_prob + (max_prob - base_prob) × match_strength × confidence`
- `match_strength × confidence`双重衰减，导致q_PKRI过于保守
- q₀=0.9的案例，q_PKRI≈0.5（几乎无信息）

**当前代码**：
```python
# scripts/pkri_train.py:298
p_sensitive = base_prob + (max_prob - base_prob) * match_strength * confidence
```

**优化方案**：

**方案2a：基于q₀调整（推荐）**
```python
def build_qpkri_from_q0(q0_p_sensitive, confidence, alpha=0.7):
    """
    基于q₀构建q_PKRI，用可信度调整
    
    Args:
        q0_p_sensitive: q₀的敏感概率
        confidence: PKRI可信度
        alpha: q₀权重（默认0.7）
    
    Returns:
        q_pkri: [p_non_sensitive, p_sensitive]
    """
    # 将confidence从[0.5, 0.9]映射到[0.5, 1.3]（增强作用）
    confidence_adjusted = 0.5 + (confidence - 0.5) * 2
    
    # 混合q₀和可信度
    qpkri_p = q0_p_sensitive * (alpha + (1-alpha) * confidence_adjusted)
    qpkri_p = np.clip(qpkri_p, 0.1, 0.9)  # 限制范围
    
    return [1.0 - qpkri_p, qpkri_p]
```

**方案2b：改进可信度加权（保持独立计算）**
```python
# 将乘积改为加权和
p_sensitive = base_prob + (max_prob - base_prob) * (
    match_strength * 0.7 + confidence * 0.3
)
```

**方案2c：可信度作为偏移而非缩放**
```python
# confidence作为偏移，而不是缩放因子
p_sensitive = base_prob + (max_prob - base_prob) * match_strength * (
    confidence + 0.3  # 偏移，让概率提升
)
```

**预期效果**：
- q_PKRI概率分布更接近q₀
- 高可信度时能充分利用知识
- q_PKRI准确率提升至接近q₀

**实施步骤**：
1. 修改`scripts/pkri_train.py`的`build_qpkri`函数
2. 如果使用方案2a，需要先加载q₀文件
3. 重新生成q_PKRI文件
4. 验证效果

---

### P1-优化3：词表清理（手动审查） ⭐⭐

**问题**：
- politics.txt中包含大量通用词（Android、App、America、Army等）
- 词表噪声导致PKRI特征偏差，影响模型训练

**优化方案**：

1. **手动审查politics.txt**：
   - 识别并移除明显通用词
   - 重点检查：技术词汇、常见人名、普通名词

2. **创建噪声词黑名单**：
   - 在`scripts/clean_lexicon.py`中添加黑名单过滤
   - 或单独创建审查脚本

3. **验证清理效果**：
   - 清理后重新训练PKRI模型
   - 对比清理前后的q_PKRI效果

**预期效果**：
- 减少误匹配
- PKRI特征质量提升
- 模型训练更稳定

**实施步骤**：
1. 手动审查politics.txt（优先级最高）
2. 创建黑名单或过滤脚本
3. 清理词表
4. 重新训练PKRI模型
5. 验证效果

---

### P1-优化4：特征工程改进 ⭐⭐

**问题**：
- 当前21个特征可能不够区分好坏匹配
- 特征选择可能冗余或不足
- 未做特征标准化

**优化方案**：

1. **特征重要性分析**：
   ```python
   # 使用随机森林分析特征重要性
   from sklearn.ensemble import RandomForestClassifier
   rf = RandomForestClassifier()
   rf.fit(X_train, y_train)
   feature_importance = pd.DataFrame({
       'feature': feature_names,
       'importance': rf.feature_importances_
   }).sort_values('importance', ascending=False)
   ```

2. **特征标准化**：
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_val_scaled = scaler.transform(X_val)
   ```

3. **新增特征**：
   - 匹配词的TF-IDF权重
   - 匹配位置特征（开头/中间/结尾）
   - 匹配词的上下文窗口
   - 匹配词的共现频率

4. **特征选择**：
   - 移除冗余特征
   - 使用LASSO正则化做特征选择

**预期效果**：
- 特征质量提升
- 模型训练更稳定
- PKRI可信度建模更准确

**实施步骤**：
1. 特征重要性分析
2. 添加特征标准化
3. 逐步新增特征
4. 验证效果

---

### P2-优化5：PKRI模型升级 ⭐

**问题**：
- 逻辑回归可能过于简单
- 非线性关系可能无法捕获

**优化方案**：

1. **尝试随机森林**：
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(
       n_estimators=100,
       max_depth=10,
       random_state=42
   )
   ```
   - 优点：特征重要性分析、非线性建模
   - 缺点：可解释性较差

2. **尝试XGBoost**：
   ```python
   import xgboost as xgb
   model = xgb.XGBClassifier(
       n_estimators=100,
       max_depth=5,
       learning_rate=0.1,
       random_state=42
   )
   ```
   - 优点：强大的非线性建模能力
   - 缺点：需要额外依赖

3. **改进校准方法**：
   - 温度缩放（Temperature Scaling）
   - Isotonic回归（比Platt scaling更灵活）

**预期效果**：
- 模型性能提升
- 可信度建模更准确
- 但可能增加复杂度

**实施步骤**：
1. 先完成P0和P1优化
2. 在优化后的特征和数据集上尝试新模型
3. 对比不同模型效果
4. 选择最优模型

---

## 📋 实施计划

### 第一阶段：快速验证（1-2天）

1. ✅ **优化1**：PKRI可信度映射改进（方案1a）
2. ✅ **优化2**：q_PKRI构建公式改进（方案2a或2b）
3. 重新训练PKRI模型并生成q_PKRI
4. 运行分析脚本验证效果

### 第二阶段：中期报告前（1周）

1. ✅ **优化3**：词表清理（至少politics.txt）
2. ✅ **优化4**：特征标准化
3. 对比优化前后效果
4. 更新文档和报告

### 第三阶段：后续优化

1. **优化4**：特征工程深度改进
2. **优化5**：模型升级（如果需要）

---

## 📝 注意事项

1. **每次优化后都要验证**：
   - 重新训练PKRI模型
   - 重新生成q_PKRI文件
   - 运行`analyze_q0_vs_qpkri.py`对比效果

2. **保持代码可回退**：
   - 使用Git版本控制
   - 重要修改前创建分支

3. **记录实验结果**：
   - 记录每次优化的效果
   - 对比指标：准确率、可信度分布、q_PKRI分布

4. **优先级排序**：
   - P0优化必须完成（核心问题）
   - P1优化尽量完成（提升质量）
   - P2优化可选（锦上添花）

---

## 🔄 验证指标

每次优化后，检查以下指标：

1. **PKRI可信度分布**：
   - 范围是否扩展（目标：[0.3, 0.9]）
   - 标准差是否提升（目标：>0.1）

2. **q_PKRI概率分布**：
   - 平均max概率是否提升（目标：>0.65）
   - 是否更接近q₀（差异<0.1）

3. **准确率**：
   - q_PKRI准确率是否提升（目标：>70%）
   - 是否接近q₀准确率

4. **EMG融合效果**：
   - EMG+q_PKRI是否优于Baseline
   - 是否接近或优于EMG+q₀

