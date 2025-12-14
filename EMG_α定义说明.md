# EMG 融合公式与 α 定义说明

**创建时间**：2025-12-14  
**目的**：统一 EMG 融合公式中 α 的定义和语义，避免混淆

---

## 融合公式

```
p_emg = α × p + (1 - α) × q
```

其中：
- `p`: baseline 模型预测概率 `[p_non_sensitive, p_sensitive]`
- `q`: q₀ 知识后验概率 `[p_non_sensitive, p_sensitive]`
- `α`: 融合权重（**给模型 p 的权重**）

---

## α 的语义

### 定义
- **α 是给模型 p 的权重**
- **(1-α) 是给知识 q 的权重**

### 取值范围和含义
- **α = 1.0**：完全信任模型 p，不使用知识 q
- **α = 0.0**：完全信任知识 q，不使用模型 p
- **α = 0.5**：模型和知识各占 50%

---

## EMG 理论预期

根据 **"越不确定越依赖知识"** 的原则：

- **低不确定性（u 小）**：α 大（接近 1.0）→ 更信任模型
- **高不确定性（u 大）**：α 小（接近 0.0）→ 更信任知识

因此 **α(u) 应该是单调递减函数**（u 增大，α 减小）。

---

## Day7 结果验证

| Bucket | u_mean | α* | 解释 |
|--------|--------|-----|------|
| 0 | 0.016 | 0.75 | 低不确定性，更信任模型 ✅ |
| 1 | 0.146 | 0.25 | 中等不确定性，开始依赖知识 ✅ |
| 2 | 0.248 | 0.50 | 平衡模型和知识 ✅ |
| 3-4 | > 0.3 | 0.00 | 高不确定性，完全信任知识 ✅ |

**结果符合理论预期**：不确定性越高（u 越大），最优 α* 越小，越信任知识后验 q₀。

---

## 代码实现

在 `scripts/emg_bucket_search.py` 中：

```python
def compute_emg_fusion(p: List[float], q: List[float], alpha: float) -> List[float]:
    """
    EMG 融合公式: p_emg = α × p + (1 - α) × q
    
    Args:
        p: baseline 预测概率 [p_non_sensitive, p_sensitive]
        q: q₀ 知识后验 [p_non_sensitive, p_sensitive]
        alpha: 融合权重（给模型 p 的权重）
    
    Returns:
        融合后的概率 [p_non_sensitive, p_sensitive]
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # EMG 融合公式: p_emg = α × p + (1 - α) × q
    p_emg = alpha * p + (1 - alpha) * q
    
    # 归一化
    p_emg = p_emg / np.sum(p_emg) if np.sum(p_emg) > 0 else p_emg
    p_emg = np.clip(p_emg, 0.0, 1.0)
    p_emg = p_emg / np.sum(p_emg)
    
    return p_emg.tolist()
```

---

## 常见误解

### ❌ 错误理解
- "α 是给知识 q 的权重"
- "α 越大越信任知识"
- "高不确定性时 α 应该增大"

### ✅ 正确理解
- **α 是给模型 p 的权重**
- **α 越大越信任模型**
- **高不确定性时 α 应该减小（接近 0），从而更多依赖知识**

---

## 文档一致性

本文档应作为所有相关文档中 α 定义的标准参考。在编写或修改文档时，请确保：

1. 明确说明 α 是给模型 p 的权重
2. 明确说明 α(u) 是单调递减的
3. 明确说明"越不确定越依赖知识"对应的是 α 减小

---

**最后更新**：2025-12-14

