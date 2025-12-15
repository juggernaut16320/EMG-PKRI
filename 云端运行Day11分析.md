# Day11 结果分析：定位q_PKRI效果不如q₀的原因

## 运行分析脚本

在云端运行以下命令来深入分析q₀和q_PKRI的差异：

```bash
cd /mnt/workspace/EMG-PKRI
source venv/bin/activate

# 运行深度分析
python scripts/analyze_q0_vs_qpkri.py \
    --q0-file data/q0_test.jsonl \
    --qpkri-file data/qpkri_test.jsonl \
    --baseline-file data/test_with_uncertainty.jsonl \
    --output-dir output
```

## 分析维度

脚本会从以下维度分析：

1. **后验概率分布分析**
   - q₀和q_PKRI的概率分布统计
   - PKRI可信度分布
   - 预测一致性
   - 相关性分析
   - 差异分析

2. **按不确定性切片分析**
   - 低/中/高不确定性切片的对比
   - 各切片上的平均概率和可信度

3. **预测准确性分析**
   - q₀和q_PKRI的准确率对比
   - 正确/错误预测的置信度分析

4. **极端差异案例分析**
   - q_PKRI显著高于q₀的案例
   - q_PKRI显著低于q₀的案例

## 可能的原因假设

基于Day11结果，可能的原因包括：

1. **PKRI模型质量不足**
   - 验证集F1=0.7864, Accuracy=0.6479（相对于q₀可能较低）
   - 特征质量不够好
   - 模型过拟合或欠拟合

2. **q_PKRI的概率分布偏差**
   - q_PKRI的概率分布可能与q₀有显著差异
   - 可信度加权可能引入噪声

3. **α(u)函数不匹配**
   - α(u)基于q₀训练，可能不适合q_PKRI
   - q_PKRI的分布特性与q₀不同

4. **门控机制影响**
   - 知识阈值门控可能对q_PKRI不利
   - 一致性门控的触发频率可能不同

## 输出文件

- `output/q0_vs_qpkri_analysis.json`：详细分析结果（JSON格式）

## 后续行动

根据分析结果，可以：
1. 如果发现PKRI模型质量问题 → 改进特征工程或模型训练
2. 如果发现分布偏差 → 调整q_PKRI构建逻辑
3. 如果发现α(u)不匹配 → 为q_PKRI单独训练α(u)函数
4. 如果差异较小且稳定 → 接受结论：q₀在当前场景下已足够好

