"""
test_calibration_analysis_logic.py - calibration_analysis.py 核心逻辑单元测试（不依赖 torch）

这个测试文件直接测试核心逻辑函数，不导入整个模块
"""

import pytest
import numpy as np
import pandas as pd


def calibration_buckets_logic(results, n_buckets=10):
    """
    按预测概率分桶，计算每个桶的校准统计（核心逻辑）
    """
    # 提取预测概率和真实标签
    probs = []
    labels = []
    
    for result in results:
        if result.get('coarse_label') is not None:
            pred_probs = np.array(result['pred_probs'])  # [prob_0, prob_1]
            prob_1 = pred_probs[1]  # 预测为敏感（类别1）的概率
            probs.append(prob_1)
            labels.append(int(result['coarse_label']))
    
    if len(probs) == 0:
        return pd.DataFrame()
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    # 等宽分桶
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_ids = np.digitize(probs, bucket_edges) - 1
    bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)
    
    # 统计每个桶的信息
    bucket_stats = []
    for bucket_id in range(n_buckets):
        mask = (bucket_ids == bucket_id)
        bucket_probs = probs[mask]
        bucket_labels = labels[mask]
        
        if len(bucket_probs) == 0:
            continue
        
        n_samples = len(bucket_probs)
        n_correct = np.sum(bucket_labels == 1)
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0
        avg_confidence = np.mean(bucket_probs)
        gap = abs(accuracy - avg_confidence)
        ece_contribution = (n_samples / len(probs)) * gap
        
        bucket_stats.append({
            'bucket_id': bucket_id,
            'prob_min': bucket_edges[bucket_id],
            'prob_max': bucket_edges[bucket_id + 1],
            'prob_mean': avg_confidence,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'gap': gap,
            'ece_contribution': ece_contribution
        })
    
    return pd.DataFrame(bucket_stats)


def compute_ece_logic(bucket_df, n_total):
    """
    计算期望校准误差（ECE）
    """
    if len(bucket_df) == 0 or n_total == 0:
        return 0.0
    
    ece = 0.0
    for _, row in bucket_df.iterrows():
        n_samples = row['n_samples']
        gap = row['gap']
        ece += (n_samples / n_total) * gap
    
    return float(ece)


def compute_mce_logic(bucket_df):
    """
    计算最大校准误差（MCE）
    """
    if len(bucket_df) == 0:
        return 0.0
    
    mce = bucket_df['gap'].max()
    return float(mce)


def compute_brier_score_logic(results):
    """
    计算 Brier Score
    """
    probs = []
    labels = []
    
    for result in results:
        if result.get('coarse_label') is not None:
            pred_probs = np.array(result['pred_probs'])
            prob_1 = pred_probs[1]
            probs.append(prob_1)
            labels.append(int(result['coarse_label']))
    
    if len(probs) == 0:
        return 0.0
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    brier_score = np.mean((probs - labels) ** 2)
    return float(brier_score)


# 测试用例
def test_calibration_buckets_basic():
    """测试基本分桶功能"""
    results = [
        {'pred_probs': [0.1, 0.9], 'coarse_label': 1},  # 高置信，正确
        {'pred_probs': [0.9, 0.1], 'coarse_label': 0},  # 高置信，正确
        {'pred_probs': [0.5, 0.5], 'coarse_label': 1},  # 低置信
        {'pred_probs': [0.3, 0.7], 'coarse_label': 0},  # 错误预测
    ]
    
    bucket_df = calibration_buckets_logic(results, n_buckets=5)
    
    assert len(bucket_df) > 0
    assert 'bucket_id' in bucket_df.columns
    assert 'accuracy' in bucket_df.columns
    assert 'avg_confidence' in bucket_df.columns
    assert 'gap' in bucket_df.columns


def test_calibration_buckets_empty():
    """测试空结果"""
    results = []
    bucket_df = calibration_buckets_logic(results)
    assert len(bucket_df) == 0


def test_calibration_buckets_no_labels():
    """测试没有标签的情况"""
    results = [
        {'pred_probs': [0.1, 0.9], 'coarse_label': None},
        {'pred_probs': [0.9, 0.1], 'coarse_label': None},
    ]
    bucket_df = calibration_buckets_logic(results)
    assert len(bucket_df) == 0


def test_compute_ece_perfect_calibration():
    """测试完美校准的情况（ECE应该为0）"""
    # 创建完美校准的数据：预测概率 = 实际准确率
    bucket_data = {
        'bucket_id': [0, 1, 2],
        'n_samples': [10, 20, 30],
        'gap': [0.0, 0.0, 0.0],  # 完美校准
    }
    bucket_df = pd.DataFrame(bucket_data)
    
    ece = compute_ece_logic(bucket_df, n_total=60)
    assert ece == 0.0


def test_compute_ece_imperfect_calibration():
    """测试不完美校准的情况"""
    bucket_data = {
        'bucket_id': [0, 1],
        'n_samples': [50, 50],
        'gap': [0.1, 0.2],  # 有校准误差
    }
    bucket_df = pd.DataFrame(bucket_data)
    
    ece = compute_ece_logic(bucket_df, n_total=100)
    # ECE = (50/100)*0.1 + (50/100)*0.2 = 0.05 + 0.1 = 0.15
    assert abs(ece - 0.15) < 1e-6


def test_compute_ece_empty():
    """测试空DataFrame"""
    bucket_df = pd.DataFrame()
    ece = compute_ece_logic(bucket_df, n_total=100)
    assert ece == 0.0


def test_compute_mce():
    """测试MCE计算"""
    bucket_data = {
        'bucket_id': [0, 1, 2],
        'gap': [0.05, 0.15, 0.10],
    }
    bucket_df = pd.DataFrame(bucket_data)
    
    mce = compute_mce_logic(bucket_df)
    assert mce == 0.15  # 最大gap


def test_compute_mce_empty():
    """测试空DataFrame的MCE"""
    bucket_df = pd.DataFrame()
    mce = compute_mce_logic(bucket_df)
    assert mce == 0.0


def test_compute_brier_score_perfect():
    """测试完美预测的Brier Score（应该为0）"""
    results = [
        {'pred_probs': [0.0, 1.0], 'coarse_label': 1},  # 完美预测
        {'pred_probs': [1.0, 0.0], 'coarse_label': 0},  # 完美预测
    ]
    
    brier_score = compute_brier_score_logic(results)
    assert abs(brier_score) < 1e-6


def test_compute_brier_score_imperfect():
    """测试不完美预测的Brier Score"""
    results = [
        {'pred_probs': [0.5, 0.5], 'coarse_label': 1},  # 预测概率0.5，真实标签1
        {'pred_probs': [0.5, 0.5], 'coarse_label': 0},  # 预测概率0.5，真实标签0
    ]
    
    brier_score = compute_brier_score_logic(results)
    # Brier = 0.5 * ((0.5-1)^2 + (0.5-0)^2) = 0.5 * (0.25 + 0.25) = 0.25
    assert abs(brier_score - 0.25) < 1e-6


def test_compute_brier_score_empty():
    """测试空结果的Brier Score"""
    results = []
    brier_score = compute_brier_score_logic(results)
    assert brier_score == 0.0


def test_compute_brier_score_no_labels():
    """测试没有标签的Brier Score"""
    results = [
        {'pred_probs': [0.5, 0.5], 'coarse_label': None},
    ]
    brier_score = compute_brier_score_logic(results)
    assert brier_score == 0.0


def test_integration():
    """集成测试：完整流程"""
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    
    results = []
    for i in range(n_samples):
        prob_1 = np.random.rand()  # 随机预测概率
        label = 1 if prob_1 > 0.5 else 0  # 简单规则：概率>0.5则标签为1
        results.append({
            'pred_probs': [1 - prob_1, prob_1],
            'coarse_label': label
        })
    
    # 分桶
    bucket_df = calibration_buckets_logic(results, n_buckets=10)
    
    # 计算指标
    n_total = len([r for r in results if r.get('coarse_label') is not None])
    ece = compute_ece_logic(bucket_df, n_total)
    mce = compute_mce_logic(bucket_df)
    brier_score = compute_brier_score_logic(results)
    
    # 验证结果
    assert len(bucket_df) > 0
    assert 0 <= ece <= 1
    assert 0 <= mce <= 1
    assert 0 <= brier_score <= 1
    assert n_total == n_samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

