"""
test_eval_baseline_logic.py - eval_baseline.py 核心逻辑单元测试（不依赖 torch）

这个测试文件直接测试核心逻辑函数，不导入整个模块
"""

import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_metrics_from_results_logic(results):
    """
    从预测结果计算评估指标的核心逻辑（从 eval_baseline.py 提取）
    """
    # 提取真实标签和预测标签
    true_labels = []
    pred_labels = []
    
    for result in results:
        if result['coarse_label'] is not None:
            true_labels.append(int(result['coarse_label']))
            pred_labels.append(int(result['pred_label']))
    
    if len(true_labels) == 0:
        return {}
    
    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    # 计算 confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0, labels=[0, 1]
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'class_0': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1': float(f1_per_class[0]),
                'support': int(support[0])
            },
            'class_1': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1': float(f1_per_class[1]),
                'support': int(support[1])
            }
        },
        'total_samples': len(true_labels)
    }
    
    return metrics


def find_high_conf_error_samples_logic(results, confidence_threshold=0.8):
    """
    找出高置信错误样本的核心逻辑（从 eval_baseline.py 提取）
    """
    high_conf_errors = []
    
    for result in results:
        if result['coarse_label'] is None:
            continue
        
        true_label = int(result['coarse_label'])
        pred_label = int(result['pred_label'])
        pred_prob = result['pred_prob']
        
        # 高置信但预测错误
        if pred_prob >= confidence_threshold and pred_label != true_label:
            high_conf_errors.append({
                'id': result['id'],
                'text': result['text'],
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'pred_probs': result['pred_probs']
            })
    
    # 按置信度排序（高置信优先）
    high_conf_errors.sort(key=lambda x: x['pred_prob'], reverse=True)
    
    return high_conf_errors


class TestEvalBaselineLogic:
    """测试基线评估核心逻辑"""
    
    def test_compute_metrics_from_results(self):
        """测试计算评估指标"""
        results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,
                'pred_label': 1,
                'pred_prob': 0.9,
                'pred_probs': [0.1, 0.9]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 0,
                'pred_label': 0,
                'pred_prob': 0.8,
                'pred_probs': [0.8, 0.2]
            },
            {
                'id': 's3',
                'text': 'text3',
                'coarse_label': 1,
                'pred_label': 0,  # 预测错误
                'pred_prob': 0.7,
                'pred_probs': [0.7, 0.3]
            },
            {
                'id': 's4',
                'text': 'text4',
                'coarse_label': 0,
                'pred_label': 1,  # 预测错误
                'pred_prob': 0.6,
                'pred_probs': [0.4, 0.6]
            },
        ]
        
        metrics = compute_metrics_from_results_logic(results)
        
        # 验证指标存在
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class' in metrics
        assert 'total_samples' in metrics
        
        # 验证准确率（2/4 = 0.5）
        assert metrics['accuracy'] == 0.5
        
        # 验证样本数
        assert metrics['total_samples'] == 4
        
        # 验证 confusion matrix 形状
        cm = np.array(metrics['confusion_matrix'])
        assert cm.shape == (2, 2)
    
    def test_compute_metrics_perfect_prediction(self):
        """测试完美预测的情况"""
        results = [
            {
                'id': f's{i}',
                'text': f'text{i}',
                'coarse_label': i % 2,
                'pred_label': i % 2,
                'pred_prob': 0.9,
                'pred_probs': [0.1, 0.9] if i % 2 == 1 else [0.9, 0.1]
            }
            for i in range(10)
        ]
        
        metrics = compute_metrics_from_results_logic(results)
        
        # 完美预测应该准确率为 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_find_high_conf_error_samples(self):
        """测试找出高置信错误样本"""
        results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,
                'pred_label': 0,  # 预测错误
                'pred_prob': 0.9,  # 高置信
                'pred_probs': [0.9, 0.1]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 0,
                'pred_label': 0,  # 预测正确
                'pred_prob': 0.85,  # 高置信
                'pred_probs': [0.85, 0.15]
            },
            {
                'id': 's3',
                'text': 'text3',
                'coarse_label': 1,
                'pred_label': 0,  # 预测错误
                'pred_prob': 0.7,  # 低置信（不满足阈值）
                'pred_probs': [0.7, 0.3]
            },
            {
                'id': 's4',
                'text': 'text4',
                'coarse_label': 0,
                'pred_label': 1,  # 预测错误
                'pred_prob': 0.95,  # 高置信
                'pred_probs': [0.05, 0.95]
            },
        ]
        
        high_conf_errors = find_high_conf_error_samples_logic(results, confidence_threshold=0.8)
        
        # 应该找到 s1 和 s4（高置信但错误）
        assert len(high_conf_errors) == 2
        assert high_conf_errors[0]['id'] in ['s1', 's4']
        assert high_conf_errors[1]['id'] in ['s1', 's4']
        
        # 验证按置信度排序
        assert high_conf_errors[0]['pred_prob'] >= high_conf_errors[1]['pred_prob']
    
    def test_find_high_conf_error_samples_no_errors(self):
        """测试没有高置信错误的情况"""
        results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,
                'pred_label': 1,  # 预测正确
                'pred_prob': 0.9,
                'pred_probs': [0.1, 0.9]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 0,
                'pred_label': 0,  # 预测正确
                'pred_prob': 0.85,
                'pred_probs': [0.85, 0.15]
            },
        ]
        
        high_conf_errors = find_high_conf_error_samples_logic(results, confidence_threshold=0.8)
        
        # 应该没有高置信错误
        assert len(high_conf_errors) == 0
    
    def test_find_high_conf_error_samples_with_none_label(self):
        """测试处理没有真实标签的情况"""
        results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': None,  # 没有真实标签
                'pred_label': 0,
                'pred_prob': 0.9,
                'pred_probs': [0.9, 0.1]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 1,
                'pred_label': 0,  # 预测错误
                'pred_prob': 0.9,  # 高置信
                'pred_probs': [0.9, 0.1]
            },
        ]
        
        high_conf_errors = find_high_conf_error_samples_logic(results, confidence_threshold=0.8)
        
        # 应该只找到 s2（s1 没有真实标签，无法判断对错）
        assert len(high_conf_errors) == 1
        assert high_conf_errors[0]['id'] == 's2'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

