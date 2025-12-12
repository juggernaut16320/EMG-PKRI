"""
test_hardset_maker_logic.py - hardset_maker.py 核心逻辑单元测试（不依赖 torch）

这个测试文件直接测试 find_disagreement_samples 函数的逻辑，不导入整个模块
"""

import json
import pytest


def find_disagreement_samples_logic(
    baseline_results,
    teacher_predictions,
    confidence_threshold=0.8,
    min_size=500,
    max_size=2000
):
    """
    找出分歧样本的核心逻辑（从 hardset_maker.py 提取）
    """
    import random
    
    disagreement_samples = []
    
    for result in baseline_results:
        sample_id = result['id']
        baseline_pred = result['pred_label']
        baseline_prob = result['pred_prob']
        true_label = result.get('coarse_label')
        teacher_pred = teacher_predictions.get(sample_id)
        
        if teacher_pred is None:
            continue
        
        # 判断是否为分歧样本
        is_disagreement = False
        
        # 情况1: baseline 预测错误但 teacher 预测正确
        if true_label is not None:
            baseline_wrong = (baseline_pred != true_label)
            teacher_correct = (teacher_pred == true_label)
            if baseline_wrong and teacher_correct:
                is_disagreement = True
        
        # 情况2: baseline 高置信但预测错误
        if true_label is not None:
            if baseline_prob >= confidence_threshold and baseline_pred != true_label:
                is_disagreement = True
        
        # 情况3: baseline 和 teacher 预测不一致（即使没有真实标签）
        if baseline_pred != teacher_pred:
            is_disagreement = True
        
        if is_disagreement:
            disagreement_samples.append({
                **result,
                'teacher_label': teacher_pred,
                'disagreement_type': []
            })
            
            # 记录分歧类型
            if true_label is not None:
                if baseline_pred != true_label and teacher_pred == true_label:
                    disagreement_samples[-1]['disagreement_type'].append('baseline_wrong_teacher_correct')
                if baseline_prob >= confidence_threshold and baseline_pred != true_label:
                    disagreement_samples[-1]['disagreement_type'].append('high_conf_error')
            if baseline_pred != teacher_pred:
                disagreement_samples[-1]['disagreement_type'].append('prediction_mismatch')
    
    # 按置信度排序（高置信错误优先）
    # 使用 (pred_prob, id) 作为排序键，确保稳定排序
    disagreement_samples.sort(key=lambda x: (x['pred_prob'], x['id']), reverse=True)
    
    # 选择样本
    selected_size = min(max_size, max(min_size, len(disagreement_samples)))
    selected_samples = disagreement_samples[:selected_size]
    
    return selected_samples


class TestHardsetMakerLogic:
    """测试困难子集构造核心逻辑"""
    
    def test_find_disagreement_samples_baseline_wrong_teacher_correct(self):
        """测试找出 baseline 预测错误但 teacher 预测正确的样本"""
        baseline_results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,  # 真实标签：敏感
                'pred_label': 0,  # baseline 预测：非敏感（错误）
                'pred_prob': 0.7,
                'pred_probs': [0.7, 0.3]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 0,  # 真实标签：非敏感
                'pred_label': 0,  # baseline 预测：非敏感（正确）
                'pred_prob': 0.8,
                'pred_probs': [0.8, 0.2]
            },
        ]
        
        teacher_predictions = {
            's1': 1,  # teacher 预测：敏感（正确）
            's2': 0,  # teacher 预测：非敏感（正确）
        }
        
        disagreement_samples = find_disagreement_samples_logic(
            baseline_results,
            teacher_predictions,
            confidence_threshold=0.8,
            min_size=1,
            max_size=10
        )
        
        # 验证：s1 应该被选中（baseline 错误但 teacher 正确）
        assert len(disagreement_samples) >= 1
        assert disagreement_samples[0]['id'] == 's1'
        assert 'baseline_wrong_teacher_correct' in disagreement_samples[0]['disagreement_type']
    
    def test_find_disagreement_samples_high_conf_error(self):
        """测试找出高置信但预测错误的样本"""
        baseline_results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,  # 真实标签：敏感
                'pred_label': 0,  # baseline 预测：非敏感（错误）
                'pred_prob': 0.9,  # 高置信度
                'pred_probs': [0.9, 0.1]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 0,  # 真实标签：非敏感
                'pred_label': 0,  # baseline 预测：非敏感（正确）
                'pred_prob': 0.8,
                'pred_probs': [0.8, 0.2]
            },
        ]
        
        teacher_predictions = {
            's1': 0,
            's2': 0,
        }
        
        disagreement_samples = find_disagreement_samples_logic(
            baseline_results,
            teacher_predictions,
            confidence_threshold=0.8,
            min_size=1,
            max_size=10
        )
        
        # 验证：s1 应该被选中（高置信但错误）
        assert len(disagreement_samples) >= 1
        assert disagreement_samples[0]['id'] == 's1'
        assert 'high_conf_error' in disagreement_samples[0]['disagreement_type']
    
    def test_find_disagreement_samples_prediction_mismatch(self):
        """测试找出 baseline 和 teacher 预测不一致的样本"""
        baseline_results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': None,  # 没有真实标签
                'pred_label': 0,  # baseline 预测：非敏感
                'pred_prob': 0.6,
                'pred_probs': [0.6, 0.4]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': None,
                'pred_label': 1,  # baseline 预测：敏感
                'pred_prob': 0.7,
                'pred_probs': [0.3, 0.7]
            },
        ]
        
        teacher_predictions = {
            's1': 1,  # teacher 预测：敏感（与 baseline 不一致）
            's2': 1,  # teacher 预测：敏感（与 baseline 一致）
        }
        
        disagreement_samples = find_disagreement_samples_logic(
            baseline_results,
            teacher_predictions,
            confidence_threshold=0.8,
            min_size=1,
            max_size=10
        )
        
        # 验证：s1 应该被选中（预测不一致）
        assert len(disagreement_samples) >= 1
        assert disagreement_samples[0]['id'] == 's1'
        assert 'prediction_mismatch' in disagreement_samples[0]['disagreement_type']
    
    def test_find_disagreement_samples_size_limit(self):
        """测试样本数量限制"""
        baseline_results = []
        teacher_predictions = {}
        
        for i in range(100):
            baseline_results.append({
                'id': f's{i}',
                'text': f'text{i}',
                'coarse_label': 1,
                'pred_label': 0,  # 全部预测错误
                'pred_prob': 0.9 - i * 0.01,  # 递减的置信度
                'pred_probs': [0.9 - i * 0.01, 0.1 + i * 0.01]
            })
            teacher_predictions[f's{i}'] = 1  # teacher 全部预测正确
        
        disagreement_samples = find_disagreement_samples_logic(
            baseline_results,
            teacher_predictions,
            confidence_threshold=0.8,
            min_size=10,
            max_size=20
        )
        
        # 验证：应该在 min_size 和 max_size 之间
        assert len(disagreement_samples) >= 10
        assert len(disagreement_samples) <= 20
    
    def test_find_disagreement_samples_sorting(self):
        """测试样本按置信度排序"""
        baseline_results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,
                'pred_label': 0,
                'pred_prob': 0.5,  # 低置信度
                'pred_probs': [0.5, 0.5]
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 1,
                'pred_label': 0,
                'pred_prob': 0.9,  # 高置信度
                'pred_probs': [0.9, 0.1]
            },
            {
                'id': 's3',
                'text': 'text3',
                'coarse_label': 1,
                'pred_label': 0,
                'pred_prob': 0.7,  # 中等置信度
                'pred_probs': [0.7, 0.3]
            },
        ]
        
        teacher_predictions = {
            's1': 1,
            's2': 1,
            's3': 1,
        }
        
        disagreement_samples = find_disagreement_samples_logic(
            baseline_results,
            teacher_predictions,
            confidence_threshold=0.8,
            min_size=3,
            max_size=10
        )
        
        # 验证：应该按置信度降序排列
        assert len(disagreement_samples) == 3
        assert disagreement_samples[0]['pred_prob'] >= disagreement_samples[1]['pred_prob']
        assert disagreement_samples[1]['pred_prob'] >= disagreement_samples[2]['pred_prob']
        assert disagreement_samples[0]['id'] == 's2'  # 最高置信度


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

