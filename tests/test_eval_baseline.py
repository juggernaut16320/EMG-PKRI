"""
test_eval_baseline.py - eval_baseline.py 单元测试
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# 尝试导入，如果失败则跳过需要这些依赖的测试
try:
    from eval_baseline import (
        compute_metrics_from_results,
        find_high_conf_error_samples,
        load_config,
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    # 如果缺少依赖（如 torch），标记为不可用
    HAS_DEPENDENCIES = False
    # 定义占位函数以便测试可以运行
    def compute_metrics_from_results(*args, **kwargs):
        pytest.skip("缺少依赖（torch/transformers/peft）")
    
    def find_high_conf_error_samples(*args, **kwargs):
        pytest.skip("缺少依赖（torch/transformers/peft）")
    
    def load_config(*args, **kwargs):
        pytest.skip("缺少依赖（torch/transformers/peft）")


class TestEvalBaseline:
    """测试基线评估功能"""
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
    def test_compute_metrics_from_results(self):
        """测试计算评估指标"""
        # 创建测试数据
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
        
        metrics = compute_metrics_from_results(results)
        
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
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
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
        
        metrics = compute_metrics_from_results(results)
        
        # 完美预测应该准确率为 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
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
        
        high_conf_errors = find_high_conf_error_samples(results, confidence_threshold=0.8)
        
        # 应该找到 s1 和 s4（高置信但错误）
        assert len(high_conf_errors) == 2
        assert high_conf_errors[0]['id'] in ['s1', 's4']
        assert high_conf_errors[1]['id'] in ['s1', 's4']
        
        # 验证按置信度排序
        assert high_conf_errors[0]['pred_prob'] >= high_conf_errors[1]['pred_prob']
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
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
        
        high_conf_errors = find_high_conf_error_samples(results, confidence_threshold=0.8)
        
        # 应该没有高置信错误
        assert len(high_conf_errors) == 0
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
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
        
        high_conf_errors = find_high_conf_error_samples(results, confidence_threshold=0.8)
        
        # 应该只找到 s2（s1 没有真实标签，无法判断对错）
        assert len(high_conf_errors) == 1
        assert high_conf_errors[0]['id'] == 's2'
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
    def test_load_config(self):
        """测试配置加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.yaml')
            
            # 创建测试配置
            config_content = """
data_dir: "./data"
output_dir: "./output"
model:
  name_or_path: "Qwen/Qwen3-1.7B"
  max_length: 512
training:
  output_dir: "checkpoints/baseline-lora"
"""
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # 加载配置
            config = load_config(config_path)
            
            # 验证
            assert config['data_dir'] == './data'
            assert config['model']['name_or_path'] == 'Qwen/Qwen3-1.7B'
            assert config['training']['output_dir'] == 'checkpoints/baseline-lora'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

