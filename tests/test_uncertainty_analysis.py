"""
test_uncertainty_analysis.py - uncertainty_analysis.py 单元测试
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# 尝试导入，如果失败则跳过需要这些依赖的测试
try:
    from uncertainty_analysis import (
        compute_uncertainty,
        bucket_analysis,
        load_config,
    )
    HAS_DEPENDENCIES = True
except ImportError as e:
    # 如果缺少依赖（如 torch），标记为不可用
    HAS_DEPENDENCIES = False
    # 定义占位函数以便测试可以运行
    def compute_uncertainty(*args, **kwargs):
        pytest.skip("缺少依赖（torch/transformers/peft）")
    
    def bucket_analysis(*args, **kwargs):
        pytest.skip("缺少依赖（torch/transformers/peft）")
    
    def load_config(*args, **kwargs):
        pytest.skip("缺少依赖（torch/transformers/peft）")


class TestUncertaintyAnalysis:
    """测试不确定性分析功能"""
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
    def test_compute_uncertainty_u_max(self):
        """测试计算 u_max 不确定性"""
        results = [
            {
                'id': 's1',
                'text': 'text1',
                'coarse_label': 1,
                'pred_label': 1,
                'pred_prob': 0.9,
                'pred_probs': [0.1, 0.9]  # 高置信
            },
            {
                'id': 's2',
                'text': 'text2',
                'coarse_label': 0,
                'pred_label': 0,
                'pred_prob': 0.5,
                'pred_probs': [0.5, 0.5]  # 低置信（高不确定性）
            },
        ]
        
        results = compute_uncertainty(results, metric='u_max')
        
        # 验证不确定性计算
        assert 'uncertainty' in results[0]
        assert 'max_prob' in results[0]
        
        # s1: u = 1 - 0.9 = 0.1 (低不确定性)
        assert abs(results[0]['uncertainty'] - 0.1) < 1e-6
        assert abs(results[0]['max_prob'] - 0.9) < 1e-6
        
        # s2: u = 1 - 0.5 = 0.5 (高不确定性)
        assert abs(results[1]['uncertainty'] - 0.5) < 1e-6
        assert abs(results[1]['max_prob'] - 0.5) < 1e-6
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
    def test_compute_uncertainty_u_entropy(self):
        """测试计算 u_entropy 不确定性"""
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
                'pred_prob': 0.5,
                'pred_probs': [0.5, 0.5]
            },
        ]
        
        results = compute_uncertainty(results, metric='u_entropy')
        
        # 验证不确定性计算
        assert 'uncertainty' in results[0]
        assert 'uncertainty' in results[1]
        
        # s2 应该比 s1 有更高的熵不确定性
        assert results[1]['uncertainty'] > results[0]['uncertainty']
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
    def test_bucket_analysis(self):
        """测试分桶分析"""
        # 创建测试数据（不同不确定性）
        results = []
        for i in range(100):
            uncertainty = i / 100.0  # 0.0 到 1.0
            max_prob = 1.0 - uncertainty
            pred_probs = [1.0 - max_prob, max_prob] if i % 2 == 0 else [max_prob, 1.0 - max_prob]
            
            # 错误率随不确定性增加
            is_error = 1 if uncertainty > 0.5 else 0
            
            results.append({
                'id': f's{i}',
                'text': f'text{i}',
                'coarse_label': 0 if i % 2 == 0 else 1,
                'pred_label': 1 if is_error else (0 if i % 2 == 0 else 1),
                'pred_prob': max_prob,
                'pred_probs': pred_probs,
                'uncertainty': uncertainty,
                'max_prob': max_prob
            })
        
        # 分桶分析
        bucket_df = bucket_analysis(results, n_buckets=10)
        
        # 验证结果
        assert len(bucket_df) > 0
        assert 'bucket_id' in bucket_df.columns
        assert 'u_mean' in bucket_df.columns
        assert 'error_rate' in bucket_df.columns
        assert 'n_samples' in bucket_df.columns
        
        # 验证错误率随不确定性增加（大致趋势）
        if len(bucket_df) >= 2:
            high_uncertainty_buckets = bucket_df[bucket_df['u_mean'] > 0.5]
            low_uncertainty_buckets = bucket_df[bucket_df['u_mean'] <= 0.5]
            
            if len(high_uncertainty_buckets) > 0 and len(low_uncertainty_buckets) > 0:
                avg_error_high = high_uncertainty_buckets['error_rate'].mean()
                avg_error_low = low_uncertainty_buckets['error_rate'].mean()
                # 高不确定性桶的平均错误率应该更高
                assert avg_error_high >= avg_error_low
    
    @pytest.mark.skipif(not HAS_DEPENDENCIES, reason="缺少依赖（torch/transformers/peft）")
    def test_bucket_analysis_empty(self):
        """测试空数据的分桶分析"""
        results = []
        bucket_df = bucket_analysis(results, n_buckets=10)
        
        # 应该返回空的 DataFrame
        assert len(bucket_df) == 0
    
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
"""
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            # 加载配置
            config = load_config(config_path)
            
            # 验证
            assert config['data_dir'] == './data'
            assert config['model']['name_or_path'] == 'Qwen/Qwen3-1.7B'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

