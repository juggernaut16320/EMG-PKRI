"""
test_emg_bucket_search.py - emg_bucket_search.py 完整功能测试

测试脚本基本功能、输入输出格式、错误处理等
"""

import pytest
import os
import json
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

# 导入被测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.emg_bucket_search import (
    load_baseline_predictions,
    load_q0_posteriors,
    load_uncertainty_buckets,
    compute_emg_fusion,
    assign_samples_to_buckets,
    compute_metrics_for_alpha,
    search_alpha_for_bucket
)


class TestLoadFunctions:
    """测试加载函数"""
    
    def test_load_baseline_predictions_valid(self):
        """测试加载有效的 baseline 预测结果"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            test_data = [
                {'id': 's1', 'coarse_label': 1, 'pred_probs': [0.2, 0.8], 'uncertainty': 0.2},
                {'id': 's2', 'coarse_label': 0, 'pred_probs': [0.9, 0.1], 'uncertainty': 0.1},
            ]
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            results = load_baseline_predictions(temp_file)
            
            assert len(results) == 2
            assert 's1' in results
            assert 's2' in results
            assert results['s1']['pred_probs'] == [0.2, 0.8]
            assert results['s1']['uncertainty'] == 0.2
        finally:
            os.unlink(temp_file)
    
    def test_load_baseline_predictions_missing_fields(self):
        """测试缺少必需字段的 baseline 预测结果"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            test_data = [
                {'id': 's1', 'coarse_label': 1, 'pred_probs': [0.2, 0.8], 'uncertainty': 0.2},  # 完整
                {'id': 's2', 'coarse_label': 0, 'pred_probs': [0.9, 0.1]},  # 缺少 uncertainty
                {'id': 's3', 'coarse_label': 1, 'uncertainty': 0.3},  # 缺少 pred_probs
            ]
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            results = load_baseline_predictions(temp_file)
            
            # 只有 s1 应该被加载（有完整字段）
            assert len(results) == 1
            assert 's1' in results
            assert 's2' not in results
            assert 's3' not in results
        finally:
            os.unlink(temp_file)
    
    def test_load_baseline_predictions_file_not_exists(self):
        """测试文件不存在的情况"""
        results = load_baseline_predictions('nonexistent_file.jsonl')
        assert len(results) == 0
    
    def test_load_q0_posteriors_valid(self):
        """测试加载有效的 q₀ 后验"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            test_data = [
                {'id': 's1', 'q0': [0.1, 0.9]},
                {'id': 's2', 'q0': [0.8, 0.2]},
            ]
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            q0_dict = load_q0_posteriors(temp_file)
            
            assert len(q0_dict) == 2
            assert 's1' in q0_dict
            assert 's2' in q0_dict
            assert q0_dict['s1'] == [0.1, 0.9]
            assert q0_dict['s2'] == [0.8, 0.2]
        finally:
            os.unlink(temp_file)
    
    def test_load_q0_posteriors_missing_q0(self):
        """测试缺少 q0 字段的情况"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            test_data = [
                {'id': 's1', 'q0': [0.1, 0.9]},
                {'id': 's2'},  # 缺少 q0
            ]
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_file = f.name
        
        try:
            q0_dict = load_q0_posteriors(temp_file)
            
            # 只有 s1 应该被加载
            assert len(q0_dict) == 1
            assert 's1' in q0_dict
            assert 's2' not in q0_dict
        finally:
            os.unlink(temp_file)
    
    def test_load_uncertainty_buckets_valid(self):
        """测试加载有效的分桶信息"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df = pd.DataFrame({
                'bucket_id': [0, 1, 2],
                'u_min': [0.0, 0.1, 0.2],
                'u_max': [0.1, 0.2, 0.3],
                'u_mean': [0.05, 0.15, 0.25],
                'n_samples': [100, 50, 30]
            })
            df.to_csv(f, index=False)
            temp_file = f.name
        
        try:
            bucket_df = load_uncertainty_buckets(temp_file)
            
            assert len(bucket_df) == 3
            assert 'bucket_id' in bucket_df.columns
            assert 'u_min' in bucket_df.columns
            assert 'u_max' in bucket_df.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_uncertainty_buckets_file_not_exists(self):
        """测试文件不存在的情况"""
        bucket_df = load_uncertainty_buckets('nonexistent_file.csv')
        assert len(bucket_df) == 0


class TestComputeEMGFusion:
    """测试 EMG 融合计算"""
    
    def test_compute_emg_fusion(self):
        """测试 EMG 融合计算函数"""
        p = [0.8, 0.2]
        q = [0.2, 0.8]
        alpha = 0.5
        
        result = compute_emg_fusion(p, q, alpha)
        
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-6
        assert all(0 <= prob <= 1 for prob in result)


class TestAssignSamplesToBuckets:
    """测试样本分配到 bucket"""
    
    def test_assign_samples_to_buckets(self):
        """测试样本分配到 bucket"""
        baseline_results = {
            's1': {'uncertainty': 0.05},
            's2': {'uncertainty': 0.15},
            's3': {'uncertainty': 0.25},
        }
        
        bucket_df = pd.DataFrame({
            'bucket_id': [0, 1, 2],
            'u_min': [0.0, 0.1, 0.2],
            'u_max': [0.1, 0.2, 0.3]
        })
        
        result = assign_samples_to_buckets(baseline_results, bucket_df)
        
        assert 0 in result
        assert 's1' in result[0]
        assert 1 in result
        assert 's2' in result[1]
        assert 2 in result
        assert 's3' in result[2]


class TestComputeMetricsForAlpha:
    """测试指标计算"""
    
    def test_compute_metrics_for_alpha(self):
        """测试计算给定 α 值下的 F1 和 NLL"""
        samples = ['s1', 's2']
        baseline_results = {
            's1': {'pred_probs': [0.2, 0.8], 'coarse_label': 1},
            's2': {'pred_probs': [0.9, 0.1], 'coarse_label': 0},
        }
        q0_dict = {
            's1': [0.1, 0.9],
            's2': [0.8, 0.2],
        }
        alpha = 0.5
        
        f1, nll = compute_metrics_for_alpha(samples, baseline_results, q0_dict, alpha)
        
        assert isinstance(f1, float)
        assert isinstance(nll, float)
        assert 0 <= f1 <= 1
        assert nll >= 0 or nll == float('inf')
    
    def test_compute_metrics_for_alpha_empty_samples(self):
        """测试空样本列表"""
        samples = []
        baseline_results = {}
        q0_dict = {}
        alpha = 0.5
        
        f1, nll = compute_metrics_for_alpha(samples, baseline_results, q0_dict, alpha)
        
        assert f1 == 0.0
        assert nll == float('inf')


class TestSearchAlphaForBucket:
    """测试 α 搜索"""
    
    def test_search_alpha_for_bucket_f1(self):
        """测试使用 F1 作为优化指标搜索最优 α"""
        bucket_id = 0
        samples = ['s1', 's2']
        baseline_results = {
            's1': {'pred_probs': [0.2, 0.8], 'coarse_label': 1},
            's2': {'pred_probs': [0.9, 0.1], 'coarse_label': 0},
        }
        q0_dict = {
            's1': [0.1, 0.9],
            's2': [0.8, 0.2],
        }
        alpha_grid = [0.0, 0.5, 1.0]
        metric = 'f1'
        
        best_alpha, alpha_metrics = search_alpha_for_bucket(
            bucket_id, samples, baseline_results, q0_dict, alpha_grid, metric
        )
        
        assert best_alpha in alpha_grid
        assert isinstance(alpha_metrics, dict)
        assert len(alpha_metrics) == len(alpha_grid)
        
        # 检查每个 α 的指标
        for alpha in alpha_grid:
            assert alpha in alpha_metrics
            assert 'f1' in alpha_metrics[alpha]
            assert 'nll' in alpha_metrics[alpha]
    
    def test_search_alpha_for_bucket_nll(self):
        """测试使用 NLL 作为优化指标搜索最优 α"""
        bucket_id = 0
        samples = ['s1', 's2']
        baseline_results = {
            's1': {'pred_probs': [0.2, 0.8], 'coarse_label': 1},
            's2': {'pred_probs': [0.9, 0.1], 'coarse_label': 0},
        }
        q0_dict = {
            's1': [0.1, 0.9],
            's2': [0.8, 0.2],
        }
        alpha_grid = [0.0, 0.5, 1.0]
        metric = 'nll'
        
        best_alpha, alpha_metrics = search_alpha_for_bucket(
            bucket_id, samples, baseline_results, q0_dict, alpha_grid, metric
        )
        
        assert best_alpha in alpha_grid
        assert isinstance(alpha_metrics, dict)


class TestInputOutputFormats:
    """测试输入输出格式"""
    
    def test_output_csv_format(self):
        """测试输出 CSV 格式（模拟）"""
        # 模拟输出数据
        results = [
            {
                'bucket_id': 0,
                'u_min': 0.0,
                'u_max': 0.1,
                'u_mean': 0.05,
                'n_samples': 100,
                'alpha_star': 0.75,
                'f1_at_alpha_star': 0.85,
                'nll_at_alpha_star': 0.3,
                'alpha_0_f1': 0.70,
                'alpha_0_nll': 0.5,
                'alpha_1_f1': 0.82,
                'alpha_1_nll': 0.32,
            }
        ]
        
        df = pd.DataFrame(results)
        
        # 检查必需字段
        required_fields = ['bucket_id', 'alpha_star', 'f1_at_alpha_star', 'nll_at_alpha_star']
        for field in required_fields:
            assert field in df.columns
        
        # 检查数据类型
        assert df['bucket_id'].dtype in [int, 'int64']
        assert df['alpha_star'].dtype in [float, 'float64']
        assert df['f1_at_alpha_star'].dtype in [float, 'float64']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

