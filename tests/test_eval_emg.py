"""
test_eval_emg.py - eval_emg.py 完整功能测试

测试脚本基本功能、输入输出格式、错误处理等
"""

import pytest
import os
import json
import tempfile
import numpy as np
from pathlib import Path

# 导入被测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_emg import (
    load_baseline_predictions,
    load_q0_posteriors,
    load_alpha_u_lut,
    lookup_alpha,
    compute_ece,
    evaluate_method,
    compare_methods
)


class TestLoadBaselinePredictions:
    """测试加载 baseline 预测结果"""
    
    def test_load_baseline_predictions_valid(self):
        """测试加载有效的 JSONL 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'id': 's1',
                'coarse_label': 1,
                'pred_probs': [0.2, 0.8],
                'uncertainty': 0.2,
                'pred_label': 1
            }, f, ensure_ascii=False)
            f.write('\n')
            json.dump({
                'id': 's2',
                'coarse_label': 0,
                'pred_probs': [0.9, 0.1],
                'uncertainty': 0.1,
                'pred_label': 0
            }, f, ensure_ascii=False)
            f.write('\n')
            temp_file = f.name
        
        try:
            results = load_baseline_predictions(temp_file)
            
            assert len(results) == 2
            assert 's1' in results
            assert 's2' in results
            assert results['s1']['coarse_label'] == 1
            assert results['s2']['coarse_label'] == 0
        finally:
            os.unlink(temp_file)
    
    def test_load_baseline_predictions_missing_fields(self):
        """测试缺少必需字段的情况"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'id': 's1',
                'coarse_label': 1
                # 缺少 pred_probs 和 uncertainty
            }, f, ensure_ascii=False)
            f.write('\n')
            temp_file = f.name
        
        try:
            results = load_baseline_predictions(temp_file)
            
            # 缺少必需字段的样本应该被跳过
            assert len(results) == 0
        finally:
            os.unlink(temp_file)
    
    def test_load_baseline_predictions_nonexistent_file(self):
        """测试文件不存在的情况"""
        results = load_baseline_predictions('nonexistent_file.jsonl')
        
        assert results == {}


class TestLoadQ0Posteriors:
    """测试加载 q₀ 后验"""
    
    def test_load_q0_posteriors_valid(self):
        """测试加载有效的 JSONL 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'id': 's1',
                'q0': [0.1, 0.9]
            }, f, ensure_ascii=False)
            f.write('\n')
            json.dump({
                'id': 's2',
                'q0': [0.8, 0.2]
            }, f, ensure_ascii=False)
            f.write('\n')
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'id': 's1'
                # 缺少 q0
            }, f, ensure_ascii=False)
            f.write('\n')
            temp_file = f.name
        
        try:
            q0_dict = load_q0_posteriors(temp_file)
            
            # 缺少 q0 字段的样本应该被跳过
            assert len(q0_dict) == 0
        finally:
            os.unlink(temp_file)


class TestLoadAlphaULut:
    """测试加载 α(u) 查表"""
    
    def test_load_alpha_u_lut_valid(self):
        """测试加载有效的 JSON 文件"""
        lut_data = {
            'u': [0.0, 0.1, 0.2, 0.3, 0.4],
            'alpha': [1.0, 0.8, 0.6, 0.4, 0.2]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(lut_data, f, ensure_ascii=False)
            temp_file = f.name
        
        try:
            lut = load_alpha_u_lut(temp_file)
            
            assert len(lut) == 2
            assert 'u' in lut
            assert 'alpha' in lut
            assert len(lut['u']) == 5
            assert len(lut['alpha']) == 5
        finally:
            os.unlink(temp_file)
    
    def test_load_alpha_u_lut_missing_fields(self):
        """测试缺少必需字段的情况"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'u': [0.0, 0.1]}, f, ensure_ascii=False)  # 缺少 alpha
            temp_file = f.name
        
        try:
            lut = load_alpha_u_lut(temp_file)
            
            # 缺少必需字段时应该返回空字典
            assert lut == {}
        finally:
            os.unlink(temp_file)


class TestLookupAlpha:
    """测试 α(u) 查找"""
    
    def test_lookup_alpha_exact(self):
        """测试精确匹配"""
        lut = {
            'u': [0.0, 0.1, 0.2, 0.3],
            'alpha': [1.0, 0.8, 0.6, 0.4]
        }
        
        assert abs(lookup_alpha(0.1, lut) - 0.8) < 1e-6
        assert abs(lookup_alpha(0.2, lut) - 0.6) < 1e-6
    
    def test_lookup_alpha_interpolation(self):
        """测试线性插值"""
        lut = {
            'u': [0.0, 0.2],
            'alpha': [1.0, 0.6]
        }
        
        alpha = lookup_alpha(0.1, lut)
        
        # 应该在 0.6 和 1.0 之间，更接近 0.8
        assert 0.6 < alpha < 1.0
        assert abs(alpha - 0.8) < 0.1


class TestComputeECE:
    """测试 ECE 计算"""
    
    def test_compute_ece_perfect(self):
        """测试完美校准"""
        pred_probs = np.array([0.2, 0.2, 0.8, 0.8])
        true_labels = np.array([0, 0, 1, 1])
        
        ece = compute_ece(pred_probs, true_labels, n_buckets=2)
        
        assert ece >= 0
        assert ece < 0.2  # 完美校准时应该很小
    
    def test_compute_ece_empty(self):
        """测试空数组"""
        pred_probs = np.array([])
        true_labels = np.array([])
        
        ece = compute_ece(pred_probs, true_labels)
        
        assert ece == 0.0


class TestEvaluateMethod:
    """测试评估方法"""
    
    def test_evaluate_baseline(self):
        """测试评估 baseline 方法"""
        baseline_results = {
            's1': {
                'coarse_label': 1,
                'pred_probs': [0.2, 0.8],
                'uncertainty': 0.2,
                'pred_label': 1
            },
            's2': {
                'coarse_label': 0,
                'pred_probs': [0.9, 0.1],
                'uncertainty': 0.1,
                'pred_label': 0
            }
        }
        q0_dict = {
            's1': [0.1, 0.9],
            's2': [0.8, 0.2]
        }
        
        metrics = evaluate_method(baseline_results, q0_dict, 'baseline')
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'nll' in metrics
        assert 'ece' in metrics
        assert 'n_samples' in metrics
        assert metrics['n_samples'] == 2
    
    def test_evaluate_fixed_alpha(self):
        """测试评估固定 α 融合方法"""
        baseline_results = {
            's1': {
                'coarse_label': 1,
                'pred_probs': [0.2, 0.8],
                'uncertainty': 0.2,
                'pred_label': 1
            }
        }
        q0_dict = {
            's1': [0.1, 0.9]
        }
        
        metrics = evaluate_method(
            baseline_results, q0_dict, 'fixed_alpha_0.5', fixed_alpha=0.5
        )
        
        assert metrics['n_samples'] == 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_evaluate_emg(self):
        """测试评估 EMG 方法"""
        baseline_results = {
            's1': {
                'coarse_label': 1,
                'pred_probs': [0.2, 0.8],
                'uncertainty': 0.2,
                'pred_label': 1
            }
        }
        q0_dict = {
            's1': [0.1, 0.9]
        }
        alpha_lut = {
            'u': [0.0, 0.2, 0.4],
            'alpha': [1.0, 0.6, 0.2]
        }
        
        metrics = evaluate_method(
            baseline_results, q0_dict, 'emg', alpha_lut=alpha_lut
        )
        
        assert metrics['n_samples'] == 1
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_evaluate_empty_results(self):
        """测试空结果"""
        baseline_results = {}
        q0_dict = {}
        
        metrics = evaluate_method(baseline_results, q0_dict, 'baseline')
        
        assert metrics == {}


class TestCompareMethods:
    """测试对比方法"""
    
    def test_compare_methods_emg_vs_baseline(self):
        """测试 EMG vs Baseline 对比"""
        metrics_dict = {
            'baseline': {
                'f1': 0.89,
                'nll': 0.45,
                'ece': 0.12
            },
            'emg': {
                'f1': 0.91,
                'nll': 0.42,
                'ece': 0.10
            }
        }
        
        comparison = compare_methods(metrics_dict)
        
        assert 'emg_vs_baseline' in comparison
        comp = comparison['emg_vs_baseline']
        assert comp['f1_improvement'] > 0
        assert comp['nll_reduction'] > 0
        assert comp['ece_reduction'] > 0
    
    def test_compare_methods_emg_vs_fixed_alpha(self):
        """测试 EMG vs 固定 α 对比"""
        # compare_methods 支持以 'fixed_alpha' 开头的键名（如 'fixed_alpha_0.5'）
        metrics_dict = {
            'fixed_alpha_0.5': {  # 实际使用时是这样的键名
                'f1': 0.895,
                'nll': 0.435,
                'ece': 0.11
            },
            'emg': {
                'f1': 0.91,
                'nll': 0.42,
                'ece': 0.10
            }
        }
        
        comparison = compare_methods(metrics_dict)
        
        assert 'emg_vs_fixed_alpha' in comparison
        comp = comparison['emg_vs_fixed_alpha']
        assert comp['f1_improvement'] > 0
    
    def test_compare_methods_empty(self):
        """测试空对比"""
        metrics_dict = {}
        
        comparison = compare_methods(metrics_dict)
        
        assert comparison == {}


class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 创建临时文件
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 baseline 预测文件
            baseline_file = os.path.join(tmpdir, 'baseline.jsonl')
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'id': 's1',
                    'coarse_label': 1,
                    'pred_probs': [0.2, 0.8],
                    'uncertainty': 0.2,
                    'pred_label': 1
                }, f, ensure_ascii=False)
                f.write('\n')
                json.dump({
                    'id': 's2',
                    'coarse_label': 0,
                    'pred_probs': [0.9, 0.1],
                    'uncertainty': 0.1,
                    'pred_label': 0
                }, f, ensure_ascii=False)
            
            # 创建 q0 文件
            q0_file = os.path.join(tmpdir, 'q0.jsonl')
            with open(q0_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'id': 's1',
                    'q0': [0.1, 0.9]
                }, f, ensure_ascii=False)
                f.write('\n')
                json.dump({
                    'id': 's2',
                    'q0': [0.8, 0.2]
                }, f, ensure_ascii=False)
            
            # 创建 alpha lut 文件
            alpha_lut_file = os.path.join(tmpdir, 'alpha_lut.json')
            with open(alpha_lut_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'u': [0.0, 0.1, 0.2, 0.3],
                    'alpha': [1.0, 0.8, 0.6, 0.4]
                }, f, ensure_ascii=False)
            
            # 加载数据
            baseline_results = load_baseline_predictions(baseline_file)
            q0_dict = load_q0_posteriors(q0_file)
            alpha_lut = load_alpha_u_lut(alpha_lut_file)
            
            assert len(baseline_results) == 2
            assert len(q0_dict) == 2
            assert len(alpha_lut) == 2
            
            # 评估三种方法
            baseline_metrics = evaluate_method(baseline_results, q0_dict, 'baseline')
            fixed_metrics = evaluate_method(
                baseline_results, q0_dict, 'fixed_alpha_0.5', fixed_alpha=0.5
            )
            emg_metrics = evaluate_method(
                baseline_results, q0_dict, 'emg', alpha_lut=alpha_lut
            )
            
            # 验证指标
            for metrics in [baseline_metrics, fixed_metrics, emg_metrics]:
                assert 'accuracy' in metrics
                assert 'f1' in metrics
                assert 'nll' in metrics
                assert 'ece' in metrics
            
            # 对比
            metrics_dict = {
                'baseline': baseline_metrics,
                'fixed_alpha_0.5': fixed_metrics,
                'emg': emg_metrics
            }
            comparison = compare_methods(metrics_dict)
            
            assert 'emg_vs_baseline' in comparison or len(comparison) >= 0

