"""
test_uncertainty_analysis_logic.py - uncertainty_analysis.py 核心逻辑单元测试（不依赖 torch）

这个测试文件直接测试核心逻辑函数，不导入整个模块
"""

import pytest
import numpy as np
import pandas as pd


def compute_uncertainty_logic(results, metric='u_max'):
    """
    计算不确定性指标的核心逻辑（从 uncertainty_analysis.py 提取）
    """
    for result in results:
        pred_probs = np.array(result['pred_probs'])
        max_prob = np.max(pred_probs)  # 总是计算 max_prob
        
        if metric == 'u_max':
            uncertainty = 1.0 - max_prob
        elif metric == 'u_entropy':
            entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10))
            uncertainty = entropy / np.log(len(pred_probs))
        elif metric == 'u_margin':
            sorted_probs = np.sort(pred_probs)[::-1]
            if len(sorted_probs) >= 2:
                uncertainty = 1.0 - (sorted_probs[0] - sorted_probs[1])
            else:
                uncertainty = 0.0
        else:
            raise ValueError(f"未知的不确定性指标: {metric}")
        
        result['uncertainty'] = float(uncertainty)
        result['max_prob'] = float(max_prob)
    
    return results


def bucket_analysis_logic(results, n_buckets=10):
    """
    分桶分析的核心逻辑（从 uncertainty_analysis.py 提取）
    """
    valid_results = [r for r in results if r.get('coarse_label') is not None]
    
    if len(valid_results) == 0:
        return pd.DataFrame()
    
    uncertainties = np.array([r['uncertainty'] for r in valid_results])
    is_error = np.array([
        int(r['pred_label']) != int(r['coarse_label']) 
        for r in valid_results
    ])
    max_probs = np.array([r['max_prob'] for r in valid_results])
    
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_indices = np.digitize(uncertainties, bucket_edges) - 1
    bucket_indices[bucket_indices < 0] = 0
    bucket_indices[bucket_indices >= n_buckets] = n_buckets - 1
    
    bucket_stats = []
    for i in range(n_buckets):
        mask = (bucket_indices == i)
        if np.sum(mask) == 0:
            continue
        
        bucket_uncertainties = uncertainties[mask]
        bucket_errors = is_error[mask]
        bucket_probs = max_probs[mask]
        
        bucket_stats.append({
            'bucket_id': i,
            'u_min': float(bucket_edges[i]),
            'u_max': float(bucket_edges[i + 1]),
            'u_mean': float(np.mean(bucket_uncertainties)),
            'u_median': float(np.median(bucket_uncertainties)),
            'n_samples': int(np.sum(mask)),
            'n_errors': int(np.sum(bucket_errors)),
            'error_rate': float(np.mean(bucket_errors)),
            'avg_confidence': float(np.mean(bucket_probs)),
            'avg_uncertainty': float(np.mean(bucket_uncertainties))
        })
    
    return pd.DataFrame(bucket_stats)


class TestUncertaintyAnalysisLogic:
    """测试不确定性分析核心逻辑"""
    
    def test_compute_uncertainty_u_max(self):
        """测试计算 u_max 不确定性"""
        results = [
            {
                'id': 's1',
                'pred_probs': [0.1, 0.9]  # 高置信
            },
            {
                'id': 's2',
                'pred_probs': [0.5, 0.5]  # 低置信
            },
        ]
        
        results = compute_uncertainty_logic(results, metric='u_max')
        
        # s1: u = 1 - 0.9 = 0.1
        assert abs(results[0]['uncertainty'] - 0.1) < 1e-6
        assert abs(results[0]['max_prob'] - 0.9) < 1e-6
        
        # s2: u = 1 - 0.5 = 0.5
        assert abs(results[1]['uncertainty'] - 0.5) < 1e-6
        assert abs(results[1]['max_prob'] - 0.5) < 1e-6
    
    def test_compute_uncertainty_u_entropy(self):
        """测试计算 u_entropy 不确定性"""
        results = [
            {
                'id': 's1',
                'pred_probs': [0.1, 0.9]  # 低熵
            },
            {
                'id': 's2',
                'pred_probs': [0.5, 0.5]  # 高熵
            },
        ]
        
        results = compute_uncertainty_logic(results, metric='u_entropy')
        
        # s2 应该比 s1 有更高的熵不确定性
        assert results[1]['uncertainty'] > results[0]['uncertainty']
    
    def test_bucket_analysis(self):
        """测试分桶分析"""
        results = []
        for i in range(100):
            uncertainty = i / 100.0
            max_prob = 1.0 - uncertainty
            pred_probs = [1.0 - max_prob, max_prob] if i % 2 == 0 else [max_prob, 1.0 - max_prob]
            is_error = 1 if uncertainty > 0.5 else 0
            
            results.append({
                'id': f's{i}',
                'coarse_label': 0 if i % 2 == 0 else 1,
                'pred_label': 1 if is_error else (0 if i % 2 == 0 else 1),
                'pred_probs': pred_probs,
                'uncertainty': uncertainty,
                'max_prob': max_prob
            })
        
        bucket_df = bucket_analysis_logic(results, n_buckets=10)
        
        assert len(bucket_df) > 0
        assert 'u_mean' in bucket_df.columns
        assert 'error_rate' in bucket_df.columns
        assert 'n_samples' in bucket_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

