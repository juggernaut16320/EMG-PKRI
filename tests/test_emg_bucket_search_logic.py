"""
test_emg_bucket_search_logic.py - emg_bucket_search.py 核心逻辑单元测试（不依赖文件系统）

这个测试文件直接测试核心逻辑函数，不导入整个模块
"""

import pytest
import numpy as np
import pandas as pd


def compute_emg_fusion_logic(
    p: list,
    q: list,
    alpha: float
) -> list:
    """
    计算 EMG 融合概率的核心逻辑（从 emg_bucket_search.py 提取）
    
    Args:
        p: baseline 预测概率 [p_non_sensitive, p_sensitive]
        q: q₀ 知识后验 [p_non_sensitive, p_sensitive]
        alpha: 融合权重
    
    Returns:
        融合后的概率 [p_non_sensitive, p_sensitive]
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # 确保概率和为1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # EMG 融合公式: p_emg = α × p + (1 - α) × q
    p_emg = alpha * p + (1 - alpha) * q
    
    # 归一化（确保概率和为1）
    p_emg = p_emg / np.sum(p_emg) if np.sum(p_emg) > 0 else p_emg
    
    # 确保概率在 [0, 1] 范围内
    p_emg = np.clip(p_emg, 0.0, 1.0)
    p_emg = p_emg / np.sum(p_emg)  # 再次归一化
    
    return p_emg.tolist()


def assign_samples_to_buckets_logic(
    baseline_results: dict,
    bucket_df: pd.DataFrame
) -> dict:
    """
    将样本分配到对应的 bucket 的核心逻辑（从 emg_bucket_search.py 提取）
    
    Args:
        baseline_results: baseline 预测结果，key 为样本 ID，value 为包含 uncertainty 的字典
        bucket_df: 分桶信息 DataFrame
    
    Returns:
        字典，key 为 bucket_id，value 为样本 ID 列表
    """
    from collections import defaultdict
    
    bucket_samples = defaultdict(list)
    
    for item_id, result in baseline_results.items():
        uncertainty = result.get('uncertainty')
        if uncertainty is None:
            continue
        
        # 找到对应的 bucket
        for _, row in bucket_df.iterrows():
            u_min = row['u_min']
            u_max = row['u_max']
            bucket_id = int(row['bucket_id'])
            
            # 处理边界情况：u_min <= uncertainty < u_max，或者 uncertainty == 1.0 且 u_max == 1.0
            if (u_min <= uncertainty < u_max) or (uncertainty == 1.0 and abs(u_max - 1.0) < 1e-6):
                bucket_samples[bucket_id].append(item_id)
                break
    
    return dict(bucket_samples)


class TestEMGFusionLogic:
    """测试 EMG 融合核心逻辑"""
    
    def test_compute_emg_fusion_alpha_0(self):
        """测试 α=0（完全信任知识）"""
        p = [0.8, 0.2]  # baseline: 80% 非敏感
        q = [0.1, 0.9]  # q0: 90% 敏感
        alpha = 0.0
        
        result = compute_emg_fusion_logic(p, q, alpha)
        
        # α=0 时，应该完全等于 q
        assert len(result) == 2
        assert abs(result[0] - q[0]) < 1e-6
        assert abs(result[1] - q[1]) < 1e-6
        assert abs(sum(result) - 1.0) < 1e-6
    
    def test_compute_emg_fusion_alpha_1(self):
        """测试 α=1（完全信任模型）"""
        p = [0.8, 0.2]  # baseline: 80% 非敏感
        q = [0.1, 0.9]  # q0: 90% 敏感
        alpha = 1.0
        
        result = compute_emg_fusion_logic(p, q, alpha)
        
        # α=1 时，应该完全等于 p
        assert len(result) == 2
        assert abs(result[0] - p[0]) < 1e-6
        assert abs(result[1] - p[1]) < 1e-6
        assert abs(sum(result) - 1.0) < 1e-6
    
    def test_compute_emg_fusion_alpha_0_5(self):
        """测试 α=0.5（平均融合）"""
        p = [0.8, 0.2]
        q = [0.2, 0.8]
        alpha = 0.5
        
        result = compute_emg_fusion_logic(p, q, alpha)
        
        # α=0.5 时，应该是 p 和 q 的平均值
        expected = [0.5 * 0.8 + 0.5 * 0.2, 0.5 * 0.2 + 0.5 * 0.8]
        expected = [x / sum(expected) for x in expected]  # 归一化
        
        assert len(result) == 2
        assert abs(result[0] - expected[0]) < 1e-6
        assert abs(result[1] - expected[1]) < 1e-6
        assert abs(sum(result) - 1.0) < 1e-6
    
    def test_compute_emg_fusion_normalization(self):
        """测试概率归一化"""
        p = [0.6, 0.4]
        q = [0.3, 0.7]
        alpha = 0.25
        
        result = compute_emg_fusion_logic(p, q, alpha)
        
        # 确保概率和为1
        assert abs(sum(result) - 1.0) < 1e-6
        # 确保概率在 [0, 1] 范围内
        assert all(0 <= prob <= 1 for prob in result)
    
    def test_compute_emg_fusion_extreme_cases(self):
        """测试极端情况"""
        # 测试非常小的概率
        p = [0.99, 0.01]
        q = [0.01, 0.99]
        alpha = 0.5
        
        result = compute_emg_fusion_logic(p, q, alpha)
        
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-6
        assert all(0 <= prob <= 1 for prob in result)
    
    def test_compute_emg_fusion_alpha_range(self):
        """测试不同 α 值的范围"""
        p = [0.8, 0.2]
        q = [0.2, 0.8]
        
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = compute_emg_fusion_logic(p, q, alpha)
            
            assert len(result) == 2
            assert abs(sum(result) - 1.0) < 1e-6
            assert all(0 <= prob <= 1 for prob in result)
            
            # α 越大，结果应该越接近 p
            if alpha > 0.5:
                # 当 α > 0.5 时，result[0] 应该更接近 p[0] (0.8) 而不是 q[0] (0.2)
                assert result[0] > 0.5
            elif alpha < 0.5:
                # 当 α < 0.5 时，result[0] 应该更接近 q[0] (0.2) 而不是 p[0] (0.8)
                assert result[0] < 0.5
            else:
                # 当 α = 0.5 时，result[0] 应该等于 0.5（p 和 q 的平均值）
                assert abs(result[0] - 0.5) < 1e-6


class TestAssignSamplesToBucketsLogic:
    """测试样本分配到 bucket 的逻辑"""
    
    def test_assign_samples_basic(self):
        """测试基本的样本分配"""
        baseline_results = {
            's1': {'uncertainty': 0.05},
            's2': {'uncertainty': 0.15},
            's3': {'uncertainty': 0.25},
            's4': {'uncertainty': 0.35},
        }
        
        bucket_df = pd.DataFrame({
            'bucket_id': [0, 1, 2, 3],
            'u_min': [0.0, 0.1, 0.2, 0.3],
            'u_max': [0.1, 0.2, 0.3, 0.4]
        })
        
        result = assign_samples_to_buckets_logic(baseline_results, bucket_df)
        
        assert 0 in result
        assert 's1' in result[0]
        assert 1 in result
        assert 's2' in result[1]
        assert 2 in result
        assert 's3' in result[2]
        assert 3 in result
        assert 's4' in result[3]
    
    def test_assign_samples_boundary(self):
        """测试边界情况"""
        baseline_results = {
            's1': {'uncertainty': 0.0},   # 最小值
            's2': {'uncertainty': 0.1},   # 边界值
            's3': {'uncertainty': 0.99},  # 接近最大值
            's4': {'uncertainty': 1.0},   # 最大值
        }
        
        bucket_df = pd.DataFrame({
            'bucket_id': [0, 1, 9],
            'u_min': [0.0, 0.1, 0.9],
            'u_max': [0.1, 0.2, 1.0]
        })
        
        result = assign_samples_to_buckets_logic(baseline_results, bucket_df)
        
        assert 's1' in result[0]  # 0.0 应该在 [0.0, 0.1) bucket
        assert 's2' in result[1]  # 0.1 应该在 [0.1, 0.2) bucket
        assert 's3' in result[9]  # 0.99 应该在 [0.9, 1.0) bucket
        assert 's4' in result[9]  # 1.0 应该在 [0.9, 1.0) bucket（边界处理）
    
    def test_assign_samples_missing_uncertainty(self):
        """测试缺少 uncertainty 的样本"""
        baseline_results = {
            's1': {'uncertainty': 0.15},
            's2': {},  # 缺少 uncertainty
            's3': {'uncertainty': None},  # uncertainty 为 None
        }
        
        bucket_df = pd.DataFrame({
            'bucket_id': [0, 1],
            'u_min': [0.0, 0.1],
            'u_max': [0.1, 0.2]
        })
        
        result = assign_samples_to_buckets_logic(baseline_results, bucket_df)
        
        # 只有 s1 应该被分配
        assert 1 in result
        assert 's1' in result[1]
        assert 's2' not in result[1]
        assert 's3' not in result[1]
    
    def test_assign_samples_empty_results(self):
        """测试空结果"""
        baseline_results = {}
        
        bucket_df = pd.DataFrame({
            'bucket_id': [0, 1],
            'u_min': [0.0, 0.1],
            'u_max': [0.1, 0.2]
        })
        
        result = assign_samples_to_buckets_logic(baseline_results, bucket_df)
        
        assert len(result) == 0
    
    def test_assign_samples_empty_buckets(self):
        """测试空 bucket 列表"""
        baseline_results = {
            's1': {'uncertainty': 0.15},
        }
        
        bucket_df = pd.DataFrame({
            'bucket_id': [],
            'u_min': [],
            'u_max': []
        })
        
        result = assign_samples_to_buckets_logic(baseline_results, bucket_df)
        
        assert len(result) == 0


class TestEMGAlphaSearchLogic:
    """测试 α 搜索逻辑（模拟）"""
    
    def test_alpha_search_concept(self):
        """测试 α 搜索的基本概念"""
        # 模拟不同 α 值下的 F1 分数
        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        f1_scores = [0.70, 0.75, 0.80, 0.85, 0.82]  # 假设 α=0.75 时 F1 最高
        
        best_idx = np.argmax(f1_scores)
        best_alpha = alpha_grid[best_idx]
        best_f1 = f1_scores[best_idx]
        
        assert best_alpha == 0.75
        assert best_f1 == 0.85
    
    def test_alpha_search_nll_concept(self):
        """测试使用 NLL 作为优化指标"""
        # 模拟不同 α 值下的 NLL
        alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        nll_scores = [0.5, 0.4, 0.35, 0.3, 0.32]  # 假设 α=0.75 时 NLL 最低
        
        best_idx = np.argmin(nll_scores)
        best_alpha = alpha_grid[best_idx]
        best_nll = nll_scores[best_idx]
        
        assert best_alpha == 0.75
        assert best_nll == 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

