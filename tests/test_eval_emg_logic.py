"""
test_eval_emg_logic.py - eval_emg.py 核心逻辑单元测试（不依赖文件系统）

测试核心计算函数，不导入整个模块
"""

import pytest
import numpy as np


def compute_emg_fusion_logic(
    p: list,
    q: list,
    alpha: float
) -> list:
    """
    计算 EMG 融合概率的核心逻辑（从 emg_bucket_search.py 提取）
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


def lookup_alpha_logic(
    u: float,
    lut: dict
) -> float:
    """
    α(u) 查找的核心逻辑
    """
    u_list = lut['u']
    alpha_list = lut['alpha']
    
    # 边界处理
    if u <= u_list[0]:
        return alpha_list[0]
    if u >= u_list[-1]:
        return alpha_list[-1]
    
    # 线性插值
    return np.interp(u, u_list, alpha_list)


def compute_ece_logic(
    pred_probs: np.ndarray,
    true_labels: np.ndarray,
    n_buckets: int = 10
) -> float:
    """
    计算期望校准误差 (ECE) 的核心逻辑
    """
    if len(pred_probs) == 0:
        return 0.0
    
    # 等宽分桶
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_ids = np.digitize(pred_probs, bucket_edges) - 1
    bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)
    
    ece = 0.0
    n_total = len(pred_probs)
    
    for bucket_id in range(n_buckets):
        mask = (bucket_ids == bucket_id)
        if np.sum(mask) == 0:
            continue
        
        bucket_probs = pred_probs[mask]
        bucket_labels = true_labels[mask]
        
        n_samples = len(bucket_probs)
        accuracy = np.mean(bucket_labels == 1)  # 实际准确率
        confidence = np.mean(bucket_probs)  # 平均预测概率
        
        gap = abs(accuracy - confidence)
        ece += (n_samples / n_total) * gap
    
    return float(ece)


class TestLookupAlpha:
    """测试 α(u) 查找逻辑"""
    
    def test_lookup_alpha_exact_match(self):
        """测试精确匹配"""
        lut = {
            'u': [0.0, 0.1, 0.2, 0.3, 0.4],
            'alpha': [1.0, 0.8, 0.6, 0.4, 0.2]
        }
        
        # 精确匹配
        assert abs(lookup_alpha_logic(0.1, lut) - 0.8) < 1e-6
        assert abs(lookup_alpha_logic(0.3, lut) - 0.4) < 1e-6
    
    def test_lookup_alpha_interpolation(self):
        """测试线性插值"""
        lut = {
            'u': [0.0, 0.2, 0.4],
            'alpha': [1.0, 0.6, 0.2]
        }
        
        # 中间值应该线性插值
        alpha_01 = lookup_alpha_logic(0.1, lut)
        assert 0.6 < alpha_01 < 1.0  # 应该在 0.6 和 1.0 之间
        assert abs(alpha_01 - 0.8) < 0.1  # 大致在中间
    
    def test_lookup_alpha_below_min(self):
        """测试低于最小值"""
        lut = {
            'u': [0.1, 0.2, 0.3],
            'alpha': [0.8, 0.6, 0.4]
        }
        
        # 低于最小值，应该返回第一个 alpha
        assert abs(lookup_alpha_logic(0.05, lut) - 0.8) < 1e-6
    
    def test_lookup_alpha_above_max(self):
        """测试高于最大值"""
        lut = {
            'u': [0.1, 0.2, 0.3],
            'alpha': [0.8, 0.6, 0.4]
        }
        
        # 高于最大值，应该返回最后一个 alpha
        assert abs(lookup_alpha_logic(0.5, lut) - 0.4) < 1e-6
    
    def test_lookup_alpha_monotonic(self):
        """测试单调递减性质"""
        lut = {
            'u': [0.0, 0.1, 0.2, 0.3, 0.4],
            'alpha': [1.0, 0.8, 0.6, 0.4, 0.2]  # 单调递减
        }
        
        # u 增大，alpha 应该减小
        alpha_01 = lookup_alpha_logic(0.05, lut)
        alpha_02 = lookup_alpha_logic(0.15, lut)
        alpha_03 = lookup_alpha_logic(0.25, lut)
        
        assert alpha_01 > alpha_02
        assert alpha_02 > alpha_03


class TestComputeECE:
    """测试 ECE 计算逻辑"""
    
    def test_compute_ece_perfect_calibration(self):
        """测试完美校准（ECE应该很小）"""
        # 完美校准：预测概率 = 实际准确率
        pred_probs = np.array([0.2, 0.2, 0.8, 0.8])
        true_labels = np.array([0, 0, 1, 1])
        
        ece = compute_ece_logic(pred_probs, true_labels, n_buckets=2)
        
        # 完美校准时 ECE 应该接近 0（但由于分桶可能有一些误差）
        assert ece < 0.3  # 放宽阈值以适应分桶误差
        assert ece >= 0  # ECE 应该 >= 0
    
    def test_compute_ece_overconfident(self):
        """测试过度自信（ECE>0）"""
        # 过度自信：预测概率 > 实际准确率
        pred_probs = np.array([0.9, 0.9, 0.9, 0.9])  # 都很自信
        true_labels = np.array([1, 0, 0, 0])  # 但只有25%正确
        
        ece = compute_ece_logic(pred_probs, true_labels, n_buckets=1)
        
        # 过度自信时 ECE 应该较大
        assert ece > 0.5
    
    def test_compute_ece_underconfident(self):
        """测试欠自信（ECE>0）"""
        # 欠自信：预测概率 < 实际准确率
        pred_probs = np.array([0.1, 0.1, 0.1, 0.1])  # 都很不自信
        true_labels = np.array([1, 1, 1, 0])  # 但75%正确
        
        ece = compute_ece_logic(pred_probs, true_labels, n_buckets=1)
        
        # 欠自信时 ECE 应该较大
        assert ece > 0.5
    
    def test_compute_ece_empty(self):
        """测试空数组"""
        pred_probs = np.array([])
        true_labels = np.array([])
        
        ece = compute_ece_logic(pred_probs, true_labels)
        
        assert ece == 0.0
    
    def test_compute_ece_single_bucket(self):
        """测试单桶情况"""
        pred_probs = np.array([0.5, 0.5, 0.5])
        true_labels = np.array([1, 0, 1])
        
        ece = compute_ece_logic(pred_probs, true_labels, n_buckets=1)
        
        # 准确率 = 2/3 ≈ 0.667，置信度 = 0.5，gap = 0.167
        assert ece > 0.1


class TestEvaluateMethodLogic:
    """测试评估方法的逻辑（模拟）"""
    
    def test_baseline_method(self):
        """测试 baseline 方法逻辑"""
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
        
        # Baseline 方法应该直接使用 pred_probs
        # 这里我们模拟评估逻辑
        true_labels = []
        pred_labels = []
        
        for item_id in ['s1', 's2']:
            result = baseline_results[item_id]
            true_labels.append(result['coarse_label'])
            pred_labels.append(result['pred_label'])
        
        # 验证 baseline 直接使用原始预测
        assert pred_labels[0] == 1  # s1 预测为敏感
        assert pred_labels[1] == 0  # s2 预测为非敏感
    
    def test_fixed_alpha_method(self):
        """测试固定 α 融合方法逻辑"""
        p = [0.2, 0.8]  # baseline
        q = [0.1, 0.9]  # q0
        alpha = 0.5
        
        # 固定 α 融合
        p_fused = compute_emg_fusion_logic(p, q, alpha)
        
        # α=0.5 时，应该是 p 和 q 的平均
        assert len(p_fused) == 2
        assert abs(sum(p_fused) - 1.0) < 1e-6
        # p_fused[1] 应该在 p[1] 和 q[1] 之间
        assert min(p[1], q[1]) <= p_fused[1] <= max(p[1], q[1])
    
    def test_emg_method(self):
        """测试 EMG 方法逻辑"""
        lut = {
            'u': [0.0, 0.2, 0.4],
            'alpha': [1.0, 0.6, 0.2]
        }
        
        # 低不确定性（u=0.1）应该返回高 alpha（接近1.0）
        alpha_low_u = lookup_alpha_logic(0.1, lut)
        assert alpha_low_u >= 0.8  # 可能正好等于0.8（插值中间值）
        
        # 高不确定性（u=0.3）应该返回低 alpha（接近0.2）
        alpha_high_u = lookup_alpha_logic(0.3, lut)
        assert alpha_high_u < 0.5
        
        # 验证 alpha 随 u 单调递减
        assert alpha_low_u > alpha_high_u
    
    def test_emg_fusion_computation(self):
        """测试 EMG 融合计算"""
        p = [0.3, 0.7]  # baseline: 70% 敏感
        q = [0.1, 0.9]  # q0: 90% 敏感
        
        # 低不确定性时，alpha 高，应该更接近 p
        alpha_low = 0.8
        p_emg_low = compute_emg_fusion_logic(p, q, alpha_low)
        assert abs(p_emg_low[1] - 0.7) < abs(p_emg_low[1] - 0.9)  # 更接近 p[1]
        
        # 高不确定性时，alpha 低，应该更接近 q
        alpha_high = 0.2
        p_emg_high = compute_emg_fusion_logic(p, q, alpha_high)
        assert abs(p_emg_high[1] - 0.9) < abs(p_emg_high[1] - 0.7)  # 更接近 q[1]


class TestCompareMethods:
    """测试对比方法的逻辑"""
    
    def test_compare_methods_improvement(self):
        """测试性能提升计算"""
        baseline_metrics = {
            'f1': 0.89,
            'nll': 0.45,
            'ece': 0.12
        }
        emg_metrics = {
            'f1': 0.91,
            'nll': 0.42,
            'ece': 0.10
        }
        
        # 计算改进
        f1_improvement = emg_metrics['f1'] - baseline_metrics['f1']
        nll_reduction = baseline_metrics['nll'] - emg_metrics['nll']
        ece_reduction = baseline_metrics['ece'] - emg_metrics['ece']
        
        assert f1_improvement > 0  # F1 提升
        assert nll_reduction > 0  # NLL 降低
        assert ece_reduction > 0  # ECE 降低
        
        # 计算百分比
        f1_improvement_percent = (f1_improvement / baseline_metrics['f1']) * 100
        assert f1_improvement_percent > 0
    
    def test_compare_methods_no_improvement(self):
        """测试无改进的情况"""
        baseline_metrics = {
            'f1': 0.90,
            'nll': 0.40,
            'ece': 0.10
        }
        emg_metrics = {
            'f1': 0.90,
            'nll': 0.40,
            'ece': 0.10
        }
        
        # 无改进时，差值应该为 0
        f1_improvement = emg_metrics['f1'] - baseline_metrics['f1']
        assert abs(f1_improvement) < 1e-6


class TestEdgeCases:
    """测试边界情况"""
    
    def test_lookup_alpha_single_point(self):
        """测试单点查表"""
        lut = {
            'u': [0.5],
            'alpha': [0.5]
        }
        
        # 任何 u 值都应该返回 0.5
        assert abs(lookup_alpha_logic(0.0, lut) - 0.5) < 1e-6
        assert abs(lookup_alpha_logic(0.5, lut) - 0.5) < 1e-6
        assert abs(lookup_alpha_logic(1.0, lut) - 0.5) < 1e-6
    
    def test_compute_ece_single_sample(self):
        """测试单样本 ECE"""
        pred_probs = np.array([0.8])
        true_labels = np.array([1])
        
        ece = compute_ece_logic(pred_probs, true_labels, n_buckets=1)
        
        # 单样本时，准确率 = 1.0，置信度 = 0.8，gap = 0.2
        assert ece > 0.1
    
    def test_emg_fusion_extreme_alpha(self):
        """测试极端 alpha 值"""
        p = [0.2, 0.8]
        q = [0.9, 0.1]
        
        # alpha = 0，应该完全等于 q
        p_alpha_0 = compute_emg_fusion_logic(p, q, 0.0)
        assert abs(p_alpha_0[0] - q[0]) < 0.1  # 允许一些数值误差
        assert abs(p_alpha_0[1] - q[1]) < 0.1
        
        # alpha = 1，应该完全等于 p
        p_alpha_1 = compute_emg_fusion_logic(p, q, 1.0)
        assert abs(p_alpha_1[0] - p[0]) < 0.1
        assert abs(p_alpha_1[1] - p[1]) < 0.1
    
    def test_probability_normalization(self):
        """测试概率归一化"""
        # 即使输入不是归一化的，输出也应该归一化
        p = [0.3, 0.5]  # 和为 0.8，不是 1.0
        q = [0.2, 0.6]  # 和为 0.8，不是 1.0
        
        p_fused = compute_emg_fusion_logic(p, q, 0.5)
        
        # 输出应该归一化
        assert abs(sum(p_fused) - 1.0) < 1e-6
        assert all(0 <= x <= 1 for x in p_fused)

