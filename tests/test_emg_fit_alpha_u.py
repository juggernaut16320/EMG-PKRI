"""
test_emg_fit_alpha_u.py - emg_fit_alpha_u.py 完整功能测试

测试脚本基本功能、输入输出格式、错误处理等
"""

import pytest
import os
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# 导入被测试的模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.emg_fit_alpha_u import (
    pav_regression,
    fit_alpha_u,
    create_lookup_table,
    load_bucket_alpha_star,
    plot_alpha_u_curve
)


class TestPAVRegression:
    """测试 PAV 回归算法"""
    
    def test_pav_regression_simple(self):
        """测试简单的 PAV 回归"""
        y = np.array([1.0, 2.0, 1.5, 3.0])
        result = pav_regression(y)
        
        # 结果应该是单调递增的
        assert np.all(result[:-1] <= result[1:])
        assert len(result) == len(y)
    
    def test_pav_regression_already_sorted(self):
        """测试已经排序的数组"""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        result = pav_regression(y)
        
        # 已经单调递增，应该不变
        assert np.allclose(result, y)
    
    def test_pav_regression_reverse(self):
        """测试反向排序的数组"""
        y = np.array([4.0, 3.0, 2.0, 1.0])
        result = pav_regression(y)
        
        # 结果应该是单调递增的（所有值可能都相同）
        assert np.all(result[:-1] <= result[1:])


class TestFitAlphaU:
    """测试 α(u) 拟合"""
    
    def test_fit_alpha_u_monotonic_decreasing(self):
        """测试单调递减的拟合"""
        # 创建模拟数据：u越大，alpha越小（符合EMG理论）
        u_values = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
        alpha_values = np.array([0.8, 0.6, 0.5, 0.3, 0.2])  # 单调递减
        
        u_fitted, alpha_fitted = fit_alpha_u(u_values, alpha_values, use_scipy=False)
        
        # u应该是排序的
        assert np.all(u_fitted[:-1] <= u_fitted[1:])
        
        # alpha应该是单调递减的（u增大，alpha减小）
        assert np.all(alpha_fitted[:-1] >= alpha_fitted[1:])
        
        # alpha应该在[0,1]范围内
        assert np.all((alpha_fitted >= 0) & (alpha_fitted <= 1))
    
    def test_fit_alpha_u_with_violations(self):
        """测试有违反单调性的数据"""
        # 创建有违反单调性的数据
        u_values = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
        alpha_values = np.array([0.8, 0.9, 0.5, 0.3, 0.2])  # 第二个值违反了单调递减
        
        u_fitted, alpha_fitted = fit_alpha_u(u_values, alpha_values, use_scipy=False)
        
        # 拟合后应该是单调递减的
        assert np.all(alpha_fitted[:-1] >= alpha_fitted[1:])
    
    def test_fit_alpha_u_single_point(self):
        """测试单点数据"""
        u_values = np.array([0.5])
        alpha_values = np.array([0.5])
        
        u_fitted, alpha_fitted = fit_alpha_u(u_values, alpha_values, use_scipy=False)
        
        assert len(u_fitted) == 1
        assert len(alpha_fitted) == 1
        assert u_fitted[0] == 0.5
        assert alpha_fitted[0] == 0.5


class TestCreateLookupTable:
    """测试查表创建"""
    
    def test_create_lookup_table(self):
        """测试创建查表"""
        u_values = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        alpha_values = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
        
        lut = create_lookup_table(u_values, alpha_values, n_points=20)
        
        assert 'u' in lut
        assert 'alpha' in lut
        assert len(lut['u']) == 20
        assert len(lut['alpha']) == 20
        
        # u应该覆盖整个范围
        assert abs(lut['u'][0] - 0.0) < 1e-6
        assert abs(lut['u'][-1] - 1.0) < 1e-6
        
        # alpha应该在[0,1]范围内
        assert all(0 <= a <= 1 for a in lut['alpha'])
    
    def test_create_lookup_table_small_range(self):
        """测试小范围的查表"""
        u_values = np.array([0.3, 0.4, 0.5])
        alpha_values = np.array([0.7, 0.6, 0.5])
        
        lut = create_lookup_table(u_values, alpha_values, n_points=10)
        
        assert len(lut['u']) == 10
        assert len(lut['alpha']) == 10


class TestLoadBucketAlphaStar:
    """测试加载 bucket alpha star 文件"""
    
    def test_load_bucket_alpha_star_valid(self):
        """测试加载有效的文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df = pd.DataFrame({
                'bucket_id': [0, 1, 2, 3],
                'u_min': [0.0, 0.1, 0.2, 0.3],
                'u_max': [0.1, 0.2, 0.3, 0.4],
                'u_mean': [0.05, 0.15, 0.25, 0.35],
                'n_samples': [100, 50, 30, 20],
                'alpha_star': [0.75, 0.5, 0.25, 0.0],
                'f1_at_alpha_star': [0.9, 0.85, 0.8, 0.75]
            })
            df.to_csv(f, index=False)
            temp_file = f.name
        
        try:
            result_df = load_bucket_alpha_star(temp_file)
            
            assert len(result_df) == 4
            assert 'bucket_id' in result_df.columns
            assert 'u_mean' in result_df.columns
            assert 'alpha_star' in result_df.columns
        finally:
            os.unlink(temp_file)
    
    def test_load_bucket_alpha_star_file_not_exists(self):
        """测试文件不存在的情况"""
        result_df = load_bucket_alpha_star('nonexistent_file.csv')
        assert len(result_df) == 0


class TestIntegration:
    """集成测试"""
    
    def test_end_to_end(self):
        """端到端测试"""
        # 创建模拟的 bucket_alpha_star.csv
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建输入文件
            input_file = os.path.join(temp_dir, 'bucket_alpha_star.csv')
            df = pd.DataFrame({
                'bucket_id': [0, 1, 2, 3, 4],
                'u_min': [0.0, 0.1, 0.2, 0.3, 0.4],
                'u_max': [0.1, 0.2, 0.3, 0.4, 0.5],
                'u_mean': [0.05, 0.15, 0.25, 0.35, 0.45],
                'n_samples': [100, 80, 60, 40, 20],
                'alpha_star': [0.8, 0.7, 0.5, 0.3, 0.1],  # 单调递减
                'f1_at_alpha_star': [0.92, 0.88, 0.85, 0.82, 0.78],
                'nll_at_alpha_star': [0.2, 0.25, 0.3, 0.35, 0.4]
            })
            df.to_csv(input_file, index=False)
            
            # 加载数据
            loaded_df = load_bucket_alpha_star(input_file)
            assert len(loaded_df) == 5
            
            # 拟合
            u_values = loaded_df['u_mean'].values
            alpha_values = loaded_df['alpha_star'].values
            
            u_fitted, alpha_fitted = fit_alpha_u(u_values, alpha_values, use_scipy=False)
            
            # 检查结果
            assert len(u_fitted) == len(alpha_fitted)
            assert np.all(alpha_fitted[:-1] >= alpha_fitted[1:])  # 单调递减
            
            # 创建查表
            lut = create_lookup_table(u_fitted, alpha_fitted, n_points=50)
            
            assert len(lut['u']) == 50
            assert len(lut['alpha']) == 50
            assert all(0 <= a <= 1 for a in lut['alpha'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

