"""
test_dataset_split.py - dataset_split.py 单元测试
"""

import os
import sys
import json
import tempfile
import pytest
import numpy as np
from pathlib import Path

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from dataset_split import split_dataset, load_config


class TestDatasetSplit:
    """测试数据集划分功能"""
    
    def test_split_basic(self):
        """测试基本划分功能"""
        # 创建测试数据
        data = []
        for i in range(100):
            data.append({
                "id": f"s{i}",
                "text": f"text {i}",
                "coarse_label": 1 if i % 3 == 0 else 0  # 约33%敏感
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            train_path = os.path.join(tmpdir, "train.jsonl")
            dev_path = os.path.join(tmpdir, "dev.jsonl")
            test_path = os.path.join(tmpdir, "test.jsonl")
            
            # 写入输入文件
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 执行划分
            stats = split_dataset(
                input_path=input_path,
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42
            )
            
            # 验证统计
            assert stats['total'] == 100
            assert stats['train'] == 80
            assert stats['dev'] == 10
            assert stats['test'] == 10
            assert stats['train'] + stats['dev'] + stats['test'] == stats['total']
            
            # 验证文件存在
            assert os.path.exists(train_path)
            assert os.path.exists(dev_path)
            assert os.path.exists(test_path)
            
            # 验证文件内容
            train_data = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        train_data.append(json.loads(line))
            
            assert len(train_data) == 80
    
    def test_split_stratify(self):
        """测试分层划分"""
        # 创建测试数据（50个敏感，50个非敏感）
        data = []
        for i in range(100):
            data.append({
                "id": f"s{i}",
                "text": f"text {i}",
                "coarse_label": 1 if i < 50 else 0
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            train_path = os.path.join(tmpdir, "train.jsonl")
            dev_path = os.path.join(tmpdir, "dev.jsonl")
            test_path = os.path.join(tmpdir, "test.jsonl")
            
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 执行分层划分
            stats = split_dataset(
                input_path=input_path,
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42,
                stratify_field="coarse_label"
            )
            
            # 验证分层效果：每个集合中敏感/非敏感比例应该接近原始比例（50%）
            train_data = []
            dev_data = []
            test_data = []
            
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        train_data.append(json.loads(line))
            
            with open(dev_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        dev_data.append(json.loads(line))
            
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
            
            # 计算各集合的敏感比例
            train_sensitive_ratio = stats['train_sensitive'] / stats['train']
            dev_sensitive_ratio = stats['dev_sensitive'] / stats['dev']
            test_sensitive_ratio = stats['test_sensitive'] / stats['test']
            
            # 验证分层：各集合的敏感比例应该接近原始比例（0.5）
            # 允许一定误差（±10%）
            assert abs(train_sensitive_ratio - 0.5) < 0.1, f"训练集敏感比例 {train_sensitive_ratio} 偏离预期"
            assert abs(dev_sensitive_ratio - 0.5) < 0.1, f"验证集敏感比例 {dev_sensitive_ratio} 偏离预期"
            assert abs(test_sensitive_ratio - 0.5) < 0.1, f"测试集敏感比例 {test_sensitive_ratio} 偏离预期"
    
    def test_split_reproducibility(self):
        """测试可复现性（相同种子产生相同结果）"""
        # 创建测试数据
        data = []
        for i in range(100):
            data.append({
                "id": f"s{i}",
                "text": f"text {i}",
                "coarse_label": 1 if i % 3 == 0 else 0
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 第一次划分
            train_path1 = os.path.join(tmpdir, "train1.jsonl")
            dev_path1 = os.path.join(tmpdir, "dev1.jsonl")
            test_path1 = os.path.join(tmpdir, "test1.jsonl")
            
            stats1 = split_dataset(
                input_path=input_path,
                train_path=train_path1,
                dev_path=dev_path1,
                test_path=test_path1,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42
            )
            
            # 第二次划分（相同种子）
            train_path2 = os.path.join(tmpdir, "train2.jsonl")
            dev_path2 = os.path.join(tmpdir, "dev2.jsonl")
            test_path2 = os.path.join(tmpdir, "test2.jsonl")
            
            stats2 = split_dataset(
                input_path=input_path,
                train_path=train_path2,
                dev_path=dev_path2,
                test_path=test_path2,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42
            )
            
            # 验证统计相同
            assert stats1 == stats2
            
            # 验证文件内容相同（通过读取ID比较）
            def get_ids(file_path):
                ids = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            ids.append(item['id'])
                return sorted(ids)
            
            assert get_ids(train_path1) == get_ids(train_path2)
            assert get_ids(dev_path1) == get_ids(dev_path2)
            assert get_ids(test_path1) == get_ids(test_path2)
    
    def test_split_different_seed(self):
        """测试不同种子产生不同结果"""
        # 创建测试数据
        data = []
        for i in range(100):
            data.append({
                "id": f"s{i}",
                "text": f"text {i}",
                "coarse_label": 1 if i % 3 == 0 else 0
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 种子42
            train_path1 = os.path.join(tmpdir, "train1.jsonl")
            dev_path1 = os.path.join(tmpdir, "dev1.jsonl")
            test_path1 = os.path.join(tmpdir, "test1.jsonl")
            
            split_dataset(
                input_path=input_path,
                train_path=train_path1,
                dev_path=dev_path1,
                test_path=test_path1,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42
            )
            
            # 种子123
            train_path2 = os.path.join(tmpdir, "train2.jsonl")
            dev_path2 = os.path.join(tmpdir, "dev2.jsonl")
            test_path2 = os.path.join(tmpdir, "test2.jsonl")
            
            split_dataset(
                input_path=input_path,
                train_path=train_path2,
                dev_path=dev_path2,
                test_path=test_path2,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=123
            )
            
            # 验证结果不同
            def get_ids(file_path):
                ids = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            ids.append(item['id'])
                return sorted(ids)
            
            train_ids1 = get_ids(train_path1)
            train_ids2 = get_ids(train_path2)
            
            # 不同种子应该产生不同的划分（虽然可能偶尔相同，但概率极低）
            # 这里只验证它们不完全相同
            assert train_ids1 != train_ids2 or len(train_ids1) == 0
    
    def test_split_empty_data(self):
        """测试空数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            train_path = os.path.join(tmpdir, "train.jsonl")
            dev_path = os.path.join(tmpdir, "dev.jsonl")
            test_path = os.path.join(tmpdir, "test.jsonl")
            
            # 创建空文件
            with open(input_path, 'w', encoding='utf-8') as f:
                pass
            
            stats = split_dataset(
                input_path=input_path,
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42
            )
            
            assert stats['total'] == 0
            assert stats['train'] == 0
            assert stats['dev'] == 0
            assert stats['test'] == 0
    
    def test_split_no_stratify_field(self):
        """测试没有分层字段时的行为（应该使用随机划分）"""
        # 创建没有 coarse_label 的数据
        data = []
        for i in range(100):
            data.append({
                "id": f"s{i}",
                "text": f"text {i}"
                # 没有 coarse_label 字段
            })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jsonl")
            train_path = os.path.join(tmpdir, "train.jsonl")
            dev_path = os.path.join(tmpdir, "dev.jsonl")
            test_path = os.path.join(tmpdir, "test.jsonl")
            
            with open(input_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 应该能正常运行（使用随机划分，不分层）
            stats = split_dataset(
                input_path=input_path,
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                train_ratio=0.8,
                dev_ratio=0.1,
                test_ratio=0.1,
                random_seed=42,
                stratify_field="coarse_label"
            )
            
            assert stats['total'] == 100
            assert stats['train'] == 80
            assert stats['dev'] == 10
            assert stats['test'] == 10


class TestLoadConfig:
    """测试配置加载"""
    
    def test_load_config(self):
        """测试加载配置文件"""
        config = load_config("configs/config.yaml")
        assert "split" in config
        assert "train" in config["split"]
        assert "dev" in config["split"]
        assert "test" in config["split"]

