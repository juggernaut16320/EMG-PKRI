"""
test_lexicon_generator_logic.py - lexicon_generator.py 核心逻辑单元测试

测试核心逻辑函数，不依赖 torch/transformers 等重依赖
"""

import unittest
import json
import os
import sys
import tempfile
import shutil

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from lexicon_generator import (
    load_data_samples,
    generate_lexicon_from_samples,
    generate_regex_patterns,
    save_lexicon_file,
    save_regex_file
)


class TestLexiconGeneratorLogic(unittest.TestCase):
    """测试 lexicon_generator 的核心逻辑"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_file = os.path.join(self.temp_dir, "test_data.jsonl")
        
        # 创建测试数据
        test_data = [
            {
                "id": "s1",
                "text": "这是一个色情内容样本",
                "coarse_label": 1,
                "subtype_label": ["porn"]
            },
            {
                "id": "s2",
                "text": "这是另一个色情内容样本",
                "coarse_label": 1,
                "subtype_label": ["porn"]
            },
            {
                "id": "s3",
                "text": "这是涉政内容样本",
                "coarse_label": 1,
                "subtype_label": ["politics"]
            },
            {
                "id": "s4",
                "text": "这是辱骂内容样本",
                "coarse_label": 1,
                "subtype_label": ["abuse"]
            },
            {
                "id": "s5",
                "text": "这是非敏感内容",
                "coarse_label": 0,
                "subtype_label": []
            },
            {
                "id": "s6",
                "text": "这是另一个色情内容",
                "coarse_label": 1,
                "subtype_label": ["porn"]
            },
        ]
        
        with open(self.test_data_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_data_samples_porn(self):
        """测试加载 porn 类别样本"""
        samples = load_data_samples(
            self.test_data_file,
            category="porn",
            samples_per_category=10
        )
        
        self.assertEqual(len(samples), 3)  # 应该有3个porn样本
        for sample in samples:
            self.assertEqual(sample.get("coarse_label"), 1)
            self.assertIn("porn", sample.get("subtype_label", []))
    
    def test_load_data_samples_politics(self):
        """测试加载 politics 类别样本"""
        samples = load_data_samples(
            self.test_data_file,
            category="politics",
            samples_per_category=10
        )
        
        self.assertEqual(len(samples), 1)  # 应该有1个politics样本
        for sample in samples:
            self.assertEqual(sample.get("coarse_label"), 1)
            self.assertIn("politics", sample.get("subtype_label", []))
    
    def test_load_data_samples_abuse(self):
        """测试加载 abuse 类别样本"""
        samples = load_data_samples(
            self.test_data_file,
            category="abuse",
            samples_per_category=10
        )
        
        self.assertEqual(len(samples), 1)  # 应该有1个abuse样本
        for sample in samples:
            self.assertEqual(sample.get("coarse_label"), 1)
            self.assertIn("abuse", sample.get("subtype_label", []))
    
    def test_load_data_samples_empty_category(self):
        """测试加载不存在的类别"""
        samples = load_data_samples(
            self.test_data_file,
            category="other",
            samples_per_category=10
        )
        
        self.assertEqual(len(samples), 0)  # 应该没有样本
    
    def test_load_data_samples_sampling(self):
        """测试采样功能"""
        # 创建更多样本
        test_data = []
        for i in range(150):
            test_data.append({
                "id": f"s{i}",
                "text": f"色情内容样本{i}",
                "coarse_label": 1,
                "subtype_label": ["porn"]
            })
        
        large_data_file = os.path.join(self.temp_dir, "large_data.jsonl")
        with open(large_data_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 测试采样功能
        samples = load_data_samples(
            large_data_file,
            category="porn",
            samples_per_category=100
        )
        self.assertEqual(len(samples), 100)  # 应该采样100个
        
        # 测试使用全部样本（默认行为）
        samples_all = load_data_samples(
            large_data_file,
            category="porn",
            samples_per_category=None
        )
        self.assertEqual(len(samples_all), 150)  # 应该使用全部150个
    
    def test_save_lexicon_file(self):
        """测试保存词表文件"""
        words = ["word1", "word2", "word3"]
        output_path = os.path.join(self.temp_dir, "test_lexicon.txt")
        
        save_lexicon_file(words, output_path, "test")
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        self.assertEqual(lines, words)
    
    def test_save_regex_file(self):
        """测试保存正则规则文件"""
        patterns = [
            "pattern1|description1",
            "pattern2|description2"
        ]
        output_path = os.path.join(self.temp_dir, "test_regex.txt")
        
        save_regex_file(patterns, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        
        self.assertEqual(lines, patterns)
    
    def test_save_lexicon_file_creates_directory(self):
        """测试保存词表时自动创建目录"""
        words = ["word1"]
        output_path = os.path.join(self.temp_dir, "subdir", "test_lexicon.txt")
        
        save_lexicon_file(words, output_path, "test")
        
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.isdir(os.path.dirname(output_path)))
    
    def test_save_regex_file_creates_directory(self):
        """测试保存正则规则时自动创建目录"""
        patterns = ["pattern|description"]
        output_path = os.path.join(self.temp_dir, "subdir", "test_regex.txt")
        
        save_regex_file(patterns, output_path)
        
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.isdir(os.path.dirname(output_path)))


if __name__ == '__main__':
    unittest.main()

