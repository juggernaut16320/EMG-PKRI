"""
test_q0_builder.py - q0_builder.py 完整功能测试

测试文件加载、数据集处理等完整功能
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path

# 添加 scripts 目录到路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from q0_builder import (
    load_lexicon,
    load_regex_patterns,
    process_dataset
)


class TestQ0Builder:
    """测试 q0_builder 完整功能"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_lexicons(self, temp_dir):
        """创建示例词表文件"""
        lexicon_dir = os.path.join(temp_dir, 'lexicons')
        os.makedirs(lexicon_dir, exist_ok=True)
        
        # 创建 porn.txt
        with open(os.path.join(lexicon_dir, 'porn.txt'), 'w', encoding='utf-8') as f:
            f.write('色情\n')
            f.write('av\n')
            f.write('xxx\n')
        
        # 创建 politics.txt
        with open(os.path.join(lexicon_dir, 'politics.txt'), 'w', encoding='utf-8') as f:
            f.write('政治\n')
            f.write('涉政\n')
        
        # 创建 abuse.txt
        with open(os.path.join(lexicon_dir, 'abuse.txt'), 'w', encoding='utf-8') as f:
            f.write('辱骂\n')
            f.write('攻击\n')
        
        return lexicon_dir
    
    @pytest.fixture
    def sample_regex_patterns(self, temp_dir):
        """创建示例正则规则文件"""
        lexicon_dir = os.path.join(temp_dir, 'lexicons')
        os.makedirs(lexicon_dir, exist_ok=True)
        
        with open(os.path.join(lexicon_dir, 'regex_patterns.txt'), 'w', encoding='utf-8') as f:
            f.write('cao|草|操|匹配拼音变体\n')
            f.write('f[u*]ck|匹配谐音变体\n')
        
        return lexicon_dir
    
    @pytest.fixture
    def sample_data(self, temp_dir):
        """创建示例数据文件"""
        data_dir = os.path.join(temp_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        data_file = os.path.join(data_dir, 'test.jsonl')
        with open(data_file, 'w', encoding='utf-8') as f:
            # 包含敏感词的样本
            f.write(json.dumps({
                'id': 's1',
                'text': '这是一段包含色情和av的内容',
                'coarse_label': 1
            }, ensure_ascii=False) + '\n')
            
            # 不包含敏感词的样本
            f.write(json.dumps({
                'id': 's2',
                'text': '这是一段正常的内容',
                'coarse_label': 0
            }, ensure_ascii=False) + '\n')
            
            # 包含多个类别敏感词的样本
            f.write(json.dumps({
                'id': 's3',
                'text': '这是一段包含色情、政治和辱骂的内容',
                'coarse_label': 1
            }, ensure_ascii=False) + '\n')
        
        return data_file
    
    def test_load_lexicon(self, sample_lexicons):
        """测试加载词表"""
        lexicon_path = os.path.join(sample_lexicons, 'porn.txt')
        words = load_lexicon(lexicon_path)
        
        assert len(words) == 3
        assert '色情' in words
        assert 'av' in words
        assert 'xxx' in words
    
    def test_load_lexicon_nonexistent(self, temp_dir):
        """测试加载不存在的词表"""
        lexicon_path = os.path.join(temp_dir, 'nonexistent.txt')
        words = load_lexicon(lexicon_path)
        
        assert len(words) == 0
    
    def test_load_lexicon_empty(self, temp_dir):
        """测试加载空词表"""
        lexicon_dir = os.path.join(temp_dir, 'lexicons')
        os.makedirs(lexicon_dir, exist_ok=True)
        
        lexicon_path = os.path.join(lexicon_dir, 'empty.txt')
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            f.write('')
        
        words = load_lexicon(lexicon_path)
        assert len(words) == 0
    
    def test_load_lexicon_with_blank_lines(self, temp_dir):
        """测试加载包含空行的词表"""
        lexicon_dir = os.path.join(temp_dir, 'lexicons')
        os.makedirs(lexicon_dir, exist_ok=True)
        
        lexicon_path = os.path.join(lexicon_dir, 'test.txt')
        with open(lexicon_path, 'w', encoding='utf-8') as f:
            f.write('word1\n')
            f.write('\n')
            f.write('word2\n')
            f.write('  \n')  # 只有空格
            f.write('word3\n')
        
        words = load_lexicon(lexicon_path)
        assert len(words) == 3
        assert 'word1' in words
        assert 'word2' in words
        assert 'word3' in words
    
    def test_load_regex_patterns(self, sample_regex_patterns):
        """测试加载正则规则"""
        regex_path = os.path.join(sample_regex_patterns, 'regex_patterns.txt')
        patterns = load_regex_patterns(regex_path)
        
        assert len(patterns) == 2
        # 注意：split('|', 1) 会将 'cao|草|操|匹配拼音变体' 分割为 pattern='cao' 和 description='草|操|匹配拼音变体'
        # 这是当前实现的逻辑，如果需要支持 pattern 中包含 |，需要使用不同的分隔符
        assert patterns[0][0] == 'cao'
        assert '匹配拼音变体' in patterns[0][1] or patterns[0][1] == '草|操|匹配拼音变体'
        assert patterns[1][0] == 'f[u*]ck'
        assert patterns[1][1] == '匹配谐音变体'
    
    def test_load_regex_patterns_nonexistent(self, temp_dir):
        """测试加载不存在的正则规则文件"""
        regex_path = os.path.join(temp_dir, 'nonexistent.txt')
        patterns = load_regex_patterns(regex_path)
        
        assert len(patterns) == 0
    
    def test_load_regex_patterns_invalid_format(self, temp_dir):
        """测试加载格式不正确的正则规则"""
        lexicon_dir = os.path.join(temp_dir, 'lexicons')
        os.makedirs(lexicon_dir, exist_ok=True)
        
        regex_path = os.path.join(lexicon_dir, 'regex_patterns.txt')
        with open(regex_path, 'w', encoding='utf-8') as f:
            f.write('pattern_without_description\n')  # 没有分隔符
            f.write('pattern|description\n')  # 正确格式
            f.write('|only_description\n')  # 只有描述
        
        patterns = load_regex_patterns(regex_path)
        
        # 应该只加载有分隔符且pattern不为空的
        assert len(patterns) == 1
        assert patterns[0][0] == 'pattern'
        assert patterns[0][1] == 'description'
    
    def test_process_dataset(self, sample_data, sample_lexicons, temp_dir):
        """测试处理数据集"""
        lexicons = {
            'porn': load_lexicon(os.path.join(sample_lexicons, 'porn.txt')),
            'politics': load_lexicon(os.path.join(sample_lexicons, 'politics.txt')),
            'abuse': load_lexicon(os.path.join(sample_lexicons, 'abuse.txt'))
        }
        
        output_path = os.path.join(temp_dir, 'output.jsonl')
        stats = process_dataset(
            sample_data,
            output_path,
            lexicons,
            [],
            use_regex=False,
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95,
            min_matches_for_sensitive=1
        )
        
        # 验证统计信息
        assert stats['total'] == 3
        assert stats['processed'] == 3
        assert stats['avg_p_sensitive'] > 0.1  # 至少有一个样本匹配
        
        # 验证输出文件
        assert os.path.exists(output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        assert len(items) == 3
        
        # 验证每个项目都有 q0 字段
        for item in items:
            assert 'q0' in item
            assert len(item['q0']) == 2
            assert 0.0 <= item['q0'][0] <= 1.0
            assert 0.0 <= item['q0'][1] <= 1.0
            assert abs(item['q0'][0] + item['q0'][1] - 1.0) < 1e-6
        
        # 验证第一个样本（包含敏感词）的敏感概率较高
        assert items[0]['q0'][1] > items[1]['q0'][1]  # s1 比 s2 更敏感
    
    def test_process_dataset_with_regex(self, sample_data, sample_lexicons, sample_regex_patterns, temp_dir):
        """测试使用正则规则处理数据集"""
        lexicons = {
            'porn': load_lexicon(os.path.join(sample_lexicons, 'porn.txt')),
            'politics': load_lexicon(os.path.join(sample_lexicons, 'politics.txt')),
            'abuse': load_lexicon(os.path.join(sample_lexicons, 'abuse.txt'))
        }
        
        regex_path = os.path.join(sample_regex_patterns, 'regex_patterns.txt')
        regex_patterns = load_regex_patterns(regex_path)
        
        # 创建一个包含正则匹配的样本
        data_file = os.path.join(temp_dir, 'data', 'test_regex.jsonl')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'id': 's1',
                'text': '这是一段包含cao的内容',
                'coarse_label': 1
            }, ensure_ascii=False) + '\n')
        
        output_path = os.path.join(temp_dir, 'output_regex.jsonl')
        stats = process_dataset(
            data_file,
            output_path,
            lexicons,
            regex_patterns,
            use_regex=True,
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95,
            min_matches_for_sensitive=1,
            include_details=True
        )
        
        # 验证输出
        assert stats['processed'] == 1
        
        with open(output_path, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        assert len(items) == 1
        assert 'q0_details' in items[0]
        assert len(items[0]['q0_details']['regex_matches']) > 0
    
    def test_process_dataset_empty_input(self, temp_dir, sample_lexicons):
        """测试处理空输入文件"""
        lexicons = {
            'porn': load_lexicon(os.path.join(sample_lexicons, 'porn.txt')),
            'politics': set(),
            'abuse': set()
        }
        
        # 创建空文件
        input_path = os.path.join(temp_dir, 'empty.jsonl')
        with open(input_path, 'w', encoding='utf-8') as f:
            pass
        
        output_path = os.path.join(temp_dir, 'output_empty.jsonl')
        stats = process_dataset(
            input_path,
            output_path,
            lexicons,
            [],
            base_sensitive_prob=0.1
        )
        
        assert stats['total'] == 0
        assert stats['processed'] == 0
    
    def test_process_dataset_with_empty_text(self, temp_dir, sample_lexicons):
        """测试处理包含空文本的样本"""
        lexicons = {
            'porn': load_lexicon(os.path.join(sample_lexicons, 'porn.txt')),
            'politics': set(),
            'abuse': set()
        }
        
        data_file = os.path.join(temp_dir, 'data', 'test_empty.jsonl')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'id': 's1',
                'text': '正常文本',
                'coarse_label': 0
            }, ensure_ascii=False) + '\n')
            f.write(json.dumps({
                'id': 's2',
                'text': '',  # 空文本
                'coarse_label': 0
            }, ensure_ascii=False) + '\n')
        
        output_path = os.path.join(temp_dir, 'output_empty.jsonl')
        stats = process_dataset(
            data_file,
            output_path,
            lexicons,
            [],
            base_sensitive_prob=0.1
        )
        
        # 应该只处理了1个样本（空文本被跳过）
        assert stats['processed'] == 1
    
    def test_process_dataset_include_details(self, sample_data, sample_lexicons, temp_dir):
        """测试包含匹配详情"""
        lexicons = {
            'porn': load_lexicon(os.path.join(sample_lexicons, 'porn.txt')),
            'politics': load_lexicon(os.path.join(sample_lexicons, 'politics.txt')),
            'abuse': load_lexicon(os.path.join(sample_lexicons, 'abuse.txt'))
        }
        
        output_path = os.path.join(temp_dir, 'output_details.jsonl')
        stats = process_dataset(
            sample_data,
            output_path,
            lexicons,
            [],
            include_details=True
        )
        
        with open(output_path, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        # 验证所有项目都有详情
        for item in items:
            assert 'q0_details' in item
            assert 'porn_matches' in item['q0_details']
            assert 'politics_matches' in item['q0_details']
            assert 'abuse_matches' in item['q0_details']
            assert 'total_matches' in item['q0_details']
    
    def test_process_dataset_no_details(self, sample_data, sample_lexicons, temp_dir):
        """测试不包含匹配详情"""
        lexicons = {
            'porn': load_lexicon(os.path.join(sample_lexicons, 'porn.txt')),
            'politics': load_lexicon(os.path.join(sample_lexicons, 'politics.txt')),
            'abuse': load_lexicon(os.path.join(sample_lexicons, 'abuse.txt'))
        }
        
        output_path = os.path.join(temp_dir, 'output_no_details.jsonl')
        stats = process_dataset(
            sample_data,
            output_path,
            lexicons,
            [],
            include_details=False
        )
        
        with open(output_path, 'r', encoding='utf-8') as f:
            items = [json.loads(line) for line in f if line.strip()]
        
        # 验证所有项目都没有详情
        for item in items:
            assert 'q0_details' not in item
            assert 'q0' in item  # 但 q0 应该存在


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

