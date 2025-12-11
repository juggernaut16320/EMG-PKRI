"""
test_data_cleaner.py - data_cleaner 模块的单元测试
"""

import os
import sys
import json
import tempfile
import pytest

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from data_cleaner import (
    replace_urls,
    replace_mentions,
    extract_retweet_content,
    clean_text,
    get_text_hash,
    deduplicate,
    clean_jsonl,
)


class TestReplaceUrls:
    """测试URL替换"""
    
    def test_replace_url_with_space(self):
        text = "访问 https://example.com 获取信息"
        result = replace_urls(text)
        assert "[URL]" in result
        assert "https://example.com" not in result
    
    def test_replace_url_with_chinese(self):
        text = "访问https://example.com这是中文"
        result = replace_urls(text)
        assert result == "访问[URL]这是中文"
    
    def test_replace_www_url(self):
        text = "网址www.example.com这是中文"
        result = replace_urls(text)
        assert result == "网址[URL]这是中文"
    
    def test_multiple_urls(self):
        text = "访问 https://example.com 和 www.test.com这是中文"
        result = replace_urls(text)
        assert result.count("[URL]") == 2


class TestReplaceMentions:
    """测试@提及替换"""
    
    def test_replace_mention_with_space(self):
        text = "感谢 @user123 的支持"
        result = replace_mentions(text)
        assert result == "感谢 [MENTION] 的支持"
    
    def test_replace_mention_with_chinese(self):
        text = "联系@admin查看"
        result = replace_mentions(text)
        assert "[MENTION]" in result
    
    def test_multiple_mentions(self):
        text = "@user1 和 @user2 都在"
        result = replace_mentions(text)
        assert result.count("[MENTION]") == 2


class TestExtractRetweet:
    """测试转发内容提取"""
    
    def test_extract_rt_mention(self):
        text = "RT @username: 这是转发内容"
        result = extract_retweet_content(text)
        assert "RT @" not in result
        assert "这是转发内容" in result
    
    def test_extract_chinese_retweet(self):
        text = "转发 @username: 这是转发内容"
        result = extract_retweet_content(text)
        assert "转发 @" not in result
        assert "这是转发内容" in result
    
    def test_no_retweet(self):
        text = "这是普通内容"
        result = extract_retweet_content(text)
        assert result == "这是普通内容"


class TestCleanText:
    """测试文本清洗"""
    
    def test_basic_cleaning(self):
        config = {
            'min_length': 5,
            'url_handling': 'replace',
            'mention_handling': 'replace',
            'retweet_handling': 'extract',
        }
        text = "访问 https://example.com 联系 @user"
        result = clean_text(text, config)
        assert result is not None
        assert "[URL]" in result
        assert "[MENTION]" in result
    
    def test_too_short(self):
        config = {'min_length': 10}
        text = "短文本"
        result = clean_text(text, config)
        assert result is None
    
    def test_empty_text(self):
        config = {}
        text = "   "
        result = clean_text(text, config)
        assert result is None
    
    def test_retweet_extraction(self):
        config = {
            'min_length': 5,
            'retweet_handling': 'extract',
        }
        text = "RT @user: 这是转发内容"
        result = clean_text(text, config)
        assert result is not None
        assert "RT @" not in result
        assert "这是转发内容" in result


class TestDeduplicate:
    """测试去重"""
    
    def test_exact_duplicate(self):
        items = [
            {'id': 's0', 'text': '这是测试'},
            {'id': 's1', 'text': '这是测试'},  # 完全重复
            {'id': 's2', 'text': '这是测试。'},  # 不同内容
        ]
        result = deduplicate(items)
        assert len(result) == 2
        assert result[0]['id'] == 's0'
        assert result[1]['id'] == 's2'
    
    def test_no_duplicate(self):
        items = [
            {'id': 's0', 'text': '文本1'},
            {'id': 's1', 'text': '文本2'},
        ]
        result = deduplicate(items)
        assert len(result) == 2


class TestCleanJsonl:
    """测试JSONL文件清洗"""
    
    def test_clean_jsonl_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "input.jsonl")
            output_file = os.path.join(tmpdir, "output.jsonl")
            
            # 创建测试数据
            with open(input_file, 'w', encoding='utf-8') as f:
                items = [
                    {'id': 's0', 'text': '访问 https://example.com 联系 @user'},
                    {'id': 's1', 'text': '短'},  # 太短
                    {'id': 's2', 'text': '访问 https://example.com 联系 @user'},  # 重复
                ]
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            config = {
                'min_length': 5,
                'url_handling': 'replace',
                'mention_handling': 'replace',
                'retweet_handling': 'extract',
                'deduplication': {'exact': True},
            }
            
            stats = clean_jsonl(input_file, output_file, config)
            
            assert stats['total'] == 3
            assert stats['filtered'] == 1  # 太短的被过滤
            assert stats['deduplicated'] == 1  # 重复的被去重
            assert stats['output'] == 1
            
            # 验证输出文件
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 1
                item = json.loads(lines[0])
                assert "[URL]" in item['text']
                assert "[MENTION]" in item['text']

