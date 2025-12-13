"""
test_lexicon_generator.py - lexicon_generator.py 完整单元测试

包含需要 LLM API 的测试（可选，如果环境不支持则跳过）
"""

import unittest
import os
import sys

# 检查依赖
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# 检查环境变量
HAS_API_KEY = bool(os.getenv("GEMINI_API_KEY"))

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestLexiconGeneratorFull(unittest.TestCase):
    """测试 lexicon_generator 的完整功能（需要 LLM API）"""
    
    @unittest.skipIf(not HAS_GENAI, "需要 google-genai 库")
    @unittest.skipIf(not HAS_API_KEY, "需要 GEMINI_API_KEY 环境变量")
    def test_generate_lexicon_from_samples(self):
        """测试使用大模型生成词表（需要 API）"""
        from lexicon_generator import generate_lexicon_from_samples
        
        samples = [
            {"text": "这是一个色情内容样本"},
            {"text": "这是另一个色情内容样本"},
        ]
        
        words = generate_lexicon_from_samples(samples, "porn")
        
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
        # 检查去重
        self.assertEqual(len(words), len(set(words)))
    
    @unittest.skipIf(not HAS_GENAI, "需要 google-genai 库")
    @unittest.skipIf(not HAS_API_KEY, "需要 GEMINI_API_KEY 环境变量")
    def test_generate_regex_patterns(self):
        """测试使用大模型生成正则规则（需要 API）"""
        from lexicon_generator import generate_regex_patterns
        
        words = ["word1", "word2", "word3", "word4", "word5"]
        
        patterns = generate_regex_patterns(words, "porn")
        
        self.assertIsInstance(patterns, list)
        # 检查格式（应该包含 | 分隔符）
        for pattern in patterns:
            self.assertIn('|', pattern)


if __name__ == '__main__':
    # 如果缺少依赖，只运行逻辑测试
    if not HAS_GENAI or not HAS_API_KEY:
        print("⚠ 跳过需要 LLM API 的测试（缺少依赖或 API Key）")
        print("  运行逻辑测试: python -m pytest tests/test_lexicon_generator_logic.py")
    
    unittest.main()

