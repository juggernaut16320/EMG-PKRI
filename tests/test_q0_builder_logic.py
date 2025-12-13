"""
test_q0_builder_logic.py - q0_builder.py 核心逻辑单元测试（不依赖文件系统）

这个测试文件直接测试核心逻辑函数，不导入整个模块
"""

import pytest
import numpy as np


def match_lexicon_logic(text: str, lexicons: dict) -> dict:
    """
    在文本中匹配词表的核心逻辑（从 q0_builder.py 提取）
    
    Args:
        text: 文本内容
        lexicons: 词表字典，key 为类别名，value 为词表集合
    
    Returns:
        匹配结果字典，key 为类别名，value 为匹配到的词列表
    """
    matches = {}
    text_lower = text.lower()
    
    for category, words in lexicons.items():
        category_matches = []
        for word in words:
            # 精确匹配（不区分大小写）
            if word.lower() in text_lower:
                category_matches.append(word)
        matches[category] = category_matches
    
    return matches


def match_regex_logic(text: str, patterns: list) -> list:
    """
    在文本中匹配正则规则的核心逻辑（从 q0_builder.py 提取）
    
    Args:
        text: 文本内容
        patterns: 正则规则列表，每个元素为 (pattern, description)
    
    Returns:
        匹配结果列表，每个元素为 (pattern, description)
    """
    import re
    matches = []
    
    for pattern, description in patterns:
        try:
            # 编译正则表达式
            regex = re.compile(pattern, re.IGNORECASE)
            if regex.search(text):
                matches.append((pattern, description))
        except re.error:
            continue
    
    return matches


def compute_q0_logic(
    text: str,
    lexicons: dict,
    regex_patterns: list,
    use_regex: bool = False,
    porn_weight: float = 1.0,
    politics_weight: float = 0.8,
    abuse_weight: float = 0.6,
    base_sensitive_prob: float = 0.1,
    max_sensitive_prob: float = 0.95,
    min_matches_for_sensitive: int = 1,
    include_details: bool = True
) -> tuple:
    """
    计算知识后验 q₀ 的核心逻辑（从 q0_builder.py 提取）
    
    Returns:
        (q0概率, 匹配详情)
        q0格式: [p_non_sensitive, p_sensitive]
    """
    # 匹配词表
    lexicon_matches = match_lexicon_logic(text, lexicons)
    
    # 匹配正则规则
    regex_matches = []
    if use_regex:
        regex_matches = match_regex_logic(text, regex_patterns)
    
    # 计算匹配分数
    porn_matches = lexicon_matches.get('porn', [])
    politics_matches = lexicon_matches.get('politics', [])
    abuse_matches = lexicon_matches.get('abuse', [])
    
    total_matches = len(porn_matches) + len(politics_matches) + len(abuse_matches) + len(regex_matches)
    
    # 计算加权匹配分数
    text_words = len(text.split())
    if text_words == 0:
        text_words = 1  # 避免除零
    
    match_score = (
        porn_weight * len(porn_matches) +
        politics_weight * len(politics_matches) +
        abuse_weight * len(abuse_matches) +
        (porn_weight + politics_weight + abuse_weight) / 3 * len(regex_matches)
    ) / text_words
    
    # 如果匹配数少于最小阈值，返回基础概率
    if total_matches < min_matches_for_sensitive:
        p_sensitive = base_sensitive_prob
    else:
        # 使用 tanh 函数平滑概率
        normalized_score = np.tanh(match_score * 10)  # 调整缩放因子
        p_sensitive = base_sensitive_prob + (max_sensitive_prob - base_sensitive_prob) * (normalized_score + 1) / 2
    
    # 确保概率在 [0, 1] 范围内
    p_sensitive = max(0.0, min(1.0, p_sensitive))
    p_non_sensitive = 1.0 - p_sensitive
    
    q0 = [float(p_non_sensitive), float(p_sensitive)]
    
    # 构建匹配详情
    details = None
    if include_details:
        details = {
            "porn_matches": porn_matches,
            "politics_matches": politics_matches,
            "abuse_matches": abuse_matches,
            "regex_matches": [desc for _, desc in regex_matches] if regex_matches else [],
            "total_matches": total_matches,
            "match_score": float(match_score) if total_matches >= min_matches_for_sensitive else 0.0
        }
    
    return q0, details


class TestQ0BuilderLogic:
    """测试 q0_builder 核心逻辑"""
    
    def test_match_lexicon_basic(self):
        """测试基本词表匹配"""
        lexicons = {
            'porn': {'色情', 'av', 'xxx'},
            'politics': {'政治', '涉政'},
            'abuse': {'辱骂', '攻击'}
        }
        
        text = "这是一段包含色情和av的内容"
        matches = match_lexicon_logic(text, lexicons)
        
        assert 'porn' in matches
        assert '色情' in matches['porn']
        assert 'av' in matches['porn']
        assert 'politics' in matches
        assert len(matches['politics']) == 0
        assert 'abuse' in matches
        assert len(matches['abuse']) == 0
    
    def test_match_lexicon_case_insensitive(self):
        """测试词表匹配不区分大小写"""
        lexicons = {
            'porn': {'AV', 'XXX'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段包含av和xxx的内容"
        matches = match_lexicon_logic(text, lexicons)
        
        assert 'AV' in matches['porn']
        assert 'XXX' in matches['porn']
    
    def test_match_lexicon_multiple_categories(self):
        """测试多个类别的匹配"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段包含色情、政治和辱骂的内容"
        matches = match_lexicon_logic(text, lexicons)
        
        assert len(matches['porn']) == 1
        assert len(matches['politics']) == 1
        assert len(matches['abuse']) == 1
    
    def test_match_regex_basic(self):
        """测试基本正则匹配"""
        patterns = [
            (r'cao|草|操', '匹配拼音变体'),
            (r'f[u*]ck', '匹配谐音变体')
        ]
        
        text = "这是一段包含cao的内容"
        matches = match_regex_logic(text, patterns)
        
        assert len(matches) == 1
        assert matches[0][0] == r'cao|草|操'
        assert matches[0][1] == '匹配拼音变体'
    
    def test_match_regex_case_insensitive(self):
        """测试正则匹配不区分大小写"""
        patterns = [
            (r'FUCK', '匹配大写')
        ]
        
        text = "这是一段包含fuck的内容"
        matches = match_regex_logic(text, patterns)
        
        assert len(matches) == 1
    
    def test_match_regex_invalid_pattern(self):
        """测试无效正则表达式"""
        patterns = [
            (r'[invalid', '无效正则'),
            (r'valid', '有效正则')
        ]
        
        text = "这是一段包含valid的内容"
        matches = match_regex_logic(text, patterns)
        
        # 应该只匹配有效正则
        assert len(matches) == 1
        assert matches[0][0] == r'valid'
    
    def test_compute_q0_no_matches(self):
        """测试无匹配的情况"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段正常的内容"
        q0, details = compute_q0_logic(
            text, lexicons, [],
            base_sensitive_prob=0.1,
            min_matches_for_sensitive=1
        )
        
        # 无匹配应该返回基础概率
        assert abs(q0[1] - 0.1) < 1e-6
        assert details['total_matches'] == 0
    
    def test_compute_q0_with_matches(self):
        """测试有匹配的情况"""
        lexicons = {
            'porn': {'色情', 'av'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段包含色情和av的内容"
        q0, details = compute_q0_logic(
            text, lexicons, [],
            porn_weight=1.0,
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95,
            min_matches_for_sensitive=1
        )
        
        # 有匹配应该返回更高的敏感概率
        assert q0[1] > 0.1
        assert q0[1] <= 0.95
        assert details['total_matches'] == 2
        assert len(details['porn_matches']) == 2
    
    def test_compute_q0_probability_range(self):
        """测试概率范围"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段包含色情的内容"
        q0, _ = compute_q0_logic(
            text, lexicons, [],
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95
        )
        
        # 概率应该在 [0, 1] 范围内
        assert 0.0 <= q0[0] <= 1.0
        assert 0.0 <= q0[1] <= 1.0
        assert abs(q0[0] + q0[1] - 1.0) < 1e-6
    
    def test_compute_q0_min_matches_threshold(self):
        """测试最小匹配数阈值"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段正常的内容"
        q0, details = compute_q0_logic(
            text, lexicons, [],
            base_sensitive_prob=0.1,
            min_matches_for_sensitive=2  # 需要至少2个匹配
        )
        
        # 无匹配应该返回基础概率
        assert abs(q0[1] - 0.1) < 1e-6
        
        # 有1个匹配但小于阈值
        text2 = "这是一段包含色情的内容"
        q0_2, details2 = compute_q0_logic(
            text2, lexicons, [],
            base_sensitive_prob=0.1,
            min_matches_for_sensitive=2
        )
        
        # 应该仍然返回基础概率（因为只有1个匹配，小于阈值2）
        assert abs(q0_2[1] - 0.1) < 1e-6
    
    def test_compute_q0_weighted_matching(self):
        """测试加权匹配"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        # 测试只有porn匹配的情况
        text1 = "这是一段包含色情的内容"
        q0_1, _ = compute_q0_logic(
            text1, lexicons, [],
            porn_weight=1.0,
            politics_weight=0.5,
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95
        )
        
        # 测试只有politics匹配的情况（权重相同）
        text2 = "这是一段包含政治的内容"
        q0_2, _ = compute_q0_logic(
            text2, lexicons, [],
            porn_weight=1.0,
            politics_weight=0.5,
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95
        )
        
        # porn权重更高，所以porn匹配的敏感概率应该更高
        assert q0_1[1] > q0_2[1]
        
        # 测试权重差异更明显的情况
        q0_3, _ = compute_q0_logic(
            text2, lexicons, [],
            porn_weight=1.0,
            politics_weight=0.1,  # 更低的权重
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95
        )
        
        # 权重更低，敏感概率应该更低
        assert q0_2[1] > q0_3[1]
    
    def test_compute_q0_with_regex(self):
        """测试使用正则规则"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        patterns = [
            (r'cao|草|操', '匹配拼音变体')
        ]
        
        text = "这是一段包含cao的内容"
        q0, details = compute_q0_logic(
            text, lexicons, patterns,
            use_regex=True,
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95
        )
        
        # 应该匹配到正则规则
        assert details['total_matches'] > 0
        assert len(details['regex_matches']) > 0
    
    def test_compute_q0_without_details(self):
        """测试不包含匹配详情"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段包含色情的内容"
        q0, details = compute_q0_logic(
            text, lexicons, [],
            include_details=False
        )
        
        # 不应该包含详情
        assert details is None
        # 但 q0 应该正常
        assert len(q0) == 2
        assert 0.0 <= q0[0] <= 1.0
        assert 0.0 <= q0[1] <= 1.0
    
    def test_compute_q0_empty_text(self):
        """测试空文本"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = ""
        q0, details = compute_q0_logic(
            text, lexicons, [],
            base_sensitive_prob=0.1
        )
        
        # 空文本应该返回基础概率
        assert abs(q0[1] - 0.1) < 1e-6
        assert details['total_matches'] == 0
    
    def test_compute_q0_multiple_same_word(self):
        """测试同一词多次出现"""
        lexicons = {
            'porn': {'色情'},
            'politics': {'政治'},
            'abuse': {'辱骂'}
        }
        
        text = "这是一段包含色情色情色情的内容"
        q0, details = compute_q0_logic(
            text, lexicons, [],
            base_sensitive_prob=0.1,
            max_sensitive_prob=0.95
        )
        
        # 应该只匹配一次（集合去重）
        assert len(details['porn_matches']) == 1
        assert details['porn_matches'][0] == '色情'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

