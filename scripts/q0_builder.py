"""
q0_builder.py - 构造规则版知识后验 q₀(c|z)

功能：
1. 基于词表匹配构造知识信号
2. 基于正则规则匹配（可选）
3. 输出 coarse 二分类概率：[p_non_sensitive, p_sensitive]
4. 对 train/dev/test/hardset 全部生成 q₀ 后验
"""

import os
import sys
import json
import logging
import argparse
import yaml
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入 pyahocorasick（高性能多模式匹配）
try:
    import ahocorasick
    HAS_AHOCORASICK = True
except ImportError:
    HAS_AHOCORASICK = False
    logger.warning("未安装 pyahocorasick，将使用较慢的字符串匹配方式。建议安装: pip install pyahocorasick")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ 核心函数 ============

def load_lexicon(lexicon_path: str) -> Set[str]:
    """
    加载词表文件
    
    Args:
        lexicon_path: 词表文件路径
    
    Returns:
        词表集合
    """
    if not os.path.exists(lexicon_path):
        logger.warning(f"词表文件不存在: {lexicon_path}")
        return set()
    
    words = set()
    try:
        with open(lexicon_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word:
                    words.add(word)
        logger.info(f"加载词表: {lexicon_path}，共 {len(words)} 个词")
    except Exception as e:
        logger.error(f"加载词表失败: {lexicon_path}, 错误: {e}")
        return set()
    
    return words


def load_regex_patterns(regex_path: str) -> List[Tuple[str, str]]:
    """
    加载正则规则文件
    
    Args:
        regex_path: 正则规则文件路径
    
    Returns:
        正则规则列表，每个元素为 (pattern, description)
    """
    if not os.path.exists(regex_path):
        logger.warning(f"正则规则文件不存在: {regex_path}")
        return []
    
    patterns = []
    try:
        with open(regex_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式：pattern|description
                if '|' in line:
                    parts = line.split('|', 1)
                    pattern = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else ""
                    if pattern:
                        patterns.append((pattern, description))
        logger.info(f"加载正则规则: {regex_path}，共 {len(patterns)} 个规则")
    except Exception as e:
        logger.error(f"加载正则规则失败: {regex_path}, 错误: {e}")
        return []
    
    return patterns


def match_lexicon(text: str, lexicons: Dict[str, Set[str]], 
                  automaton_cache: Optional[Dict] = None) -> Dict[str, List[str]]:
    """
    在文本中匹配词表（优化版本）
    
    Args:
        text: 文本内容
        lexicons: 词表字典，key 为类别名，value 为词表集合
        automaton_cache: 可选的自动机缓存（用于加速）
    
    Returns:
        匹配结果字典，key 为类别名，value 为匹配到的词列表
    """
    matches = {}
    text_lower = text.lower()
    
    # 如果使用 Aho-Corasick 且提供了缓存
    if HAS_AHOCORASICK and automaton_cache is not None:
        for category, automaton in automaton_cache.items():
            if category not in lexicons:
                continue
            
            category_matches = []
            # 使用 Aho-Corasick 自动机进行匹配（O(m + k)）
            for end_index, (original_word, word_lower) in automaton.iter(text_lower):
                category_matches.append(original_word)
            matches[category] = category_matches
    else:
        # 回退到简单的字符串匹配（O(n × m)）
        for category, words in lexicons.items():
            category_matches = []
            for word in words:
                # 精确匹配（不区分大小写）
                if word.lower() in text_lower:
                    category_matches.append(word)
            matches[category] = category_matches
    
    return matches


def build_automaton_cache(lexicons: Dict[str, Set[str]]) -> Optional[Dict]:
    """
    构建 Aho-Corasick 自动机缓存（用于加速词表匹配）
    
    Args:
        lexicons: 词表字典
    
    Returns:
        自动机缓存字典，如果 pyahocorasick 未安装则返回 None
    """
    if not HAS_AHOCORASICK:
        return None
    
    automaton_cache = {}
    
    for category, words in lexicons.items():
        automaton = ahocorasick.Automaton()
        
        for word in words:
            word_lower = word.lower()
            # 存储 (原始词, 小写词) 的元组
            automaton.add_word(word_lower, (word, word_lower))
        
        automaton.make_automaton()
        automaton_cache[category] = automaton
    
    logger.info(f"✓ 构建 Aho-Corasick 自动机缓存，共 {len(automaton_cache)} 个类别")
    return automaton_cache


def match_regex(text: str, patterns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    在文本中匹配正则规则
    
    Args:
        text: 文本内容
        patterns: 正则规则列表，每个元素为 (pattern, description)
    
    Returns:
        匹配结果列表，每个元素为 (pattern, description)
    """
    matches = []
    
    for pattern, description in patterns:
        try:
            # 编译正则表达式
            regex = re.compile(pattern, re.IGNORECASE)
            if regex.search(text):
                matches.append((pattern, description))
        except re.error as e:
            logger.warning(f"正则表达式编译失败: {pattern}, 错误: {e}")
            continue
    
    return matches


def compute_q0(
    text: str,
    lexicons: Dict[str, Set[str]],
    regex_patterns: List[Tuple[str, str]],
    use_regex: bool = False,
    porn_weight: float = 1.0,
    politics_weight: float = 0.8,
    abuse_weight: float = 0.6,
    base_sensitive_prob: float = 0.1,
    max_sensitive_prob: float = 0.95,
    min_matches_for_sensitive: int = 1,
    include_details: bool = True
) -> Tuple[List[float], Optional[Dict]]:
    """
    计算知识后验 q₀
    
    Args:
        text: 文本内容
        lexicons: 词表字典
        regex_patterns: 正则规则列表
        use_regex: 是否使用正则规则
        porn_weight: 色情类别权重
        politics_weight: 涉政类别权重
        abuse_weight: 辱骂类别权重
        base_sensitive_prob: 基础敏感概率（无匹配时）
        max_sensitive_prob: 最大敏感概率
        min_matches_for_sensitive: 触发敏感的最小匹配数
        include_details: 是否包含匹配详情
    
    Returns:
        (q0概率, 匹配详情)
        q0格式: [p_non_sensitive, p_sensitive]
    """
    # 匹配词表
    lexicon_matches = match_lexicon(text, lexicons, automaton_cache=automaton_cache)
    
    # 匹配正则规则
    regex_matches = []
    if use_regex:
        regex_matches = match_regex(text, regex_patterns)
    
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
        # 使用 sigmoid 函数平滑概率
        # sigmoid(x) = 1 / (1 + exp(-x))
        # 调整参数使 match_score 在合理范围内映射到 [base_sensitive_prob, max_sensitive_prob]
        # 使用 tanh 函数，然后映射到目标范围
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


def process_dataset(
    input_path: str,
    output_path: str,
    lexicons: Dict[str, Set[str]],
    regex_patterns: List[Tuple[str, str]],
    use_regex: bool = False,
    porn_weight: float = 1.0,
    politics_weight: float = 0.8,
    abuse_weight: float = 0.6,
    base_sensitive_prob: float = 0.1,
    max_sensitive_prob: float = 0.95,
    min_matches_for_sensitive: int = 1,
    include_details: bool = True,
    automaton_cache: Optional[Dict] = None
) -> Dict:
    """
    处理单个数据集，生成 q₀ 后验
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        lexicons: 词表字典
        regex_patterns: 正则规则列表
        use_regex: 是否使用正则规则
        porn_weight: 色情类别权重
        politics_weight: 涉政类别权重
        abuse_weight: 辱骂类别权重
        base_sensitive_prob: 基础敏感概率
        max_sensitive_prob: 最大敏感概率
        min_matches_for_sensitive: 触发敏感的最小匹配数
        include_details: 是否包含匹配详情
    
    Returns:
        统计信息字典
    """
    logger.info(f"处理数据集: {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        return {}
    
    # 读取输入数据
    items = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"跳过无效JSON行: {e}")
                    continue
    
    logger.info(f"读取 {len(items)} 条数据")
    
    # 处理每条数据
    stats = {
        "total": len(items),
        "processed": 0,
        "avg_p_sensitive": 0.0,
        "sensitive_samples": 0,
        "non_sensitive_samples": 0
    }
    
    total_p_sensitive = 0.0
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in items:
            text = item.get('text', '')
            if not text:
                logger.warning(f"跳过空文本: {item.get('id', 'unknown')}")
                continue
            
            # 计算 q₀
            q0, details = compute_q0(
                text,
                lexicons,
                regex_patterns,
                use_regex=use_regex,
                porn_weight=porn_weight,
                politics_weight=politics_weight,
                abuse_weight=abuse_weight,
                base_sensitive_prob=base_sensitive_prob,
                max_sensitive_prob=max_sensitive_prob,
                min_matches_for_sensitive=min_matches_for_sensitive,
                include_details=include_details,
                automaton_cache=automaton_cache
            )
            
            # 添加 q₀ 到输出
            item['q0'] = q0
            if details:
                item['q0_details'] = details
            
            # 写入输出
            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 更新统计
            p_sensitive = q0[1]
            total_p_sensitive += p_sensitive
            if p_sensitive > 0.5:
                stats["sensitive_samples"] += 1
            else:
                stats["non_sensitive_samples"] += 1
            stats["processed"] += 1
    
    # 计算平均概率
    if stats["processed"] > 0:
        stats["avg_p_sensitive"] = total_p_sensitive / stats["processed"]
    
    logger.info(f"✓ 处理完成: {output_path}")
    logger.info(f"  总样本: {stats['total']}, 处理: {stats['processed']}")
    logger.info(f"  平均敏感概率: {stats['avg_p_sensitive']:.4f}")
    logger.info(f"  敏感样本: {stats['sensitive_samples']}, 非敏感样本: {stats['non_sensitive_samples']}")
    
    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构造规则版知识后验 q₀(c|z)')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据目录（默认从config读取）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认与data-dir相同）'
    )
    parser.add_argument(
        '--lexicon-dir',
        type=str,
        default=None,
        help='词表目录（默认从config读取）'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        default=None,
        help='训练集文件（默认: train.jsonl）'
    )
    parser.add_argument(
        '--dev-file',
        type=str,
        default=None,
        help='验证集文件（默认: dev.jsonl）'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default=None,
        help='测试集文件（默认: test.jsonl）'
    )
    parser.add_argument(
        '--hard-file',
        type=str,
        default=None,
        help='困难子集文件（默认: hard_eval_set.jsonl）'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['train', 'dev', 'test', 'hard'],
        help='要处理的数据集列表（默认: train dev test hard）'
    )
    parser.add_argument(
        '--use-regex',
        action='store_true',
        help='是否使用正则规则'
    )
    parser.add_argument(
        '--porn-weight',
        type=float,
        default=None,
        help='色情类别权重（默认从config读取）'
    )
    parser.add_argument(
        '--politics-weight',
        type=float,
        default=None,
        help='涉政类别权重（默认从config读取）'
    )
    parser.add_argument(
        '--abuse-weight',
        type=float,
        default=None,
        help='辱骂类别权重（默认从config读取）'
    )
    parser.add_argument(
        '--base-sensitive-prob',
        type=float,
        default=None,
        help='基础敏感概率（默认从config读取）'
    )
    parser.add_argument(
        '--max-sensitive-prob',
        type=float,
        default=None,
        help='最大敏感概率（默认从config读取）'
    )
    parser.add_argument(
        '--min-matches',
        type=int,
        default=None,
        help='触发敏感的最小匹配数（默认从config读取）'
    )
    parser.add_argument(
        '--no-details',
        action='store_true',
        help='不包含匹配详情'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        config = {}
    
    # 获取参数
    data_dir = args.data_dir or config.get('data_dir', './data')
    output_dir = args.output_dir or data_dir
    lexicon_dir = args.lexicon_dir or config.get('q0', {}).get('lexicon_dir', 'configs/lexicons')
    
    # q0 配置
    q0_config = config.get('q0', {})
    use_regex = args.use_regex or q0_config.get('use_regex', False)
    porn_weight = args.porn_weight if args.porn_weight is not None else q0_config.get('porn_weight', 1.0)
    politics_weight = args.politics_weight if args.politics_weight is not None else q0_config.get('politics_weight', 0.8)
    abuse_weight = args.abuse_weight if args.abuse_weight is not None else q0_config.get('abuse_weight', 0.6)
    base_sensitive_prob = args.base_sensitive_prob if args.base_sensitive_prob is not None else q0_config.get('base_sensitive_prob', 0.1)
    max_sensitive_prob = args.max_sensitive_prob if args.max_sensitive_prob is not None else q0_config.get('max_sensitive_prob', 0.95)
    min_matches_for_sensitive = args.min_matches if args.min_matches is not None else q0_config.get('min_matches_for_sensitive', 1)
    include_details = not args.no_details and q0_config.get('include_details', True)
    
    logger.info("=" * 60)
    logger.info("开始构建知识后验 q₀")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"词表目录: {lexicon_dir}")
    logger.info(f"使用正则规则: {use_regex}")
    logger.info(f"权重配置: porn={porn_weight}, politics={politics_weight}, abuse={abuse_weight}")
    logger.info(f"概率配置: base={base_sensitive_prob}, max={max_sensitive_prob}, min_matches={min_matches_for_sensitive}")
    logger.info("")
    
    # 加载词表
    logger.info("加载词表...")
    lexicons = {}
    for category in ['porn', 'politics', 'abuse']:
        lexicon_path = os.path.join(lexicon_dir, f"{category}.txt")
        words = load_lexicon(lexicon_path)
        if words:
            lexicons[category] = words
    
    if not lexicons:
        logger.error("没有找到任何词表文件！")
        return 1
    
    total_words = sum(len(words) for words in lexicons.values())
    logger.info(f"✓ 共加载 {len(lexicons)} 个词表，总计 {total_words} 个词")
    
    # 构建 Aho-Corasick 自动机缓存（用于加速匹配）
    logger.info("构建自动机缓存（用于加速词表匹配）...")
    automaton_cache = build_automaton_cache(lexicons)
    if automaton_cache:
        logger.info(f"✓ 自动机缓存构建完成，将使用高性能匹配算法")
    else:
        logger.info("⚠ 将使用较慢的字符串匹配方式（建议安装 pyahocorasick: pip install pyahocorasick）")
    logger.info("")
    
    # 加载正则规则
    regex_patterns = []
    if use_regex:
        regex_path = os.path.join(lexicon_dir, "regex_patterns.txt")
        regex_patterns = load_regex_patterns(regex_path)
        if regex_patterns:
            logger.info(f"✓ 加载 {len(regex_patterns)} 个正则规则")
        logger.info("")
    
    # 处理数据集
    dataset_files = {
        'train': args.train_file or 'train.jsonl',
        'dev': args.dev_file or 'dev.jsonl',
        'test': args.test_file or 'test.jsonl',
        'hard': args.hard_file or 'hard_eval_set.jsonl'
    }
    
    all_stats = {}
    
    for dataset_name in args.datasets:
        if dataset_name not in dataset_files:
            logger.warning(f"未知的数据集: {dataset_name}，跳过")
            continue
        
        input_file = dataset_files[dataset_name]
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(output_dir, f"q0_{dataset_name}.jsonl")
        
        logger.info("-" * 60)
        stats = process_dataset(
            input_path,
            output_path,
            lexicons,
            regex_patterns,
            use_regex=use_regex,
            porn_weight=porn_weight,
            politics_weight=politics_weight,
            abuse_weight=abuse_weight,
            base_sensitive_prob=base_sensitive_prob,
            max_sensitive_prob=max_sensitive_prob,
            min_matches_for_sensitive=min_matches_for_sensitive,
            include_details=include_details,
            automaton_cache=automaton_cache
        )
        all_stats[dataset_name] = stats
        logger.info("")
    
    # 总结
    logger.info("=" * 60)
    logger.info("构建完成")
    logger.info("=" * 60)
    logger.info("统计信息:")
    for dataset_name, stats in all_stats.items():
        logger.info(f"  {dataset_name}:")
        logger.info(f"    总样本: {stats.get('total', 0)}")
        logger.info(f"    处理: {stats.get('processed', 0)}")
        logger.info(f"    平均敏感概率: {stats.get('avg_p_sensitive', 0.0):.4f}")
        logger.info(f"    敏感样本: {stats.get('sensitive_samples', 0)}")
        logger.info(f"    非敏感样本: {stats.get('non_sensitive_samples', 0)}")
    
    logger.info("")
    logger.info("输出文件:")
    for dataset_name in args.datasets:
        if dataset_name in dataset_files:
            output_path = os.path.join(output_dir, f"q0_{dataset_name}.jsonl")
            logger.info(f"  - {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

