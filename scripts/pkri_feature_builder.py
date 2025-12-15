"""
PKRI 特征构建脚本（方案一：简化版）
构建知识边特征用于PKRI模型训练
"""
import json
import argparse
import sys
import os
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from q0_builder import (
    load_lexicon, 
    match_lexicon,
    build_automaton_cache
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_lexicon_features(
    text: str,
    lexicons: Dict[str, Set[str]],
    automaton_cache: Optional[Dict] = None
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    提取词表命中特征
    
    Args:
        text: 文本内容
        lexicons: 词表字典
        automaton_cache: 自动机缓存（可选）
    
    Returns:
        (特征字典, 匹配结果)
    """
    matches = match_lexicon(text, lexicons, automaton_cache)
    
    # 各类别匹配数
    porn_matches = len(matches.get('porn', []))
    politics_matches = len(matches.get('politics', []))
    abuse_matches = len(matches.get('abuse', []))
    total_matches = porn_matches + politics_matches + abuse_matches
    
    # 文本词数（简单分词）
    text_words = len(text.split())
    if text_words == 0:
        text_words = 1
    
    # 匹配比例
    match_ratio = total_matches / text_words if text_words > 0 else 0.0
    
    features = {
        'lexicon_match_porn': float(porn_matches),
        'lexicon_match_politics': float(politics_matches),
        'lexicon_match_abuse': float(abuse_matches),
        'total_matches': float(total_matches),
        'match_ratio': match_ratio,
        'has_porn_match': 1.0 if porn_matches > 0 else 0.0,
        'has_politics_match': 1.0 if politics_matches > 0 else 0.0,
        'has_abuse_match': 1.0 if abuse_matches > 0 else 0.0,
    }
    
    return features, matches


def extract_subtype_match_features(
    subtype_labels: List[str],
    matches: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    提取子标签匹配特征
    
    Args:
        subtype_labels: 子标签列表
        matches: 词表匹配结果
    
    Returns:
        特征字典
    """
    subtype_set = set(subtype_labels) if subtype_labels else set()
    
    # 各类别是否匹配（子标签和词表都匹配）
    subtype_match_porn = 1.0 if ('porn' in subtype_set and len(matches.get('porn', [])) > 0) else 0.0
    subtype_match_politics = 1.0 if ('politics' in subtype_set and len(matches.get('politics', [])) > 0) else 0.0
    subtype_match_abuse = 1.0 if ('abuse' in subtype_set and len(matches.get('abuse', [])) > 0) else 0.0
    
    # 计算匹配一致性分数
    # 如果子标签和词表匹配一致，分数高
    match_score = 0.0
    if subtype_set:
        matched_categories = []
        if len(matches.get('porn', [])) > 0:
            matched_categories.append('porn')
        if len(matches.get('politics', [])) > 0:
            matched_categories.append('politics')
        if len(matches.get('abuse', [])) > 0:
            matched_categories.append('abuse')
        
        # 计算交集比例
        if matched_categories:
            intersection = len(set(matched_categories) & subtype_set)
            union = len(set(matched_categories) | subtype_set)
            match_score = intersection / union if union > 0 else 0.0
        else:
            # 如果没有词表匹配，但子标签存在，分数为0（不一致）
            match_score = 0.0
    else:
        # 如果没有子标签，但词表有匹配，分数为0（不一致）
        if any(len(matches.get(cat, [])) > 0 for cat in ['porn', 'politics', 'abuse']):
            match_score = 0.0
        else:
            # 都没有，视为一致（非敏感）
            match_score = 1.0
    
    features = {
        'subtype_match_porn': subtype_match_porn,
        'subtype_match_politics': subtype_match_politics,
        'subtype_match_abuse': subtype_match_abuse,
        'subtype_match_score': match_score,
    }
    
    return features


def extract_match_density_features(
    text: str,
    matches: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    提取匹配密度特征
    
    Args:
        text: 文本内容
        matches: 词表匹配结果
    
    Returns:
        特征字典
    """
    total_matches = sum(len(matches.get(cat, [])) for cat in ['porn', 'politics', 'abuse'])
    text_length = len(text)
    
    # 匹配密度
    match_density = total_matches / text_length if text_length > 0 else 0.0
    
    # 匹配跨度比例（简化：假设匹配词平均长度）
    # 这里简化处理，使用匹配数作为代理
    avg_match_length = 3.0  # 假设平均匹配词长度为3
    match_span = total_matches * avg_match_length
    match_span_ratio = match_span / text_length if text_length > 0 else 0.0
    
    # 匹配数最多的类别（one-hot编码）
    porn_count = len(matches.get('porn', []))
    politics_count = len(matches.get('politics', []))
    abuse_count = len(matches.get('abuse', []))
    
    max_category = max(
        [('porn', porn_count), ('politics', politics_count), ('abuse', abuse_count)],
        key=lambda x: x[1]
    )[0]
    
    features = {
        'match_density': match_density,
        'match_span_ratio': match_span_ratio,
        'max_match_category_porn': 1.0 if max_category == 'porn' else 0.0,
        'max_match_category_politics': 1.0 if max_category == 'politics' else 0.0,
        'max_match_category_abuse': 1.0 if max_category == 'abuse' else 0.0,
    }
    
    return features


def extract_source_confidence_features(
    matches: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    提取源图谱置信特征
    
    Args:
        matches: 词表匹配结果
    
    Returns:
        特征字典
    """
    # 固定置信度（根据config.yaml中的权重）
    source_confidence_porn = 1.0
    source_confidence_politics = 0.8  # 对应politics_weight=0.6，但这里用0.8作为置信度
    source_confidence_abuse = 0.6  # 对应abuse_weight=0.5，但这里用0.6作为置信度
    
    # 计算加权平均置信度
    porn_count = len(matches.get('porn', []))
    politics_count = len(matches.get('politics', []))
    abuse_count = len(matches.get('abuse', []))
    total_count = porn_count + politics_count + abuse_count
    
    if total_count > 0:
        weighted_confidence = (
            source_confidence_porn * porn_count +
            source_confidence_politics * politics_count +
            source_confidence_abuse * abuse_count
        ) / total_count
    else:
        weighted_confidence = 0.0
    
    features = {
        'source_confidence_porn': source_confidence_porn,
        'source_confidence_politics': source_confidence_politics,
        'source_confidence_abuse': source_confidence_abuse,
        'weighted_source_confidence': weighted_confidence,
    }
    
    return features


def extract_all_features(
    text: str,
    subtype_labels: List[str],
    lexicons: Dict[str, Set[str]],
    automaton_cache: Optional[Dict] = None
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    提取所有PKRI特征
    
    Args:
        text: 文本内容
        subtype_labels: 子标签列表
        lexicons: 词表字典
        automaton_cache: 自动机缓存（可选）
    
    Returns:
        (特征字典, 匹配结果)
    """
    # 1. 词表命中特征
    lexicon_features, matches = extract_lexicon_features(text, lexicons, automaton_cache)
    
    # 2. 子标签匹配特征
    subtype_features = extract_subtype_match_features(subtype_labels, matches)
    
    # 3. 匹配密度特征
    density_features = extract_match_density_features(text, matches)
    
    # 4. 源图谱置信特征
    confidence_features = extract_source_confidence_features(matches)
    
    # 合并所有特征
    all_features = {
        **lexicon_features,
        **subtype_features,
        **density_features,
        **confidence_features,
    }
    
    return all_features, matches


def build_features_for_dataset(
    dataset_file: str,
    lexicons: Dict[str, Set[str]],
    output_file: str,
    automaton_cache: Optional[Dict] = None
):
    """
    为整个数据集构建特征
    
    Args:
        dataset_file: 数据集文件路径（JSONL格式）
        lexicons: 词表字典
        output_file: 输出CSV文件路径
        automaton_cache: 自动机缓存（可选）
    """
    logger.info("=" * 60)
    logger.info("构建PKRI特征")
    logger.info("=" * 60)
    logger.info(f"输入文件: {dataset_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info("")
    
    # 读取数据
    data_items = []
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data_items.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效JSON行: {e}")
                continue
    
    logger.info(f"读取 {len(data_items)} 条数据")
    logger.info("")
    
    # 构建特征
    feature_rows = []
    feature_names = None
    
    for i, item in enumerate(data_items):
        if (i + 1) % 1000 == 0:
            logger.info(f"处理进度: {i + 1}/{len(data_items)}")
        
        item_id = item.get('id', f'item_{i}')
        text = item.get('text', '')
        coarse_label = item.get('coarse_label', None)
        subtype_labels = item.get('subtype_label', [])
        
        if not isinstance(subtype_labels, list):
            subtype_labels = []
        
        # 提取特征
        try:
            features, matches = extract_all_features(
                text, 
                subtype_labels, 
                lexicons, 
                automaton_cache
            )
        except Exception as e:
            logger.warning(f"样本 {item_id} 特征提取失败: {e}")
            continue
        
        # 构建特征行
        feature_row = {
            'id': item_id,
            'text': text,
            'coarse_label': coarse_label,
            **features
        }
        
        # 如果coarse_label存在，作为标签
        if coarse_label is not None:
            feature_row['label'] = int(coarse_label)
        
        feature_rows.append(feature_row)
        
        # 记录特征名（第一次）
        if feature_names is None:
            feature_names = list(features.keys())
    
    logger.info(f"✓ 特征提取完成，共 {len(feature_rows)} 条")
    logger.info("")
    
    # 保存为CSV
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    df = pd.DataFrame(feature_rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info("=" * 60)
    logger.info("特征构建完成")
    logger.info("=" * 60)
    logger.info(f"输出文件: {output_file}")
    logger.info(f"特征数: {len(feature_names)}")
    logger.info(f"样本数: {len(feature_rows)}")
    logger.info("")
    logger.info("特征列表:")
    for feat_name in feature_names:
        logger.info(f"  - {feat_name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='构建PKRI特征')
    parser.add_argument('--dataset-file', type=str, required=True,
                       help='数据集文件路径（JSONL格式，需包含subtype_label字段）')
    parser.add_argument('--lexicon-dir', type=str, default='configs/lexicons',
                       help='词表目录')
    parser.add_argument('--output-file', type=str, required=True,
                       help='特征输出文件（CSV格式）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.dataset_file):
        logger.error(f"❌ 数据集文件不存在: {args.dataset_file}")
        sys.exit(1)
    
    # 加载词表
    logger.info("加载词表...")
    lexicons = {}
    lexicon_categories = ['porn', 'politics', 'abuse']
    
    for category in lexicon_categories:
        lexicon_path = os.path.join(args.lexicon_dir, f'{category}.txt')
        if os.path.exists(lexicon_path):
            words = load_lexicon(lexicon_path)
            lexicons[category] = words
            logger.info(f"  {category}: {len(words)} 个词")
        else:
            logger.warning(f"  ⚠ {category} 词表不存在: {lexicon_path}")
            lexicons[category] = set()
    
    if not any(lexicons.values()):
        logger.error("❌ 没有找到任何词表")
        sys.exit(1)
    
    logger.info("")
    
    # 构建自动机缓存（可选，加速匹配）
    automaton_cache = build_automaton_cache(lexicons)
    if automaton_cache:
        logger.info("✓ 已构建自动机缓存（加速匹配）")
    else:
        logger.info("⚠ 未安装 pyahocorasick，将使用较慢的字符串匹配")
    logger.info("")
    
    # 构建特征
    build_features_for_dataset(
        args.dataset_file,
        lexicons,
        args.output_file,
        automaton_cache
    )


if __name__ == '__main__':
    main()

