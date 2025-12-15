"""
clean_lexicon.py - 词表去重工具

功能：
1. 使用set去重
2. 备份原文件
3. 保存清理后的词表（按字母顺序排序）
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def deduplicate_lexicon_file(
    input_file: str,
    output_file: str = None,
    backup: bool = True
) -> dict:
    """
    对词表文件去重
    
    Args:
        input_file: 输入词表文件路径
        output_file: 输出词表文件路径（默认覆盖原文件）
        backup: 是否备份原文件
    
    Returns:
        统计信息字典
    """
    if not os.path.exists(input_file):
        logger.error(f"❌ 文件不存在: {input_file}")
        return {}
    
    # 如果没有指定输出文件，覆盖原文件
    if output_file is None:
        output_file = input_file
    
    logger.info(f"开始处理词表: {input_file}")
    
    # 读取原始词表
    original_words = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:  # 只保留非空行
                original_words.append(word)
    
    original_count = len(original_words)
    logger.info(f"原始词数: {original_count}")
    
    # 使用set去重
    unique_words = set(original_words)
    duplicate_count = original_count - len(unique_words)
    
    logger.info(f"去重后词数: {len(unique_words)} (移除 {duplicate_count} 个重复词)")
    
    # 备份原文件
    if backup and output_file == input_file:
        backup_file = f"{input_file}.backup"
        shutil.copy2(input_file, backup_file)
        logger.info(f"✓ 已备份原文件: {backup_file}")
    
    # 保存去重后的词表（按字母顺序排序，便于查看）
    unique_words_sorted = sorted(unique_words, key=lambda x: (len(x), x.lower()))
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for word in unique_words_sorted:
            f.write(word + '\n')
    
    logger.info(f"✓ 去重后的词表已保存: {output_file}")
    
    # 统计信息
    stats = {
        'original_count': original_count,
        'unique_count': len(unique_words),
        'duplicate_count': duplicate_count,
        'removal_rate': duplicate_count / original_count * 100 if original_count > 0 else 0
    }
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("去重统计:")
    logger.info(f"  原始词数: {stats['original_count']}")
    logger.info(f"  重复词数: {stats['duplicate_count']}")
    logger.info(f"  去重后词数: {stats['unique_count']}")
    logger.info(f"  重复率: {stats['removal_rate']:.2f}%")
    logger.info("=" * 60)
    
    return stats


def deduplicate_all_lexicons(
    lexicon_dir: str,
    backup: bool = True
) -> dict:
    """
    对所有词表文件去重
    
    Args:
        lexicon_dir: 词表目录
        backup: 是否备份原文件
    
    Returns:
        所有文件的统计信息
    """
    lexicon_dir = Path(lexicon_dir)
    if not lexicon_dir.exists():
        logger.error(f"❌ 目录不存在: {lexicon_dir}")
        return {}
    
    categories = ['porn', 'politics', 'abuse']
    all_stats = {}
    
    for category in categories:
        lexicon_file = lexicon_dir / f"{category}.txt"
        if lexicon_file.exists():
            logger.info("")
            logger.info("-" * 60)
            logger.info(f"处理类别: {category}")
            logger.info("-" * 60)
            stats = deduplicate_lexicon_file(
                str(lexicon_file),
                backup=backup
            )
            all_stats[category] = stats
        else:
            logger.warning(f"⚠ 词表文件不存在: {lexicon_file}")
    
    # 总结
    if all_stats:
        logger.info("")
        logger.info("=" * 60)
        logger.info("全部处理完成")
        logger.info("=" * 60)
        total_original = sum(s.get('original_count', 0) for s in all_stats.values())
        total_unique = sum(s.get('unique_count', 0) for s in all_stats.values())
        total_duplicate = sum(s.get('duplicate_count', 0) for s in all_stats.values())
        logger.info(f"总计:")
        logger.info(f"  原始词数: {total_original}")
        logger.info(f"  重复词数: {total_duplicate}")
        logger.info(f"  去重后词数: {total_unique}")
        logger.info(f"  总重复率: {total_duplicate / total_original * 100:.2f}%" if total_original > 0 else "  总重复率: 0.00%")
        logger.info("=" * 60)
    
    return all_stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='词表去重工具（使用set去重）')
    parser.add_argument('--lexicon-file', type=str, default=None,
                       help='单个词表文件路径')
    parser.add_argument('--lexicon-dir', type=str, default='configs/lexicons',
                       help='词表目录（如果指定，将处理目录下所有词表）')
    parser.add_argument('--output-file', type=str, default=None,
                       help='输出文件路径（默认覆盖原文件）')
    parser.add_argument('--no-backup', action='store_true',
                       help='不备份原文件')
    
    args = parser.parse_args()
    
    backup = not args.no_backup
    
    if args.lexicon_file:
        # 处理单个文件
        deduplicate_lexicon_file(
            args.lexicon_file,
            output_file=args.output_file,
            backup=backup
        )
    else:
        # 处理整个目录
        deduplicate_all_lexicons(
            args.lexicon_dir,
            backup=backup
        )


if __name__ == '__main__':
    main()
