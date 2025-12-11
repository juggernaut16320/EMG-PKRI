"""
check_token_length.py - 检查数据集中超过指定token长度的样本数量

功能：
统计数据集中超过max_length（默认512）token的样本数量和分布
"""

import os
import sys
import json
import logging
import argparse
import yaml
from collections import Counter
from typing import Dict, List

# 设置标准输出编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def count_tokens(text: str, tokenizer) -> int:
    """统计文本的token数量"""
    encoded = tokenizer(text, return_tensors=None, add_special_tokens=True)
    return len(encoded['input_ids'])


def analyze_file(file_path: str, tokenizer, max_length: int = 512) -> Dict:
    """
    分析单个JSONL文件中的token长度分布
    
    Args:
        file_path: JSONL文件路径
        tokenizer: Tokenizer实例
        max_length: 最大长度阈值
    
    Returns:
        统计信息字典
    """
    if not os.path.exists(file_path):
        logger.warning(f"文件不存在: {file_path}")
        return None
    
    stats = {
        'total': 0,
        'over_max': 0,
        'under_max': 0,
        'token_lengths': [],
        'max_token_length': 0,
        'min_token_length': float('inf'),
        'avg_token_length': 0.0,
        'over_samples': []  # 超过max_length的样本（前10个）
    }
    
    logger.info(f"Analyzing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                item = json.loads(line)
                text = item.get('text', '')
                
                if not text:
                    continue
                
                # Count tokens
                token_count = count_tokens(text, tokenizer)
                stats['token_lengths'].append(token_count)
                stats['total'] += 1
                
                # Update min/max
                if token_count > stats['max_token_length']:
                    stats['max_token_length'] = token_count
                if token_count < stats['min_token_length']:
                    stats['min_token_length'] = token_count
                
                # Count samples over max_length
                if token_count > max_length:
                    stats['over_max'] += 1
                    if len(stats['over_samples']) < 10:
                        stats['over_samples'].append({
                            'id': item.get('id', line_num),
                            'token_count': token_count,
                            'text_preview': text[:100] + '...' if len(text) > 100 else text
                        })
                else:
                    stats['under_max'] += 1
                
                # Progress update every 1000 samples
                if stats['total'] % 1000 == 0:
                    logger.info(f"Processed {stats['total']} samples...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} JSON decode error: {e}")
                continue
            except Exception as e:
                logger.warning(f"Line {line_num} processing error: {e}")
                continue
    
    # Calculate average length
    if stats['total'] > 0:
        stats['avg_token_length'] = sum(stats['token_lengths']) / stats['total']
        stats['over_ratio'] = stats['over_max'] / stats['total'] * 100
    else:
        stats['min_token_length'] = 0
    
    return stats


def print_statistics(stats: Dict, file_name: str, max_length: int = 512):
    """Print statistics"""
    if stats is None:
        return
    
    print("\n" + "=" * 80)
    print(f"File: {file_name}")
    print("=" * 80)
    print(f"Total samples: {stats['total']:,}")
    print(f"Samples over {max_length} tokens: {stats['over_max']:,} ({stats.get('over_ratio', 0):.2f}%)")
    print(f"Samples under {max_length} tokens: {stats['under_max']:,} ({100 - stats.get('over_ratio', 0):.2f}%)")
    print(f"\nToken length statistics:")
    print(f"  Min length: {stats['min_token_length']}")
    print(f"  Max length: {stats['max_token_length']}")
    print(f"  Avg length: {stats['avg_token_length']:.2f}")
    
    # Length distribution
    if stats['token_lengths']:
        length_dist = Counter(stats['token_lengths'])
        print(f"\nTop 10 most common lengths:")
        for length, count in length_dist.most_common(10):
            print(f"  {length} tokens: {count:,} samples ({count/stats['total']*100:.2f}%)")
        
        # Range distribution
        ranges = [
            (0, 100, "0-100"),
            (100, 256, "100-256"),
            (256, 512, "256-512"),
            (512, 1024, "512-1024"),
            (1024, float('inf'), "1024+")
        ]
        print(f"\nLength range distribution:")
        for min_len, max_len, label in ranges:
            count = sum(1 for l in stats['token_lengths'] if min_len <= l < max_len)
            if count > 0:
                print(f"  {label}: {count:,} samples ({count/stats['total']*100:.2f}%)")
    
    # Show samples over max_length
    if stats['over_samples']:
        print(f"\nSamples over {max_length} tokens (first {len(stats['over_samples'])}):")
        for i, sample in enumerate(stats['over_samples'], 1):
            print(f"  [{i}] ID: {sample['id']}, Tokens: {sample['token_count']}")
            print(f"      Text preview: {sample['text_preview']}")


def main():
    parser = argparse.ArgumentParser(description='检查数据集中超过指定token长度的样本')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--max-length', type=int, default=512,
                        help='最大token长度阈值（默认512）')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                        help='要分析的文件列表（相对于data_dir），默认分析train/dev/test.jsonl')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    data_dir = config.get('data_dir', './data')
    model_config = config.get('model', {})
    model_name_or_path = model_config.get('name_or_path', 'Qwen/Qwen3-1.7B')
    max_length = args.max_length
    
    logger.info("=" * 80)
    logger.info("Token Length Statistics Tool")
    logger.info(f"Model: {model_name_or_path}")
    logger.info(f"Max length threshold: {max_length}")
    logger.info("=" * 80)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return
    
    # 确定要分析的文件
    if args.files:
        files_to_analyze = args.files
    else:
        files_to_analyze = ['train.jsonl', 'dev.jsonl', 'test.jsonl']
    
    # 分析每个文件
    all_stats = {}
    for file_name in files_to_analyze:
        file_path = os.path.join(data_dir, file_name)
        stats = analyze_file(file_path, tokenizer, max_length)
        if stats:
            all_stats[file_name] = stats
            print_statistics(stats, file_name, max_length)
    
    # Summary statistics
    if all_stats:
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        total_samples = sum(s['total'] for s in all_stats.values())
        total_over = sum(s['over_max'] for s in all_stats.values())
        total_under = sum(s['under_max'] for s in all_stats.values())
        
        print(f"Total samples across all files: {total_samples:,}")
        print(f"Total samples over {max_length} tokens: {total_over:,} ({total_over/total_samples*100:.2f}%)")
        print(f"Total samples under {max_length} tokens: {total_under:,} ({total_under/total_samples*100:.2f}%)")
        
        # File distribution
        print(f"\nFile distribution:")
        for file_name, stats in all_stats.items():
            print(f"  {file_name}: {stats['total']:,} samples ({stats['total']/total_samples*100:.2f}%)")
            if stats['over_max'] > 0:
                print(f"    - Over {max_length} tokens: {stats['over_max']:,} samples ({stats.get('over_ratio', 0):.2f}%)")


if __name__ == '__main__':
    main()

