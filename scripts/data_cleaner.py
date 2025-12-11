"""
data_cleaner.py - 数据清洗工具

功能：
1. 基础过滤：空文本、最小长度
2. URL替换为占位符
3. @提及替换为占位符
4. 转发内容提取
5. 精确去重
"""

import os
import json
import re
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 预编译正则表达式（性能优化）
# URL模式：能正确处理URL后直接跟中文的情况
URL_PATTERN = re.compile(
    r'(?:https?://|www\.)'           # http://, https://, www.
    r'[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+'  # URL有效字符（ASCII only）
    r'(?=\s|$|[\u4e00-\u9fff\u3400-\u4dbf]|[^\x00-\x7F])',  # 前瞻：空格/结束/中文/非ASCII
    re.IGNORECASE | re.UNICODE
)

# @提及模式：推特用户名规则（1-15个字母数字下划线）
# 注意：前瞻断言需要排除中文字符，因为\w在Unicode模式下包含中文
MENTION_PATTERN = re.compile(
    r'@[a-zA-Z0-9_]{1,15}'           # @后1-15个字母数字下划线
    r'(?=\s|$|[\u4e00-\u9fff]|[^\w])',  # 前瞻：空格/结束/中文字符/非单词字符
    re.UNICODE
)

# 转发标记模式
RETWEET_PATTERNS = [
    re.compile(r'RT\s+@\w+[:：]\s*', re.IGNORECASE),  # RT @username:
    re.compile(r'转发\s*@\w+[:：]\s*', re.IGNORECASE),  # 转发 @username:
    re.compile(r'Retweet\s+@\w+[:：]\s*', re.IGNORECASE),  # Retweet @username:
]


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def replace_urls(text: str, placeholder: str = "[URL]") -> str:
    """
    替换文本中的URL为占位符
    
    Args:
        text: 原始文本
        placeholder: 占位符，默认 "[URL]"
    
    Returns:
        替换后的文本
    """
    return URL_PATTERN.sub(placeholder, text)


def replace_mentions(text: str, placeholder: str = "[MENTION]") -> str:
    """
    替换文本中的@提及为占位符
    
    Args:
        text: 原始文本
        placeholder: 占位符，默认 "[MENTION]"
    
    Returns:
        替换后的文本
    """
    return MENTION_PATTERN.sub(placeholder, text)


def extract_retweet_content(text: str, patterns: List[re.Pattern] = None) -> str:
    """
    提取转发内容（移除转发标记）
    
    Args:
        text: 原始文本
        patterns: 转发标记模式列表，如果为None则使用默认模式
    
    Returns:
        提取后的文本（如果包含转发标记则移除，否则返回原文本）
    """
    if patterns is None:
        patterns = RETWEET_PATTERNS
    
    result = text
    for pattern in patterns:
        result = pattern.sub('', result)
    
    return result.strip()


def clean_text(text: str, config: dict) -> Optional[str]:
    """
    清洗单条文本
    
    Args:
        text: 原始文本
        config: 清洗配置
    
    Returns:
        清洗后的文本，如果被过滤则返回 None
    """
    if not text:
        return None
    
    # 1. 基础清理：去除首尾空白
    text = text.strip()
    
    # 2. 空文本过滤
    if not text:
        return None
    
    # 3. 长度过滤
    min_length = config.get('min_length', 5)
    if len(text) < min_length:
        return None
    
    # max_length为None或不存在表示无限制
    max_length = config.get('max_length')
    if max_length is not None and max_length > 0 and len(text) > max_length:
        return None
    
    # 4. URL替换
    url_handling = config.get('url_handling', 'replace')
    if url_handling == 'replace':
        url_placeholder = config.get('url_placeholder', '[URL]')
        text = replace_urls(text, url_placeholder)
    
    # 5. @提及替换
    mention_handling = config.get('mention_handling', 'replace')
    if mention_handling == 'replace':
        mention_placeholder = config.get('mention_placeholder', '[MENTION]')
        text = replace_mentions(text, mention_placeholder)
    
    # 6. 转发内容提取
    retweet_handling = config.get('retweet_handling', 'extract')
    if retweet_handling == 'extract':
        text = extract_retweet_content(text)
        # 提取后可能变空，需要再次检查
        if not text.strip():
            return None
    
    # 7. 再次去除首尾空白
    text = text.strip()
    
    # 8. 最终长度检查
    if len(text) < min_length:
        return None
    
    return text


def get_text_hash(text: str) -> str:
    """
    计算文本的MD5哈希值
    
    Args:
        text: 文本内容
    
    Returns:
        MD5哈希值的十六进制字符串
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def deduplicate(items: List[Dict], text_field: str = 'text') -> List[Dict]:
    """
    精确去重（基于文本哈希值）
    
    Args:
        items: 数据项列表
        text_field: 用于去重的文本字段名
    
    Returns:
        去重后的列表（保留第一次出现的）
    """
    seen_hashes = set()
    unique_items = []
    
    for item in items:
        text = item.get(text_field, '')
        text_hash = get_text_hash(text)
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_items.append(item)
    
    return unique_items


def clean_jsonl(
    input_path: str,
    output_path: str,
    config: dict,
    text_field: str = 'text',
    skip_existing: bool = False
) -> dict:
    """
    清洗JSONL文件
    
    Args:
        input_path: 输入JSONL文件路径
        output_path: 输出JSONL文件路径
        config: 清洗配置
        text_field: 文本字段名
        skip_existing: 是否跳过已存在的输出文件
    
    Returns:
        统计信息字典
    """
    # 检查输出文件是否存在
    if skip_existing and os.path.exists(output_path):
        logger.info(f"输出文件已存在: {output_path}，跳过处理")
        return {
            "total": 0,
            "processed": 0,
            "filtered": 0,
            "deduplicated": 0,
            "output": 0
        }
    
    # 读取输入数据
    input_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    input_data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON失败: {e}")
                    continue
    
    logger.info(f"读取 {len(input_data)} 条数据")
    
    # 统计信息
    stats = {
        "total": len(input_data),
        "processed": 0,
        "filtered": 0,
        "deduplicated": 0,
        "output": 0
    }
    
    # 清洗数据
    cleaned_data = []
    for item in input_data:
        text = item.get(text_field, '')
        if not text:
            stats["filtered"] += 1
            continue
        
        cleaned_text = clean_text(text, config)
        if cleaned_text is None:
            stats["filtered"] += 1
            continue
        
        # 更新文本字段
        item[text_field] = cleaned_text
        cleaned_data.append(item)
        stats["processed"] += 1
    
    logger.info(f"清洗完成：处理 {stats['processed']} 条，过滤 {stats['filtered']} 条")
    
    # 去重
    dedup_config = config.get('deduplication', {})
    if dedup_config.get('exact', True):
        original_count = len(cleaned_data)
        cleaned_data = deduplicate(cleaned_data, text_field)
        stats["deduplicated"] = original_count - len(cleaned_data)
        logger.info(f"去重完成：移除 {stats['deduplicated']} 条重复数据")
    
    # 写入输出
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            stats["output"] += 1
    
    logger.info(f"完成！输出 {stats['output']} 条清洗后的数据")
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据清洗工具")
    parser.add_argument(
        "--input",
        default="data/dataset_raw.jsonl",
        help="输入JSONL文件路径（默认: data/dataset_raw.jsonl）"
    )
    parser.add_argument(
        "--output",
        default="data/cleaned_raw.jsonl",
        help="输出JSONL文件路径（默认: data/cleaned_raw.jsonl）"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="文本字段名（默认: text）"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出文件已存在则跳过"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
        cleaning_config = config.get('data_cleaning', {})
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        exit(1)
    
    # 从配置文件读取data_dir（如果指定了）
    data_dir = config.get("data_dir", "./data")
    
    # 如果路径是相对路径，使用data_dir
    if not os.path.isabs(args.input):
        args.input = os.path.join(data_dir, os.path.basename(args.input))
    if not os.path.isabs(args.output):
        args.output = os.path.join(data_dir, os.path.basename(args.output))
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        exit(1)
    
    # 执行清洗
    try:
        stats = clean_jsonl(
            input_path=args.input,
            output_path=args.output,
            config=cleaning_config,
            text_field=args.text_field,
            skip_existing=args.skip_existing
        )
        
        print("\n清洗结果:")
        print(f"  总数据: {stats['total']}")
        print(f"  成功处理: {stats['processed']}")
        print(f"  被过滤: {stats['filtered']}")
        print(f"  去重移除: {stats['deduplicated']}")
        print(f"  最终输出: {stats['output']}")
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        exit(1)

