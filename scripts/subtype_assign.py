"""
subtype_assign.py - 子标签打标工具

功能：
给敏感样本添加 subtypes 多标签字段（porn, politics, abuse, other）
"""

import os
import sys
import json
import logging
import argparse
import yaml

# 添加 scripts 目录到路径（用于导入 llm_labeler）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_labeler import (
    load_config,
    run_label_task,
    SUBTYPE_LABEL_TASK,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_sensitive_items(input_path: str, output_path: str, label_field: str = "coarse_label"):
    """
    过滤出敏感样本（label=1），用于只对敏感样本打子标签
    
    Args:
        input_path: 输入JSONL文件路径
        output_path: 临时输出文件路径（只包含敏感样本）
        label_field: 标签字段名
    
    Returns:
        敏感样本数量
    """
    sensitive_count = 0
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    item = json.loads(line)
                    label = item.get(label_field)
                    # 只处理敏感样本（label=1）
                    if label == 1:
                        f_out.write(line)
                        sensitive_count += 1
    return sensitive_count


def merge_subtype_labels(
    input_path: str,
    subtype_output_path: str,
    final_output_path: str,
    label_field: str = "coarse_label"
):
    """
    合并子标签到原始数据
    
    Args:
        input_path: 原始输入文件路径
        subtype_output_path: 子标签打标后的文件路径（只包含敏感样本）
        final_output_path: 最终输出文件路径
        label_field: 标签字段名
    """
    # 读取子标签结果（建立id到subtypes的映射）
    subtype_map = {}
    with open(subtype_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item_id = item.get('id')
                if item_id is not None:
                    subtype_map[item_id] = {
                        'subtype_label': item.get('subtype_label', []),
                        'subtype_response': item.get('subtype_response', '')
                    }
    
    # 合并到原始数据
    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(final_output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    item = json.loads(line)
                    item_id = item.get('id')
                    
                    # 如果是敏感样本且有子标签结果，添加子标签
                    if item.get(label_field) == 1 and item_id in subtype_map:
                        item['subtype_label'] = subtype_map[item_id]['subtype_label']
                        item['subtype_response'] = subtype_map[item_id]['subtype_response']
                    # 如果是非敏感样本，设置空子标签
                    elif item.get(label_field) == 0:
                        item['subtype_label'] = []
                        item['subtype_response'] = ''
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="子标签打标工具")
    parser.add_argument(
        "--input",
        default=None,
        help="输入JSONL文件路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出JSONL文件路径（默认从config.yaml读取）"
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
        "--label-field",
        default="coarse_label",
        help="粗粒度标签字段名（默认: coarse_label）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批处理大小（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="最大重试次数（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--only-sensitive",
        action="store_true",
        help="只对敏感样本（label=1）调用LLM打子标签"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="跳过已存在的记录（默认: True）"
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="不跳过已存在的记录"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return 1
    
    # 获取 data_dir
    data_dir = config.get("data_dir", "./data")
    
    # 确定输入输出路径
    if args.input:
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.join(data_dir, input_path)
    else:
        # 默认输入：with_coarse_label.jsonl
        input_path = os.path.join(data_dir, "with_coarse_label.jsonl")
    
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(data_dir, output_path)
    else:
        # 默认输出：with_coarse_and_subtypes.jsonl
        output_path = os.path.join(data_dir, "with_coarse_and_subtypes.jsonl")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        return 1
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取 LLM 配置
    llm_config = config.get("llm", {})
    batch_size = args.batch_size or llm_config.get("batch_size", 1)
    max_retries = args.max_retries or llm_config.get("max_retries", 3)
    request_interval = llm_config.get("request_interval", 2.5)
    
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出文件: {output_path}")
    logger.info(f"只处理敏感样本: {args.only_sensitive}")
    logger.info(f"批处理大小: {batch_size}")
    logger.info(f"最大重试次数: {max_retries}")
    logger.info(f"请求间隔: {request_interval}秒")
    
    try:
        if args.only_sensitive:
            # 只对敏感样本打子标签
            import tempfile
            temp_sensitive_path = output_path + ".sensitive.tmp"
            temp_subtype_path = output_path + ".subtype.tmp"
            
            try:
                # 1. 过滤出敏感样本
                sensitive_count = filter_sensitive_items(
                    input_path, 
                    temp_sensitive_path, 
                    args.label_field
                )
                logger.info(f"找到 {sensitive_count} 条敏感样本")
                
                if sensitive_count == 0:
                    logger.warning("没有敏感样本，直接复制输入文件到输出")
                    import shutil
                    shutil.copy(input_path, output_path)
                    return 0
                
                # 2. 对敏感样本打子标签
                stats = run_label_task(
                    task=SUBTYPE_LABEL_TASK,
                    input_path=temp_sensitive_path,
                    output_path=temp_subtype_path,
                    text_field=args.text_field,
                    batch_size=batch_size,
                    max_retries=max_retries,
                    skip_existing=args.skip_existing,
                    request_interval=request_interval
                )
                
                logger.info(f"子标签打标完成：成功 {stats['success']}, 失败 {stats['failed']}")
                
                # 3. 合并子标签到原始数据
                merge_subtype_labels(
                    input_path,
                    temp_subtype_path,
                    output_path,
                    args.label_field
                )
                
                logger.info(f"合并完成，输出文件: {output_path}")
                
            finally:
                # 清理临时文件
                for temp_file in [temp_sensitive_path, temp_subtype_path]:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
        else:
            # 对所有样本打子标签（包括非敏感样本）
            stats = run_label_task(
                task=SUBTYPE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                text_field=args.text_field,
                batch_size=batch_size,
                max_retries=max_retries,
                skip_existing=args.skip_existing,
                request_interval=request_interval
            )
            
            logger.info(f"打标完成：成功 {stats['success']}, 失败 {stats['failed']}, 跳过 {stats['skipped']}")
        
        print("\n打标结果:")
        print(f"  输出文件: {output_path}")
        if not args.only_sensitive:
            print(f"  总数据: {stats['total']}")
            print(f"  成功: {stats['success']}")
            print(f"  失败: {stats['failed']}")
            print(f"  跳过: {stats['skipped']}")
        else:
            print(f"  敏感样本数: {sensitive_count}")
            print(f"  成功打标: {stats['success']}")
            print(f"  失败: {stats['failed']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

