"""
coarse_label.py - 粗粒度标签打标工具

功能：
利用 LLM 给每条样本打主标签：label = 1（敏感）或 0（非敏感）
"""

import os
import sys
import logging
import argparse
import yaml

# 添加 scripts 目录到路径（用于导入 llm_labeler）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_labeler import (
    load_config,
    run_label_task,
    COARSE_LABEL_TASK,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="粗粒度标签打标工具")
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
        # 默认输入：cleaned_raw.jsonl
        input_path = os.path.join(data_dir, "cleaned_raw.jsonl")
    
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(data_dir, output_path)
    else:
        # 默认输出：with_coarse_label.jsonl
        output_path = os.path.join(data_dir, "with_coarse_label.jsonl")
    
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
    logger.info(f"批处理大小: {batch_size}")
    logger.info(f"最大重试次数: {max_retries}")
    logger.info(f"请求间隔: {request_interval}秒")
    logger.info(f"跳过已存在: {args.skip_existing}")
    
    # 执行打标任务
    try:
        stats = run_label_task(
            task=COARSE_LABEL_TASK,
            input_path=input_path,
            output_path=output_path,
            text_field=args.text_field,
            batch_size=batch_size,
            max_retries=max_retries,
            skip_existing=args.skip_existing,
            request_interval=request_interval
        )
        
        print("\n打标结果:")
        print(f"  总数据: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        print(f"  跳过: {stats['skipped']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

