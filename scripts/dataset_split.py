"""
dataset_split.py - 数据集划分工具

功能：
将打好标签的数据按比例划分 train/dev/test（支持分层划分）
采用完全重新划分策略：每次运行都会重新划分所有数据
"""

import os
import sys
import json
import logging
import argparse
import yaml
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

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


def split_dataset(
    input_path: str,
    train_path: str,
    dev_path: str,
    test_path: str,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify_field: str = "coarse_label",
    label_field: str = "coarse_label"
) -> dict:
    """
    划分数据集为 train/dev/test
    
    Args:
        input_path: 输入 JSONL 文件路径
        train_path: 训练集输出路径
        dev_path: 验证集输出路径
        test_path: 测试集输出路径
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        test_ratio: 测试集比例
        random_seed: 随机种子（固定种子确保可复现）
        stratify_field: 分层字段名（用于分层划分）
        label_field: 标签字段名（用于统计）
    
    Returns:
        统计信息字典
    """
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 读取数据
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    logger.info(f"读取 {len(data)} 条数据")
    
    if len(data) == 0:
        logger.error("输入数据为空")
        return {
            "total": 0,
            "train": 0,
            "dev": 0,
            "test": 0,
            "train_sensitive": 0,
            "dev_sensitive": 0,
            "test_sensitive": 0
        }
    
    # 检查分层字段是否存在
    has_stratify = all(item.get(stratify_field) is not None for item in data)
    
    if has_stratify:
        # 提取标签用于分层
        labels = [item.get(stratify_field) for item in data]
        logger.info(f"使用分层划分（按 {stratify_field} 分层）")
        
        # 统计标签分布
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"标签分布: {label_counts}")
        
        # 第一次划分：train 和 (dev+test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data,
            labels,
            test_size=(dev_ratio + test_ratio),
            stratify=labels,
            random_state=random_seed
        )
        
        # 第二次划分：dev 和 test
        # 计算 dev 在 (dev+test) 中的比例
        dev_ratio_in_temp = dev_ratio / (dev_ratio + test_ratio)
        dev_data, test_data, dev_labels, test_labels = train_test_split(
            temp_data,
            temp_labels,
            test_size=(1 - dev_ratio_in_temp),
            stratify=temp_labels,
            random_state=random_seed
        )
    else:
        logger.warning(f"字段 {stratify_field} 不存在，使用随机划分（不分层）")
        # 随机划分，不分层
        train_data, temp_data = train_test_split(
            data,
            test_size=(dev_ratio + test_ratio),
            random_state=random_seed
        )
        
        dev_ratio_in_temp = dev_ratio / (dev_ratio + test_ratio)
        dev_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - dev_ratio_in_temp),
            random_state=random_seed
        )
    
    # 写入文件
    def write_jsonl(data_list: List[Dict], output_path: str):
        """写入 JSONL 文件"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    write_jsonl(train_data, train_path)
    write_jsonl(dev_data, dev_path)
    write_jsonl(test_data, test_path)
    
    # 统计信息
    def count_sensitive(data_list: List[Dict], label_field: str) -> int:
        """统计敏感样本数量"""
        return sum(1 for item in data_list if item.get(label_field) == 1)
    
    stats = {
        "total": len(data),
        "train": len(train_data),
        "dev": len(dev_data),
        "test": len(test_data),
        "train_sensitive": count_sensitive(train_data, label_field),
        "dev_sensitive": count_sensitive(dev_data, label_field),
        "test_sensitive": count_sensitive(test_data, label_field),
        "train_ratio": len(train_data) / len(data),
        "dev_ratio": len(dev_data) / len(data),
        "test_ratio": len(test_data) / len(data)
    }
    
    logger.info(f"划分完成:")
    logger.info(f"  训练集: {stats['train']} 条 (敏感: {stats['train_sensitive']}, 比例: {stats['train_ratio']:.2%})")
    logger.info(f"  验证集: {stats['dev']} 条 (敏感: {stats['dev_sensitive']}, 比例: {stats['dev_ratio']:.2%})")
    logger.info(f"  测试集: {stats['test']} 条 (敏感: {stats['test_sensitive']}, 比例: {stats['test_ratio']:.2%})")
    
    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集划分工具")
    parser.add_argument(
        "--input",
        default=None,
        help="输入JSONL文件路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--train-output",
        default=None,
        help="训练集输出路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--dev-output",
        default=None,
        help="验证集输出路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--test-output",
        default=None,
        help="测试集输出路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="训练集比例（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=None,
        help="验证集比例（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="测试集比例（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机种子（默认从config.yaml读取，默认42）"
    )
    parser.add_argument(
        "--stratify-field",
        default="coarse_label",
        help="分层字段名（默认: coarse_label）"
    )
    parser.add_argument(
        "--label-field",
        default="coarse_label",
        help="标签字段名（用于统计，默认: coarse_label）"
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
    
    # 获取划分配置
    split_config = config.get("split", {})
    train_ratio = args.train_ratio or split_config.get("train", 0.8)
    dev_ratio = args.dev_ratio or split_config.get("dev", 0.1)
    test_ratio = args.test_ratio or split_config.get("test", 0.1)
    random_seed = args.random_seed or split_config.get("random_seed", 42)
    
    # 验证比例
    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        logger.error(f"比例之和必须为1.0，当前: {train_ratio + dev_ratio + test_ratio}")
        return 1
    
    # 确定输入输出路径
    if args.input:
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.join(data_dir, input_path)
    else:
        # 默认输入：with_coarse_and_subtypes.jsonl 或 with_coarse.jsonl
        input_path = os.path.join(data_dir, "with_coarse_and_subtypes.jsonl")
        if not os.path.exists(input_path):
            input_path = os.path.join(data_dir, "dataset_with_coarse.jsonl")
    
    if args.train_output:
        train_path = args.train_output
        if not os.path.isabs(train_path):
            train_path = os.path.join(data_dir, train_path)
    else:
        train_path = os.path.join(data_dir, "train.jsonl")
    
    if args.dev_output:
        dev_path = args.dev_output
        if not os.path.isabs(dev_path):
            dev_path = os.path.join(data_dir, dev_path)
    else:
        dev_path = os.path.join(data_dir, "dev.jsonl")
    
    if args.test_output:
        test_path = args.test_output
        if not os.path.isabs(test_path):
            test_path = os.path.join(data_dir, test_path)
    else:
        test_path = os.path.join(data_dir, "test.jsonl")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        return 1
    
    logger.info(f"输入文件: {input_path}")
    logger.info(f"训练集输出: {train_path}")
    logger.info(f"验证集输出: {dev_path}")
    logger.info(f"测试集输出: {test_path}")
    logger.info(f"划分比例: train={train_ratio}, dev={dev_ratio}, test={test_ratio}")
    logger.info(f"随机种子: {random_seed}")
    logger.info(f"分层字段: {args.stratify_field}")
    
    # 执行划分
    try:
        stats = split_dataset(
            input_path=input_path,
            train_path=train_path,
            dev_path=dev_path,
            test_path=test_path,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            stratify_field=args.stratify_field,
            label_field=args.label_field
        )
        
        print("\n划分结果:")
        print(f"  总数据: {stats['total']}")
        print(f"  训练集: {stats['train']} (敏感: {stats['train_sensitive']}, 比例: {stats['train_ratio']:.2%})")
        print(f"  验证集: {stats['dev']} (敏感: {stats['dev_sensitive']}, 比例: {stats['dev_ratio']:.2%})")
        print(f"  测试集: {stats['test']} (敏感: {stats['test_sensitive']}, 比例: {stats['test_ratio']:.2%})")
        
        return 0
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

