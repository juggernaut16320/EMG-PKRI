"""
txt_to_jsonl.py - TXT 文件转 JSONL 工具

功能：
1. 从 unprocessed 目录读取所有 txt 文件
2. 使用全局序号（从 s0 开始，s+num 格式）命名每条记录
3. 如果 new_jsonl 已存在，继续接着最大序号添加
4. 处理完成后删除已处理的 txt 文件
5. 性能优化：大文件时使用临时文件策略
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple
import yaml

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


def get_max_id_from_jsonl(jsonl_path: str) -> int:
    """
    从 JSONL 文件中获取最大的 id 值
    
    Args:
        jsonl_path: JSONL 文件路径
    
    Returns:
        最大 id 值，如果文件不存在或没有 id 字段，返回 -1
    """
    if not os.path.exists(jsonl_path):
        return -1
    
    max_id = -1
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        item_id = item.get('id', '')
                        # 解析 s+num 格式（如 "s0", "s1", "s123"）
                        if isinstance(item_id, str) and item_id.startswith('s'):
                            try:
                                item_id_int = int(item_id[1:])
                            except ValueError:
                                item_id_int = -1
                        elif isinstance(item_id, (int, float)):
                            item_id_int = int(item_id)
                        else:
                            item_id_int = -1
                        
                        if item_id_int > max_id:
                            max_id = item_id_int
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"解析 JSONL 行失败: {e}")
                        continue
    except Exception as e:
        logger.error(f"读取 JSONL 文件失败: {e}")
        return -1
    
    return max_id


def format_id(id_num: int) -> str:
    """
    格式化 id 为 s+num 形式
    
    Args:
        id_num: id 数字（从 0 开始）
    
    Returns:
        格式化后的 id 字符串，如 "s0", "s1", "s2"
    """
    return f"s{id_num}"


def estimate_jsonl_size(jsonl_path: str) -> int:
    """
    估算 JSONL 文件的大小（行数）
    
    Args:
        jsonl_path: JSONL 文件路径
    
    Returns:
        文件行数，如果文件不存在返回 0
    """
    if not os.path.exists(jsonl_path):
        return 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.warning(f"估算文件大小失败: {e}")
        return 0


def process_txt_files(
    unprocessed_dir: str,
    output_jsonl: str,
    start_id: int = 0,
    use_temp_file: bool = False,
    temp_file_threshold: int = 10000
) -> dict:
    """
    处理 unprocessed 目录下的所有 txt 文件，转换为 JSONL 格式
    
    Args:
        unprocessed_dir: 未处理的 txt 文件目录
        output_jsonl: 输出 JSONL 文件路径
        start_id: 起始 id（默认 12345，会格式化为 "012345"）
        use_temp_file: 是否使用临时文件策略（当输出文件很大时）
        temp_file_threshold: 使用临时文件的阈值（行数）
    
    Returns:
        统计信息字典
    """
    unprocessed_path = Path(unprocessed_dir)
    if not unprocessed_path.exists():
        logger.warning(f"目录不存在: {unprocessed_dir}")
        return {
            "total_txt_files": 0,
            "processed": 0,
            "failed": 0,
            "deleted": 0,
            "start_id": start_id,
            "end_id": start_id - 1
        }
    
    # 获取所有 txt 文件
    txt_files = list(unprocessed_path.glob("*.txt"))
    if not txt_files:
        logger.info(f"未找到 txt 文件: {unprocessed_dir}")
        return {
            "total_txt_files": 0,
            "processed": 0,
            "failed": 0,
            "deleted": 0,
            "start_id": start_id,
            "end_id": start_id - 1
        }
    
    logger.info(f"找到 {len(txt_files)} 个 txt 文件")
    
    # 确定起始 id
    current_id = start_id
    if os.path.exists(output_jsonl):
        max_id = get_max_id_from_jsonl(output_jsonl)
        if max_id >= 0:
            current_id = max_id + 1
            logger.info(f"检测到已存在的 JSONL 文件，最大 id: {max_id}，从 {current_id} 开始")
        else:
            logger.info(f"检测到已存在的 JSONL 文件，但无法读取 id，从 {start_id} 开始")
    
    # 决定是否使用临时文件策略
    if not use_temp_file and os.path.exists(output_jsonl):
        file_size = estimate_jsonl_size(output_jsonl)
        if file_size > temp_file_threshold:
            logger.info(f"输出文件较大（{file_size} 行），使用临时文件策略")
            use_temp_file = True
    
    # 统计信息
    stats = {
        "total_txt_files": len(txt_files),
        "processed": 0,
        "failed": 0,
        "deleted": 0,
        "start_id": current_id,
        "end_id": current_id - 1
    }
    
    # 处理文件
    temp_jsonl = None
    if use_temp_file:
        temp_jsonl = output_jsonl + ".tmp"
        logger.info(f"使用临时文件: {temp_jsonl}")
    
    output_path = temp_jsonl if use_temp_file else output_jsonl
    mode = 'a' if os.path.exists(output_path) and not use_temp_file else 'w'
    
    try:
        with open(output_path, mode, encoding='utf-8') as f_out:
            for txt_file in txt_files:
                try:
                    # 读取 txt 文件内容
                    with open(txt_file, 'r', encoding='utf-8') as f_in:
                        text = f_in.read().strip()
                    
                    if not text:
                        logger.warning(f"跳过空文件: {txt_file}")
                        stats["failed"] += 1
                        continue
                    
                    # 创建 JSON 记录
                    record = {
                        "id": format_id(current_id),
                        "text": text
                    }
                    
                    # 写入 JSONL
                    f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f_out.flush()  # 确保数据写入磁盘后再删除 txt 文件
                    
                    # 删除已处理的 txt 文件
                    try:
                        txt_file.unlink()
                        stats["deleted"] += 1
                        logger.debug(f"已删除: {txt_file}")
                    except Exception as e:
                        logger.warning(f"删除文件失败 {txt_file}: {e}")
                    
                    stats["processed"] += 1
                    current_id += 1
                    
                    if stats["processed"] % 100 == 0:
                        logger.info(f"进度: {stats['processed']}/{stats['total_txt_files']}")
                
                except Exception as e:
                    logger.error(f"处理文件失败 {txt_file}: {e}")
                    stats["failed"] += 1
        
        stats["end_id"] = current_id - 1
        
        # 如果使用了临时文件，需要合并
        if use_temp_file:
            logger.info("合并临时文件到输出文件...")
            merge_jsonl_files(output_jsonl, temp_jsonl)
            # 删除临时文件
            if os.path.exists(temp_jsonl):
                os.remove(temp_jsonl)
                logger.info(f"已删除临时文件: {temp_jsonl}")
    
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        # 如果使用了临时文件，清理它
        if temp_jsonl and os.path.exists(temp_jsonl):
            try:
                os.remove(temp_jsonl)
            except:
                pass
        raise
    
    logger.info(f"完成！处理: {stats['processed']}, 失败: {stats['failed']}, 删除: {stats['deleted']}")
    if stats['processed'] > 0:
        logger.info(f"ID 范围: {format_id(stats['start_id'])} - {format_id(stats['end_id'])}")
    
    return stats


def merge_jsonl_files(existing_jsonl: str, new_jsonl: str):
    """
    合并两个 JSONL 文件（将 new_jsonl 追加到 existing_jsonl）
    
    Args:
        existing_jsonl: 已存在的 JSONL 文件路径
        new_jsonl: 新的 JSONL 文件路径（临时文件）
    """
    if not os.path.exists(new_jsonl):
        logger.warning(f"临时文件不存在: {new_jsonl}")
        return
    
    # 如果 existing_jsonl 不存在，直接重命名
    if not os.path.exists(existing_jsonl):
        import shutil
        shutil.move(new_jsonl, existing_jsonl)
        logger.info(f"重命名临时文件: {new_jsonl} -> {existing_jsonl}")
        return
    
    # 合并文件
    try:
        with open(existing_jsonl, 'a', encoding='utf-8') as f_out:
            with open(new_jsonl, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
        
        # 删除新文件
        os.remove(new_jsonl)
        logger.info(f"成功合并文件: {new_jsonl} -> {existing_jsonl}")
    except Exception as e:
        logger.error(f"合并文件失败: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TXT 文件转 JSONL 工具")
    parser.add_argument(
        "--unprocessed-dir",
        default="data/unprocessed",
        help="未处理的 txt 文件目录（默认: data/unprocessed）"
    )
    parser.add_argument(
        "--output",
        default="data/dataset_raw.jsonl",
        help="输出 JSONL 文件路径（默认: data/dataset_raw.jsonl）"
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="起始 ID（默认: 0，会格式化为 s0）"
    )
    parser.add_argument(
        "--use-temp",
        action="store_true",
        help="强制使用临时文件策略"
    )
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=10000,
        help="使用临时文件的阈值（行数，默认: 10000）"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    
    args = parser.parse_args()
    
    # 从配置文件读取 data_dir（如果指定了）
    try:
        config = load_config(args.config)
        data_dir = config.get("data_dir", "./data")
        
        # 如果路径是相对路径，使用 data_dir
        if not os.path.isabs(args.unprocessed_dir):
            args.unprocessed_dir = os.path.join(data_dir, "unprocessed")
        if not os.path.isabs(args.output):
            args.output = os.path.join(data_dir, os.path.basename(args.output))
    except Exception as e:
        logger.warning(f"读取配置文件失败，使用默认路径: {e}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 执行处理
    try:
        stats = process_txt_files(
            unprocessed_dir=args.unprocessed_dir,
            output_jsonl=args.output,
            start_id=args.start_id,
            use_temp_file=args.use_temp,
            temp_file_threshold=args.temp_threshold
        )
        
        print("\n处理结果:")
        print(f"  总文件数: {stats['total_txt_files']}")
        print(f"  成功处理: {stats['processed']}")
        print(f"  失败: {stats['failed']}")
        print(f"  已删除: {stats['deleted']}")
        if stats['processed'] > 0:
            print(f"  ID 范围: {format_id(stats['start_id'])} - {format_id(stats['end_id'])}")
    
    except Exception as e:
        logger.error(f"执行失败: {e}")
        exit(1)

