"""
q0_hyperparameter_search.py - q0参数调优搜索

功能：
1. 网格搜索q0的最优参数组合
2. 评估每个参数组合的q0质量（F1、Precision、Recall）
3. 输出最优参数配置和排名结果
"""

import os
import sys
import json
import yaml
import logging
import argparse
import tempfile
import shutil
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入q0_builder
from q0_builder import (
    load_lexicon, 
    build_automaton_cache,
    process_dataset
)

# 导入eval_q0的评估函数
# 注意：eval_q0是一个脚本，需要直接调用其函数
import eval_q0

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_param_combinations(mode: str = 'fast') -> List[Dict[str, Any]]:
    """
    生成参数组合
    
    Args:
        mode: 搜索模式
            - 'fast': 简化搜索空间（12种组合，快速验证）
            - 'refined': 精细搜索（25种组合，围绕最优值细化）
            - 'full': 完整搜索空间（384种组合，全面探索）
    
    Returns:
        参数组合列表
    """
    if mode == 'fast':
        # 简化搜索空间：只搜索关键参数
        param_grid = {
            'min_matches_for_sensitive': [1, 2],
            'max_sensitive_prob': [0.65, 0.75, 0.85],
            'base_sensitive_prob': [0.1, 0.15],
        }
    elif mode == 'refined':
        # 精细搜索：围绕已找到的最优值，范围缩小但细粒度更高
        param_grid = {
            'min_matches_for_sensitive': [1],  # 固定为1（已确认最优）
            'max_sensitive_prob': [0.80, 0.82, 0.85, 0.87, 0.90],  # 围绕0.85，步长0.02
            'base_sensitive_prob': [0.12, 0.13, 0.15, 0.17, 0.18],  # 围绕0.15，步长0.01-0.02
        }
    else:  # mode == 'full'
        # 完整搜索空间
        param_grid = {
            'min_matches_for_sensitive': [1, 2],
            'max_sensitive_prob': [0.65, 0.75, 0.85, 0.95],
            'base_sensitive_prob': [0.05, 0.1, 0.15, 0.2],
            'politics_weight': [0.5, 0.6, 0.7, 0.8],
            'abuse_weight': [0.4, 0.5, 0.6, 0.7],
        }
    
    # 生成所有组合
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = []
    
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        # 设置默认值
        params.setdefault('porn_weight', 1.0)
        params.setdefault('politics_weight', 0.6)
        params.setdefault('abuse_weight', 0.5)
        combinations.append(params)
    
    logger.info(f"生成了 {len(combinations)} 种参数组合（模式: {mode}）")
    return combinations


def evaluate_q0_with_params(
    params: Dict[str, Any],
    dataset_file: str,
    baseline_file: str,
    lexicon_dir: str,
    temp_output_file: str
) -> Dict[str, float]:
    """
    使用给定参数评估q0质量
    
    Args:
        params: q0参数字典
        dataset_file: 数据集文件
        baseline_file: baseline文件（包含真实标签）
        lexicon_dir: 词表目录
        temp_output_file: 临时输出文件路径
    
    Returns:
        评估指标字典
    """
    # 加载词表
    lexicons = {}
    for category in ['porn', 'politics', 'abuse']:
        lexicon_path = os.path.join(lexicon_dir, f'{category}.txt')
        if os.path.exists(lexicon_path):
            lexicons[category] = load_lexicon(lexicon_path)
    
    if not lexicons:
        logger.error("没有找到任何词表")
        return {}
    
    # 构建自动机缓存
    automaton_cache = build_automaton_cache(lexicons)
    
    # 使用给定参数生成q0
    try:
        process_dataset(
            input_path=dataset_file,
            output_path=temp_output_file,
            lexicons=lexicons,
            regex_patterns=[],
            use_regex=False,
            porn_weight=params.get('porn_weight', 1.0),
            politics_weight=params.get('politics_weight', 0.6),
            abuse_weight=params.get('abuse_weight', 0.5),
            base_sensitive_prob=params.get('base_sensitive_prob', 0.1),
            max_sensitive_prob=params.get('max_sensitive_prob', 0.75),
            min_matches_for_sensitive=params.get('min_matches_for_sensitive', 1),
            include_details=False,
            automaton_cache=automaton_cache
        )
        
        # 评估q0质量
        result = eval_q0.evaluate_q0(temp_output_file, baseline_file)
        if result is None:
            return {}
        
        return result
    except Exception as e:
        logger.error(f"评估参数组合失败: {params}, 错误: {e}")
        return {}


def search_optimal_params(
    dataset_file: str,
    baseline_file: str,
    lexicon_dir: str,
    mode: str = 'fast',
    output_dir: str = './output'
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    搜索最优参数
    
    Args:
        dataset_file: 数据集文件（dev集）
        baseline_file: baseline文件（包含真实标签）
        lexicon_dir: 词表目录
        mode: 搜索模式（'fast' 或 'full'）
        output_dir: 输出目录
    
    Returns:
        (最优参数字典, 所有结果DataFrame)
    """
    logger.info("=" * 60)
    logger.info("q0参数调优搜索")
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_file}")
    logger.info(f"搜索模式: {mode}")
    logger.info("")
    
    # 生成参数组合
    param_combinations = generate_param_combinations(mode)
    
    # 创建临时输出文件
    os.makedirs(output_dir, exist_ok=True)
    temp_output_file = os.path.join(output_dir, 'q0_temp_eval.jsonl')
    
    # 评估每个参数组合
    results = []
    best_params = None
    best_f1 = -float('inf')
    
    total = len(param_combinations)
    for idx, params in enumerate(param_combinations, 1):
        logger.info(f"[{idx}/{total}] 评估参数组合: {params}")
        
        metrics = evaluate_q0_with_params(
            params, dataset_file, baseline_file, lexicon_dir, temp_output_file
        )
        
        if not metrics:
            logger.warning(f"  参数组合评估失败，跳过")
            continue
        
        # 记录结果
        result = {
            **params,
            'f1': metrics.get('f1', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'avg_p_sensitive': metrics.get('avg_p_sensitive', 0.0),
        }
        results.append(result)
        
        # 更新最优参数（基于F1）
        if metrics.get('f1', 0.0) > best_f1:
            best_f1 = metrics.get('f1', 0.0)
            best_params = params
        
        logger.info(
            f"  F1={metrics.get('f1', 0.0):.4f}, "
            f"Precision={metrics.get('precision', 0.0):.4f}, "
            f"Recall={metrics.get('recall', 0.0):.4f}"
        )
        logger.info("")
    
    # 清理临时文件
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        logger.error("没有成功评估任何参数组合")
        return {}, pd.DataFrame()
    
    # 按F1排序
    results_df = results_df.sort_values('f1', ascending=False).reset_index(drop=True)
    
    # 输出最优参数
    logger.info("=" * 60)
    logger.info("搜索完成")
    logger.info("=" * 60)
    logger.info(f"评估了 {len(results_df)} 种参数组合")
    logger.info(f"最优参数（F1={best_f1:.4f}）:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("")
    
    # 显示Top 5结果
    logger.info("Top 5 参数组合:")
    top5 = results_df.head(5)
    for idx, row in top5.iterrows():
        logger.info(
            f"  {idx+1}. F1={row['f1']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"参数: {dict(row[['min_matches_for_sensitive', 'max_sensitive_prob', 'base_sensitive_prob']])}"
        )
    
    return best_params, results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='q0参数调优搜索')
    parser.add_argument(
        '--dataset-file',
        type=str,
        required=True,
        help='数据集文件（dev集）'
    )
    parser.add_argument(
        '--baseline-file',
        type=str,
        required=True,
        help='Baseline文件（包含真实标签）'
    )
    parser.add_argument(
        '--lexicon-dir',
        type=str,
        default='configs/lexicons',
        help='词表目录'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='fast',
        choices=['fast', 'refined', 'full'],
        help='搜索模式：fast（快速，12种组合）、refined（精细，25种组合）或 full（完整，384种组合）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='输出目录'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='结果CSV输出文件（默认: output/q0_param_search_results.csv）'
    )
    parser.add_argument(
        '--output-best-config',
        type=str,
        default=None,
        help='最优参数JSON输出文件（默认: output/q0_best_params.json）'
    )
    
    args = parser.parse_args()
    
    # 搜索最优参数
    best_params, results_df = search_optimal_params(
        dataset_file=args.dataset_file,
        baseline_file=args.baseline_file,
        lexicon_dir=args.lexicon_dir,
        mode=args.mode,
        output_dir=args.output_dir
    )
    
    if len(results_df) == 0:
        logger.error("搜索失败")
        return 1
    
    # 保存结果
    output_csv = args.output_csv or os.path.join(args.output_dir, 'q0_param_search_results.csv')
    results_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"✓ 结果已保存: {output_csv}")
    
    # 保存最优参数
    output_config = args.output_best_config or os.path.join(args.output_dir, 'q0_best_params.json')
    with open(output_config, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 最优参数已保存: {output_config}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

