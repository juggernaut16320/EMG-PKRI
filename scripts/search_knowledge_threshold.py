"""
search_knowledge_threshold.py - 搜索最优知识阈值

功能：
- 在验证集上搜索最优的知识阈值
- 规则：当 max(q₀) < threshold 时，设置 α=1.0（完全信任模型）
- 输出最优阈值和对应的指标
"""

import os
import sys
import json
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    log_loss
)

# 导入 Day7 的 compute_emg_fusion 函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emg_bucket_search import compute_emg_fusion

# 导入 eval_emg 的工具函数
from eval_emg import (
    load_baseline_predictions,
    load_q0_posteriors,
    load_alpha_u_lut,
    lookup_alpha,
    compute_ece
)

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


def apply_knowledge_threshold_gating(
    p: List[float],
    q0: List[float],
    u: float,
    alpha_lut: Dict[str, List[float]],
    knowledge_threshold: float
) -> Tuple[float, List[float]]:
    """
    应用知识阈值门控的EMG融合
    
    Args:
        p: baseline 预测概率 [p_non_sensitive, p_sensitive]
        q0: q₀ 知识后验 [p_non_sensitive, p_sensitive]
        u: 不确定性值
        alpha_lut: α(u) 查表
        knowledge_threshold: 知识阈值
        
    Returns:
        (使用的alpha值, 融合后的概率)
    """
    max_q0 = max(q0)
    
    if max_q0 < knowledge_threshold:
        # 知识太弱，完全信任模型
        alpha = 1.0
    else:
        # 知识足够强，使用正常的 α(u)
        alpha = lookup_alpha(u, alpha_lut)
    
    p_emg = compute_emg_fusion(p, q0, alpha)
    return alpha, p_emg


def evaluate_with_threshold(
    baseline_results: Dict[str, Dict],
    q0_dict: Dict[str, List[float]],
    alpha_lut: Dict[str, List[float]],
    knowledge_threshold: float
) -> Dict:
    """
    使用给定的知识阈值评估效果
    
    Args:
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        alpha_lut: α(u) 查表
        knowledge_threshold: 知识阈值
        
    Returns:
        评估指标字典
    """
    true_labels = []
    pred_labels = []
    pred_probs = []
    
    for item_id, result in baseline_results.items():
        if result['coarse_label'] is None:
            continue
        
        if item_id not in q0_dict:
            continue
        
        true_label = int(result['coarse_label'])
        p = result['pred_probs']
        q0 = q0_dict[item_id]
        u = result['uncertainty']
        
        # 应用知识阈值门控
        _, p_emg = apply_knowledge_threshold_gating(
            p, q0, u, alpha_lut, knowledge_threshold
        )
        
        true_labels.append(true_label)
        pred_labels.append(np.argmax(p_emg))
        pred_probs.append(p_emg[1])  # 类别1的概率
    
    if len(true_labels) == 0:
        logger.warning("没有有效样本，无法计算指标")
        return {}
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)
    
    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    # 计算 NLL
    try:
        nll = log_loss(true_labels, pred_probs, labels=[0, 1])
    except Exception as e:
        logger.warning(f"计算 NLL 失败: {e}，使用 inf")
        nll = float('inf')
    
    # 计算 ECE
    ece = compute_ece(pred_probs, true_labels, n_buckets=10)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'nll': float(nll),
        'ece': float(ece),
        'n_samples': len(true_labels)
    }


def search_knowledge_threshold(
    baseline_results: Dict[str, Dict],
    q0_dict: Dict[str, List[float]],
    alpha_lut: Dict[str, List[float]],
    threshold_grid: List[float] = None,
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    搜索最优的知识阈值
    
    Args:
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        alpha_lut: α(u) 查表
        threshold_grid: 候选阈值列表
        metric: 优化指标（'f1' 或 'nll'）
        
    Returns:
        (最优阈值, 所有阈值的指标字典)
    """
    if threshold_grid is None:
        threshold_grid = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    logger.info(f"搜索最优知识阈值（指标: {metric}，候选阈值: {threshold_grid}）...")
    
    threshold_metrics = {}
    best_threshold = None
    best_value = -float('inf') if metric == 'f1' else float('inf')
    
    for threshold in threshold_grid:
        logger.info(f"评估阈值 {threshold:.2f}...")
        metrics = evaluate_with_threshold(
            baseline_results, q0_dict, alpha_lut, threshold
        )
        
        if not metrics:
            logger.warning(f"阈值 {threshold:.2f} 评估失败，跳过")
            continue
        
        threshold_metrics[threshold] = metrics
        
        # 选择最优值
        value = metrics[metric]
        if metric == 'f1':
            if value > best_value:
                best_value = value
                best_threshold = threshold
        else:  # metric == 'nll'
            if value < best_value:
                best_value = value
                best_threshold = threshold
        
        logger.info(
            f"  阈值 {threshold:.2f}: F1={metrics['f1']:.4f}, "
            f"NLL={metrics['nll']:.4f}, ECE={metrics['ece']:.4f}"
        )
    
    if best_threshold is None:
        raise ValueError("未能找到最优阈值")
    
    logger.info(
        f"✓ 最优阈值 = {best_threshold:.4f} ({metric}={best_value:.4f})"
    )
    
    return best_threshold, threshold_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='搜索最优知识阈值')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--dev-file',
        type=str,
        default=None,
        help='验证集文件（包含 baseline 预测和 uncertainty）'
    )
    parser.add_argument(
        '--q0-file',
        type=str,
        default=None,
        help='q₀ 后验文件'
    )
    parser.add_argument(
        '--alpha-lut-file',
        type=str,
        default=None,
        help='α(u) 查表文件'
    )
    parser.add_argument(
        '--threshold-grid',
        type=float,
        nargs='+',
        default=None,
        help='候选阈值列表（默认: 0.4 0.5 0.6 0.7 0.8 0.9）'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='f1',
        choices=['f1', 'nll'],
        help='优化指标（默认: f1）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    output_dir = args.output_dir or config.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 解析参数
    dev_file = args.dev_file or config.get('evaluation', {}).get(
        'dev_baseline_file', 'output/dev_with_uncertainty.jsonl'
    )
    q0_file = args.q0_file or config.get('evaluation', {}).get(
        'dev_q0_file', 'data/q0_dev.jsonl'
    )
    alpha_lut_file = args.alpha_lut_file or config.get('evaluation', {}).get(
        'alpha_lut_file', 'output/alpha_u_lut.json'
    )
    
    threshold_grid = args.threshold_grid or [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 加载数据
    logger.info("加载数据...")
    baseline_results = load_baseline_predictions(dev_file)
    q0_dict = load_q0_posteriors(q0_file)
    alpha_lut = load_alpha_u_lut(alpha_lut_file)
    
    logger.info(f"加载了 {len(baseline_results)} 个 baseline 预测结果")
    logger.info(f"加载了 {len(q0_dict)} 个 q₀ 后验")
    
    # 搜索最优阈值
    best_threshold, threshold_metrics = search_knowledge_threshold(
        baseline_results, q0_dict, alpha_lut, threshold_grid, args.metric
    )
    
    # 保存结果
    output_data = {
        'best_threshold': float(best_threshold),
        'metric': args.metric,
        'threshold_grid': [float(t) for t in threshold_grid],
        'threshold_metrics': {
            str(k): v for k, v in threshold_metrics.items()
        },
        'best_metrics': threshold_metrics[best_threshold]
    }
    
    output_file = os.path.join(output_dir, 'knowledge_threshold.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ 结果已保存: {output_file}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("知识阈值搜索结果")
    print("=" * 60)
    print(f"最优阈值: {best_threshold:.4f}")
    print(f"优化指标: {args.metric}")
    print(f"\n最优阈值对应的指标:")
    best_metrics = threshold_metrics[best_threshold]
    print(f"  F1: {best_metrics['f1']:.4f}")
    print(f"  NLL: {best_metrics['nll']:.4f}")
    print(f"  ECE: {best_metrics['ece']:.4f}")
    print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"  样本数: {best_metrics['n_samples']}")
    
    print(f"\n所有阈值结果:")
    print(f"{'阈值':<8} {'F1':<10} {'NLL':<10} {'ECE':<10}")
    print("-" * 40)
    for threshold in sorted(threshold_metrics.keys()):
        m = threshold_metrics[threshold]
        marker = " ←" if threshold == best_threshold else ""
        print(
            f"{threshold:<8.2f} {m['f1']:<10.4f} "
            f"{m['nll']:<10.4f} {m['ece']:<10.4f}{marker}"
        )
    
    return best_threshold, threshold_metrics


if __name__ == '__main__':
    main()

