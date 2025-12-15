"""
eval_emg.py - EMG 效果验证工具

功能：
- 在 test 和 hardset 上对比三种方法：
  1. Baseline：只使用模型预测 p(c|x)
  2. 固定 α 融合：使用固定的 α 值进行融合
  3. EMG：使用不确定性自适应的 α(u) 进行融合
- 计算并对比各种指标：Accuracy、F1、Precision、Recall、NLL、ECE 等
- 生成对比表格和可视化图表
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
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    log_loss
)
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入 Day7 的 compute_emg_fusion 函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emg_bucket_search import compute_emg_fusion


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_baseline_predictions(file_path: str) -> Dict[str, Dict]:
    """
    加载 baseline 预测结果
    
    Args:
        file_path: JSONL 文件路径（包含 baseline 预测和 uncertainty）
    
    Returns:
        字典，key 为样本 ID，value 为预测结果
    """
    logger.info(f"加载 baseline 预测结果: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return {}
    
    results = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if item_id is None:
                    continue
                
                # 确保包含必需字段
                if 'pred_probs' not in item or 'uncertainty' not in item:
                    logger.warning(f"样本 {item_id} 缺少必需字段，跳过")
                    continue
                
                results[item_id] = {
                    'id': item_id,
                    'coarse_label': item.get('coarse_label'),
                    'pred_probs': item['pred_probs'],  # [p_non_sensitive, p_sensitive]
                    'uncertainty': item['uncertainty'],  # u 值
                    'pred_label': item.get('pred_label', np.argmax(item['pred_probs']))
                }
            except Exception as e:
                logger.warning(f"解析行失败: {e}")
                continue
    
    logger.info(f"✓ 加载 {len(results)} 条 baseline 预测结果")
    return results


def load_q0_posteriors(file_path: str) -> Dict[str, List[float]]:
    """
    加载 q₀ 后验
    
    Args:
        file_path: JSONL 文件路径（包含 q₀ 后验）
    
    Returns:
        字典，key 为样本 ID，value 为 q₀ 概率 [p_non_sensitive, p_sensitive]
    """
    logger.info(f"加载 q₀ 后验: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return {}
    
    q0_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if item_id is None:
                    continue
                
                if 'q0' not in item:
                    logger.warning(f"样本 {item_id} 缺少 q0 字段，跳过")
                    continue
                
                q0_dict[item_id] = item['q0']  # [p_non_sensitive, p_sensitive]
            except Exception as e:
                logger.warning(f"解析行失败: {e}")
                continue
    
    logger.info(f"✓ 加载 {len(q0_dict)} 条 q₀ 后验")
    return q0_dict


def load_alpha_u_lut(file_path: str) -> Dict[str, List[float]]:
    """
    加载 α(u) 查表
    
    Args:
        file_path: JSON 文件路径（包含 α(u) 查表）
    
    Returns:
        字典，包含 'u' 和 'alpha' 列表
    """
    logger.info(f"加载 α(u) 查表: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lut = json.load(f)
    
    if 'u' not in lut or 'alpha' not in lut:
        logger.error("查表格式错误：缺少 'u' 或 'alpha' 字段")
        return {}
    
    logger.info(f"✓ 加载 {len(lut['u'])} 个查表点")
    return lut


def lookup_alpha(u: float, lut: Dict[str, List[float]]) -> float:
    """
    线性插值查找 α(u)
    
    Args:
        u: 不确定性值
        lut: α(u) 查表
    
    Returns:
        α 值
    """
    u_list = lut['u']
    alpha_list = lut['alpha']
    
    # 边界处理
    if u <= u_list[0]:
        return alpha_list[0]
    if u >= u_list[-1]:
        return alpha_list[-1]
    
    # 线性插值
    return np.interp(u, u_list, alpha_list)


def compute_ece(pred_probs: np.ndarray, true_labels: np.ndarray, n_buckets: int = 10) -> float:
    """
    计算期望校准误差 (ECE)
    
    Args:
        pred_probs: 预测概率数组（类别1的概率）
        true_labels: 真实标签数组
        n_buckets: 分桶数量
    
    Returns:
        ECE 值
    """
    if len(pred_probs) == 0:
        return 0.0
    
    # 等宽分桶
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_ids = np.digitize(pred_probs, bucket_edges) - 1
    bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)
    
    ece = 0.0
    n_total = len(pred_probs)
    
    for bucket_id in range(n_buckets):
        mask = (bucket_ids == bucket_id)
        if np.sum(mask) == 0:
            continue
        
        bucket_probs = pred_probs[mask]
        bucket_labels = true_labels[mask]
        
        n_samples = len(bucket_probs)
        accuracy = np.mean(bucket_labels == 1)  # 实际准确率
        confidence = np.mean(bucket_probs)  # 平均预测概率
        
        gap = abs(accuracy - confidence)
        ece += (n_samples / n_total) * gap
    
    return float(ece)


def evaluate_method(
    baseline_results: Dict[str, Dict],
    q0_dict: Dict[str, List[float]],
    method: str,
    alpha_lut: Optional[Dict[str, List[float]]] = None,
    fixed_alpha: Optional[float] = None,
    knowledge_threshold: Optional[float] = None,
    use_consistency_gating: bool = False
) -> Dict:
    """
    评估一种方法（Baseline、固定α融合、EMG）
    
    Args:
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        method: 方法名称（'baseline', 'fixed_alpha_0.5'等, 'emg'）
        alpha_lut: α(u) 查表（仅 EMG 需要）
        fixed_alpha: 固定 α 值（仅 fixed_alpha 需要）
    
    Returns:
        评估指标字典
    """
    """
    评估一种方法（Baseline、固定α融合、EMG）
    
    Args:
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        method: 方法名称（'baseline', 'fixed_alpha', 'emg'）
        alpha_lut: α(u) 查表（仅 EMG 需要）
        fixed_alpha: 固定 α 值（仅 fixed_alpha 需要）
    
    Returns:
        评估指标字典
    """
    logger.info(f"评估 {method}...")
    
    true_labels = []
    pred_labels = []
    pred_probs = []  # 类别1的概率
    
    for item_id, result in baseline_results.items():
        if result['coarse_label'] is None:
            continue
        
        if item_id not in q0_dict:
            continue
        
        true_label = int(result['coarse_label'])
        p = result['pred_probs']  # [p_non_sensitive, p_sensitive]
        q0 = q0_dict[item_id]  # [p_non_sensitive, p_sensitive]
        
        # 根据方法选择融合方式
        if method == 'baseline':
            p_final = p
        elif method.startswith('fixed_alpha'):
            if fixed_alpha is None:
                raise ValueError("fixed_alpha 方法需要指定 fixed_alpha 参数")
            p_final = compute_emg_fusion(p, q0, fixed_alpha)
        elif method == 'emg':
            if alpha_lut is None:
                raise ValueError("emg 方法需要指定 alpha_lut 参数")
            u = result['uncertainty']
            alpha = lookup_alpha(u, alpha_lut)
            
            # 应用门控机制
            # 优先级1: 知识阈值门控
            if knowledge_threshold is not None:
                max_q0 = max(q0)
                if max_q0 < knowledge_threshold:
                    # 知识太弱，完全信任模型
                    alpha = 1.0
            
            # 优先级2: 一致性门控
            if use_consistency_gating:
                if np.argmax(p) != np.argmax(q0):
                    # 预测不一致，强制使用小alpha（或0）
                    alpha = 0.0
            
            p_final = compute_emg_fusion(p, q0, alpha)
        else:
            raise ValueError(f"未知方法: {method}")
        
        true_labels.append(true_label)
        pred_labels.append(np.argmax(p_final))
        pred_probs.append(p_final[1])  # 类别1的概率
    
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
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'nll': float(nll),
        'ece': float(ece),
        'n_samples': len(true_labels)
    }
    
    logger.info(f"  - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  - F1: {f1:.4f} ({f1*100:.2f}%)")
    logger.info(f"  - NLL: {nll:.4f}")
    logger.info(f"  - ECE: {ece:.4f}")
    
    return metrics


def evaluate_by_uncertainty_slices(
    baseline_results: Dict[str, Dict],
    q0_dict: Dict[str, List[float]],
    alpha_lut: Dict[str, List[float]],
    thresholds: List[float] = [0.1, 0.3],
    knowledge_threshold: Optional[float] = None,
    use_consistency_gating: bool = False
) -> Dict:
    """
    按不确定性切片评估 EMG 效果
    
    Args:
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        alpha_lut: α(u) 查表
        thresholds: u 的阈值列表，例如 [0.1, 0.3] 表示：
            - u < 0.1: 低不确定性
            - 0.1 ≤ u < 0.3: 中等不确定性
            - u ≥ 0.3: 高不确定性
        knowledge_threshold: 知识阈值（可选）
        use_consistency_gating: 是否使用一致性门控
    
    Returns:
        切片评估结果字典
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("按不确定性切片评估")
    logger.info("=" * 60)
    
    slices = {}
    
    # 低不确定性切片
    if len(thresholds) > 0:
        u_max = thresholds[0]
        low_u_samples = {
            id: result for id, result in baseline_results.items()
            if result['uncertainty'] < u_max and id in q0_dict
        }
        
        if len(low_u_samples) > 0:
            logger.info(f"\n【低不确定性切片】u < {u_max} (样本数: {len(low_u_samples)})")
            baseline_metrics = evaluate_method(low_u_samples, q0_dict, 'baseline')
            emg_metrics = evaluate_method(
                low_u_samples, q0_dict, 'emg', alpha_lut=alpha_lut,
                knowledge_threshold=knowledge_threshold,
                use_consistency_gating=use_consistency_gating
            )
            
            slices[f'u_<_{u_max}'] = {
                'baseline': baseline_metrics,
                'emg': emg_metrics,
                'n_samples': len(low_u_samples)
            }
    
    # 中等不确定性切片
    for i in range(len(thresholds) - 1):
        u_min = thresholds[i]
        u_max = thresholds[i + 1]
        
        mid_u_samples = {
            id: result for id, result in baseline_results.items()
            if u_min <= result['uncertainty'] < u_max and id in q0_dict
        }
        
        if len(mid_u_samples) > 0:
            logger.info(f"\n【中等不确定性切片】{u_min} ≤ u < {u_max} (样本数: {len(mid_u_samples)})")
            baseline_metrics = evaluate_method(mid_u_samples, q0_dict, 'baseline')
            emg_metrics = evaluate_method(
                mid_u_samples, q0_dict, 'emg', alpha_lut=alpha_lut,
                knowledge_threshold=knowledge_threshold,
                use_consistency_gating=use_consistency_gating
            )
            
            slices[f'u_{u_min}_{u_max}'] = {
                'baseline': baseline_metrics,
                'emg': emg_metrics,
                'n_samples': len(mid_u_samples)
            }
    
    # 高不确定性切片
    if len(thresholds) > 0:
        u_min = thresholds[-1]
        high_u_samples = {
            id: result for id, result in baseline_results.items()
            if result['uncertainty'] >= u_min and id in q0_dict
        }
        
        if len(high_u_samples) > 0:
            logger.info(f"\n【高不确定性切片】u ≥ {u_min} (样本数: {len(high_u_samples)})")
            baseline_metrics = evaluate_method(high_u_samples, q0_dict, 'baseline')
            emg_metrics = evaluate_method(
                high_u_samples, q0_dict, 'emg', alpha_lut=alpha_lut,
                knowledge_threshold=knowledge_threshold,
                use_consistency_gating=use_consistency_gating
            )
            
            slices[f'u_≥_{u_min}'] = {
                'baseline': baseline_metrics,
                'emg': emg_metrics,
                'n_samples': len(high_u_samples)
            }
            
            # 计算高不确定性切片的改进
            f1_improvement = emg_metrics['f1'] - baseline_metrics['f1']
            nll_improvement = baseline_metrics['nll'] - emg_metrics['nll']
            ece_improvement = baseline_metrics['ece'] - emg_metrics['ece']
            
            logger.info(f"\n高不确定性切片改进:")
            logger.info(f"  - F1: {f1_improvement:+.4f} ({f1_improvement*100:+.2f}%)")
            logger.info(f"  - NLL: {nll_improvement:+.4f} ({nll_improvement/baseline_metrics['nll']*100:+.2f}%)")
            logger.info(f"  - ECE: {ece_improvement:+.4f} ({ece_improvement/baseline_metrics['ece']*100:+.2f}%)")
    
    return slices


def compare_methods(metrics_dict: Dict[str, Dict]) -> Dict:
    """
    对比不同方法的性能
    
    Args:
        metrics_dict: 方法名到指标的字典
    
    Returns:
        对比结果字典
    """
    comparison = {}
    
    if 'baseline' in metrics_dict and 'emg' in metrics_dict:
        baseline = metrics_dict['baseline']
        emg = metrics_dict['emg']
        
        comparison['emg_vs_baseline'] = {
            'f1_improvement': emg['f1'] - baseline['f1'],
            'f1_improvement_percent': ((emg['f1'] - baseline['f1']) / baseline['f1'] * 100) if baseline['f1'] > 0 else 0.0,
            'nll_reduction': baseline['nll'] - emg['nll'],
            'nll_reduction_percent': ((baseline['nll'] - emg['nll']) / baseline['nll'] * 100) if baseline['nll'] > 0 else 0.0,
            'ece_reduction': baseline['ece'] - emg['ece'],
            'ece_reduction_percent': ((baseline['ece'] - emg['ece']) / baseline['ece'] * 100) if baseline['ece'] > 0 else 0.0
        }
    
    # 查找 fixed_alpha 相关的键（可能是 'fixed_alpha' 或 'fixed_alpha_0.5' 等）
    fixed_alpha_key = None
    for key in metrics_dict.keys():
        if key.startswith('fixed_alpha'):
            fixed_alpha_key = key
            break
    
    if fixed_alpha_key and 'emg' in metrics_dict:
        fixed = metrics_dict[fixed_alpha_key]
        emg = metrics_dict['emg']
        
        comparison['emg_vs_fixed_alpha'] = {
            'f1_improvement': emg['f1'] - fixed['f1'],
            'f1_improvement_percent': ((emg['f1'] - fixed['f1']) / fixed['f1'] * 100) if fixed['f1'] > 0 else 0.0,
            'nll_reduction': fixed['nll'] - emg['nll'],
            'nll_reduction_percent': ((fixed['nll'] - emg['nll']) / fixed['nll'] * 100) if fixed['nll'] > 0 else 0.0,
            'ece_reduction': fixed['ece'] - emg['ece'],
            'ece_reduction_percent': ((fixed['ece'] - emg['ece']) / fixed['ece'] * 100) if fixed['ece'] > 0 else 0.0
        }
    
    return comparison


def plot_comparison(metrics_dict: Dict[str, Dict], output_path: str):
    """
    绘制对比图表
    
    Args:
        metrics_dict: 方法名到指标的字典
        output_path: 输出图片路径
    """
    logger.info(f"绘制对比图表: {output_path}")
    
    methods = list(metrics_dict.keys())
    if len(methods) == 0:
        logger.warning("没有数据，无法绘制图表")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 提取数据
    f1_scores = [metrics_dict[m]['f1'] for m in methods]
    nll_scores = [metrics_dict[m]['nll'] for m in methods]
    ece_scores = [metrics_dict[m]['ece'] for m in methods]
    acc_scores = [metrics_dict[m]['accuracy'] for m in methods]
    
    # 图1: F1 对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, f1_scores, color=['#3498db', '#2ecc71', '#e74c3c'][:len(methods)])
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([min(f1_scores) * 0.99, max(f1_scores) * 1.01])
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars1, f1_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 图2: NLL 对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, nll_scores, color=['#3498db', '#2ecc71', '#e74c3c'][:len(methods)])
    ax2.set_ylabel('NLL (Negative Log Likelihood)', fontsize=12)
    ax2.set_title('NLL Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, max(nll_scores) * 1.1])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars2, nll_scores)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nll_scores) * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 图3: ECE 对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, ece_scores, color=['#3498db', '#2ecc71', '#e74c3c'][:len(methods)])
    ax3.set_ylabel('ECE (Expected Calibration Error)', fontsize=12)
    ax3.set_title('ECE Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, max(ece_scores) * 1.1])
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars3, ece_scores)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ece_scores) * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 图4: Accuracy 对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, acc_scores, color=['#3498db', '#2ecc71', '#e74c3c'][:len(methods)])
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylim([min(acc_scores) * 0.99, max(acc_scores) * 1.01])
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars4, acc_scores)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ 图表已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EMG 效果验证工具')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--baseline-file',
        type=str,
        default=None,
        help='Baseline 预测结果文件（包含 uncertainty）'
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
        '--fixed-alpha',
        type=float,
        default=0.5,
        help='固定 α 值（默认: 0.5）'
    )
    parser.add_argument(
        '--hard-file',
        type=str,
        default=None,
        help='困难集 baseline 预测结果文件（可选）'
    )
    parser.add_argument(
        '--hard-q0-file',
        type=str,
        default=None,
        help='困难集 q₀ 后验文件（可选）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录'
    )
    parser.add_argument(
        '--knowledge-threshold-file',
        type=str,
        default=None,
        help='知识阈值文件（JSON，包含best_threshold字段）'
    )
    parser.add_argument(
        '--knowledge-threshold',
        type=float,
        default=None,
        help='知识阈值（直接指定，优先级高于文件）'
    )
    parser.add_argument(
        '--use-consistency-gating',
        action='store_true',
        help='使用一致性门控（argmax(p) != argmax(q0) 时使用 alpha=0）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    output_dir = args.output_dir or config.get('output_dir', './output')
    eval_config = config.get('evaluation', {})
    
    # 解析参数
    baseline_file = args.baseline_file or eval_config.get('baseline_file', 'output/test_with_uncertainty.jsonl')
    q0_file = args.q0_file or eval_config.get('q0_file', 'data/q0_test.jsonl')
    alpha_lut_file = args.alpha_lut_file or eval_config.get('alpha_lut_file', 'output/alpha_u_lut.json')
    fixed_alpha = args.fixed_alpha if args.fixed_alpha is not None else eval_config.get('fixed_alpha', 0.5)
    hard_file = args.hard_file or eval_config.get('hard_file')
    hard_q0_file = args.hard_q0_file or eval_config.get('hard_q0_file')
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("EMG 效果验证")
    logger.info("=" * 60)
    logger.info(f"Baseline 文件: {baseline_file}")
    logger.info(f"q₀ 文件: {q0_file}")
    logger.info(f"α(u) 查表文件: {alpha_lut_file}")
    logger.info(f"固定 α 值: {fixed_alpha}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("")
    
    # 加载数据
    baseline_results = load_baseline_predictions(baseline_file)
    if len(baseline_results) == 0:
        logger.error("无法加载 baseline 预测结果")
        return 1
    
    q0_dict = load_q0_posteriors(q0_file)
    if len(q0_dict) == 0:
        logger.error("无法加载 q₀ 后验")
        return 1
    
    alpha_lut = load_alpha_u_lut(alpha_lut_file)
    if len(alpha_lut) == 0:
        logger.error("无法加载 α(u) 查表")
        return 1
    
    # 加载知识阈值
    knowledge_threshold = args.knowledge_threshold
    if knowledge_threshold is None and args.knowledge_threshold_file:
        threshold_file = args.knowledge_threshold_file
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r', encoding='utf-8') as f:
                threshold_data = json.load(f)
                knowledge_threshold = threshold_data.get('best_threshold')
                logger.info(f"从文件加载知识阈值: {knowledge_threshold:.4f}")
        else:
            logger.warning(f"知识阈值文件不存在: {threshold_file}")
    
    logger.info(f"知识阈值门控: {'启用' if knowledge_threshold is not None else '禁用'}")
    if knowledge_threshold is not None:
        logger.info(f"  阈值: {knowledge_threshold:.4f}")
    logger.info(f"一致性门控: {'启用' if args.use_consistency_gating else '禁用'}")
    logger.info("")
    
    # 评估三种方法
    metrics_dict = {}
    
    # 1. Baseline
    metrics_dict['baseline'] = evaluate_method(
        baseline_results, q0_dict, 'baseline'
    )
    
    # 2. 固定 α 融合
    metrics_dict[f'fixed_alpha_{fixed_alpha}'] = evaluate_method(
        baseline_results, q0_dict, 'fixed_alpha', fixed_alpha=fixed_alpha
    )
    
    # 3. EMG（可能带门控）
    metrics_dict['emg'] = evaluate_method(
        baseline_results, q0_dict, 'emg', 
        alpha_lut=alpha_lut,
        knowledge_threshold=knowledge_threshold,
        use_consistency_gating=args.use_consistency_gating
    )
    
    # 4. 按不确定性切片评估
    slice_results = evaluate_by_uncertainty_slices(
        baseline_results, q0_dict, alpha_lut, thresholds=[0.1, 0.3],
        knowledge_threshold=knowledge_threshold,
        use_consistency_gating=args.use_consistency_gating
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("对比分析")
    logger.info("=" * 60)
    
    # 对比分析
    comparison = compare_methods(metrics_dict)
    
    if 'emg_vs_baseline' in comparison:
        comp = comparison['emg_vs_baseline']
        logger.info("EMG vs Baseline:")
        logger.info(f"  - F1 提升: {comp['f1_improvement']:+.4f} ({comp['f1_improvement_percent']:+.2f}%)")
        logger.info(f"  - NLL 降低: {comp['nll_reduction']:+.4f} ({comp['nll_reduction_percent']:+.2f}%)")
        logger.info(f"  - ECE 降低: {comp['ece_reduction']:+.4f} ({comp['ece_reduction_percent']:+.2f}%)")
    
    if 'emg_vs_fixed_alpha' in comparison:
        comp = comparison['emg_vs_fixed_alpha']
        logger.info("EMG vs 固定 α 融合:")
        logger.info(f"  - F1 提升: {comp['f1_improvement']:+.4f} ({comp['f1_improvement_percent']:+.2f}%)")
        logger.info(f"  - NLL 降低: {comp['nll_reduction']:+.4f} ({comp['nll_reduction_percent']:+.2f}%)")
        logger.info(f"  - ECE 降低: {comp['ece_reduction']:+.4f} ({comp['ece_reduction_percent']:+.2f}%)")
    
    # 处理困难集（如果提供）
    hard_metrics_dict = {}
    if hard_file and hard_q0_file:
        logger.info("")
        logger.info("=" * 60)
        logger.info("评估困难集")
        logger.info("=" * 60)
        
        hard_baseline_results = load_baseline_predictions(hard_file)
        hard_q0_dict = load_q0_posteriors(hard_q0_file)
        
        if len(hard_baseline_results) > 0 and len(hard_q0_dict) > 0:
            hard_metrics_dict['baseline'] = evaluate_method(
                hard_baseline_results, hard_q0_dict, 'baseline'
            )
            hard_metrics_dict[f'fixed_alpha_{fixed_alpha}'] = evaluate_method(
                hard_baseline_results, hard_q0_dict, 'fixed_alpha', fixed_alpha=fixed_alpha
            )
            hard_metrics_dict['emg'] = evaluate_method(
                hard_baseline_results, hard_q0_dict, 'emg', alpha_lut=alpha_lut
            )
    
    # 保存结果
    logger.info("")
    logger.info("=" * 60)
    logger.info("保存结果")
    logger.info("=" * 60)
    
    # 保存 JSON 指标文件
    output_metrics = {
        'test_set': metrics_dict
    }
    if hard_metrics_dict:
        output_metrics['hard_set'] = hard_metrics_dict
    output_metrics['comparison'] = comparison
    # 添加切片评估结果
    if slice_results:
        output_metrics['uncertainty_slices'] = slice_results
    
    metrics_file = os.path.join(output_dir, 'metrics_emg.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(output_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 指标已保存: {metrics_file}")
    
    # 保存 CSV 对比表格
    comparison_rows = []
    for dataset_name, dataset_metrics in output_metrics.items():
        if dataset_name in ['comparison', 'uncertainty_slices']:
            continue
        # 处理普通数据集（test_set, hard_set）
        if isinstance(dataset_metrics, dict):
            for method, metrics in dataset_metrics.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    comparison_rows.append({
                        'method': method,
                        'dataset': dataset_name,
                        'accuracy': metrics['accuracy'],
                        'f1': metrics['f1'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'nll': metrics['nll'],
                        'ece': metrics['ece']
                    })
    
    # 单独处理切片结果
    if 'uncertainty_slices' in output_metrics:
        for slice_name, slice_data in output_metrics['uncertainty_slices'].items():
            if isinstance(slice_data, dict):
                for method in ['baseline', 'emg']:
                    if method in slice_data and isinstance(slice_data[method], dict):
                        metrics = slice_data[method]
                        if 'accuracy' in metrics:
                            comparison_rows.append({
                                'method': method,
                                'dataset': f'slice_{slice_name}',
                                'accuracy': metrics['accuracy'],
                                'f1': metrics['f1'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'nll': metrics['nll'],
                                'ece': metrics['ece']
                            })
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = os.path.join(output_dir, 'emg_comparison_table.csv')
    comparison_df.to_csv(comparison_csv, index=False, encoding='utf-8')
    logger.info(f"✓ 对比表格已保存: {comparison_csv}")
    
    # 绘制对比图表
    chart_file = os.path.join(output_dir, 'emg_comparison_charts.png')
    plot_comparison(metrics_dict, chart_file)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

