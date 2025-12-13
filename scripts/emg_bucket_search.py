"""
emg_bucket_search.py - EMG 分桶 α 搜索

功能：
1. 对每个不确定性 u bucket，枚举不同的融合权重 α
2. 在 dev 集上计算 NLL / F1，选出每个 bucket 的最优 α*
3. 为 Day8 的 PAV 保序回归提供数据
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
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from sklearn.metrics import f1_score, log_loss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ 核心函数 ============

def load_baseline_predictions(dev_file: str, uncertainty_file: Optional[str] = None) -> Dict[str, Dict]:
    """
    加载 baseline 预测结果
    
    Args:
        dev_file: dev 集文件路径
        uncertainty_file: 可选，包含 uncertainty 信息的文件路径（如果 dev_file 不包含 uncertainty）
    
    Returns:
        字典，key 为样本 ID，value 为预测结果
    """
    logger.info(f"加载 baseline 预测结果: {dev_file}")
    
    if not os.path.exists(dev_file):
        logger.error(f"文件不存在: {dev_file}")
        return {}
    
    results = {}
    missing_pred_probs = []
    missing_uncertainty = []
    
    # 首先尝试从 dev_file 加载
    with open(dev_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if not item_id:
                    continue
                
                result = {
                    'id': item_id,
                    'coarse_label': item.get('coarse_label')
                }
                
                # 检查 pred_probs
                if 'pred_probs' in item:
                    result['pred_probs'] = item['pred_probs']
                else:
                    missing_pred_probs.append(item_id)
                    continue
                
                # 检查 uncertainty
                if 'uncertainty' in item:
                    result['uncertainty'] = item['uncertainty']
                else:
                    missing_uncertainty.append(item_id)
                
                results[item_id] = result
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效JSON行: {e}")
                continue
    
    # 如果有缺失的 uncertainty，尝试从 uncertainty_file 加载
    if missing_uncertainty and uncertainty_file and os.path.exists(uncertainty_file):
        logger.info(f"尝试从 {uncertainty_file} 加载 uncertainty 信息...")
        # 注意：uncertainty_file 通常是 CSV，不包含详细结果
        # 这里我们假设如果 dev_file 没有 uncertainty，需要重新计算
        logger.warning(f"dev_file 中 {len(missing_uncertainty)} 个样本缺少 uncertainty，需要重新计算")
    
    if missing_pred_probs:
        logger.warning(f"有 {len(missing_pred_probs)} 个样本缺少 pred_probs，这些样本将被跳过")
        logger.warning("提示：请先运行 uncertainty_analysis.py 生成包含 pred_probs 和 uncertainty 的结果")
    
    if missing_uncertainty:
        logger.warning(f"有 {len(missing_uncertainty)} 个样本缺少 uncertainty，这些样本将被跳过")
        logger.warning("提示：请先运行 uncertainty_analysis.py 生成包含 uncertainty 的结果")
    
    # 过滤掉缺少必需字段的样本
    valid_results = {
        item_id: result 
        for item_id, result in results.items() 
        if 'pred_probs' in result and 'uncertainty' in result
    }
    
    logger.info(f"✓ 加载 {len(valid_results)} 条有效的 baseline 预测结果")
    return valid_results


def load_q0_posteriors(q0_file: str) -> Dict[str, List[float]]:
    """
    加载 q₀ 后验
    
    Args:
        q0_file: q₀ 后验文件路径
    
    Returns:
        字典，key 为样本 ID，value 为 q₀ 概率 [p_non_sensitive, p_sensitive]
    """
    logger.info(f"加载 q₀ 后验: {q0_file}")
    
    if not os.path.exists(q0_file):
        logger.error(f"文件不存在: {q0_file}")
        return {}
    
    q0_dict = {}
    with open(q0_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if not item_id:
                    logger.warning("跳过没有 ID 的样本")
                    continue
                
                if 'q0' not in item:
                    logger.warning(f"样本 {item_id} 缺少 q0 字段，跳过")
                    continue
                
                q0_dict[item_id] = item['q0']  # [p_non_sensitive, p_sensitive]
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效JSON行: {e}")
                continue
    
    logger.info(f"✓ 加载 {len(q0_dict)} 条 q₀ 后验")
    return q0_dict


def load_uncertainty_buckets(buckets_file: str) -> pd.DataFrame:
    """
    加载不确定性分桶信息
    
    Args:
        buckets_file: 分桶结果文件路径
    
    Returns:
        DataFrame，包含 bucket_id, u_min, u_max, u_mean 等
    """
    logger.info(f"加载不确定性分桶信息: {buckets_file}")
    
    if not os.path.exists(buckets_file):
        logger.error(f"文件不存在: {buckets_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(buckets_file)
        logger.info(f"✓ 加载 {len(df)} 个 bucket")
        return df
    except Exception as e:
        logger.error(f"加载分桶信息失败: {e}")
        return pd.DataFrame()


def compute_emg_fusion(
    p: List[float],
    q: List[float],
    alpha: float
) -> List[float]:
    """
    计算 EMG 融合概率
    
    Args:
        p: baseline 预测概率 [p_non_sensitive, p_sensitive]
        q: q₀ 知识后验 [p_non_sensitive, p_sensitive]
        alpha: 融合权重
    
    Returns:
        融合后的概率 [p_non_sensitive, p_sensitive]
    """
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # 确保概率和为1
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # EMG 融合公式: p_emg = α × p + (1 - α) × q
    p_emg = alpha * p + (1 - alpha) * q
    
    # 归一化（确保概率和为1）
    p_emg = p_emg / np.sum(p_emg) if np.sum(p_emg) > 0 else p_emg
    
    # 确保概率在 [0, 1] 范围内
    p_emg = np.clip(p_emg, 0.0, 1.0)
    p_emg = p_emg / np.sum(p_emg)  # 再次归一化
    
    return p_emg.tolist()


def assign_samples_to_buckets(
    baseline_results: Dict[str, Dict],
    bucket_df: pd.DataFrame
) -> Dict[int, List[str]]:
    """
    将样本分配到对应的 bucket
    
    Args:
        baseline_results: baseline 预测结果
        bucket_df: 分桶信息 DataFrame
    
    Returns:
        字典，key 为 bucket_id，value 为样本 ID 列表
    """
    logger.info("分配样本到 bucket...")
    
    bucket_samples = defaultdict(list)
    
    for item_id, result in baseline_results.items():
        uncertainty = result.get('uncertainty')
        if uncertainty is None:
            continue
        
        # 找到对应的 bucket
        for _, row in bucket_df.iterrows():
            u_min = row['u_min']
            u_max = row['u_max']
            bucket_id = int(row['bucket_id'])
            
            # 处理边界情况：u_min <= uncertainty < u_max，或者 uncertainty == 1.0 且 u_max == 1.0
            if (u_min <= uncertainty < u_max) or (uncertainty == 1.0 and abs(u_max - 1.0) < 1e-6):
                bucket_samples[bucket_id].append(item_id)
                break
    
    total_assigned = sum(len(samples) for samples in bucket_samples.values())
    logger.info(f"✓ 分配 {total_assigned} 个样本到 {len(bucket_samples)} 个 bucket")
    
    return bucket_samples


def compute_metrics_for_alpha(
    samples: List[str],
    baseline_results: Dict[str, Dict],
    q0_dict: Dict[str, List[float]],
    alpha: float
) -> Tuple[float, float]:
    """
    计算给定 α 值下的 F1 和 NLL
    
    Args:
        samples: 样本 ID 列表
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        alpha: 融合权重
    
    Returns:
        (F1 分数, NLL)
    """
    true_labels = []
    pred_labels = []
    pred_probs = []
    
    for item_id in samples:
        if item_id not in baseline_results:
            continue
        if item_id not in q0_dict:
            continue
        
        result = baseline_results[item_id]
        q0 = q0_dict[item_id]
        
        # 获取 baseline 预测概率
        p = result['pred_probs']
        
        # 计算 EMG 融合概率
        p_emg = compute_emg_fusion(p, q0, alpha)
        
        # 获取真实标签
        true_label = result.get('coarse_label')
        if true_label is None:
            continue
        
        true_labels.append(int(true_label))
        pred_labels.append(np.argmax(p_emg))
        pred_probs.append(p_emg[1])  # 敏感类别的概率
    
    if len(true_labels) == 0:
        return 0.0, float('inf')
    
    # 计算 F1
    f1 = f1_score(true_labels, pred_labels, average='binary', zero_division=0)
    
    # 计算 NLL
    try:
        nll = log_loss(true_labels, pred_probs, labels=[0, 1])
    except Exception as e:
        logger.warning(f"计算 NLL 失败: {e}，使用 inf")
        nll = float('inf')
    
    return float(f1), float(nll)


def search_alpha_for_bucket(
    bucket_id: int,
    samples: List[str],
    baseline_results: Dict[str, Dict],
    q0_dict: Dict[str, List[float]],
    alpha_grid: List[float],
    metric: str = 'f1'
) -> Tuple[float, Dict]:
    """
    为单个 bucket 搜索最优 α
    
    Args:
        bucket_id: bucket ID
        samples: bucket 内的样本 ID 列表
        baseline_results: baseline 预测结果
        q0_dict: q₀ 后验字典
        alpha_grid: α 网格
        metric: 优化指标（'f1' 或 'nll'）
    
    Returns:
        (最优 α, 所有 α 的指标字典)
    """
    logger.info(f"搜索 bucket {bucket_id} 的最优 α（样本数: {len(samples)}）...")
    
    alpha_metrics = {}
    best_alpha = None
    best_value = -float('inf') if metric == 'f1' else float('inf')
    
    for alpha in alpha_grid:
        f1, nll = compute_metrics_for_alpha(samples, baseline_results, q0_dict, alpha)
        
        alpha_metrics[alpha] = {
            'f1': f1,
            'nll': nll
        }
        
        # 选择最优值
        if metric == 'f1':
            if f1 > best_value:
                best_value = f1
                best_alpha = alpha
        else:  # metric == 'nll'
            if nll < best_value:
                best_value = nll
                best_alpha = alpha
        
        logger.debug(f"  α={alpha:.2f}: F1={f1:.4f}, NLL={nll:.4f}")
    
    logger.info(f"✓ bucket {bucket_id} 最优 α*={best_alpha:.2f} ({metric}={best_value:.4f})")
    
    return best_alpha, alpha_metrics


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EMG 分桶 α 搜索')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='数据目录（默认从config读取）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认从config读取）'
    )
    parser.add_argument(
        '--dev-file',
        type=str,
        default=None,
        help='dev 集文件（需要包含 pred_probs 和 uncertainty，或先运行 uncertainty_analysis.py）'
    )
    parser.add_argument(
        '--baseline-predictions-file',
        type=str,
        default=None,
        help='baseline 预测结果文件（可选，如果 dev_file 不包含 pred_probs）'
    )
    parser.add_argument(
        '--q0-file',
        type=str,
        default=None,
        help='q₀ 后验文件（默认: q0_dev.jsonl）'
    )
    parser.add_argument(
        '--uncertainty-file',
        type=str,
        default=None,
        help='不确定性分桶文件（默认: uncertainty_buckets.csv）'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='输出文件（默认: bucket_alpha_star.csv）'
    )
    parser.add_argument(
        '--alpha-grid',
        type=float,
        nargs='+',
        default=None,
        help='α 网格（默认从config读取）'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=None,
        choices=['f1', 'nll'],
        help='优化指标（默认从config读取）'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        config = {}
    
    # 获取参数
    data_dir = args.data_dir or config.get('data_dir', './data')
    output_dir = args.output_dir or config.get('output_dir', './output')
    
    # emg_bucket_search 配置
    emg_config = config.get('emg_bucket_search', {})
    alpha_grid = args.alpha_grid if args.alpha_grid is not None else emg_config.get('alpha_grid', [0, 0.25, 0.5, 0.75, 1.0])
    metric = args.metric if args.metric is not None else emg_config.get('metric', 'f1')
    
    # 文件路径
    # 优先使用 uncertainty_analysis.py 的输出文件
    default_dev_file = os.path.join(output_dir, 'dev_with_uncertainty.jsonl')
    if not os.path.exists(default_dev_file):
        default_dev_file = os.path.join(data_dir, 'dev.jsonl')
    
    dev_file = args.dev_file or default_dev_file
    q0_file = args.q0_file or os.path.join(data_dir, 'q0_dev.jsonl')
    uncertainty_file = args.uncertainty_file or os.path.join(output_dir, 'uncertainty_buckets.csv')
    output_file = args.output_file or os.path.join(output_dir, 'bucket_alpha_star.csv')
    
    logger.info("=" * 60)
    logger.info("开始 EMG 分桶 α 搜索")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Dev 文件: {dev_file}")
    logger.info(f"Q₀ 文件: {q0_file}")
    logger.info(f"不确定性分桶文件: {uncertainty_file}")
    logger.info(f"α 网格: {alpha_grid}")
    logger.info(f"优化指标: {metric}")
    logger.info("")
    
    # 加载数据
    # 如果指定了 baseline_predictions_file，使用它；否则使用 dev_file
    baseline_file = args.baseline_predictions_file if args.baseline_predictions_file else dev_file
    baseline_results = load_baseline_predictions(baseline_file)
    if not baseline_results:
        logger.error("无法加载 baseline 预测结果")
        logger.error("提示：请先运行 uncertainty_analysis.py 生成包含 pred_probs 和 uncertainty 的结果")
        return 1
    
    q0_dict = load_q0_posteriors(q0_file)
    if not q0_dict:
        logger.error("无法加载 q₀ 后验")
        return 1
    
    bucket_df = load_uncertainty_buckets(uncertainty_file)
    if len(bucket_df) == 0:
        logger.error("无法加载不确定性分桶信息")
        return 1
    
    # 检查样本 ID 匹配
    baseline_ids = set(baseline_results.keys())
    q0_ids = set(q0_dict.keys())
    common_ids = baseline_ids & q0_ids
    
    logger.info(f"Baseline 样本数: {len(baseline_ids)}")
    logger.info(f"Q₀ 样本数: {len(q0_ids)}")
    logger.info(f"共同样本数: {len(common_ids)}")
    
    if len(common_ids) == 0:
        logger.error("没有找到匹配的样本 ID")
        return 1
    
    if len(common_ids) < len(baseline_ids) * 0.9:
        logger.warning(f"只有 {len(common_ids)}/{len(baseline_ids)} 样本匹配，可能影响结果")
    
    logger.info("")
    
    # 分配样本到 bucket
    bucket_samples = assign_samples_to_buckets(baseline_results, bucket_df)
    
    if len(bucket_samples) == 0:
        logger.error("没有样本被分配到 bucket")
        return 1
    
    logger.info("")
    
    # 对每个 bucket 搜索最优 α
    results = []
    
    for _, row in bucket_df.iterrows():
        bucket_id = int(row['bucket_id'])
        
        if bucket_id not in bucket_samples:
            logger.warning(f"Bucket {bucket_id} 没有样本，跳过")
            continue
        
        samples = bucket_samples[bucket_id]
        
        # 搜索最优 α
        alpha_star, alpha_metrics = search_alpha_for_bucket(
            bucket_id,
            samples,
            baseline_results,
            q0_dict,
            alpha_grid,
            metric
        )
        
        # 获取最优指标
        best_metrics = alpha_metrics[alpha_star]
        
        # 构建结果行
        result_row = {
            'bucket_id': bucket_id,
            'u_min': row['u_min'],
            'u_max': row['u_max'],
            'u_mean': row['u_mean'],
            'n_samples': len(samples),
            'alpha_star': alpha_star,
            'f1_at_alpha_star': best_metrics['f1'],
            'nll_at_alpha_star': best_metrics['nll']
        }
        
        # 添加所有 α 的指标
        for alpha in alpha_grid:
            alpha_key = f"alpha_{alpha}".replace('.', '_')
            metrics = alpha_metrics[alpha]
            result_row[f'{alpha_key}_f1'] = metrics['f1']
            result_row[f'{alpha_key}_nll'] = metrics['nll']
        
        results.append(result_row)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("搜索完成")
    logger.info("=" * 60)
    logger.info(f"输出文件: {output_file}")
    logger.info("")
    logger.info("最优 α* 统计:")
    for _, row in results_df.iterrows():
        logger.info(f"  Bucket {int(row['bucket_id'])} (u={row['u_mean']:.3f}): α*={row['alpha_star']:.2f}, F1={row['f1_at_alpha_star']:.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

