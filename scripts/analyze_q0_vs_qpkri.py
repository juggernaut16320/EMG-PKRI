"""
分析 q₀ 和 q_PKRI 的差异，定位EMG融合效果差异的原因
"""
import json
import numpy as np
import pandas as pd
import argparse
import os
import sys
import logging
from typing import Dict, List, Tuple
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_q0(file_path: str) -> Dict[str, List[float]]:
    """加载q₀后验"""
    q0_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q0_dict[item['id']] = item['q0']
    return q0_dict


def load_qpkri(file_path: str) -> Dict[str, Tuple[List[float], float]]:
    """加载q_PKRI后验和可信度"""
    qpkri_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qpkri_dict[item['id']] = (item['qpkri'], item.get('confidence', 0.0))
    return qpkri_dict


def load_baseline(file_path: str) -> Dict[str, Dict]:
    """加载baseline预测结果"""
    baseline_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            baseline_dict[item['id']] = {
                'pred_probs': item['pred_probs'],
                'uncertainty': item['uncertainty'],
                'coarse_label': item.get('coarse_label')
            }
    return baseline_dict


def analyze_distributions(
    q0_dict: Dict[str, List[float]],
    qpkri_dict: Dict[str, Tuple[List[float], float]],
    baseline_dict: Dict[str, Dict]
) -> Dict:
    """分析q₀和q_PKRI的分布差异"""
    logger.info("=" * 60)
    logger.info("1. 后验概率分布分析")
    logger.info("=" * 60)
    
    q0_probs = []
    qpkri_probs = []
    confidences = []
    disagreements = []
    q0_confidence = []
    qpkri_confidence = []
    
    for item_id in q0_dict:
        if item_id not in qpkri_dict or item_id not in baseline_dict:
            continue
        
        q0 = q0_dict[item_id]
        qpkri, confidence = qpkri_dict[item_id]
        baseline = baseline_dict[item_id]
        
        q0_max = max(q0)
        qpkri_max = max(qpkri)
        p_max = max(baseline['pred_probs'])
        
        q0_probs.append(q0_max)
        qpkri_probs.append(qpkri_max)
        confidences.append(confidence)
        q0_confidence.append(q0_max)  # 使用max作为"置信度"的代理
        
        # 检查预测一致性
        q0_pred = np.argmax(q0)
        qpkri_pred = np.argmax(qpkri)
        p_pred = np.argmax(baseline['pred_probs'])
        
        if q0_pred != qpkri_pred:
            disagreements.append({
                'id': item_id,
                'q0_pred': q0_pred,
                'qpkri_pred': qpkri_pred,
                'p_pred': p_pred,
                'q0_max': q0_max,
                'qpkri_max': qpkri_max,
                'confidence': confidence,
                'uncertainty': baseline['uncertainty']
            })
    
    q0_probs = np.array(q0_probs)
    qpkri_probs = np.array(qpkri_probs)
    confidences = np.array(confidences)
    
    logger.info(f"\n样本数: {len(q0_probs)}")
    logger.info(f"\n【q₀ 分布统计】")
    logger.info(f"  平均max概率: {np.mean(q0_probs):.4f}")
    logger.info(f"  中位数: {np.median(q0_probs):.4f}")
    logger.info(f"  标准差: {np.std(q0_probs):.4f}")
    logger.info(f"  最小值: {np.min(q0_probs):.4f}")
    logger.info(f"  最大值: {np.max(q0_probs):.4f}")
    logger.info(f"  >0.9的比例: {np.mean(q0_probs > 0.9):.2%}")
    logger.info(f"  >0.7的比例: {np.mean(q0_probs > 0.7):.2%}")
    logger.info(f"  <0.3的比例: {np.mean(q0_probs < 0.3):.2%}")
    
    logger.info(f"\n【q_PKRI 分布统计】")
    logger.info(f"  平均max概率: {np.mean(qpkri_probs):.4f}")
    logger.info(f"  中位数: {np.median(qpkri_probs):.4f}")
    logger.info(f"  标准差: {np.std(qpkri_probs):.4f}")
    logger.info(f"  最小值: {np.min(qpkri_probs):.4f}")
    logger.info(f"  最大值: {np.max(qpkri_probs):.4f}")
    logger.info(f"  >0.9的比例: {np.mean(qpkri_probs > 0.9):.2%}")
    logger.info(f"  >0.7的比例: {np.mean(qpkri_probs > 0.7):.2%}")
    logger.info(f"  <0.3的比例: {np.mean(qpkri_probs < 0.3):.2%}")
    
    logger.info(f"\n【PKRI可信度分布】")
    logger.info(f"  平均可信度: {np.mean(confidences):.4f}")
    logger.info(f"  中位数: {np.median(confidences):.4f}")
    logger.info(f"  标准差: {np.std(confidences):.4f}")
    logger.info(f"  最小值: {np.min(confidences):.4f}")
    logger.info(f"  最大值: {np.max(confidences):.4f}")
    logger.info(f"  >0.8的比例: {np.mean(confidences > 0.8):.2%}")
    logger.info(f"  >0.6的比例: {np.mean(confidences > 0.6):.2%}")
    logger.info(f"  <0.4的比例: {np.mean(confidences < 0.4):.2%}")
    
    logger.info(f"\n【预测一致性】")
    logger.info(f"  q₀ vs q_PKRI 预测不一致: {len(disagreements)} ({len(disagreements)/len(q0_probs):.2%})")
    
    # 计算相关系数
    correlation = np.corrcoef(q0_probs, qpkri_probs)[0, 1]
    logger.info(f"\n【相关性】")
    logger.info(f"  q₀_max vs q_PKRI_max 相关系数: {correlation:.4f}")
    
    # 分析差异
    diff = qpkri_probs - q0_probs
    logger.info(f"\n【差异分析】")
    logger.info(f"  平均差异 (q_PKRI - q₀): {np.mean(diff):.4f}")
    logger.info(f"  中位数差异: {np.median(diff):.4f}")
    logger.info(f"  差异标准差: {np.std(diff):.4f}")
    logger.info(f"  q_PKRI > q₀ 的比例: {np.mean(diff > 0):.2%}")
    logger.info(f"  q_PKRI < q₀ 的比例: {np.mean(diff < 0):.2%}")
    logger.info(f"  差异>0.1的比例: {np.mean(np.abs(diff) > 0.1):.2%}")
    logger.info(f"  差异>0.2的比例: {np.mean(np.abs(diff) > 0.2):.2%}")
    
    return {
        'q0_probs': q0_probs,
        'qpkri_probs': qpkri_probs,
        'confidences': confidences,
        'disagreements': disagreements,
        'correlation': correlation,
        'diff': diff
    }


def analyze_by_uncertainty(
    q0_dict: Dict[str, List[float]],
    qpkri_dict: Dict[str, Tuple[List[float], float]],
    baseline_dict: Dict[str, Dict],
    thresholds: List[float] = [0.1, 0.3]
) -> Dict:
    """按不确定性切片分析"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("2. 按不确定性切片分析")
    logger.info("=" * 60)
    
    slices = {
        'low': {'u': [], 'q0': [], 'qpkri': [], 'conf': []},
        'medium': {'u': [], 'q0': [], 'qpkri': [], 'conf': []},
        'high': {'u': [], 'q0': [], 'qpkri': [], 'conf': []}
    }
    
    for item_id in q0_dict:
        if item_id not in qpkri_dict or item_id not in baseline_dict:
            continue
        
        u = baseline_dict[item_id]['uncertainty']
        q0_max = max(q0_dict[item_id])
        qpkri_max, confidence = qpkri_dict[item_id]
        qpkri_max = max(qpkri_max)
        
        if u < thresholds[0]:
            slices['low']['u'].append(u)
            slices['low']['q0'].append(q0_max)
            slices['low']['qpkri'].append(qpkri_max)
            slices['low']['conf'].append(confidence)
        elif u < thresholds[1]:
            slices['medium']['u'].append(u)
            slices['medium']['q0'].append(q0_max)
            slices['medium']['qpkri'].append(qpkri_max)
            slices['medium']['conf'].append(confidence)
        else:
            slices['high']['u'].append(u)
            slices['high']['q0'].append(q0_max)
            slices['high']['qpkri'].append(qpkri_max)
            slices['high']['conf'].append(confidence)
    
    for slice_name, slice_data in slices.items():
        if len(slice_data['q0']) == 0:
            continue
        
        q0_mean = np.mean(slice_data['q0'])
        qpkri_mean = np.mean(slice_data['qpkri'])
        conf_mean = np.mean(slice_data['conf'])
        diff_mean = qpkri_mean - q0_mean
        
        slice_label = {
            'low': '低u (u < 0.1)',
            'medium': '中等u (0.1 ≤ u < 0.3)',
            'high': '高u (u ≥ 0.3)'
        }[slice_name]
        
        logger.info(f"\n【{slice_label}】样本数: {len(slice_data['q0'])}")
        logger.info(f"  平均q₀_max: {q0_mean:.4f}")
        logger.info(f"  平均q_PKRI_max: {qpkri_mean:.4f}")
        logger.info(f"  平均PKRI可信度: {conf_mean:.4f}")
        logger.info(f"  平均差异 (q_PKRI - q₀): {diff_mean:+.4f}")
    
    return slices


def analyze_accuracy(
    q0_dict: Dict[str, List[float]],
    qpkri_dict: Dict[str, Tuple[List[float], float]],
    baseline_dict: Dict[str, Dict]
) -> Dict:
    """分析准确性差异"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("3. 预测准确性分析")
    logger.info("=" * 60)
    
    correct_q0 = 0
    correct_qpkri = 0
    correct_baseline = 0
    total = 0
    
    q0_correct_confidence = []
    qpkri_correct_confidence = []
    q0_wrong_confidence = []
    qpkri_wrong_confidence = []
    
    for item_id in q0_dict:
        if item_id not in qpkri_dict or item_id not in baseline_dict:
            continue
        
        true_label = baseline_dict[item_id].get('coarse_label')
        if true_label is None:
            continue
        
        true_label = int(true_label)
        q0_pred = np.argmax(q0_dict[item_id])
        qpkri_pred = np.argmax(qpkri_dict[item_id][0])
        p_pred = np.argmax(baseline_dict[item_id]['pred_probs'])
        
        q0_max = max(q0_dict[item_id])
        qpkri_max = max(qpkri_dict[item_id][0])
        confidence = qpkri_dict[item_id][1]
        
        total += 1
        
        if q0_pred == true_label:
            correct_q0 += 1
            q0_correct_confidence.append(q0_max)
        else:
            q0_wrong_confidence.append(q0_max)
        
        if qpkri_pred == true_label:
            correct_qpkri += 1
            qpkri_correct_confidence.append(qpkri_max)
        else:
            qpkri_wrong_confidence.append(qpkri_max)
        
        if p_pred == true_label:
            correct_baseline += 1
    
    logger.info(f"\n总样本数: {total}")
    logger.info(f"\n【准确率对比】")
    logger.info(f"  Baseline准确率: {correct_baseline/total:.4f} ({correct_baseline}/{total})")
    logger.info(f"  q₀准确率: {correct_q0/total:.4f} ({correct_q0}/{total})")
    logger.info(f"  q_PKRI准确率: {correct_qpkri/total:.4f} ({correct_qpkri}/{total})")
    
    if q0_correct_confidence:
        logger.info(f"\n【q₀置信度分析】")
        logger.info(f"  正确预测的平均置信度: {np.mean(q0_correct_confidence):.4f}")
        logger.info(f"  错误预测的平均置信度: {np.mean(q0_wrong_confidence) if q0_wrong_confidence else 0:.4f}")
    
    if qpkri_correct_confidence:
        # 计算正确预测的PKRI可信度
        correct_confidences = []
        wrong_confidences = []
        for item_id in q0_dict:
            if item_id not in qpkri_dict or item_id not in baseline_dict:
                continue
            true_label = baseline_dict[item_id].get('coarse_label')
            if true_label is None:
                continue
            true_label = int(true_label)
            qpkri_pred = np.argmax(qpkri_dict[item_id][0])
            confidence = qpkri_dict[item_id][1]
            if qpkri_pred == true_label:
                correct_confidences.append(confidence)
            else:
                wrong_confidences.append(confidence)
        
        logger.info(f"\n【q_PKRI置信度分析】")
        logger.info(f"  正确预测的平均置信度: {np.mean(qpkri_correct_confidence):.4f}")
        logger.info(f"  错误预测的平均置信度: {np.mean(qpkri_wrong_confidence) if qpkri_wrong_confidence else 0:.4f}")
        if correct_confidences:
            logger.info(f"  正确预测的平均PKRI可信度: {np.mean(correct_confidences):.4f}")
        if wrong_confidences:
            logger.info(f"  错误预测的平均PKRI可信度: {np.mean(wrong_confidences):.4f}")
    
    return {
        'q0_accuracy': correct_q0 / total if total > 0 else 0,
        'qpkri_accuracy': correct_qpkri / total if total > 0 else 0,
        'baseline_accuracy': correct_baseline / total if total > 0 else 0
    }


def analyze_extreme_cases(
    q0_dict: Dict[str, List[float]],
    qpkri_dict: Dict[str, Tuple[List[float], float]],
    baseline_dict: Dict[str, Dict],
    n_cases: int = 10
):
    """分析极端差异案例"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("4. 极端差异案例分析")
    logger.info("=" * 60)
    
    differences = []
    for item_id in q0_dict:
        if item_id not in qpkri_dict or item_id not in baseline_dict:
            continue
        
        q0_max = max(q0_dict[item_id])
        qpkri_max = max(qpkri_dict[item_id][0])
        diff = qpkri_max - q0_max
        
        differences.append({
            'id': item_id,
            'q0_max': q0_max,
            'qpkri_max': qpkri_max,
            'diff': diff,
            'confidence': qpkri_dict[item_id][1],
            'uncertainty': baseline_dict[item_id]['uncertainty']
        })
    
    differences.sort(key=lambda x: abs(x['diff']), reverse=True)
    
    logger.info(f"\n【q_PKRI >> q₀ 的案例（前{n_cases}个）】")
    large_positive = [d for d in differences if d['diff'] > 0.1]
    for i, case in enumerate(large_positive[:n_cases], 1):
        logger.info(f"  {i}. ID={case['id']}: q₀={case['q0_max']:.4f}, q_PKRI={case['qpkri_max']:.4f}, "
                   f"差异=+{case['diff']:.4f}, 可信度={case['confidence']:.4f}, u={case['uncertainty']:.4f}")
    
    logger.info(f"\n【q_PKRI << q₀ 的案例（前{n_cases}个）】")
    large_negative = [d for d in differences if d['diff'] < -0.1]
    for i, case in enumerate(large_negative[:n_cases], 1):
        logger.info(f"  {i}. ID={case['id']}: q₀={case['q0_max']:.4f}, q_PKRI={case['qpkri_max']:.4f}, "
                   f"差异={case['diff']:.4f}, 可信度={case['confidence']:.4f}, u={case['uncertainty']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='分析q₀和q_PKRI的差异')
    parser.add_argument('--q0-file', type=str, required=True, help='q₀文件路径')
    parser.add_argument('--qpkri-file', type=str, required=True, help='q_PKRI文件路径')
    parser.add_argument('--baseline-file', type=str, required=True, help='Baseline预测文件路径')
    parser.add_argument('--output-dir', type=str, default='output', help='输出目录')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    logger.info("加载数据...")
    q0_dict = load_q0(args.q0_file)
    logger.info(f"✓ 加载 {len(q0_dict)} 条q₀后验")
    
    qpkri_dict = load_qpkri(args.qpkri_file)
    logger.info(f"✓ 加载 {len(qpkri_dict)} 条q_PKRI后验")
    
    baseline_dict = load_baseline(args.baseline_file)
    logger.info(f"✓ 加载 {len(baseline_dict)} 条baseline预测")
    
    # 分析
    dist_results = analyze_distributions(q0_dict, qpkri_dict, baseline_dict)
    slice_results = analyze_by_uncertainty(q0_dict, qpkri_dict, baseline_dict)
    accuracy_results = analyze_accuracy(q0_dict, qpkri_dict, baseline_dict)
    analyze_extreme_cases(q0_dict, qpkri_dict, baseline_dict)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, 'q0_vs_qpkri_analysis.json')
    results = {
        'distribution': {
            'q0_mean': float(np.mean(dist_results['q0_probs'])),
            'qpkri_mean': float(np.mean(dist_results['qpkri_probs'])),
            'confidence_mean': float(np.mean(dist_results['confidences'])),
            'correlation': float(dist_results['correlation']),
            'mean_diff': float(np.mean(dist_results['diff']))
        },
        'accuracy': accuracy_results,
        'slices': {
            name: {
                'q0_mean': float(np.mean(data['q0'])),
                'qpkri_mean': float(np.mean(data['qpkri'])),
                'confidence_mean': float(np.mean(data['conf'])),
                'n_samples': len(data['q0'])
            }
            for name, data in slice_results.items() if len(data['q0']) > 0
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("")
    logger.info(f"✓ 分析结果已保存: {output_file}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("分析完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

