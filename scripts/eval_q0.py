"""
评估 q₀ 的质量（Precision/Recall/F1/Accuracy）
"""
import json
import argparse
import sys
import os

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
except ImportError:
    print("⚠️ 需要安装 sklearn: pip install scikit-learn")
    sys.exit(1)


def load_q0_and_labels(q0_file: str, baseline_file: str = None):
    """加载 q₀ 和真实标签"""
    q0_dict = {}
    true_labels = []
    q0_labels = []
    sample_ids = []
    
    # 从 q0_file 加载 q₀
    with open(q0_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            q0 = item['q0']
            q0_dict[item['id']] = q0
            sample_ids.append(item['id'])
            # q₀ 预测：如果 p_sensitive > 0.5，预测为敏感
            q0_labels.append(1 if q0[1] > 0.5 else 0)
            if 'coarse_label' in item:
                true_labels.append(item['coarse_label'])
    
    # 如果 baseline_file 提供，从那里获取真实标签
    if baseline_file and len(true_labels) == 0:
        baseline_dict = {}
        with open(baseline_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                baseline_dict[item['id']] = item
        
        # 重新构建标签列表
        true_labels = []
        q0_labels = []
        for sample_id in sample_ids:
            if sample_id in baseline_dict:
                true_labels.append(baseline_dict[sample_id]['coarse_label'])
                q0 = q0_dict[sample_id]
                q0_labels.append(1 if q0[1] > 0.5 else 0)
    
    return q0_dict, true_labels, q0_labels


def evaluate_q0(q0_file: str, baseline_file: str = None):
    """评估 q₀ 质量"""
    q0_dict, true_labels, q0_labels = load_q0_and_labels(q0_file, baseline_file)
    
    if len(true_labels) == 0:
        print("⚠️ 无法加载真实标签")
        print("   请提供 --baseline-file 参数，或确保 q0_file 包含 coarse_label 字段")
        return None
    
    if len(true_labels) != len(q0_labels):
        print(f"⚠️ 标签数量不匹配: true_labels={len(true_labels)}, q0_labels={len(q0_labels)}")
        return None
    
    # 计算指标
    precision = precision_score(true_labels, q0_labels, zero_division=0)
    recall = recall_score(true_labels, q0_labels, zero_division=0)
    f1 = f1_score(true_labels, q0_labels, zero_division=0)
    accuracy = accuracy_score(true_labels, q0_labels)
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, q0_labels)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # 处理只有一类的情况
        if cm.shape == (1, 1):
            if true_labels[0] == 0:
                tn, fp, fn, tp = cm[0, 0], 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    
    # 计算平均敏感概率
    avg_p_sensitive = sum(q0[1] for q0 in q0_dict.values()) / len(q0_dict) if q0_dict else 0
    
    # 输出结果
    print("=" * 60)
    print("q₀ 质量评估")
    print("=" * 60)
    print(f"样本数: {len(true_labels)}")
    print()
    print("评估指标:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1:        {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    print("混淆矩阵:")
    print(f"  TN (真阴性): {tn}")
    print(f"  FP (假阳性): {fp}  ← 误报，需要减少")
    print(f"  FN (假阴性): {fn}")
    print(f"  TP (真阳性): {tp}")
    print()
    print(f"平均敏感概率: {avg_p_sensitive:.4f}")
    print()
    
    # 判断质量
    print("质量评估:")
    if precision < 0.70:
        print("  ⚠️ q₀ Precision < 0.70，质量较差，需要改进")
        print("     建议：清洗词表、调整参数、改进匹配策略")
    elif precision < 0.75:
        print("  ⚠️ q₀ Precision < 0.75，质量一般，建议改进")
        print("     建议：调整 max_sensitive_prob 或清洗词表")
    else:
        print("  ✅ q₀ Precision ≥ 0.75，质量良好")
    
    if avg_p_sensitive > 0.80:
        print(f"  ⚠️ 平均敏感概率 {avg_p_sensitive:.4f} 过高（> 0.80）")
        print("     建议：降低 max_sensitive_prob 或提高 min_matches_for_sensitive")
    elif avg_p_sensitive < 0.50:
        print(f"  ⚠️ 平均敏感概率 {avg_p_sensitive:.4f} 过低（< 0.50）")
        print("     建议：检查词表匹配是否正常")
    else:
        print(f"  ✅ 平均敏感概率 {avg_p_sensitive:.4f} 合理")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'avg_p_sensitive': avg_p_sensitive,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'n_samples': len(true_labels)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估 q₀ 质量')
    parser.add_argument('--q0-file', type=str, required=True, help='q₀ 文件路径')
    parser.add_argument('--baseline-file', type=str, default=None, 
                       help='Baseline 文件路径（包含真实标签），如果 q0_file 没有 coarse_label 字段则需要提供')
    args = parser.parse_args()
    
    if not os.path.exists(args.q0_file):
        print(f"❌ q₀ 文件不存在: {args.q0_file}")
        sys.exit(1)
    
    if args.baseline_file and not os.path.exists(args.baseline_file):
        print(f"⚠️ Baseline 文件不存在: {args.baseline_file}")
        args.baseline_file = None
    
    evaluate_q0(args.q0_file, args.baseline_file)

