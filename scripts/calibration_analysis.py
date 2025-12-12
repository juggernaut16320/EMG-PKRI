"""
calibration_analysis.py - 校准分析工具：Reliability Diagram + ECE

功能：
- 在 dev/test 上绘制 Reliability Diagram（可靠性图）
- 计算 ECE（Expected Calibration Error）和 MCE（Maximum Calibration Error）
- 量化 baseline 的"过度自信"问题，为门控/校准提供动机
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

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
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


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class TextDataset(Dataset):
    """文本数据集（用于模型推理）"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        Args:
            data_path: JSONL 文件路径
            tokenizer: Tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if 'text' in item:
                    self.data.append({
                        'id': item.get('id', ''),
                        'text': item['text'],
                        'coarse_label': item.get('coarse_label', None)
                    })
        
        logger.info(f"加载数据集: {data_path}, 样本数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'id': item['id'],
            'text': text,
            'coarse_label': item['coarse_label']
        }


def load_baseline_model(checkpoint_path: str, base_model_path: str, device: str = None):
    """
    加载训练好的 baseline 模型（PEFT/LoRA）
    
    Args:
        checkpoint_path: checkpoint 路径
        base_model_path: 基础模型路径
        device: 设备（'cuda' 或 'cpu'）
    
    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"加载 tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    
    logger.info(f"加载基础模型: {base_model_path}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )
    
    # 确保模型配置中也设置了 pad_token_id
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    
    logger.info(f"加载 LoRA 权重: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    logger.info(f"模型加载完成，设备: {device}")
    return model, tokenizer


def predict_with_baseline(model, tokenizer, dataloader, device: str = None) -> List[Dict]:
    """
    使用 baseline 模型进行预测
    
    Args:
        model: 模型
        tokenizer: tokenizer
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        预测结果列表，每个元素包含：id, text, coarse_label, pred_label, pred_prob, pred_probs
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = []
    
    logger.info("开始使用 baseline 模型进行预测...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ids = batch['id']
            texts = batch['text']
            labels = batch['coarse_label']
            
            # 预测
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算概率
            probs = torch.softmax(logits, dim=-1)
            pred_probs = probs.cpu().numpy()
            pred_labels = pred_probs.argmax(axis=-1)
            
            # 保存结果
            for i in range(len(ids)):
                results.append({
                    'id': ids[i],
                    'text': texts[i],
                    'coarse_label': labels[i] if labels[i] is not None else None,
                    'pred_label': int(pred_labels[i]),
                    'pred_prob': float(pred_probs[i][pred_labels[i]]),
                    'pred_probs': pred_probs[i].tolist()  # [prob_0, prob_1]
                })
            
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"已处理 {batch_idx + 1} 个批次")
    
    logger.info(f"Baseline 预测完成，共 {len(results)} 条样本")
    return results


def calibration_buckets(results: List[Dict], n_buckets: int = 10) -> pd.DataFrame:
    """
    按预测概率分桶，计算每个桶的校准统计
    
    Args:
        results: 预测结果列表，每个元素包含 pred_probs 和 coarse_label
        n_buckets: 分桶数量
    
    Returns:
        分桶统计 DataFrame
    """
    logger.info(f"按预测概率分桶，桶数: {n_buckets}")
    
    # 提取预测概率和真实标签
    # 对于二分类，我们使用预测为类别1的概率（即 pred_probs[1]）
    probs = []
    labels = []
    
    for result in results:
        if result['coarse_label'] is not None:
            pred_probs = np.array(result['pred_probs'])  # [prob_0, prob_1]
            prob_1 = pred_probs[1]  # 预测为敏感（类别1）的概率
            probs.append(prob_1)
            labels.append(int(result['coarse_label']))
    
    if len(probs) == 0:
        logger.warning("没有找到有效样本，无法进行分桶分析")
        return pd.DataFrame()
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    # 等宽分桶：[0, 1] 等分为 n_buckets 个区间
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_ids = np.digitize(probs, bucket_edges) - 1
    # 处理边界情况：prob=1.0 会被分到最后一个桶
    bucket_ids = np.clip(bucket_ids, 0, n_buckets - 1)
    
    # 统计每个桶的信息
    bucket_stats = []
    for bucket_id in range(n_buckets):
        mask = (bucket_ids == bucket_id)
        bucket_probs = probs[mask]
        bucket_labels = labels[mask]
        
        if len(bucket_probs) == 0:
            continue
        
        # 计算统计量
        n_samples = len(bucket_probs)
        n_correct = np.sum(bucket_labels == 1)  # 真实标签为1（敏感）的数量
        accuracy = n_correct / n_samples if n_samples > 0 else 0.0
        avg_confidence = np.mean(bucket_probs)  # 平均预测概率
        gap = abs(accuracy - avg_confidence)  # 校准误差
        ece_contribution = (n_samples / len(probs)) * gap  # ECE 贡献
        
        bucket_stats.append({
            'bucket_id': bucket_id,
            'prob_min': bucket_edges[bucket_id],
            'prob_max': bucket_edges[bucket_id + 1],
            'prob_mean': avg_confidence,
            'n_samples': n_samples,
            'n_correct': n_correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'gap': gap,
            'ece_contribution': ece_contribution
        })
    
    df = pd.DataFrame(bucket_stats)
    logger.info(f"分桶完成，有效桶数: {len(df)}")
    return df


def compute_ece(bucket_df: pd.DataFrame, n_total: int) -> float:
    """
    计算期望校准误差（Expected Calibration Error, ECE）
    
    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|
    
    Args:
        bucket_df: 分桶统计 DataFrame
        n_total: 总样本数
    
    Returns:
        ECE 值
    """
    if len(bucket_df) == 0 or n_total == 0:
        return 0.0
    
    ece = 0.0
    for _, row in bucket_df.iterrows():
        n_samples = row['n_samples']
        gap = row['gap']
        ece += (n_samples / n_total) * gap
    
    return float(ece)


def compute_mce(bucket_df: pd.DataFrame) -> float:
    """
    计算最大校准误差（Maximum Calibration Error, MCE）
    
    MCE = max_m |acc(B_m) - conf(B_m)|
    
    Args:
        bucket_df: 分桶统计 DataFrame
    
    Returns:
        MCE 值
    """
    if len(bucket_df) == 0:
        return 0.0
    
    mce = bucket_df['gap'].max()
    return float(mce)


def compute_brier_score(results: List[Dict]) -> float:
    """
    计算 Brier Score
    
    Brier Score = (1/n) * Σ (p_i - y_i)²
    
    Args:
        results: 预测结果列表
    
    Returns:
        Brier Score 值
    """
    probs = []
    labels = []
    
    for result in results:
        if result['coarse_label'] is not None:
            pred_probs = np.array(result['pred_probs'])
            prob_1 = pred_probs[1]  # 预测为敏感（类别1）的概率
            probs.append(prob_1)
            labels.append(int(result['coarse_label']))
    
    if len(probs) == 0:
        return 0.0
    
    probs = np.array(probs)
    labels = np.array(labels)
    
    brier_score = np.mean((probs - labels) ** 2)
    return float(brier_score)


def plot_reliability_diagram(bucket_df: pd.DataFrame, output_path: str, dataset_name: str = ""):
    """
    绘制可靠性图（Reliability Diagram）
    
    Args:
        bucket_df: 分桶统计 DataFrame
        output_path: 输出图片路径
        dataset_name: 数据集名称（用于标题）
    """
    logger.info(f"绘制可靠性图: {output_path}")
    
    if len(bucket_df) == 0:
        logger.warning("分桶数据为空，无法绘制图表")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 图1: Reliability Diagram
    ax1 = axes[0]
    
    # 绘制完美校准对角线
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # 绘制实际校准曲线
    prob_means = bucket_df['prob_mean'].values
    accuracies = bucket_df['accuracy'].values
    
    ax1.plot(prob_means, accuracies, 'o-', linewidth=2, markersize=8, 
             label='Model Calibration', color='steelblue')
    
    # 添加误差条（使用标准误差）
    # 对于二项分布，标准误差 = sqrt(p(1-p)/n)
    for _, row in bucket_df.iterrows():
        n_samples = row['n_samples']
        accuracy = row['accuracy']
        if n_samples > 0:
            std_error = np.sqrt(accuracy * (1 - accuracy) / n_samples)
            ax1.errorbar(row['prob_mean'], row['accuracy'], 
                        yerr=std_error, fmt='none', color='steelblue', alpha=0.5)
    
    ax1.set_xlabel('Mean Predicted Probability (Confidence)', fontsize=12)
    ax1.set_ylabel('Observed Accuracy', fontsize=12)
    title = 'Reliability Diagram'
    if dataset_name:
        title += f' ({dataset_name})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 图2: 样本数量分布（柱状图）
    ax2 = axes[1]
    ax2.bar(bucket_df['prob_mean'], bucket_df['n_samples'], 
            width=0.08, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution Across Confidence Bins', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"可靠性图已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='校准分析：Reliability Diagram + ECE')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint 路径（覆盖 config）')
    parser.add_argument('--base-model', type=str, default=None,
                       help='基础模型路径（覆盖 config）')
    parser.add_argument('--dev-file', type=str, default=None,
                       help='验证集文件（覆盖 config）')
    parser.add_argument('--test-file', type=str, default=None,
                       help='测试集文件（覆盖 config）')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据目录（覆盖 config）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（覆盖 config）')
    parser.add_argument('--n-buckets', type=int, default=None,
                       help='分桶数量（覆盖 config）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批处理大小（覆盖 config）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，覆盖 config）')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取参数
    data_dir = args.data_dir or config.get('data_dir', './data')
    output_dir = args.output_dir or config.get('output_dir', './output')
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    calibration_config = config.get('calibration', {})
    
    checkpoint_path = args.checkpoint or training_config.get('output_dir', 'checkpoints/baseline-lora')
    base_model_path = args.base_model or model_config.get('name_or_path', 'Qwen/Qwen3-1.7B')
    dev_file = args.dev_file or training_config.get('dev_file', 'dev.jsonl')
    test_file = args.test_file or 'test.jsonl'
    n_buckets = args.n_buckets or calibration_config.get('n_buckets', 10)
    batch_size = args.batch_size or calibration_config.get('batch_size', 16)
    device = args.device or calibration_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    max_length = model_config.get('max_length', 512)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("校准分析：Reliability Diagram + ECE")
    logger.info("=" * 60)
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"分桶数量: {n_buckets}")
    logger.info(f"批处理大小: {batch_size}")
    logger.info(f"设备: {device}")
    logger.info("=" * 60)
    
    # 加载模型
    logger.info("\n加载 baseline 模型...")
    model, tokenizer = load_baseline_model(checkpoint_path, base_model_path, device)
    
    # 处理验证集和测试集
    all_results = {}
    
    for dataset_name, dataset_file in [("dev", dev_file), ("test", test_file)]:
        dataset_path = os.path.join(data_dir, dataset_file)
        
        if not os.path.exists(dataset_path):
            logger.warning(f"{dataset_name}文件不存在: {dataset_path}，跳过")
            continue
        
        logger.info(f"\n处理{dataset_name}集: {dataset_path}")
        dataset = TextDataset(dataset_path, tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        results = predict_with_baseline(model, tokenizer, dataloader, device)
        all_results[dataset_name] = results
    
    # 计算校准指标
    logger.info("\n计算校准指标...")
    ece_results = {}
    
    for dataset_name, results in all_results.items():
        if len(results) == 0:
            continue
        
        logger.info(f"\n分析 {dataset_name} 集...")
        
        # 分桶分析
        bucket_df = calibration_buckets(results, n_buckets=n_buckets)
        
        if len(bucket_df) == 0:
            logger.warning(f"{dataset_name} 集分桶结果为空，跳过")
            continue
        
        # 计算指标
        n_total = len([r for r in results if r['coarse_label'] is not None])
        ece = compute_ece(bucket_df, n_total)
        mce = compute_mce(bucket_df)
        brier_score = compute_brier_score(results)
        
        ece_results[dataset_name] = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'n_samples': n_total,
            'n_buckets': len(bucket_df)
        }
        
        logger.info(f"{dataset_name} 集校准指标:")
        logger.info(f"  ECE: {ece:.4f}")
        logger.info(f"  MCE: {mce:.4f}")
        logger.info(f"  Brier Score: {brier_score:.4f}")
        logger.info(f"  样本数: {n_total}")
        logger.info(f"  有效桶数: {len(bucket_df)}")
        
        # 保存分桶统计（可选）
        bucket_csv_path = os.path.join(output_dir, f'calibration_buckets_{dataset_name}.csv')
        bucket_df.to_csv(bucket_csv_path, index=False, encoding='utf-8')
        logger.info(f"分桶统计已保存: {bucket_csv_path}")
        
        # 绘制可靠性图
        plot_path = os.path.join(output_dir, f'reliability_diagram_{dataset_name}.png')
        plot_reliability_diagram(bucket_df, plot_path, dataset_name=dataset_name)
    
    # 计算整体指标（如果有多个数据集）
    if len(ece_results) > 1:
        # 合并所有结果计算整体指标
        all_results_combined = []
        for results in all_results.values():
            all_results_combined.extend(results)
        
        bucket_df_combined = calibration_buckets(all_results_combined, n_buckets=n_buckets)
        if len(bucket_df_combined) > 0:
            n_total_combined = len([r for r in all_results_combined if r['coarse_label'] is not None])
            ece_combined = compute_ece(bucket_df_combined, n_total_combined)
            mce_combined = compute_mce(bucket_df_combined)
            brier_score_combined = compute_brier_score(all_results_combined)
            
            ece_results['overall'] = {
                'ece': ece_combined,
                'mce': mce_combined,
                'brier_score': brier_score_combined,
                'n_samples': n_total_combined,
                'n_buckets': len(bucket_df_combined)
            }
            
            logger.info(f"\n整体校准指标:")
            logger.info(f"  ECE: {ece_combined:.4f}")
            logger.info(f"  MCE: {mce_combined:.4f}")
            logger.info(f"  Brier Score: {brier_score_combined:.4f}")
            logger.info(f"  样本数: {n_total_combined}")
            
            # 绘制整体可靠性图
            plot_path_combined = os.path.join(output_dir, 'reliability_diagram_overall.png')
            plot_reliability_diagram(bucket_df_combined, plot_path_combined, dataset_name='Overall')
    
    # 保存结果
    ece_json_path = os.path.join(output_dir, 'ece_results.json')
    with open(ece_json_path, 'w', encoding='utf-8') as f:
        json.dump(ece_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n校准结果已保存: {ece_json_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("校准分析完成！")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

