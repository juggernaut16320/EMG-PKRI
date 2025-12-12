"""
uncertainty_analysis.py - 不确定性指标 u 构建 + 分桶分析工具

功能：
- 对 dev/test 的每条样本，计算不确定性指标 u = 1 - max_c pθ(c|x)
- 按 u 分桶（10 桶），统计样本数、错误率、平均置信度等
- 绘制 u_vs_error.png 图表，证明不确定性 u 与错误率有相关性
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


def compute_uncertainty(results: List[Dict], metric: str = 'u_max') -> List[Dict]:
    """
    计算不确定性指标 u
    
    Args:
        results: 预测结果列表
        metric: 不确定性指标类型
            - 'u_max': u = 1 - max_c pθ(c|x) (默认)
            - 'u_entropy': 熵不确定性（预留）
            - 'u_margin': 边际不确定性（预留）
    
    Returns:
        添加了 uncertainty 字段的结果列表
    """
    logger.info(f"计算不确定性指标: {metric}")
    
    for result in results:
        pred_probs = np.array(result['pred_probs'])  # [prob_0, prob_1]
        max_prob = np.max(pred_probs)  # 总是计算 max_prob
        
        if metric == 'u_max':
            # u = 1 - max_c pθ(c|x)
            uncertainty = 1.0 - max_prob
        elif metric == 'u_entropy':
            # 熵不确定性（预留）
            entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10))
            uncertainty = entropy / np.log(len(pred_probs))  # 归一化到 [0, 1]
        elif metric == 'u_margin':
            # 边际不确定性（预留）
            sorted_probs = np.sort(pred_probs)[::-1]
            if len(sorted_probs) >= 2:
                uncertainty = 1.0 - (sorted_probs[0] - sorted_probs[1])
            else:
                uncertainty = 0.0
        else:
            raise ValueError(f"未知的不确定性指标: {metric}")
        
        result['uncertainty'] = float(uncertainty)
        result['max_prob'] = float(max_prob)
    
    logger.info(f"不确定性计算完成，共 {len(results)} 条样本")
    return results


def bucket_analysis(results: List[Dict], n_buckets: int = 10) -> pd.DataFrame:
    """
    按不确定性 u 分桶分析
    
    Args:
        results: 包含 uncertainty 字段的结果列表
        n_buckets: 分桶数量（默认 10）
    
    Returns:
        分桶统计 DataFrame
    """
    logger.info(f"开始分桶分析，桶数: {n_buckets}")
    
    # 提取有真实标签的样本
    valid_results = [r for r in results if r.get('coarse_label') is not None]
    
    if len(valid_results) == 0:
        logger.warning("没有找到真实标签，无法进行分桶分析")
        return pd.DataFrame()
    
    # 提取不确定性和错误信息
    uncertainties = np.array([r['uncertainty'] for r in valid_results])
    is_error = np.array([
        int(r['pred_label']) != int(r['coarse_label']) 
        for r in valid_results
    ])
    max_probs = np.array([r['max_prob'] for r in valid_results])
    
    # 分桶（等宽分桶）
    bucket_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_indices = np.digitize(uncertainties, bucket_edges) - 1
    # 处理边界情况
    bucket_indices[bucket_indices < 0] = 0
    bucket_indices[bucket_indices >= n_buckets] = n_buckets - 1
    
    # 统计每个桶
    bucket_stats = []
    for i in range(n_buckets):
        mask = (bucket_indices == i)
        if np.sum(mask) == 0:
            continue
        
        bucket_uncertainties = uncertainties[mask]
        bucket_errors = is_error[mask]
        bucket_probs = max_probs[mask]
        
        bucket_stats.append({
            'bucket_id': i,
            'u_min': float(bucket_edges[i]),
            'u_max': float(bucket_edges[i + 1]),
            'u_mean': float(np.mean(bucket_uncertainties)),
            'u_median': float(np.median(bucket_uncertainties)),
            'n_samples': int(np.sum(mask)),
            'n_errors': int(np.sum(bucket_errors)),
            'error_rate': float(np.mean(bucket_errors)),
            'avg_confidence': float(np.mean(bucket_probs)),
            'avg_uncertainty': float(np.mean(bucket_uncertainties))
        })
    
    df = pd.DataFrame(bucket_stats)
    logger.info(f"分桶分析完成，共 {len(df)} 个有效桶")
    
    return df


def plot_u_vs_error(bucket_df: pd.DataFrame, output_path: str):
    """
    绘制 u vs error rate 图表
    
    Args:
        bucket_df: 分桶统计 DataFrame
        output_path: 输出图片路径
    """
    logger.info(f"绘制 u vs error rate 图表: {output_path}")
    
    if len(bucket_df) == 0:
        logger.warning("分桶数据为空，无法绘制图表")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # 图1: u vs error rate
    ax1 = axes[0]
    ax1.plot(bucket_df['u_mean'], bucket_df['error_rate'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Uncertainty (u)', fontsize=12)
    ax1.set_ylabel('Error Rate', fontsize=12)
    ax1.set_title('Uncertainty vs Error Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 添加样本数标注
    for _, row in bucket_df.iterrows():
        ax1.annotate(
            f"n={row['n_samples']}",
            (row['u_mean'], row['error_rate']),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=8
        )
    
    # 图2: u vs sample count (bar chart)
    ax2 = axes[1]
    ax2.bar(bucket_df['u_mean'], bucket_df['n_samples'], width=0.08, alpha=0.7)
    ax2.set_xlabel('Uncertainty (u)', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"图表已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="不确定性指标 u 构建 + 分桶分析工具")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="配置文件路径（默认: configs/config.yaml）"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Baseline checkpoint 路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="基础模型路径（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--dev-file",
        default=None,
        help="验证集文件（默认: dev.jsonl）"
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="测试集文件（默认: test.jsonl）"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录（默认从config.yaml读取）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批处理大小（默认: 16）"
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=10,
        help="分桶数量（默认: 10）"
    )
    parser.add_argument(
        "--metric",
        default="u_max",
        choices=["u_max", "u_entropy", "u_margin"],
        help="不确定性指标类型（默认: u_max）"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="设备（cuda/cpu，默认自动选择）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    data_dir = config.get('data_dir', './data')
    output_dir = args.output_dir or config.get('output_dir', './output')
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # 解析参数
    checkpoint_path = args.checkpoint or training_config.get('output_dir', 'checkpoints/baseline-lora')
    base_model_path = args.base_model or model_config.get('name_or_path', 'Qwen/Qwen3-1.7B')
    dev_file = args.dev_file or 'dev.jsonl'
    test_file = args.test_file or 'test.jsonl'
    batch_size = args.batch_size
    n_buckets = args.n_buckets
    metric = args.metric
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建完整路径
    dev_path = os.path.join(data_dir, dev_file)
    test_path = os.path.join(data_dir, test_file)
    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("开始不确定性分析")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"验证集: {dev_path}")
    logger.info(f"测试集: {test_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"不确定性指标: {metric}")
    logger.info(f"分桶数量: {n_buckets}")
    logger.info("=" * 60)
    
    # 加载模型
    model, tokenizer = load_baseline_model(checkpoint_path, base_model_path, device)
    
    # 处理验证集和测试集
    all_results = []
    
    for dataset_name, dataset_path in [("验证集", dev_path), ("测试集", test_path)]:
        if not os.path.exists(dataset_path):
            logger.warning(f"{dataset_name}文件不存在: {dataset_path}，跳过")
            continue
        
        logger.info(f"\n处理{dataset_name}: {dataset_path}")
        dataset = TextDataset(dataset_path, tokenizer, model_config.get('max_length', 512))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        results = predict_with_baseline(model, tokenizer, dataloader, device)
        all_results.extend(results)
    
    logger.info(f"\n总共处理 {len(all_results)} 条样本")
    
    # 计算不确定性
    all_results = compute_uncertainty(all_results, metric=metric)
    
    # 分桶分析
    bucket_df = bucket_analysis(all_results, n_buckets=n_buckets)
    
    # 保存分桶结果
    bucket_csv_path = os.path.join(output_dir, 'uncertainty_buckets.csv')
    logger.info(f"\n保存分桶结果到: {bucket_csv_path}")
    bucket_df.to_csv(bucket_csv_path, index=False, encoding='utf-8')
    
    # 绘制图表
    plot_path = os.path.join(output_dir, 'u_vs_error.png')
    plot_u_vs_error(bucket_df, plot_path)
    
    # 计算相关性
    if len(bucket_df) > 0:
        correlation = np.corrcoef(bucket_df['u_mean'], bucket_df['error_rate'])[0, 1]
        logger.info(f"\n不确定性 u 与错误率的相关系数: {correlation:.4f}")
        
        # 保存统计信息
        stats = {
            'correlation': float(correlation),
            'n_buckets': len(bucket_df),
            'total_samples': len(all_results),
            'metric': metric
        }
        stats_path = os.path.join(output_dir, 'uncertainty_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存: {stats_path}")
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("不确定性分析完成")
    logger.info("=" * 60)
    logger.info(f"总样本数: {len(all_results)}")
    logger.info(f"有效桶数: {len(bucket_df)}")
    if len(bucket_df) > 0:
        logger.info(f"不确定性 u 与错误率相关系数: {correlation:.4f}")
    logger.info(f"分桶结果: {bucket_csv_path}")
    logger.info(f"可视化图表: {plot_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

