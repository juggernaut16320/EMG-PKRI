"""
eval_baseline.py - 基线评估 + 高置信错误分析工具

功能：
- 在 test.jsonl 上评估 baseline：Accuracy / F1 / confusion matrix
- 在 hard_eval_set.jsonl 上单独评估，比较 hard vs 普通样本表现
- 选出一批高置信错误样本（如预测概率>0.8但错了），存成样本库
"""

import os
import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as np

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


def compute_metrics_from_results(results: List[Dict]) -> Dict:
    """
    从预测结果计算评估指标
    
    Args:
        results: 预测结果列表
    
    Returns:
        评估指标字典
    """
    # 提取真实标签和预测标签
    true_labels = []
    pred_labels = []
    
    for result in results:
        if result['coarse_label'] is not None:
            true_labels.append(int(result['coarse_label']))
            pred_labels.append(int(result['pred_label']))
    
    if len(true_labels) == 0:
        logger.warning("没有找到真实标签，无法计算指标")
        return {}
    
    # 计算指标
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    # 计算 confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, zero_division=0, labels=[0, 1]
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'class_0': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1': float(f1_per_class[0]),
                'support': int(support[0])
            },
            'class_1': {
                'precision': float(precision_per_class[1]),
                'recall': float(recall_per_class[1]),
                'f1': float(f1_per_class[1]),
                'support': int(support[1])
            }
        },
        'total_samples': len(true_labels)
    }
    
    return metrics


def find_high_conf_error_samples(
    results: List[Dict],
    confidence_threshold: float = 0.8
) -> List[Dict]:
    """
    找出高置信错误样本
    
    Args:
        results: 预测结果列表
        confidence_threshold: 置信度阈值
    
    Returns:
        高置信错误样本列表
    """
    high_conf_errors = []
    
    for result in results:
        if result['coarse_label'] is None:
            continue
        
        true_label = int(result['coarse_label'])
        pred_label = int(result['pred_label'])
        pred_prob = result['pred_prob']
        
        # 高置信但预测错误
        if pred_prob >= confidence_threshold and pred_label != true_label:
            high_conf_errors.append({
                'id': result['id'],
                'text': result['text'],
                'true_label': true_label,
                'pred_label': pred_label,
                'pred_prob': pred_prob,
                'pred_probs': result['pred_probs']
            })
    
    # 按置信度排序（高置信优先）
    high_conf_errors.sort(key=lambda x: x['pred_prob'], reverse=True)
    
    logger.info(f"找到 {len(high_conf_errors)} 个高置信错误样本（置信度 >= {confidence_threshold}）")
    
    return high_conf_errors


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基线评估 + 高置信错误分析工具")
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
        "--test-file",
        default=None,
        help="测试集文件（默认: test.jsonl）"
    )
    parser.add_argument(
        "--hard-file",
        default=None,
        help="困难集文件（默认: hard_eval_set.jsonl）"
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
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="高置信度阈值（默认: 0.8）"
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
    hardset_config = config.get('hardset', {})
    
    # 解析参数
    checkpoint_path = args.checkpoint or training_config.get('output_dir', 'checkpoints/baseline-lora')
    base_model_path = args.base_model or model_config.get('name_or_path', 'Qwen/Qwen3-1.7B')
    test_file = args.test_file or 'test.jsonl'
    hard_file = args.hard_file or 'hard_eval_set.jsonl'
    batch_size = args.batch_size
    confidence_threshold = args.confidence_threshold or hardset_config.get('confidence_threshold', 0.8)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建完整路径
    test_path = os.path.join(data_dir, test_file)
    hard_path = os.path.join(data_dir, hard_file)
    checkpoint_path = os.path.abspath(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)
    if not os.path.exists(test_path):
        logger.error(f"测试集文件不存在: {test_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("开始基线评估")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"测试集: {test_path}")
    logger.info(f"困难集: {hard_path if os.path.exists(hard_path) else '不存在'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 加载模型
    model, tokenizer = load_baseline_model(checkpoint_path, base_model_path, device)
    
    # 评估测试集
    logger.info("\n" + "=" * 60)
    logger.info("评估测试集")
    logger.info("=" * 60)
    
    test_dataset = TextDataset(test_path, tokenizer, model_config.get('max_length', 512))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_results = predict_with_baseline(model, tokenizer, test_dataloader, device)
    test_metrics = compute_metrics_from_results(test_results)
    
    logger.info("\n测试集评估结果:")
    logger.info(f"Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    logger.info(f"Precision: {test_metrics.get('precision', 0):.4f}")
    logger.info(f"Recall: {test_metrics.get('recall', 0):.4f}")
    logger.info(f"F1: {test_metrics.get('f1', 0):.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(test_metrics.get('confusion_matrix', []))}")
    
    # 评估困难集（如果存在）
    hard_metrics = {}
    if os.path.exists(hard_path):
        logger.info("\n" + "=" * 60)
        logger.info("评估困难集")
        logger.info("=" * 60)
        
        hard_dataset = TextDataset(hard_path, tokenizer, model_config.get('max_length', 512))
        hard_dataloader = DataLoader(
            hard_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        hard_results = predict_with_baseline(model, tokenizer, hard_dataloader, device)
        hard_metrics = compute_metrics_from_results(hard_results)
        
        logger.info("\n困难集评估结果:")
        logger.info(f"Accuracy: {hard_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {hard_metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {hard_metrics.get('recall', 0):.4f}")
        logger.info(f"F1: {hard_metrics.get('f1', 0):.4f}")
        logger.info(f"Confusion Matrix:\n{np.array(hard_metrics.get('confusion_matrix', []))}")
    
    # 找出高置信错误样本
    logger.info("\n" + "=" * 60)
    logger.info("识别高置信错误样本")
    logger.info("=" * 60)
    
    high_conf_errors = find_high_conf_error_samples(test_results, confidence_threshold)
    
    # 保存结果
    metrics_output = {
        'test_set': test_metrics,
        'hard_set': hard_metrics if hard_metrics else None,
        'high_conf_error_count': len(high_conf_errors),
        'confidence_threshold': confidence_threshold
    }
    
    metrics_path = os.path.join(output_dir, 'metrics_baseline.json')
    logger.info(f"\n保存评估指标到: {metrics_path}")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)
    
    high_conf_errors_path = os.path.join(output_dir, 'high_conf_error_samples.jsonl')
    logger.info(f"保存高置信错误样本到: {high_conf_errors_path}")
    with open(high_conf_errors_path, 'w', encoding='utf-8') as f:
        for sample in high_conf_errors:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("评估完成")
    logger.info("=" * 60)
    logger.info(f"测试集样本数: {test_metrics.get('total_samples', 0)}")
    logger.info(f"测试集 F1: {test_metrics.get('f1', 0):.4f}")
    if hard_metrics:
        logger.info(f"困难集样本数: {hard_metrics.get('total_samples', 0)}")
        logger.info(f"困难集 F1: {hard_metrics.get('f1', 0):.4f}")
    logger.info(f"高置信错误样本数: {len(high_conf_errors)}")
    logger.info(f"评估指标: {metrics_path}")
    logger.info(f"高置信错误样本: {high_conf_errors_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

