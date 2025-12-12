"""
hardset_maker.py - 困难子集构造工具

功能：
基于 teacher–student 分歧构造困难子集。
- 使用训练好的 baseline 模型（student）进行预测
- 使用 gemma-3-27b 模型（teacher）进行打标
- 找出分歧样本：baseline 预测错误但 teacher 预测正确，或 baseline 高置信但预测错误
- 从分歧样本中抽取 500-2000 条作为 hard_eval_set.jsonl
"""

import os
import sys
import json
import logging
import argparse
import yaml
import random
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

# 添加 scripts 目录到路径（用于导入 llm_labeler）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_labeler import (
    load_config as load_llm_config,
    run_label_task,
    COARSE_LABEL_TASK,
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
        预测结果列表，每个元素包含：id, text, coarse_label, pred_label, pred_prob
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


def get_teacher_predictions(input_data: List[Dict], config: dict, temp_output_path: str) -> Dict[str, int]:
    """
    使用 LLM (teacher) 对样本进行打标
    
    Args:
        input_data: 输入数据列表，每个元素应包含 'id' 和 'text' 字段
        config: 配置字典
        temp_output_path: 临时输出文件路径
    
    Returns:
        字典，key 为 id，value 为预测标签 (0 或 1)
    """
    logger.info("使用 LLM (teacher) 进行打标...")
    
    # 创建临时输入文件
    temp_input_path = temp_output_path.replace('.jsonl', '_input.jsonl')
    os.makedirs(os.path.dirname(temp_output_path) or '.', exist_ok=True)
    
    with open(temp_input_path, 'w', encoding='utf-8') as f:
        for item in input_data:
            # 提取 id 和 text
            item_id = item.get('id', '')
            text = item.get('text', '')
            if not text:
                continue
            f.write(json.dumps({
                'id': item_id,
                'text': text
            }, ensure_ascii=False) + '\n')
    
    # 使用 run_label_task 进行打标
    llm_config = load_llm_config(config.get('config', 'configs/config.yaml'))
    batch_size = llm_config.get('llm', {}).get('batch_size', 10)
    request_interval = llm_config.get('llm', {}).get('request_interval', 2.5)
    max_retries = llm_config.get('llm', {}).get('max_retries', 3)
    
    run_label_task(
        task=COARSE_LABEL_TASK,
        input_path=temp_input_path,
        output_path=temp_output_path,
        text_field='text',
        batch_size=batch_size,
        max_retries=max_retries,
        skip_existing=False,
        request_interval=request_interval
    )
    
    # 读取结果
    teacher_predictions = {}
    if os.path.exists(temp_output_path):
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'id' in item and 'coarse_label' in item:
                        teacher_predictions[item['id']] = int(item['coarse_label'])
    
    # 清理临时文件
    if os.path.exists(temp_input_path):
        os.remove(temp_input_path)
    
    logger.info(f"Teacher 预测完成，共 {len(teacher_predictions)} 条样本")
    return teacher_predictions


def find_disagreement_samples(
    baseline_results: List[Dict],
    teacher_predictions: Dict[str, int],
    confidence_threshold: float = 0.8,
    min_size: int = 500,
    max_size: int = 2000
) -> List[Dict]:
    """
    找出分歧样本
    
    Args:
        baseline_results: baseline 预测结果
        teacher_predictions: teacher 预测结果
        confidence_threshold: 高置信度阈值
        min_size: 最小样本数
        max_size: 最大样本数
    
    Returns:
        分歧样本列表
    """
    disagreement_samples = []
    
    for result in baseline_results:
        sample_id = result['id']
        baseline_pred = result['pred_label']
        baseline_prob = result['pred_prob']
        true_label = result.get('coarse_label')
        teacher_pred = teacher_predictions.get(sample_id)
        
        if teacher_pred is None:
            continue
        
        # 判断是否为分歧样本
        is_disagreement = False
        
        # 情况1: baseline 预测错误但 teacher 预测正确
        if true_label is not None:
            baseline_wrong = (baseline_pred != true_label)
            teacher_correct = (teacher_pred == true_label)
            if baseline_wrong and teacher_correct:
                is_disagreement = True
        
        # 情况2: baseline 高置信但预测错误
        if true_label is not None:
            if baseline_prob >= confidence_threshold and baseline_pred != true_label:
                is_disagreement = True
        
        # 情况3: baseline 和 teacher 预测不一致（即使没有真实标签）
        if baseline_pred != teacher_pred:
            is_disagreement = True
        
        if is_disagreement:
            disagreement_samples.append({
                **result,
                'teacher_label': teacher_pred,
                'disagreement_type': []
            })
            
            # 记录分歧类型
            if true_label is not None:
                if baseline_pred != true_label and teacher_pred == true_label:
                    disagreement_samples[-1]['disagreement_type'].append('baseline_wrong_teacher_correct')
                if baseline_prob >= confidence_threshold and baseline_pred != true_label:
                    disagreement_samples[-1]['disagreement_type'].append('high_conf_error')
            if baseline_pred != teacher_pred:
                disagreement_samples[-1]['disagreement_type'].append('prediction_mismatch')
    
    logger.info(f"找到 {len(disagreement_samples)} 个分歧样本")
    
    # 按置信度排序（高置信错误优先）
    # 使用 (pred_prob, id) 作为排序键，确保稳定排序
    disagreement_samples.sort(key=lambda x: (x['pred_prob'], x['id']), reverse=True)
    
    # 选择样本
    selected_size = min(max_size, max(min_size, len(disagreement_samples)))
    selected_samples = disagreement_samples[:selected_size]
    
    logger.info(f"选择 {len(selected_samples)} 个样本作为困难子集")
    
    return selected_samples


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="困难子集构造工具")
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
        "--output",
        default=None,
        help="输出文件路径（默认: hard_eval_set.jsonl）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批处理大小（默认: 16）"
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
    output_dir = config.get('output_dir', './output')
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    hardset_config = config.get('hardset', {})
    
    # 解析参数
    checkpoint_path = args.checkpoint or training_config.get('output_dir', 'checkpoints/baseline-lora')
    base_model_path = args.base_model or model_config.get('name_or_path', 'Qwen/Qwen3-1.7B')
    dev_file = args.dev_file or 'dev.jsonl'
    test_file = args.test_file or 'test.jsonl'
    output_file = args.output or 'hard_eval_set.jsonl'
    batch_size = args.batch_size
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 构建完整路径
    dev_path = os.path.join(data_dir, dev_file)
    test_path = os.path.join(data_dir, test_file)
    output_path = os.path.join(data_dir, output_file)
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)
    if not os.path.exists(dev_path):
        logger.error(f"验证集文件不存在: {dev_path}")
        sys.exit(1)
    if not os.path.exists(test_path):
        logger.error(f"测试集文件不存在: {test_path}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("开始构造困难子集")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"基础模型: {base_model_path}")
    logger.info(f"验证集: {dev_path}")
    logger.info(f"测试集: {test_path}")
    logger.info(f"输出: {output_path}")
    logger.info("=" * 60)
    
    # 加载模型
    model, tokenizer = load_baseline_model(checkpoint_path, base_model_path, device)
    
    # 加载数据
    logger.info("加载数据集...")
    dev_dataset = TextDataset(dev_path, tokenizer, model_config.get('max_length', 512))
    test_dataset = TextDataset(test_path, tokenizer, model_config.get('max_length', 512))
    
    # 合并数据集
    all_data = []
    for dataset in [dev_dataset, test_dataset]:
        for i in range(len(dataset)):
            all_data.append(dataset[i])
    
    logger.info(f"合并数据集，共 {len(all_data)} 条样本")
    
    # 创建 DataLoader
    dataloader = DataLoader(
        all_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Windows 上建议设为 0
    )
    
    # Baseline 预测
    baseline_results = predict_with_baseline(model, tokenizer, dataloader, device)
    
    # Teacher 预测（使用 LLM）
    temp_output_path = os.path.join(output_dir, 'temp_teacher_labels.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备 teacher 预测的输入数据（只需要 id 和 text）
    teacher_input_data = [
        {
            'id': result['id'],
            'text': result['text']
        }
        for result in baseline_results
    ]
    
    teacher_predictions = get_teacher_predictions(
        teacher_input_data,
        config,
        temp_output_path
    )
    
    # 找出分歧样本
    confidence_threshold = hardset_config.get('confidence_threshold', 0.8)
    min_size = hardset_config.get('min_size', 500)
    max_size = hardset_config.get('max_size', 2000)
    
    disagreement_samples = find_disagreement_samples(
        baseline_results,
        teacher_predictions,
        confidence_threshold,
        min_size,
        max_size
    )
    
    # 保存结果
    logger.info(f"保存困难子集到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in disagreement_samples:
            # 只保存必要字段
            output_item = {
                'id': sample['id'],
                'text': sample['text'],
                'coarse_label': sample.get('coarse_label'),
                'baseline_pred': sample['pred_label'],
                'baseline_prob': sample['pred_prob'],
                'teacher_label': sample.get('teacher_label'),
                'disagreement_type': sample.get('disagreement_type', [])
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
    
    # 清理临时文件
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    # 统计信息
    logger.info("=" * 60)
    logger.info("困难子集构造完成")
    logger.info(f"总样本数: {len(baseline_results)}")
    logger.info(f"分歧样本数: {len(disagreement_samples)}")
    logger.info(f"输出文件: {output_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

