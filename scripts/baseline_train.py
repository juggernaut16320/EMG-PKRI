"""
baseline_train.py - 基线模型训练工具

功能：
使用 Qwen 模型 + LoRA 微调训练二分类基线模型（敏感/非敏感）
"""

import os
import sys
import json
import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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


class TextClassificationDataset(Dataset):
    """文本分类数据集"""
    
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
                # 只保留必需字段
                if 'text' in item and 'coarse_label' in item:
                    self.data.append({
                        'text': item['text'],
                        'label': int(item['coarse_label'])
                    })
        
        logger.info(f"加载数据集: {data_path}, 样本数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def setup_lora_model(model, lora_config: dict):
    """设置 LoRA 模型"""
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('alpha', 16),
        lora_dropout=lora_config.get('dropout', 0.1),
        target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
        bias='none',
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description='训练基线模型（Qwen + LoRA）')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--train-file', type=str, default=None,
                        help='训练集文件（相对于 data_dir）')
    parser.add_argument('--dev-file', type=str, default=None,
                        help='验证集文件（相对于 data_dir）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--model-name-or-path', type=str, default=None,
                        help='模型路径或 HuggingFace ID')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                        help='从 checkpoint 继续训练')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    data_dir = config.get('data_dir', './data')
    model_config = config.get('model', {})
    lora_config = config.get('lora', {})
    training_config = config.get('training', {})
    
    # 解析参数（命令行优先）
    train_file = args.train_file or training_config.get('train_file', 'train.jsonl')
    dev_file = args.dev_file or training_config.get('dev_file', 'dev.jsonl')
    output_dir = args.output_dir or training_config.get('output_dir', 'checkpoints/baseline-lora')
    model_name_or_path = args.model_name_or_path or model_config.get('name_or_path', 'Qwen/Qwen1.5-1.7B')
    max_length = model_config.get('max_length', 512)
    
    # 构建完整路径
    train_path = os.path.join(data_dir, train_file)
    dev_path = os.path.join(data_dir, dev_file)
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        logger.error(f"训练集文件不存在: {train_path}")
        sys.exit(1)
    if not os.path.exists(dev_path):
        logger.error(f"验证集文件不存在: {dev_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("开始训练基线模型")
    logger.info(f"模型: {model_name_or_path}")
    logger.info(f"训练集: {train_path}")
    logger.info(f"验证集: {dev_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    # 加载 tokenizer 和模型
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # 设置 pad_token（如果不存在）- 修复 batch_size > 1 时的 padding 问题
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # 如果 eos_token 也不存在，使用 unk_token
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
    
    logger.info("加载模型...")
    # 根据 GPU 支持情况选择数据类型
    # 如果支持 bf16，使用 bfloat16；否则使用 float32（让 Trainer 的混合精度处理）
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        logger.info("使用 bfloat16 精度")
    elif torch.cuda.is_available():
        # 不支持 bf16 时，使用 float32，让 Trainer 的 fp16 自动处理
        model_dtype = torch.float32
        logger.info("使用 float32 精度（训练时将自动转换为 fp16）")
    else:
        model_dtype = torch.float32
        logger.info("使用 float32 精度（CPU 模式）")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map='auto' if torch.cuda.is_available() else None
    )
    
    # 确保模型配置中也设置了 pad_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # 设置 LoRA
    logger.info("设置 LoRA...")
    model = setup_lora_model(model, lora_config)
    
    # 加载数据集
    logger.info("加载数据集...")
    train_dataset = TextClassificationDataset(train_path, tokenizer, max_length)
    dev_dataset = TextClassificationDataset(dev_path, tokenizer, max_length)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get('num_epochs', 3),
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 16),
        learning_rate=float(training_config.get('learning_rate', 2e-4)),
        warmup_steps=training_config.get('warmup_steps', 100),
        logging_steps=training_config.get('logging_steps', 50),
        eval_steps=training_config.get('eval_steps', 200),
        save_steps=training_config.get('save_steps', 500),
        save_strategy=training_config.get('save_strategy', 'steps'),
        eval_strategy=training_config.get('evaluation_strategy', 'steps'),  # 新版本使用 eval_strategy
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'f1'),
        greater_is_better=training_config.get('greater_is_better', True),
        save_total_limit=training_config.get('save_total_limit', 3),
        seed=training_config.get('seed', 42),
        # 根据 GPU 支持情况选择混合精度类型
        # 如果支持 bf16，使用 bf16；否则使用 fp16
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to='none',  # 不使用 wandb/tensorboard
        logging_dir=os.path.join(output_dir, 'logs'),
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 训练
    logger.info("开始训练...")
    if args.resume_from_checkpoint:
        logger.info(f"从 checkpoint 继续训练: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # 保存最终模型
    logger.info("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("=" * 60)
    logger.info("训练完成！")
    logger.info(f"模型保存在: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

