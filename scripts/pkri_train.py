"""
PKRI 模型训练脚本（方案一：简化版）
训练逻辑回归模型预测知识可信度，并生成q_PKRI后验
"""
import json
import argparse
import sys
import os
import logging
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
except ImportError:
    print("❌ 需要安装 sklearn: pip install scikit-learn")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_features(feature_file: str) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    加载特征文件
    
    Args:
        feature_file: CSV特征文件路径
    
    Returns:
        (特征DataFrame, 标签Series, 特征列名列表)
    """
    df = pd.read_csv(feature_file, encoding='utf-8')
    
    # 分离特征和标签
    feature_cols = [
        'lexicon_match_porn', 'lexicon_match_politics', 'lexicon_match_abuse',
        'total_matches', 'match_ratio',
        'has_porn_match', 'has_politics_match', 'has_abuse_match',
        'subtype_match_porn', 'subtype_match_politics', 'subtype_match_abuse',
        'subtype_match_score',
        'match_density', 'match_span_ratio',
        'max_match_category_porn', 'max_match_category_politics', 'max_match_category_abuse',
        'source_confidence_porn', 'source_confidence_politics', 'source_confidence_abuse',
        'weighted_source_confidence',
    ]
    
    # 只保留存在的特征列
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].fillna(0.0)
    
    # 标签（如果存在）
    y = None
    if 'label' in df.columns:
        y = df['label']
    elif 'coarse_label' in df.columns:
        y = df['coarse_label'].fillna(0).astype(int)
    
    logger.info(f"加载特征: {len(X)} 条样本, {len(available_cols)} 个特征")
    if y is not None:
        logger.info(f"标签分布: {y.value_counts().to_dict()}")
    
    return X, y, available_cols


def train_pkri_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = 'lr',
    calibration_method: str = 'temperature'
) -> Tuple[object, Dict]:
    """
    训练PKRI模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_val: 验证特征（用于校准）
        y_val: 验证标签
        model_type: 模型类型（'lr'）
        calibration_method: 校准方法（'temperature' 或 'platt'）
    
    Returns:
        (训练好的模型, 模型配置)
    """
    logger.info("=" * 60)
    logger.info("训练PKRI模型")
    logger.info("=" * 60)
    logger.info(f"模型类型: {model_type}")
    logger.info(f"校准方法: {calibration_method}")
    logger.info(f"训练样本数: {len(X_train)}")
    logger.info("")
    
    # 训练基础模型
    if model_type == 'lr':
        base_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'  # 适合小数据集
        )
        base_model.fit(X_train, y_train)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 评估基础模型
    train_pred = base_model.predict(X_train)
    train_proba = base_model.predict_proba(X_train)[:, 1]
    
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, zero_division=0)
    
    logger.info("训练集指标:")
    logger.info(f"  Accuracy: {train_acc:.4f}")
    logger.info(f"  F1: {train_f1:.4f}")
    
    # 校准模型
    calibrated_model = base_model
    if X_val is not None and y_val is not None:
        logger.info("")
        logger.info("校准模型...")
        
        if calibration_method in ['temperature', 'platt']:
            # 手动实现 Platt scaling（兼容所有 sklearn 版本）
            # 在验证集上获取基础模型的预测概率
            val_proba_uncalibrated = base_model.predict_proba(X_val)[:, 1]
            
            # 使用 LogisticRegression 在预测概率上训练 Platt scaling
            # Platt scaling: sigmoid(ax + b)，其中 x 是未校准的概率
            from sklearn.linear_model import LogisticRegression as PlattScaler
            
            # 将概率作为特征（需要reshape）
            X_platt = val_proba_uncalibrated.reshape(-1, 1)
            
            # 训练 Platt scaler
            platt_scaler = PlattScaler()
            platt_scaler.fit(X_platt, y_val)
            
            # 创建包装类以保持与原始模型接口一致
            class CalibratedModel:
                def __init__(self, base_model, platt_scaler):
                    self.base_model = base_model
                    self.platt_scaler = platt_scaler
                
                def predict(self, X):
                    # 预测类别
                    return self.base_model.predict(X)
                
                def predict_proba(self, X):
                    # 获取基础模型预测概率
                    base_proba = self.base_model.predict_proba(X)
                    # 对类别1的概率进行校准
                    proba_1_uncalibrated = base_proba[:, 1].reshape(-1, 1)
                    proba_1_calibrated = self.platt_scaler.predict_proba(proba_1_uncalibrated)[:, 1]
                    # 重新构建概率数组
                    proba_0_calibrated = 1.0 - proba_1_calibrated
                    return np.column_stack([proba_0_calibrated, proba_1_calibrated])
            
            calibrated_model = CalibratedModel(base_model, platt_scaler)
        else:
            logger.warning(f"未知的校准方法: {calibration_method}，跳过校准")
        
        # 评估校准后的模型
        val_pred = calibrated_model.predict(X_val)
        val_proba = calibrated_model.predict_proba(X_val)[:, 1]
        
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        
        try:
            val_auc = roc_auc_score(y_val, val_proba) if len(np.unique(y_val)) > 1 else 0.0
        except ValueError:
            val_auc = 0.0
        
        logger.info("验证集指标（校准后）:")
        logger.info(f"  Accuracy: {val_acc:.4f}")
        logger.info(f"  F1: {val_f1:.4f}")
        logger.info(f"  AUC: {val_auc:.4f}")
    else:
        val_acc = None
        val_f1 = None
        val_auc = None
    
    # 模型配置
    model_config = {
        'model_type': model_type,
        'calibration_method': calibration_method,
        'feature_names': list(X_train.columns),
        'n_features': len(X_train.columns),
        'train_accuracy': float(train_acc),
        'train_f1': float(train_f1),
    }
    
    if val_acc is not None:
        model_config['val_accuracy'] = float(val_acc)
        model_config['val_f1'] = float(val_f1)
        if val_auc is not None:
            model_config['val_auc'] = float(val_auc)
    
    logger.info("")
    logger.info("✓ 模型训练完成")
    
    return calibrated_model, model_config


def predict_confidence(
    features: pd.DataFrame,
    model: object
) -> np.ndarray:
    """
    预测知识可信度
    
    Args:
        features: 特征DataFrame
        model: 训练好的模型
    
    Returns:
        可信度数组（0-1）
    """
    # 预测概率（类别1的概率，即敏感的概率）
    proba = model.predict_proba(features)[:, 1]
    
    # 将概率作为可信度
    # 如果模型预测为敏感（高概率），则知识可信度高
    confidence = proba
    
    return confidence


def build_qpkri(
    matches: Dict[str, List[str]],
    confidence: float,
    base_prob: float = 0.1,
    max_prob: float = 0.75
) -> List[float]:
    """
    根据匹配结果和可信度构建q_PKRI
    
    Args:
        matches: 词表匹配结果 {category: [matched_words]}
        confidence: PKRI预测的可信度（0-1）
        base_prob: 基础敏感概率
        max_prob: 最大敏感概率
    
    Returns:
        q_pkri: [p_non_sensitive, p_sensitive]
    """
    # 计算匹配强度（类似q₀的逻辑）
    porn_count = len(matches.get('porn', []))
    politics_count = len(matches.get('politics', []))
    abuse_count = len(matches.get('abuse', []))
    total_matches = porn_count + politics_count + abuse_count
    
    # 如果无匹配，返回基础概率
    if total_matches == 0:
        p_sensitive = base_prob
    else:
        # 计算加权匹配分数（简化版，类似q₀）
        porn_weight = 1.0
        politics_weight = 0.6
        abuse_weight = 0.5
        
        match_score = (
            porn_weight * porn_count +
            politics_weight * politics_count +
            abuse_weight * abuse_count
        ) / max(total_matches, 1)
        
        # 使用tanh映射到[0, 1]
        normalized_score = np.tanh(match_score * 5)
        
        # 基础匹配强度
        match_strength = (normalized_score + 1) / 2
        
        # 应用可信度加权
        # confidence高时，更信任匹配结果
        # confidence低时，更接近基础概率
        p_sensitive = base_prob + (max_prob - base_prob) * match_strength * confidence
    
    # 确保概率在[0, 1]范围内
    p_sensitive = max(0.0, min(1.0, p_sensitive))
    p_non_sensitive = 1.0 - p_sensitive
    
    return [float(p_non_sensitive), float(p_sensitive)]


def generate_qpkri_for_dataset(
    feature_file: str,
    dataset_file: str,
    model: object,
    output_file: str,
    feature_cols: List[str]
):
    """
    为数据集生成q_PKRI
    
    Args:
        feature_file: 特征文件路径
        dataset_file: 原始数据集文件路径（用于获取文本和匹配信息）
        model: 训练好的模型
        output_file: 输出q_PKRI文件路径
        feature_cols: 特征列名列表
    """
    logger.info("=" * 60)
    logger.info("生成q_PKRI后验")
    logger.info("=" * 60)
    logger.info(f"特征文件: {feature_file}")
    logger.info(f"数据集文件: {dataset_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info("")
    
    # 加载特征
    X, _, _ = load_features(feature_file)
    
    # 加载原始数据（用于匹配信息）
    data_dict = {}
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                item_id = item.get('id')
                if item_id:
                    data_dict[item_id] = item
            except json.JSONDecodeError:
                continue
    
    # 加载词表（用于重新匹配，获取matches）
    from q0_builder import load_lexicon, match_lexicon, build_automaton_cache
    
    lexicon_dir = 'configs/lexicons'
    lexicons = {}
    for category in ['porn', 'politics', 'abuse']:
        lexicon_path = os.path.join(lexicon_dir, f'{category}.txt')
        if os.path.exists(lexicon_path):
            lexicons[category] = load_lexicon(lexicon_path)
    
    automaton_cache = build_automaton_cache(lexicons)
    
    # 预测可信度
    logger.info("预测知识可信度...")
    confidences = predict_confidence(X, model)
    
    # 生成q_PKRI
    logger.info("构建q_PKRI后验...")
    qpkri_items = []
    
    # 从特征文件读取id
    df = pd.read_csv(feature_file, encoding='utf-8')
    
    for idx, row in df.iterrows():
        item_id = row['id']
        
        # 获取原始数据
        if item_id not in data_dict:
            logger.warning(f"样本 {item_id} 在原始数据中不存在，跳过")
            continue
        
        item = data_dict[item_id]
        text = item.get('text', '')
        confidence = confidences[idx]
        
        # 重新匹配（获取matches）
        matches = match_lexicon(text, lexicons, automaton_cache)
        
        # 构建q_PKRI
        qpkri = build_qpkri(matches, confidence)
        
        # 构建输出项
        qpkri_item = {
            'id': item_id,
            'qpkri': qpkri,
            'confidence': float(confidence),
        }
        
        qpkri_items.append(qpkri_item)
        
        if (idx + 1) % 1000 == 0:
            logger.info(f"处理进度: {idx + 1}/{len(df)}")
    
    logger.info(f"✓ 生成完成，共 {len(qpkri_items)} 条")
    logger.info("")
    
    # 保存q_PKRI
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in qpkri_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ q_PKRI已保存: {output_file}")
    
    # 统计信息
    if qpkri_items:
        avg_confidence = np.mean([item['confidence'] for item in qpkri_items])
        avg_p_sensitive = np.mean([item['qpkri'][1] for item in qpkri_items])
        
        logger.info("")
        logger.info("q_PKRI统计:")
        logger.info(f"  平均可信度: {avg_confidence:.4f}")
        logger.info(f"  平均敏感概率: {avg_p_sensitive:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练PKRI模型并生成q_PKRI')
    parser.add_argument('--train-features', type=str, required=True,
                       help='训练集特征文件（CSV）')
    parser.add_argument('--dev-features', type=str, required=True,
                       help='验证集特征文件（CSV，用于校准）')
    parser.add_argument('--test-features', type=str, default=None,
                       help='测试集特征文件（CSV，可选）')
    parser.add_argument('--train-dataset', type=str, required=True,
                       help='训练集原始数据文件（JSONL，用于生成q_PKRI）')
    parser.add_argument('--dev-dataset', type=str, required=True,
                       help='验证集原始数据文件（JSONL，用于生成q_PKRI）')
    parser.add_argument('--test-dataset', type=str, default=None,
                       help='测试集原始数据文件（JSONL，可选）')
    parser.add_argument('--model-type', type=str, default='lr',
                       choices=['lr'],
                       help='模型类型')
    parser.add_argument('--calibration-method', type=str, default='temperature',
                       choices=['temperature', 'platt'],
                       help='校准方法')
    parser.add_argument('--output-model', type=str, required=True,
                       help='模型输出路径（.pkl）')
    parser.add_argument('--output-qpkri-dir', type=str, default='data',
                       help='q_PKRI输出目录')
    parser.add_argument('--output-metrics', type=str, default=None,
                       help='评估指标输出文件（JSON，可选）')
    
    args = parser.parse_args()
    
    # 加载特征
    logger.info("加载特征...")
    X_train, y_train, feature_cols = load_features(args.train_features)
    X_val, y_val, _ = load_features(args.dev_features)
    
    if y_train is None or y_val is None:
        logger.error("❌ 特征文件必须包含 'label' 或 'coarse_label' 字段")
        sys.exit(1)
    
    logger.info("")
    
    # 训练模型
    model, model_config = train_pkri_model(
        X_train, y_train,
        X_val, y_val,
        model_type=args.model_type,
        calibration_method=args.calibration_method
    )
    
    # 保存模型
    logger.info("")
    logger.info("保存模型...")
    output_model_dir = os.path.dirname(args.output_model)
    if output_model_dir:
        os.makedirs(output_model_dir, exist_ok=True)
    
    with open(args.output_model, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✓ 模型已保存: {args.output_model}")
    
    # 保存模型配置
    config_file = args.output_model.replace('.pkl', '_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ 模型配置已保存: {config_file}")
    
    # 生成q_PKRI
    logger.info("")
    datasets = [
        ('train', args.train_features, args.train_dataset, 'qpkri_train.jsonl'),
        ('dev', args.dev_features, args.dev_dataset, 'qpkri_dev.jsonl'),
    ]
    
    if args.test_features and args.test_dataset:
        datasets.append(('test', args.test_features, args.test_dataset, 'qpkri_test.jsonl'))
    
    for dataset_name, feature_file, dataset_file, output_filename in datasets:
        logger.info("")
        output_file = os.path.join(args.output_qpkri_dir, output_filename)
        generate_qpkri_for_dataset(
            feature_file,
            dataset_file,
            model,
            output_file,
            feature_cols
        )
    
    # 保存评估指标
    if args.output_metrics:
        metrics = {
            'model_config': model_config,
        }
        output_metrics_dir = os.path.dirname(args.output_metrics)
        if output_metrics_dir:
            os.makedirs(output_metrics_dir, exist_ok=True)
        with open(args.output_metrics, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ 评估指标已保存: {args.output_metrics}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("PKRI训练完成")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

