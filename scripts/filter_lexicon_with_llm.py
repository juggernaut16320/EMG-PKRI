"""
filter_lexicon_with_llm.py - 使用大模型过滤词表

功能：
1. 读取词表文件（politics.txt, porn.txt, abuse.txt）
2. 分批处理，每组100个词
3. 调用大模型判断是否为敏感词（严格标准）
4. 只保留极大概率是敏感词的词汇
5. 支持断点续传
"""

import os
import json
import logging
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime

# 导入llm_labeler的异步接口
from llm_labeler import (
    AsyncGeminiBackend,
    call_llm_backend_async,
    call_llm_backend_batch_with_fallback_async
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 类别定义（严格标准）
CATEGORY_DEFINITIONS = {
    "politics": """政治敏感词判断标准（极严格）：
- 只保留：明确涉及敏感政治人物、敏感政治事件、敏感政治立场的词汇
- 必须排除：
  * 所有通用缩写（如G20、BBC、CNN、FBI、CIA等）
  * 正常政治术语（如"民主"、"自由"等中性词）
  * 普通人名（除非是明确的政治敏感人物）
  * 正常组织机构名称
  * 通用词汇（如app、law、red等）
- 判断原则：如果词汇在正常语境下可以合法使用，则必须排除""",
    
    "porn": """色情敏感词判断标准（极严格）：
- 只保留：明确涉及性行为、性器官、色情内容的词汇
- 必须排除：
  * 所有通用缩写（如app、BBC、BMW、DIY、cos等）
  * 正常生理词汇（如bra、gay、les等正常性取向词汇）
  * 医学术语
  * 正常身体部位描述（除非明确用于色情语境）
  * 通用词汇（如hit、mad、mom等）
  * 正常网络用语（如awa、ovo等）
- 判断原则：如果词汇在正常语境下可以合法使用，则必须排除。只有明确用于色情表达的词汇才保留""",
    
    "abuse": """辱骂敏感词判断标准（极严格）：
- 只保留：明确的脏话、侮辱性词汇、恶意攻击性表达
- 必须排除：
  * 所有通用缩写（如CCP、GFW、GMD、MDC等）
  * 正常词汇（如awa、ban、dis、hit、mad、mom、ovo等）
  * 中性表达
  * 正常网络用语
  * 正常语气词、感叹词
- 判断原则：如果词汇在正常语境下可以合法使用，则必须排除。只有明确用于侮辱、攻击的词汇才保留"""
}


def load_lexicon(lexicon_path: str) -> List[str]:
    """加载词表文件"""
    if not os.path.exists(lexicon_path):
        logger.warning(f"词表文件不存在: {lexicon_path}")
        return []
    
    words = []
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word:
                words.append(word)
    
    logger.info(f"加载词表: {lexicon_path}，共 {len(words)} 个词")
    return words


def save_lexicon(words: List[str], output_path: str):
    """保存词表文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')
    logger.info(f"保存词表: {output_path}，共 {len(words)} 个词")


def build_prompt(words: List[str], category: str) -> str:
    """构建提示词"""
    category_definition = CATEGORY_DEFINITIONS.get(category, "")
    words_list = "\n".join([f"{i+1}. {word}" for i, word in enumerate(words)])
    
    prompt = f"""你是一个内容安全专家。请判断以下词汇列表中的每个词是否属于{category}类敏感词。

【极严格过滤标准】
1. 只保留极大概率是敏感词的词汇（严格标准，宁可少留不可多留）
2. 如果词汇在正常语境下可以合法使用，则必须排除
3. 所有通用缩写、正常词汇、中性表达都必须排除
4. 对于{category}类别，判断标准：
{category_definition}

【重要原则】
- 当不确定时，选择排除（保守策略）
- 只有明确用于敏感内容的词汇才保留
- 通用词、缩写、正常词汇一律排除

词汇列表（共{len(words)}个）：
{words_list}

请以JSON格式输出结果，格式如下：
{{
  "sensitive_words": ["词1", "词2", ...],
  "removed_words": ["词1", "词2", ...],
  "reason": "简要说明过滤原因"
}}

只输出JSON，不要其他内容。"""
    
    return prompt


def parse_response(response: str, original_words: List[str]) -> Tuple[List[str], List[str]]:
    """解析大模型响应"""
    try:
        # 尝试提取JSON
        response = response.strip()
        
        # 如果响应包含```json，提取其中的内容
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end != -1:
                response = response[start:end].strip()
        
        # 解析JSON
        result = json.loads(response)
        
        sensitive_words = result.get("sensitive_words", [])
        removed_words = result.get("removed_words", [])
        
        # 验证：确保所有词都在原始列表中
        sensitive_words = [w for w in sensitive_words if w in original_words]
        removed_words = [w for w in removed_words if w in original_words]
        
        return sensitive_words, removed_words
        
    except json.JSONDecodeError as e:
        logger.warning(f"JSON解析失败: {e}")
        logger.debug(f"响应内容: {response[:500]}")
        # 如果解析失败，返回空列表（保守策略）
        return [], original_words
    except Exception as e:
        logger.error(f"解析响应时出错: {e}")
        return [], original_words


async def filter_batch(
    words: List[str],
    category: str,
    batch_idx: int,
    backend: AsyncGeminiBackend
) -> Tuple[List[str], Dict]:
    """过滤一批词汇"""
    logger.info(f"处理批次 {batch_idx}，共 {len(words)} 个词")
    
    # 构建批量提示词
    batch_prompt = build_prompt(words, category)
    
    # 构建单条提示词（用于回退）
    single_prompts = [build_prompt([word], category) for word in words]
    
    try:
        # 尝试批量处理
        batch_response, single_results = await call_llm_backend_batch_with_fallback_async(
            batch_prompt=batch_prompt,
            single_prompts=single_prompts,
            max_retries=3,
            retry_delay=1.0,
            max_concurrent=5,
            backend=backend
        )
        
        sensitive_words = []
        removed_words = []
        
        if batch_response:
            # 批量处理成功
            logger.info(f"批次 {batch_idx} 批量处理成功")
            sensitive_words, removed_words = parse_response(batch_response, words)
        else:
            # 回退到逐条处理
            logger.info(f"批次 {batch_idx} 回退到逐条处理，共 {len(single_results)} 条")
            for idx, (word_idx, response, error) in enumerate(single_results):
                if error:
                    logger.warning(f"批次 {batch_idx} 词 {word_idx} ({words[word_idx]}) 处理失败: {error}")
                    # 失败时保守处理：不保留
                    removed_words.append(words[word_idx])
                else:
                    word_sensitive, word_removed = parse_response(response, [words[word_idx]])
                    sensitive_words.extend(word_sensitive)
                    removed_words.extend(word_removed)
        
        stats = {
            "batch_idx": batch_idx,
            "total": len(words),
            "sensitive": len(sensitive_words),
            "removed": len(removed_words),
            "filter_rate": len(removed_words) / len(words) if words else 0
        }
        
        logger.info(f"批次 {batch_idx} 完成: 保留 {len(sensitive_words)} 个，过滤 {len(removed_words)} 个")
        
        return sensitive_words, stats
        
    except Exception as e:
        logger.error(f"批次 {batch_idx} 处理失败: {e}")
        # 失败时保守处理：不保留任何词
        stats = {
            "batch_idx": batch_idx,
            "total": len(words),
            "sensitive": 0,
            "removed": len(words),
            "filter_rate": 1.0,
            "error": str(e)
        }
        return [], stats


def load_progress(progress_file: str) -> Dict:
    """加载进度"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "processed_batches": [],
        "sensitive_words": [],
        "stats": []
    }


def save_progress(progress_file: str, progress: Dict):
    """保存进度"""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


async def filter_lexicon(
    lexicon_path: str,
    category: str,
    output_path: str,
    batch_size: int = 100,
    resume: bool = True
):
    """过滤词表"""
    logger.info("=" * 60)
    logger.info(f"开始过滤词表: {lexicon_path}")
    logger.info(f"类别: {category}")
    logger.info(f"批次大小: {batch_size}")
    logger.info("=" * 60)
    
    # 加载词表
    all_words = load_lexicon(lexicon_path)
    if not all_words:
        logger.error("词表为空，退出")
        return
    
    # 进度文件
    progress_file = output_path + ".progress.json"
    progress = load_progress(progress_file) if resume else {
        "processed_batches": [],
        "sensitive_words": [],
        "stats": []
    }
    
    # 初始化后端
    backend = AsyncGeminiBackend()
    
    # 分批处理
    total_batches = (len(all_words) + batch_size - 1) // batch_size
    processed_batches = set(progress["processed_batches"])
    all_sensitive_words = set(progress["sensitive_words"])
    all_stats = progress["stats"]
    
    logger.info(f"总词数: {len(all_words)}")
    logger.info(f"总批次数: {total_batches}")
    logger.info(f"已处理批次数: {len(processed_batches)}")
    logger.info("")
    
    for batch_idx in range(total_batches):
        if batch_idx in processed_batches:
            logger.info(f"跳过已处理的批次 {batch_idx}")
            continue
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_words))
        batch_words = all_words[start_idx:end_idx]
        
        # 过滤批次
        sensitive_words, stats = await filter_batch(
            batch_words,
            category,
            batch_idx,
            backend
        )
        
        # 更新进度
        all_sensitive_words.update(sensitive_words)
        processed_batches.add(batch_idx)
        all_stats.append(stats)
        
        progress = {
            "processed_batches": list(processed_batches),
            "sensitive_words": list(all_sensitive_words),
            "stats": all_stats
        }
        save_progress(progress_file, progress)
        
        logger.info("")
    
    # 保存最终结果
    final_words = sorted(list(all_sensitive_words))
    save_lexicon(final_words, output_path)
    
    # 统计信息
    total_removed = len(all_words) - len(final_words)
    filter_rate = total_removed / len(all_words) if all_words else 0
    
    logger.info("=" * 60)
    logger.info("过滤完成！")
    logger.info(f"原始词数: {len(all_words)}")
    logger.info(f"保留词数: {len(final_words)}")
    logger.info(f"过滤词数: {total_removed}")
    logger.info(f"过滤率: {filter_rate:.2%}")
    logger.info(f"输出文件: {output_path}")
    logger.info("=" * 60)
    
    # 删除进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)
        logger.info(f"已删除进度文件: {progress_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用大模型过滤词表')
    parser.add_argument('--lexicon-dir', type=str, default='configs/lexicons',
                       help='词表目录')
    parser.add_argument('--categories', type=str, nargs='+', 
                       default=['politics', 'porn', 'abuse'],
                       help='要处理的类别')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='批次大小（每组词数）')
    parser.add_argument('--no-resume', action='store_true',
                       help='不恢复进度，从头开始')
    
    args = parser.parse_args()
    
    # 检查环境变量
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("❌ GEMINI_API_KEY 环境变量未设置")
        logger.error("请设置环境变量或创建 .env 文件")
        return
    
    # 处理每个类别
    for category in args.categories:
        lexicon_path = os.path.join(args.lexicon_dir, f"{category}.txt")
        output_path = os.path.join(args.lexicon_dir, f"{category}.txt.filtered")
        
        if not os.path.exists(lexicon_path):
            logger.warning(f"词表文件不存在: {lexicon_path}，跳过")
            continue
        
        try:
            asyncio.run(filter_lexicon(
                lexicon_path=lexicon_path,
                category=category,
                output_path=output_path,
                batch_size=args.batch_size,
                resume=not args.no_resume
            ))
        except KeyboardInterrupt:
            logger.info("用户中断，已保存进度")
        except Exception as e:
            logger.error(f"处理 {category} 时出错: {e}", exc_info=True)


if __name__ == '__main__':
    main()

