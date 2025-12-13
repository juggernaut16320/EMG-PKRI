"""
lexicon_generator.py - 使用大模型生成词表、关键词和正则规则

功能：
1. 从标注数据中按类别采样样本
2. 调用大模型生成词表（porn/politics/abuse）
3. 调用大模型生成正则规则
4. 保存到 configs/lexicons/ 目录
"""

import os
import sys
import json
import logging
import argparse
import yaml
import random
import asyncio
import shutil
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
import time

# 添加 scripts 目录到路径（用于导入 llm_labeler）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_labeler import (
    call_llm_backend, 
    load_config,
    call_llm_backend_batch_with_fallback_async,
    call_llm_backend_async,
    AsyncGeminiBackend
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============ Prompt 模板 ============

LEXICON_GENERATION_PROMPT = """请分析以下敏感文本样本（类别：{category}），提取出该类别相关的关键词。

类别说明：
- porn: 色情内容相关
- politics: 涉政内容相关
- abuse: 辱骂/攻击性内容相关

样本列表（JSON格式）：
{texts_json}

要求：
1. 提取直接相关的敏感词（包括中英文）
2. 提取相关的同义词、近义词
3. 提取相关的变体词（拼音、谐音等）
4. 每个词一行，只输出词，不要其他内容
5. 去除重复词
6. 至少提取 20 个词，最多 500 个词

输出格式（每行一个词，不要编号，不要说明）：
"""

REGEX_GENERATION_PROMPT = """请为以下敏感词类别设计正则表达式规则，用于匹配常见的变体形式。

类别：{category}
示例敏感词（前10个）：
{sample_words}

要求：
1. 设计匹配拼音变体的正则（如：cao -> cao|草|操）
2. 设计匹配谐音变体的正则（如：fuck -> f[u*]ck|f[u*]k）
3. 设计匹配特殊字符插入的正则（如：f*u*c*k）
4. 设计匹配数字替代的正则（如：f0ck, f4ck）
5. 每个正则一行，格式：pattern|description
6. 至少设计 15 个正则规则

输出格式（每行一个正则，格式：pattern|description）：
"""

LEXICON_CLEANING_PROMPT = """请清洗以下敏感词表（类别：{category}），只返回应该保留的词。

类别说明：
- porn: 色情内容相关
- politics: 涉政内容相关  
- abuse: 辱骂/攻击性内容相关

词表（每行一个词）：
{words_list}

清洗规则（应删除的词，不要返回这些）：
1. 单字符词（如：A, B, +, @）
2. 表情符号（如：😂, 😍, 🔥）
3. 无意义的编号/代码（如：ABF-061, ALDN-290）
4. 明显正常的日常词汇（如：连衣裙, 刷牙, 攻略）
5. 纯数字或纯字母且长度≤2的词
6. 标签符号开头的词（如：#tag, @user）
7. 重复词（同一词出现多次）

保留规则（应返回的词）：
1. 真正的敏感词（与类别相关）
2. 敏感词的变体、谐音、拼音
3. 长度≥2且有实际含义的词

输出格式（只输出要保留的词，每行一个词，不要编号，不要说明，不要JSON格式）：
"""


# ============ 核心函数 ============

def load_data_samples(
    input_path: str,
    category: str,
    samples_per_category: Optional[int] = None,
    subtype_field: str = "subtype_label",
    coarse_field: str = "coarse_label"
) -> List[Dict]:
    """
    从数据文件中加载指定类别的样本
    
    Args:
        input_path: 输入JSONL文件路径
        category: 类别名称（porn/politics/abuse）
        samples_per_category: 每个类别采样数量，None表示使用全部样本
        subtype_field: 子标签字段名
        coarse_field: 粗标签字段名
    
    Returns:
        样本列表
    """
    logger.info(f"开始加载 {category} 类别的样本...")
    samples = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                # 只处理敏感样本（coarse_label=1）
                if item.get(coarse_field) != 1:
                    continue
                
                # 检查子标签
                subtype_labels = item.get(subtype_field, [])
                if not isinstance(subtype_labels, list):
                    subtype_labels = []
                
                # 如果样本包含该类别，则加入
                if category in subtype_labels:
                    samples.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效JSON行: {e}")
                continue
    
    logger.info(f"找到 {len(samples)} 个 {category} 类别样本")
    
    # 如果指定了采样数量，则进行采样；否则使用全部样本
    if samples_per_category is not None and len(samples) > samples_per_category:
        samples = random.sample(samples, samples_per_category)
        logger.info(f"随机采样 {samples_per_category} 个样本")
    else:
        logger.info(f"使用全部 {len(samples)} 个样本")
    
    return samples


async def generate_lexicon_from_samples_async(
    samples: List[Dict],
    category: str,
    text_field: str = "text",
    max_retries: int = 3,
    retry_delay: float = 2.5,
    request_interval: float = 2.5,
    batch_size: int = 10,
    output_path: Optional[str] = None,
    use_async: bool = True
) -> List[str]:
    """
    异步版本：使用大模型从样本中生成词表（批量处理，支持动态追加到文件）
    支持批量失败后自动回退到逐条处理
    """
    logger.info(f"开始为 {category} 类别生成词表（样本数：{len(samples)}，批量大小：{batch_size}，异步模式：{use_async}）...")
    
    # 准备文本列表
    texts = [item.get(text_field, "") for item in samples if item.get(text_field)]
    
    all_words = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # 如果提供输出路径，准备动态追加模式
    existing_words = set()
    if output_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 如果文件已存在，读取已有词（用于去重）
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_words = {line.strip() for line in f if line.strip()}
                logger.info(f"从现有文件读取 {len(existing_words)} 个已有词，将继续追加新词...")
            except Exception as e:
                logger.warning(f"读取现有文件时出错: {e}，将创建新文件")
    
    if not use_async:
        # 回退到同步版本
        return generate_lexicon_from_samples(
            samples, category, text_field, max_retries, retry_delay, 
            request_interval, batch_size, output_path
        )
    
    # 创建异步后端实例（共享，确保速率限制统一）
    async_backend = AsyncGeminiBackend(max_rate=30.0)
    
    # 使用列表来存储所有批次的结果（避免闭包问题）
    batch_results = []
    
    # 异步处理单个批次
    async def process_batch_async(batch_idx: int, batch_texts: List[str], batch_num: int):
        """异步处理单个批次，支持批量失败后回退到逐条处理"""
        batch_start_time = time.time()
        
        # 准备批量Prompt
        prompt_prep_start = time.time()
        texts_json = json.dumps(batch_texts, ensure_ascii=False, indent=2)
        batch_prompt = LEXICON_GENERATION_PROMPT.format(
            category=category,
            texts_json=texts_json
        )
        
        # 准备逐条Prompt（用于回退）
        single_prompts = [
            LEXICON_GENERATION_PROMPT.format(
                category=category,
                texts_json=json.dumps([text], ensure_ascii=False, indent=2)
            )
            for text in batch_texts
        ]
        prompt_prep_elapsed = time.time() - prompt_prep_start
        logger.debug(f"批次 {batch_num} Prompt准备耗时: {prompt_prep_elapsed:.3f}秒")
        
        # 调用异步批量处理（支持失败回退）
        api_call_start = time.time()
        batch_response, single_results = await call_llm_backend_batch_with_fallback_async(
            batch_prompt=batch_prompt,
            single_prompts=single_prompts,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_concurrent=5,  # 逐条处理时的最大并发数
            backend=async_backend
        )
        api_call_elapsed = time.time() - api_call_start
        logger.info(f"批次 {batch_num} API调用总耗时: {api_call_elapsed:.2f}秒")
        
        batch_words = []
        
        if batch_response:
            # 批量处理成功
            # 解析批量响应
            for line in batch_response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 移除可能的编号（如 "1. word" -> "word"）
                line = line.lstrip('0123456789. \t-')
                if line:
                    batch_words.append(line)
            
            # 检查返回的词表数量是否合理
            if len(batch_words) < 5:
                logger.warning(f"批次 {batch_num} 返回的词表数量过少（{len(batch_words)} 个），使用逐条处理结果...")
                # 批量响应词太少，使用逐条处理结果
                batch_words = []  # 清空，使用逐条处理结果
                for idx, response, error in single_results:
                    if error is None and response:
                        for line in response.strip().split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            line = line.lstrip('0123456789. \t-')
                            if line:
                                batch_words.append(line)
                    elif error:
                        # 检查是否是超时错误
                        error_msg = str(error)
                        if "超时" in error_msg or "Timeout" in error_msg or isinstance(error, TimeoutError):
                            logger.warning(f"批次 {batch_num} 样本 {idx} 超时跳过（>20秒）")
                        else:
                            logger.warning(f"批次 {batch_num} 样本 {idx} 处理失败: {error}")
            else:
                logger.info(f"批次 {batch_num} 成功生成 {len(batch_words)} 个词")
        else:
            # 批量处理失败，使用逐条处理结果
            logger.info(f"批次 {batch_num} 批量处理失败，使用逐条处理结果")
            for idx, response, error in single_results:
                if error is None and response:
                    for line in response.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        line = line.lstrip('0123456789. \t-')
                        if line:
                            batch_words.append(line)
                elif error:
                    logger.warning(f"批次 {batch_num} 样本 {idx} 处理失败: {error}")
            
            if batch_words:
                logger.info(f"批次 {batch_num}（逐条处理模式）共生成 {len(batch_words)} 个词")
        
        # 追加到总词表
        all_words.extend(batch_words)
        
        # 如果提供了输出路径，立即追加新词到文件（去重后）
        if output_path:
            new_words = [w for w in batch_words if w not in existing_words]
            if new_words:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for word in new_words:
                        f.write(word + '\n')
                existing_words.update(new_words)
                logger.info(f"批次 {batch_num} 追加 {len(new_words)} 个新词到文件（跳过 {len(batch_words) - len(new_words)} 个重复词）")
        
        batch_results.append(batch_words)
        
        batch_total_elapsed = time.time() - batch_start_time
        logger.info(f"批次 {batch_num} 总耗时: {batch_total_elapsed:.2f}秒（API: {api_call_elapsed:.2f}秒，其他: {batch_total_elapsed - api_call_elapsed:.2f}秒）")
        
        return batch_words
    
    # 限制并发批次数量，避免所有批次同时启动导致速率限制器串行化
    max_concurrent_batches = min(5, total_batches)  # 最多同时处理5个批次
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_batch_with_limit(batch_idx: int, batch_texts: List[str], batch_num: int):
        """带并发限制的批次处理"""
        async with semaphore:
            return await process_batch_async(batch_idx, batch_texts, batch_num)
    
    # 创建所有批次任务
    tasks = []
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        logger.info(f"准备批次 {batch_num}/{total_batches}（{len(batch_texts)} 条样本）...")
        tasks.append(process_batch_with_limit(batch_idx, batch_texts, batch_num))
    
    # 并发执行所有批次（受并发限制和速率限制器控制）
    total_start_time = time.time()
    logger.info(f"开始并发处理 {len(tasks)} 个批次（最多同时 {max_concurrent_batches} 个）...")
    results = await asyncio.gather(*tasks)
    total_elapsed = time.time() - total_start_time
    logger.info(f"所有批次处理完成，总耗时: {total_elapsed:.2f}秒，平均每批次: {total_elapsed/len(tasks):.2f}秒")
    
    # 去重并排序
    all_words = sorted(list(set(all_words)))
    
    # 如果使用了动态追加模式，重新整理文件（去重、排序）
    if output_path:
        logger.info(f"整理文件：去重并排序...")
        # 读取文件中所有词（包括刚才追加的）
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                file_words = sorted(list(set(line.strip() for line in f if line.strip())))
            # 重新写入（覆盖）
            with open(output_path, 'w', encoding='utf-8') as f:
                for word in file_words:
                    f.write(word + '\n')
            logger.info(f"✓ 文件已整理，共 {len(file_words)} 个词")
            all_words = file_words
    
    logger.info(f"✓ {category} 词表生成完成，共 {len(all_words)} 个词（来自 {total_batches} 个批次）")
    return all_words


def generate_lexicon_from_samples(
    samples: List[Dict],
    category: str,
    text_field: str = "text",
    max_retries: int = 3,
    retry_delay: float = 2.5,
    request_interval: float = 2.5,
    batch_size: int = 10,
    output_path: Optional[str] = None,
    use_async: bool = True
) -> List[str]:
    """
    使用大模型从样本中生成词表（批量处理，支持动态追加到文件）
    
    Args:
        samples: 样本列表
        category: 类别名称
        text_field: 文本字段名
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        request_interval: 请求间隔（秒），确保不超过API限制（30次/分钟 = 2秒/次）
        batch_size: 批量处理大小（默认10条）
        output_path: 输出文件路径，如果提供则每批生成后立即追加到文件
    
    Returns:
        词表列表（去重、排序）
    """
    # 如果使用异步模式，调用异步版本
    if use_async:
        try:
            # 检查是否已有事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有事件循环在运行，需要特殊处理
                logger.warning("检测到运行中的事件循环，尝试使用异步版本...")
                # 创建一个新的异步任务
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(generate_lexicon_from_samples_async(
                    samples, category, text_field, max_retries, retry_delay,
                    request_interval, batch_size, output_path, use_async=True
                ))
            else:
                # 没有运行中的事件循环，可以直接使用 asyncio.run
                return asyncio.run(generate_lexicon_from_samples_async(
                    samples, category, text_field, max_retries, retry_delay,
                    request_interval, batch_size, output_path, use_async=True
                ))
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(generate_lexicon_from_samples_async(
                samples, category, text_field, max_retries, retry_delay,
                request_interval, batch_size, output_path, use_async=True
            ))
        except ImportError:
            # nest_asyncio 未安装，回退到同步版本
            logger.warning("nest_asyncio 未安装，回退到同步模式。安装命令: pip install nest-asyncio")
            use_async = False
        except Exception as e:
            logger.warning(f"异步模式调用失败，回退到同步模式: {e}")
            use_async = False
    
    # 同步版本（原有逻辑）
    logger.info(f"开始为 {category} 类别生成词表（样本数：{len(samples)}，批量大小：{batch_size}）...")
    
    # 准备文本列表
    texts = [item.get(text_field, "") for item in samples if item.get(text_field)]
    
    all_words = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # 如果提供输出路径，准备动态追加模式
    existing_words = set()
    if output_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # 如果文件已存在，读取已有词（用于去重）
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_words = {line.strip() for line in f if line.strip()}
                logger.info(f"从现有文件读取 {len(existing_words)} 个已有词，将继续追加新词...")
            except Exception as e:
                logger.warning(f"读取现有文件时出错: {e}，将创建新文件")
    
    # 分批处理
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"处理批次 {batch_num}/{total_batches}（{len(batch_texts)} 条样本）...")
        
        # 准备批量Prompt
        texts_json = json.dumps(batch_texts, ensure_ascii=False, indent=2)
        prompt = LEXICON_GENERATION_PROMPT.format(
            category=category,
            texts_json=texts_json
        )
        
        # 调用大模型
        try:
            response = call_llm_backend(prompt, max_retries=max_retries, retry_delay=retry_delay)
            
            # 解析响应
            batch_words = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # 移除可能的编号（如 "1. word" -> "word"）
                line = line.lstrip('0123456789. \t-')
                if line:
                    batch_words.append(line)
            
            # 检查返回的词表数量是否合理
            # 如果返回的词太少（少于5个），可能是解析失败，需要单独处理
            if len(batch_words) < 5:
                logger.warning(f"批次 {batch_num} 返回的词表数量过少（{len(batch_words)} 个），尝试单独处理...")
                # 单独处理这一批的每条样本
                for text in batch_texts:
                    try:
                        single_prompt = LEXICON_GENERATION_PROMPT.format(
                            category=category,
                            texts_json=json.dumps([text], ensure_ascii=False, indent=2)
                        )
                        single_response = call_llm_backend(single_prompt, max_retries=max_retries, retry_delay=retry_delay)
                        
                        # 解析单条响应
                        for line in single_response.strip().split('\n'):
                            line = line.strip()
                            if not line:
                                continue
                            line = line.lstrip('0123456789. \t-')
                            if line:
                                batch_words.append(line)
                        
                        # 单条处理间隔
                        if request_interval > 0:
                            time.sleep(request_interval)
                    except Exception as e:
                        logger.warning(f"单独处理样本时出错: {e}")
                        continue
            else:
                logger.info(f"批次 {batch_num} 成功生成 {len(batch_words)} 个词")
            
            all_words.extend(batch_words)
            
            # 如果提供了输出路径，立即追加新词到文件（去重后）
            if output_path:
                new_words = [w for w in batch_words if w not in existing_words]
                if new_words:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        for word in new_words:
                            f.write(word + '\n')
                    existing_words.update(new_words)
                    logger.info(f"批次 {batch_num} 追加 {len(new_words)} 个新词到文件（跳过 {len(batch_words) - len(new_words)} 个重复词）")
            
        except Exception as e:
            logger.error(f"批次 {batch_num} 处理失败: {e}，尝试单独处理...")
            # 批量失败，单独处理这一批的每条样本
            batch_words_from_single = []
            for text in batch_texts:
                try:
                    single_prompt = LEXICON_GENERATION_PROMPT.format(
                        category=category,
                        texts_json=json.dumps([text], ensure_ascii=False, indent=2)
                    )
                    single_response = call_llm_backend(single_prompt, max_retries=max_retries, retry_delay=retry_delay)
                    
                    # 解析单条响应
                    for line in single_response.strip().split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        line = line.lstrip('0123456789. \t-')
                        if line:
                            batch_words_from_single.append(line)
                            all_words.append(line)
                    
                    # 单条处理间隔
                    if request_interval > 0:
                        time.sleep(request_interval)
                except Exception as e2:
                    logger.warning(f"单独处理样本时出错: {e2}")
                    continue
            
            # 批量失败后单独处理时，也动态追加
            if output_path and batch_words_from_single:
                new_words = [w for w in batch_words_from_single if w not in existing_words]
                if new_words:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        for word in new_words:
                            f.write(word + '\n')
                    existing_words.update(new_words)
                    logger.info(f"批次 {batch_num}（单独处理模式）追加 {len(new_words)} 个新词到文件")
        
        # 批量请求间隔
        if request_interval > 0 and batch_idx + batch_size < len(texts):
            time.sleep(request_interval)
    
    # 去重并排序
    all_words = sorted(list(set(all_words)))
    
    # 如果使用了动态追加模式，重新整理文件（去重、排序）
    if output_path:
        logger.info(f"整理文件：去重并排序...")
        # 读取文件中所有词（包括刚才追加的）
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                file_words = sorted(list(set(line.strip() for line in f if line.strip())))
            # 重新写入（覆盖）
            with open(output_path, 'w', encoding='utf-8') as f:
                for word in file_words:
                    f.write(word + '\n')
            logger.info(f"✓ 文件已整理，共 {len(file_words)} 个词")
            all_words = file_words
    
    logger.info(f"✓ {category} 词表生成完成，共 {len(all_words)} 个词（来自 {total_batches} 个批次）")
    return all_words


async def generate_regex_patterns_async(
    words: List[str],
    category: str,
    max_retries: int = 3,
    retry_delay: float = 2.5,
    sample_words_count: int = 20,
    output_path: Optional[str] = None,
    backend: Optional[AsyncGeminiBackend] = None,
    timeout: float = 20.0
) -> List[str]:
    """
    异步版本：使用大模型生成正则规则（支持动态追加到文件，带超时跳过）
    
    Args:
        words: 词表列表
        category: 类别名称
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        sample_words_count: 示例词数量
        output_path: 输出文件路径，如果提供则生成后立即追加到文件
        backend: 可选的异步后端实例
        timeout: API调用超时时间（秒），超过则跳过
    
    Returns:
        正则规则列表（格式：pattern|description）
    """
    logger.info(f"开始为 {category} 类别生成正则规则（异步模式，超时: {timeout}秒）...")
    
    # 随机采样示例词，确保多样性
    if len(words) >= sample_words_count:
        sample_words = random.sample(words, sample_words_count)
        logger.info(f"随机采样 {sample_words_count} 个词作为示例")
    else:
        sample_words = words
        logger.info(f"词表数量不足，使用全部 {len(words)} 个词作为示例")
    
    sample_words_str = '\n'.join(sample_words)
    
    # 构建Prompt
    prompt = REGEX_GENERATION_PROMPT.format(
        category=category,
        sample_words=sample_words_str
    )
    
    # 调用大模型（异步版本，带超时）
    logger.info(f"调用大模型生成 {category} 正则规则...")
    try:
        if backend is None:
            backend = AsyncGeminiBackend(max_rate=30.0)
        
        # 添加超时限制
        response = await asyncio.wait_for(
            call_llm_backend_async(
                prompt, 
                max_retries=max_retries, 
                retry_delay=retry_delay,
                backend=backend
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"生成 {category} 正则规则超时（>{timeout}秒），跳过本次调用")
        return []
    except Exception as e:
        logger.error(f"生成 {category} 正则规则失败: {e}")
        return []
    
    # 解析响应
    patterns = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # 移除可能的编号
        line = line.lstrip('0123456789. \t-')
        if line and '|' in line:
            patterns.append(line)
    
    # 如果提供了输出路径，立即追加到文件（去重）
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        existing_patterns = set()
        # 读取已有规则
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_patterns = {line.strip() for line in f if line.strip()}
            except Exception as e:
                logger.warning(f"读取现有正则规则文件时出错: {e}")
        
        # 追加新规则（去重）
        new_patterns = [p for p in patterns if p not in existing_patterns]
        if new_patterns:
            with open(output_path, 'a', encoding='utf-8') as f:
                for pattern in new_patterns:
                    f.write(pattern + '\n')
            logger.info(f"追加 {len(new_patterns)} 个新正则规则到文件（跳过 {len(patterns) - len(new_patterns)} 个重复规则）")
        
        # 合并所有规则（包括已有的）
        patterns = list(existing_patterns) + new_patterns
    
    logger.info(f"✓ {category} 正则规则生成完成，共 {len(patterns)} 个规则")
    return patterns


def generate_regex_patterns(
    words: List[str],
    category: str,
    max_retries: int = 3,
    retry_delay: float = 2.5,
    request_interval: float = 2.5,
    sample_words_count: int = 20,
    output_path: Optional[str] = None,
    use_async: bool = True,
    timeout: float = 20.0
) -> List[str]:
    """
    使用大模型生成正则规则（支持动态追加到文件）
    
    Args:
        words: 词表列表
        category: 类别名称
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        request_interval: 请求间隔（秒），确保不超过API限制（30次/分钟 = 2秒/次）（仅同步模式使用）
        sample_words_count: 示例词数量
        output_path: 输出文件路径，如果提供则生成后立即追加到文件
        use_async: 是否使用异步模式（默认True）
        timeout: API调用超时时间（秒），超过则跳过（仅异步模式使用）
    
    Returns:
        正则规则列表（格式：pattern|description）
    """
    # 如果使用异步模式，调用异步版本
    if use_async:
        try:
            # 检查是否已有事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有事件循环在运行，需要特殊处理
                logger.warning("检测到运行中的事件循环，尝试使用异步版本...")
                # 创建一个新的异步任务
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(generate_regex_patterns_async(
                        words, category, max_retries, retry_delay,
                        sample_words_count, output_path, None, timeout
                    ))
                except ImportError:
                    logger.warning("nest_asyncio 未安装，回退到同步模式。安装命令: pip install nest-asyncio")
                    use_async = False
            else:
                # 没有运行中的事件循环，可以直接使用 asyncio.run
                return asyncio.run(generate_regex_patterns_async(
                    words, category, max_retries, retry_delay,
                    sample_words_count, output_path, None, timeout
                ))
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(generate_regex_patterns_async(
                words, category, max_retries, retry_delay,
                sample_words_count, output_path, None, timeout
            ))
        except Exception as e:
            logger.warning(f"异步模式调用失败，回退到同步模式: {e}")
            use_async = False
    
    # 同步版本（回退）
    logger.info(f"开始为 {category} 类别生成正则规则（同步模式）...")
    
    # 随机采样示例词，确保多样性
    if len(words) >= sample_words_count:
        sample_words = random.sample(words, sample_words_count)
        logger.info(f"随机采样 {sample_words_count} 个词作为示例")
    else:
        sample_words = words
        logger.info(f"词表数量不足，使用全部 {len(words)} 个词作为示例")
    
    sample_words_str = '\n'.join(sample_words)
    
    # 构建Prompt
    prompt = REGEX_GENERATION_PROMPT.format(
        category=category,
        sample_words=sample_words_str
    )
    
    # 调用大模型
    logger.info(f"调用大模型生成 {category} 正则规则...")
    response = call_llm_backend(prompt, max_retries=max_retries, retry_delay=retry_delay)
    
    # 添加请求间隔，确保不超过API限制（30次/分钟）
    if request_interval > 0:
        logger.debug(f"等待 {request_interval} 秒后继续（API限制：30次/分钟）")
        time.sleep(request_interval)
    
    # 解析响应
    patterns = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # 移除可能的编号
        line = line.lstrip('0123456789. \t-')
        if line and '|' in line:
            patterns.append(line)
    
    # 如果提供了输出路径，立即追加到文件（去重）
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        existing_patterns = set()
        # 读取已有规则
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_patterns = {line.strip() for line in f if line.strip()}
            except Exception as e:
                logger.warning(f"读取现有正则规则文件时出错: {e}")
        
        # 追加新规则（去重）
        new_patterns = [p for p in patterns if p not in existing_patterns]
        if new_patterns:
            with open(output_path, 'a', encoding='utf-8') as f:
                for pattern in new_patterns:
                    f.write(pattern + '\n')
            logger.info(f"追加 {len(new_patterns)} 个新正则规则到文件（跳过 {len(patterns) - len(new_patterns)} 个重复规则）")
        
        # 合并所有规则（包括已有的）
        patterns = list(existing_patterns) + new_patterns
    
    logger.info(f"✓ {category} 正则规则生成完成，共 {len(patterns)} 个规则")
    return patterns


def save_lexicon_file(
    words: List[str],
    output_path: str,
    category: str
):
    """
    保存词表到文件
    
    Args:
        words: 词表列表
        output_path: 输出文件路径
        category: 类别名称
    """
    logger.info(f"保存 {category} 词表到: {output_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word + '\n')
    
    logger.info(f"✓ {category} 词表已保存，共 {len(words)} 个词")


def save_regex_file(
    patterns: List[str],
    output_path: str
):
    """
    保存正则规则到文件
    
    Args:
        patterns: 正则规则列表
        output_path: 输出文件路径
    """
    logger.info(f"保存正则规则到: {output_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pattern in patterns:
            f.write(pattern + '\n')
    
    logger.info(f"✓ 正则规则已保存，共 {len(patterns)} 个规则")


# ============ 词表清洗功能 ============

async def clean_lexicon_batch_async(
    batch_words: List[str],
    category: str,
    batch_idx: int,
    total_batches: int,
    async_backend: AsyncGeminiBackend,
    max_retries: int = 3,
    retry_delay: float = 2.5
) -> List[str]:
    """
    异步清洗单个批次的词表
    
    Args:
        batch_words: 批次词表列表
        category: 类别名称
        batch_idx: 批次索引
        total_batches: 总批次数
        async_backend: 异步后端实例
        max_retries: 最大重试次数
        retry_delay: 重试延迟
    
    Returns:
        清洗后保留的词列表
    """
    prompt = LEXICON_CLEANING_PROMPT.format(
        category=category,
        words_list='\n'.join(batch_words)
    )
    
    cleaned_words = []
    
    for attempt in range(max_retries):
        try:
            response = await call_llm_backend_async(
                prompt,
                max_retries=1,  # 这里只重试一次，外层循环处理重试
                retry_delay=retry_delay,
                backend=async_backend
            )
            
            if response:
                # 解析响应，提取保留的词
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    # 移除可能的编号（如 "1. word" -> "word"）
                    line = line.lstrip('0123456789. \t-')
                    if line and line in batch_words:  # 只保留原始词表中的词
                        cleaned_words.append(line)
                
                logger.info(f"批次 {batch_idx + 1}/{total_batches} 清洗完成，保留 {len(cleaned_words)}/{len(batch_words)} 个词")
                return cleaned_words
            else:
                logger.warning(f"批次 {batch_idx + 1}/{total_batches} 返回空响应（尝试 {attempt + 1}/{max_retries}）")
        
        except Exception as e:
            logger.warning(f"批次 {batch_idx + 1}/{total_batches} 清洗失败（尝试 {attempt + 1}/{max_retries}）: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"批次 {batch_idx + 1}/{total_batches} 清洗失败，已重试 {max_retries} 次，跳过该批次")
    
    return cleaned_words


async def clean_lexicon_with_llm_async(
    words: List[str],
    category: str,
    batch_size: int = 100,
    max_retries: int = 3,
    retry_delay: float = 2.5,
    temp_file_path: Optional[str] = None
) -> Tuple[List[str], int]:
    """
    使用 LLM 异步批量清洗词表
    
    Args:
        words: 原始词表列表
        category: 类别名称
        batch_size: 每批处理的词数
        max_retries: 最大重试次数
        retry_delay: 重试延迟
        temp_file_path: 临时文件路径（用于逐次追加结果）
    
    Returns:
        (清洗后的词表, 删除的词数)
    """
    logger.info(f"开始清洗 {category} 词表（原始词数：{len(words)}，批次大小：{batch_size}）...")
    
    # 检查API key是否设置
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("=" * 60)
        logger.error("GEMINI_API_KEY 环境变量未设置，无法使用LLM清洗")
        logger.error("=" * 60)
        logger.error("请设置环境变量后再运行：")
        logger.error("  PowerShell: $env:GEMINI_API_KEY = 'your_api_key'")
        logger.error("  CMD: set GEMINI_API_KEY=your_api_key")
        logger.error("  Linux/Mac: export GEMINI_API_KEY=your_api_key")
        raise ValueError("GEMINI_API_KEY 环境变量未设置")
    
    # 过滤空词
    filtered_words = [w.strip() for w in words if w.strip()]
    if len(filtered_words) == 0:
        logger.warning(f"词表为空，清洗完成")
        return [], len(words)
    
    # 准备临时文件
    if temp_file_path:
        os.makedirs(os.path.dirname(temp_file_path) if os.path.dirname(temp_file_path) else '.', exist_ok=True)
        # 清空临时文件（如果存在）
        if os.path.exists(temp_file_path):
            open(temp_file_path, 'w', encoding='utf-8').close()
    
    # 分批处理（LLM清洗）
    total_batches = (len(filtered_words) + batch_size - 1) // batch_size
    logger.info(f"使用LLM清洗词表（共 {total_batches} 个批次）...")
    
    # 创建异步后端实例（共享，确保速率限制统一）
    async_backend = AsyncGeminiBackend(max_rate=30.0)
    
    # 创建批次任务
    batch_tasks = []
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(filtered_words))
        batch_words = filtered_words[start_idx:end_idx]
        
        task = clean_lexicon_batch_async(
            batch_words,
            category,
            batch_idx,
            total_batches,
            async_backend,
            max_retries,
            retry_delay
        )
        batch_tasks.append((batch_idx, task))
    
    # 并发执行批次任务（但受速率限制）
    all_cleaned_words = []
    for batch_idx, task in batch_tasks:
        try:
            batch_cleaned = await task
            
            # 追加到临时文件
            if temp_file_path and batch_cleaned:
                with open(temp_file_path, 'a', encoding='utf-8') as f:
                    for word in batch_cleaned:
                        f.write(word + '\n')
            
            all_cleaned_words.extend(batch_cleaned)
            
        except Exception as e:
            logger.error(f"批次 {batch_idx + 1} 处理异常: {e}")
            continue
    
    # 4. 从临时文件读取并去重（如果使用临时文件）
    if temp_file_path and os.path.exists(temp_file_path):
        logger.info("步骤3: 从临时文件读取并去重...")
        with open(temp_file_path, 'r', encoding='utf-8') as f:
            temp_words = [line.strip() for line in f if line.strip()]
        cleaned_words = list(set(temp_words))  # 去重
        logger.info(f"✓ 去重完成，最终保留 {len(cleaned_words)} 个词")
    else:
        # 如果没有临时文件，直接去重
        cleaned_words = list(set(all_cleaned_words))
    
    removed_count = len(words) - len(cleaned_words)
    removal_rate = removed_count / len(words) if len(words) > 0 else 0.0
    
    logger.info(f"✓ {category} 词表清洗完成")
    logger.info(f"  - 原始词数: {len(words)}")
    logger.info(f"  - 清洗后词数: {len(cleaned_words)}")
    logger.info(f"  - 删除词数: {removed_count}")
    logger.info(f"  - 删除率: {removal_rate:.2%}")
    
    return cleaned_words, removed_count


def clean_lexicon_with_llm(
    words: List[str],
    category: str,
    batch_size: int = 100,
    max_retries: int = 3,
    retry_delay: float = 2.5,
    temp_file_path: Optional[str] = None,
    use_async: bool = True
) -> Tuple[List[str], int]:
    """
    使用 LLM 清洗词表（同步包装函数）
    
    Args:
        words: 原始词表列表
        category: 类别名称
        batch_size: 每批处理的词数
        max_retries: 最大重试次数
        retry_delay: 重试延迟
        temp_file_path: 临时文件路径
        use_async: 是否使用异步模式
    
    Returns:
        (清洗后的词表, 删除的词数)
    """
    if use_async:
        try:
            # 尝试使用异步版本
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # 没有事件循环，创建新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # 如果事件循环正在运行，使用 nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    logger.warning("事件循环正在运行且未安装 nest_asyncio，创建新的事件循环")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            
            return loop.run_until_complete(
                clean_lexicon_with_llm_async(
                    words, category, batch_size, max_retries, retry_delay, temp_file_path
                )
            )
        except Exception as e:
            logger.warning(f"异步模式失败，回退到同步模式: {e}")
            use_async = False
    
    # 同步模式（简化版，不推荐用于大批量）
    logger.warning("使用同步模式清洗词表（性能较慢，建议使用异步模式）")
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            clean_lexicon_with_llm_async(
                words, category, batch_size, max_retries, retry_delay, temp_file_path
            )
        )
    finally:
        loop.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用大模型生成词表和正则规则')
    parser.add_argument(
        '--input',
        type=str,
        default='data/train.jsonl',
        help='输入数据文件路径（JSONL格式，包含subtype_label字段）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='configs/lexicons',
        help='输出目录（词表文件将保存到此目录）'
    )
    parser.add_argument(
        '--samples-per-category',
        type=int,
        default=None,
        help='每个类别采样样本数（默认：None，使用全部样本）'
    )
    parser.add_argument(
        '--categories',
        type=str,
        nargs='+',
        default=['porn', 'politics', 'abuse'],
        help='要生成的类别列表（默认：porn politics abuse）'
    )
    parser.add_argument(
        '--generate-regex',
        action='store_true',
        help='是否生成正则规则（默认：False）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认：42）'
    )
    parser.add_argument(
        '--clean-lexicon',
        action='store_true',
        help='清洗现有词表（使用LLM判断，覆盖所有词表）'
    )
    parser.add_argument(
        '--lexicon-dir',
        type=str,
        default='configs/lexicons',
        help='词表目录（清洗模式使用，默认：configs/lexicons）'
    )
    parser.add_argument(
        '--clean-batch-size',
        type=int,
        default=100,
        help='清洗批次大小（默认：100）'
    )
    parser.add_argument(
        '--backup-original',
        action='store_true',
        help='清洗前备份原始词表文件（默认：False）'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 加载配置，获取请求间隔
    try:
        config = load_config()
        request_interval = config.get('llm', {}).get('request_interval', 2.5)
        logger.info(f"从配置文件读取请求间隔: {request_interval} 秒")
    except Exception as e:
        logger.warning(f"无法加载配置文件，使用默认请求间隔 2.5 秒: {e}")
        request_interval = 2.5
    
    # 如果是清洗模式，执行清洗逻辑
    if args.clean_lexicon:
        logger.info("=" * 60)
        logger.info("开始清洗词表（使用LLM）")
        logger.info("=" * 60)
        logger.info(f"词表目录: {args.lexicon_dir}")
        logger.info(f"清洗类别: {args.categories}")
        logger.info(f"批次大小: {args.clean_batch_size}")
        logger.info(f"备份原始文件: {args.backup_original}")
        logger.info("")
        
        # 确保词表目录存在
        if not os.path.exists(args.lexicon_dir):
            logger.error(f"词表目录不存在: {args.lexicon_dir}")
            return 1
        
        total_original = 0
        total_cleaned = 0
        total_removed = 0
        
        for category in args.categories:
            lexicon_path = os.path.join(args.lexicon_dir, f"{category}.txt")
            
            if not os.path.exists(lexicon_path):
                logger.warning(f"词表文件不存在: {lexicon_path}，跳过")
                continue
            
            logger.info("")
            logger.info("-" * 60)
            logger.info(f"清洗类别: {category}")
            logger.info("-" * 60)
            
            # 备份原始文件（如果需要）
            if args.backup_original:
                backup_path = os.path.join(args.lexicon_dir, f"{category}.txt.backup")
                shutil.copy2(lexicon_path, backup_path)
                logger.info(f"✓ 已备份原始文件到: {backup_path}")
            
            # 读取原始词表
            logger.info(f"读取词表: {lexicon_path}")
            original_words = []
            with open(lexicon_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        original_words.append(word)
            
            logger.info(f"原始词数: {len(original_words)}")
            
            if len(original_words) == 0:
                logger.warning(f"词表为空，跳过清洗")
                continue
            
            # 创建临时文件路径
            temp_file_path = os.path.join(args.lexicon_dir, f"{category}.txt.temp")
            
            # 清洗词表（包含LLM清洗）
            try:
                cleaned_words, removed_count = clean_lexicon_with_llm(
                    original_words,
                    category,
                    batch_size=args.clean_batch_size,
                    max_retries=3,
                    retry_delay=request_interval,
                    temp_file_path=temp_file_path,
                    use_async=True
                )
                
                # 保存清洗后的词表（覆盖原文件）
                logger.info(f"保存清洗后的词表到: {lexicon_path}")
                with open(lexicon_path, 'w', encoding='utf-8') as f:
                    for word in sorted(cleaned_words):  # 排序后保存
                        f.write(word + '\n')
                
                # 删除临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
                total_original += len(original_words)
                total_cleaned += len(cleaned_words)
                total_removed += removed_count
                
                logger.info(f"✓ {category} 词表清洗完成")
                
            except Exception as e:
                logger.error(f"✗ 清洗 {category} 词表时出错: {e}", exc_info=True)
                # 如果出错，删除临时文件
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                continue
        
        # 总结
        logger.info("")
        logger.info("=" * 60)
        logger.info("词表清洗完成")
        logger.info("=" * 60)
        logger.info(f"总原始词数: {total_original}")
        logger.info(f"总清洗后词数: {total_cleaned}")
        logger.info(f"总删除词数: {total_removed}")
        if total_original > 0:
            logger.info(f"总删除率: {total_removed / total_original:.2%}")
        logger.info("")
        logger.info("提示：清洗后请人工审核词表，确保没有误删重要敏感词")
        
        return 0
    
    # 生成词表模式（原有逻辑）
    logger.info("=" * 60)
    logger.info("开始生成词表和正则规则")
    logger.info("=" * 60)
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出目录: {args.output_dir}")
    if args.samples_per_category is None:
        logger.info(f"采样策略: 使用全部样本")
    else:
        logger.info(f"每个类别采样数: {args.samples_per_category}")
    logger.info(f"生成类别: {args.categories}")
    logger.info(f"生成正则规则: {args.generate_regex}")
    logger.info(f"API请求间隔: {request_interval} 秒（确保不超过30次/分钟）")
    logger.info("")
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return 1
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 为每个类别生成词表
    all_words = {}
    all_patterns = []
    
    for category in args.categories:
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"处理类别: {category}")
        logger.info("-" * 60)
        
        try:
            # 1. 加载样本
            samples = load_data_samples(
                args.input,
                category,
                samples_per_category=args.samples_per_category
            )
            
            if len(samples) == 0:
                logger.warning(f"⚠ {category} 类别没有找到样本，跳过")
                continue
            
            # 2. 生成词表（动态追加模式，默认使用异步）
            output_path = os.path.join(args.output_dir, f"{category}.txt")
            words = generate_lexicon_from_samples(
                samples, 
                category,
                request_interval=request_interval,
                output_path=output_path,  # 传入输出路径，启用动态追加
                use_async=True  # 使用异步模式
            )
            all_words[category] = words
            
            # 3. 词表已在生成过程中保存，这里只记录日志
            logger.info(f"✓ {category} 词表已保存到: {output_path}（共 {len(words)} 个词）")
            
            # 4. 生成正则规则（可选，动态追加模式）
            if args.generate_regex:
                regex_path = os.path.join(args.output_dir, "regex_patterns.txt")
                patterns = generate_regex_patterns(
                    words, 
                    category,
                    request_interval=request_interval,
                    sample_words_count=20,
                    output_path=regex_path  # 传入输出路径，启用动态追加
                )
                all_patterns.extend(patterns)
            
            logger.info(f"✓ {category} 类别处理完成")
            
        except Exception as e:
            logger.error(f"✗ 处理 {category} 类别时出错: {e}", exc_info=True)
            continue
    
    # 整理正则规则文件（如果生成，已在生成过程中追加，这里只需去重整理）
    if args.generate_regex and all_patterns:
        logger.info("")
        logger.info("-" * 60)
        logger.info("整理正则规则文件")
        logger.info("-" * 60)
        regex_path = os.path.join(args.output_dir, "regex_patterns.txt")
        if os.path.exists(regex_path):
            # 读取所有规则，去重并排序
            with open(regex_path, 'r', encoding='utf-8') as f:
                all_patterns = sorted(list(set(line.strip() for line in f if line.strip())))
            # 重新写入（覆盖）
            with open(regex_path, 'w', encoding='utf-8') as f:
                for pattern in all_patterns:
                    f.write(pattern + '\n')
            logger.info(f"✓ 正则规则文件已整理，共 {len(all_patterns)} 个规则")
    
    # 总结
    logger.info("")
    logger.info("=" * 60)
    logger.info("生成完成")
    logger.info("=" * 60)
    logger.info(f"生成的词表文件:")
    for category in args.categories:
        if category in all_words:
            count = len(all_words[category])
            file_path = os.path.join(args.output_dir, f"{category}.txt")
            logger.info(f"  - {file_path} ({count} 个词)")
    
    if args.generate_regex and all_patterns:
        regex_path = os.path.join(args.output_dir, "regex_patterns.txt")
        logger.info(f"  - {regex_path} ({len(all_patterns)} 个规则)")
    
    logger.info("")
    logger.info("提示：生成后请人工审核和优化词表，删除误报，补充遗漏")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

