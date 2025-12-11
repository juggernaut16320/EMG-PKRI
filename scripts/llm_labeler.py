"""
llm_labeler.py - LLM 打标接口

封装调用 Gemini 模型打标签的逻辑，当前锁定使用 gemma-3-27b-it。
所有 API Key 从环境变量读取，配置从 config.yaml 读取。
"""

import os
import json
import time
import yaml
import logging
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List
from pathlib import Path

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


@dataclass
class LabelTask:
    """标签任务定义"""
    name: str
    prompt_template: str
    parse_fn: Callable[[str], Any]
    valid_labels: list = field(default_factory=list)
    batch_prompt_template: Optional[str] = None  # 批量处理Prompt模板
    batch_parse_fn: Optional[Callable[[str, int], Any]] = None  # 批量解析函数
    

class LLMBackend(ABC):
    """LLM 后端抽象基类"""
    
    @abstractmethod
    def call(self, prompt: str) -> str:
        """调用 LLM 并返回响应文本"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """后端名称"""
        pass


class GeminiBackend(LLMBackend):
    """Google Gemini 后端（使用新版 SDK）"""
    
    def __init__(self, model: str = "gemma-3-27b-it"):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 环境变量未设置")
        
        # 使用新版 SDK
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 google-genai: pip install google-genai")
    
    def call(self, prompt: str) -> str:
        """调用 Gemini API"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        # 检查安全过滤器是否阻止了响应
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            # 检查是否有安全评分（safety ratings）
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                blocked = any(
                    rating.category in ['HARM_CATEGORY_SEXUALLY_EXPLICIT', 
                                      'HARM_CATEGORY_HATE_SPEECH',
                                      'HARM_CATEGORY_HARASSMENT',
                                      'HARM_CATEGORY_DANGEROUS_CONTENT'] 
                    and rating.probability in ['HIGH', 'MEDIUM']
                    for rating in candidate.safety_ratings
                )
                if blocked:
                    logger.warning("API 安全过滤器阻止了响应（内容可能违规）")
                    raise ValueError("API 安全过滤器阻止了响应（内容可能违规）")
        
        # 检查响应是否为空
        if response is None:
            raise ValueError("API 返回 None 响应")
        
        # 检查 response.text 是否存在
        if not hasattr(response, 'text') or response.text is None:
            # 尝试获取阻止原因
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == 'SAFETY':
                        raise ValueError("API 因安全原因阻止了响应（内容违规）")
                    elif finish_reason == 'RECITATION':
                        raise ValueError("API 因版权原因阻止了响应")
                    else:
                        raise ValueError(f"API 返回空响应，finish_reason: {finish_reason}")
            raise ValueError("API 返回空响应（可能因内容违规被阻止）")
        
        if not response.text.strip():
            raise ValueError("API 返回空文本（可能因内容违规被阻止）")
        
        return response.text
    
    @property
    def name(self) -> str:
        return f"gemini:{self.model}"


def get_backend(backend_name: str = "gemma-3-27b-it") -> LLMBackend:
    """
    获取 LLM 后端实例（当前仅支持 gemma-3-27b-it）
    
    Args:
        backend_name: 后端名称，默认 "gemma-3-27b-it"（当前仅支持此模型）
    
    Returns:
        GeminiBackend 实例
    """
    return GeminiBackend(model="gemma-3-27b-it")


def call_llm_backend(prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """
    统一的 LLM 后端调用接口（使用 gemini-2.5-flash-live）
    
    Args:
        prompt: 提示词
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
    
    Returns:
        LLM 响应文本
    """
    llm_backend = get_backend()
    
    for attempt in range(max_retries):
        try:
            response = llm_backend.call(prompt)
            return response
        except Exception as e:
            logger.warning(f"调用 gemma-3-27b-it 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                raise RuntimeError(f"调用 gemma-3-27b-it 失败，已重试 {max_retries} 次: {e}")


def run_label_task(
    task: LabelTask,
    input_path: str,
    output_path: str,
    text_field: str = "text",
    batch_size: int = 1,
    max_retries: int = 3,
    skip_existing: bool = True,
    request_interval: float = 2.5
) -> dict:
    """
    运行标签任务，处理 JSONL 文件（使用 gemini-2.5-flash-live）
    
    Args:
        task: 标签任务定义
        input_path: 输入 JSONL 文件路径
        output_path: 输出 JSONL 文件路径
        text_field: 文本字段名
        batch_size: 批处理大小（预留，当前逐条处理）
        max_retries: 最大重试次数
        skip_existing: 是否跳过已存在的输出文件
    
    Returns:
        统计信息字典
    """
    # 全局变量用于信号处理
    global _interrupt_flag, _current_stats, _current_f_out
    _interrupt_flag = False
    _current_stats = None
    _current_f_out = None
    
    def signal_handler(sig, frame):
        """信号处理函数，支持优雅退出"""
        global _interrupt_flag, _current_stats
        _interrupt_flag = True
        logger.info("\n收到中断信号（Ctrl+C），正在安全退出...")
        if _current_stats:
            logger.info(f"已处理: {_current_stats['success']} 条成功, {_current_stats['failed']} 条失败, {_current_stats['skipped']} 条跳过")
        logger.info(f"数据已保存到: {output_path}")
        logger.info("下次运行时可以使用 --skip-existing 跳过已处理的记录")
        sys.exit(0)
    
    # 注册信号处理器（仅支持 Unix 系统，Windows 上可能不支持 SIGTERM）
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except (AttributeError, ValueError):
        # Windows 上可能不支持某些信号
        signal.signal(signal.SIGINT, signal_handler)
    
    # 检查输出文件是否存在
    output_file = Path(output_path)
    existing_ids = set()
    
    if skip_existing and output_file.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'id' in item:
                        existing_ids.add(item['id'])
        logger.info(f"已存在 {len(existing_ids)} 条记录，将跳过")
    
    # 读取输入数据
    input_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                input_data.append(json.loads(line))
    
    logger.info(f"读取 {len(input_data)} 条数据")
    
    # 统计
    stats = {
        "total": len(input_data),
        "processed": 0,
        "skipped": 0,
        "success": 0,
        "failed": 0
    }
    _current_stats = stats
    
    # 处理数据
    with open(output_path, 'a', encoding='utf-8') as f_out:
        _current_f_out = f_out
        # 过滤出需要处理的数据
        items_to_process = []
        for i, item in enumerate(input_data):
            item_id = item.get('id', i)
            
            # 跳过已处理的
            if item_id in existing_ids:
                stats["skipped"] += 1
                continue
            
            text = item.get(text_field, "")
            if not text:
                logger.warning(f"第 {i} 条数据缺少文本字段")
                stats["failed"] += 1
                continue
            
            items_to_process.append((i, item))
        
        # 批量处理模式
        if batch_size > 1 and task.batch_prompt_template and task.batch_parse_fn:
            logger.info(f"使用批量处理模式，批量大小: {batch_size}")
            
            # 分批处理
            for batch_start in range(0, len(items_to_process), batch_size):
                # 检查中断标志
                if _interrupt_flag:
                    logger.info("检测到中断信号，退出处理循环")
                    break
                
                batch = items_to_process[batch_start:batch_start + batch_size]
                batch_indices = [idx for idx, _ in batch]
                batch_items = [item for _, item in batch]
                
                # 构造批量 prompt
                texts = [item.get(text_field, "") for item in batch_items]
                texts_data = [{"id": i, "text": text} for i, text in enumerate(texts)]
                texts_json = json.dumps(texts_data, ensure_ascii=False)
                
                prompt = task.batch_prompt_template.format(texts_json=texts_json)
                
                try:
                    # 调用 LLM 打标
                    response = call_llm_backend(prompt, max_retries)
                    
                    # 批量请求完成后，添加延迟避免API限流
                    if request_interval > 0:
                        time.sleep(request_interval)
                    
                    # 解析批量标签
                    batch_labels = task.batch_parse_fn(response, len(batch_items))
                    
                    if batch_labels is None:
                        # 批量解析失败，回退到逐条处理
                        logger.warning(f"批量解析失败，回退到逐条处理（批次 {batch_start // batch_size + 1}）")
                        # 批量失败后，在开始逐条处理前添加延迟，避免API限流
                        if request_interval > 0:
                            time.sleep(request_interval)
                        for idx, item in batch:
                            # 检查中断标志
                            if _interrupt_flag:
                                logger.info("检测到中断信号，退出处理循环")
                                break
                            
                            try:
                                single_prompt = task.prompt_template.format(text=item.get(text_field, ""))
                                single_response = call_llm_backend(single_prompt, max_retries)
                                label = task.parse_fn(single_response)
                                item[f"{task.name}_label"] = label
                                item[f"{task.name}_response"] = single_response
                                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                f_out.flush()  # 每次写入后立即 flush
                                stats["success"] += 1
                                
                                if request_interval > 0:
                                    time.sleep(request_interval)
                            except ValueError as e:
                                # API 安全过滤器阻止或内容违规
                                error_msg = str(e)
                                if "安全" in error_msg or "违规" in error_msg or "阻止" in error_msg:
                                    logger.warning(f"第 {idx} 条数据被API安全过滤器阻止（可能违规）: {error_msg}")
                                    # 对于被阻止的内容，标记为敏感（1），因为被阻止通常意味着内容有问题
                                    item[f"{task.name}_label"] = 1
                                    item[f"{task.name}_response"] = f"[BLOCKED: {error_msg}]"
                                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                    f_out.flush()
                                    stats["success"] += 1
                                else:
                                    logger.error(f"处理第 {idx} 条数据失败: {error_msg}")
                                    stats["failed"] += 1
                            except Exception as e:
                                logger.error(f"处理第 {idx} 条数据失败: {e}")
                                stats["failed"] += 1
                            stats["processed"] += 1
                    else:
                        # 批量解析成功，拼接回原始数据
                        for i, (idx, item) in enumerate(batch):
                            if i < len(batch_labels) and batch_labels[i] is not None:
                                item[f"{task.name}_label"] = batch_labels[i]
                                item[f"{task.name}_response"] = response  # 保存完整响应
                                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                f_out.flush()  # 每次写入后立即 flush
                                stats["success"] += 1
                            else:
                                logger.warning(f"第 {idx} 条数据标签解析失败")
                                stats["failed"] += 1
                            stats["processed"] += 1
                        
                        # 批量写入后再次 flush，确保所有数据保存到磁盘
                        f_out.flush()
                        
                        # 检查中断标志
                        if _interrupt_flag:
                            logger.info("检测到中断信号，退出处理循环")
                            break
                        
                        # 批量处理进度日志
                        if (batch_start // batch_size + 1) % 10 == 0:
                            logger.info(f"进度: {min(batch_start + batch_size, len(items_to_process))}/{len(items_to_process)}")
                        
                        # 注意：批量请求完成后已经在第258-259行添加了延迟
                        # 这里不需要再次延迟，因为延迟已经在请求完成后立即添加
                        # 下一个批次会在循环开始时处理，间隔已经足够
                    
                except Exception as e:
                    logger.error(f"批量处理失败（批次 {batch_start // batch_size + 1}）: {e}")
                    # 批量处理失败，回退到逐条处理
                    # 批量失败后，在开始逐条处理前添加延迟，避免API限流
                    if request_interval > 0:
                        time.sleep(request_interval)
                    for idx, item in batch:
                        try:
                            single_prompt = task.prompt_template.format(text=item.get(text_field, ""))
                            single_response = call_llm_backend(single_prompt, max_retries)
                            label = task.parse_fn(single_response)
                            item[f"{task.name}_label"] = label
                            item[f"{task.name}_response"] = single_response
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            f_out.flush()  # 每次写入后立即 flush
                            stats["success"] += 1
                            
                            # 检查中断标志
                            if _interrupt_flag:
                                logger.info("检测到中断信号，退出处理循环")
                                break
                            
                            if request_interval > 0:
                                time.sleep(request_interval)
                        except Exception as e2:
                            logger.error(f"处理第 {idx} 条数据失败: {e2}")
                            stats["failed"] += 1
                        stats["processed"] += 1
        
        else:
            # 逐条处理模式（batch_size = 1 或不支持批量处理）
            logger.info("使用逐条处理模式")
            
            for i, item in items_to_process:
                # 检查中断标志
                if _interrupt_flag:
                    logger.info("检测到中断信号，退出处理循环")
                    break
                
                idx, item = i, item
                item_id = item.get('id', idx)
                
                text = item.get(text_field, "")
                
                # 构造 prompt
                prompt = task.prompt_template.format(text=text)
                
                try:
                    # 调用 LLM 打标
                    response = call_llm_backend(prompt, max_retries)
                    label = task.parse_fn(response)
                    item[f"{task.name}_label"] = label
                    item[f"{task.name}_response"] = response
                    
                    # 写入输出
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f_out.flush()  # 每次写入后立即 flush，确保数据保存到磁盘
                    stats["success"] += 1
                    
                    # 检查中断标志
                    if _interrupt_flag:
                        logger.info("检测到中断信号，退出处理循环")
                        break
                    
                    if (stats["processed"] + 1) % 10 == 0:
                        logger.info(f"进度: {stats['processed'] + 1}/{len(items_to_process)}")
                    
                    # 添加请求间隔
                    if request_interval > 0:
                        time.sleep(request_interval)
                        
                except ValueError as e:
                    # API 安全过滤器阻止或内容违规
                    error_msg = str(e)
                    if "安全" in error_msg or "违规" in error_msg or "阻止" in error_msg:
                        logger.warning(f"第 {idx} 条数据被API安全过滤器阻止（可能违规）: {error_msg}")
                        # 对于被阻止的内容，标记为敏感（1）
                        item[f"{task.name}_label"] = 1
                        item[f"{task.name}_response"] = f"[BLOCKED: {error_msg}]"
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        f_out.flush()
                        stats["success"] += 1
                    else:
                        logger.error(f"处理第 {idx} 条数据失败: {error_msg}")
                        stats["failed"] += 1
                except Exception as e:
                    logger.error(f"处理第 {idx} 条数据失败: {e}")
                    stats["failed"] += 1
                
                stats["processed"] += 1
    
    logger.info(f"完成！成功: {stats['success']}, 失败: {stats['failed']}, 跳过: {stats['skipped']}")
    return stats


# ============ 预定义的解析函数 ============

def parse_binary_label(response: str) -> int:
    """
    解析二分类标签（敏感/非敏感）
    
    Returns:
        1 表示敏感，0 表示非敏感，None 表示解析失败
    """
    # 检查 response 是否为 None 或空
    if response is None:
        logger.warning("parse_binary_label: response 为 None")
        return None
    if not isinstance(response, str):
        logger.warning(f"parse_binary_label: response 类型错误: {type(response)}")
        return None
    
    response_lower = response.lower().strip()
    
    # 优先检查否定形式（避免"不敏感"被"敏感"误匹配）
    if '非敏感' in response_lower or '不敏感' in response_lower or response_lower.startswith('不'):
        return 0
    if 'non-sensitive' in response_lower or 'non_sensitive' in response_lower or 'nonsensitive' in response_lower:
        return 0
    
    # 尝试直接匹配
    if '1' in response_lower or 'sensitive' in response_lower or '敏感' in response_lower:
        if '0' in response_lower or 'non' in response_lower:
            # 两者都有，尝试更精确匹配
            if response_lower.startswith('1') or response_lower.startswith('sensitive') or response_lower.startswith('敏感'):
                return 1
            elif response_lower.startswith('0') or response_lower.startswith('non') or response_lower.startswith('非'):
                return 0
        else:
            return 1
    elif '0' in response_lower or 'non' in response_lower:
        return 0
    
    # 尝试 JSON 解析
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            label = data.get('label', data.get('sensitive', data.get('is_sensitive')))
            if label is not None:
                return int(label)
    except json.JSONDecodeError:
        pass
    
    return None


def parse_batch_coarse_labels(response: str, batch_size: int) -> Optional[List[int]]:
    """
    解析批量粗粒度标签
    
    Args:
        response: LLM 响应（JSON格式）
        batch_size: 批量大小
    
    Returns:
        标签列表，如果解析失败返回 None
    """
    # 检查 response 是否为 None 或空
    if response is None:
        logger.error("parse_batch_coarse_labels: response 为 None")
        return None
    if not isinstance(response, str):
        logger.error(f"parse_batch_coarse_labels: response 类型错误: {type(response)}")
        return None
    
    try:
        # 清理响应：移除代码块标记
        cleaned_response = response.strip()
        original_response = response  # 保存原始响应用于日志
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # 移除 ```json
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]  # 移除 ```
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # 移除结尾的 ```
        cleaned_response = cleaned_response.strip()
        
        # 尝试解析JSON
        data = json.loads(cleaned_response)
        
        # 支持多种格式
        if isinstance(data, dict):
            labels = data.get('labels', data.get('label', []))
        elif isinstance(data, list):
            labels = data
        else:
            logger.warning(f"批量标签格式不正确: {type(data)}")
            logger.debug(f"原始响应: {original_response[:500]}")
            return None
        
        # 验证长度
        if len(labels) != batch_size:
            logger.warning(f"标签数量不匹配：期望 {batch_size}，得到 {len(labels)}")
            logger.info(f"实际返回的标签数组: {labels}")
            logger.info(f"原始响应长度: {len(original_response)} 字符")
            logger.info(f"清理后响应长度: {len(cleaned_response)} 字符")
            logger.info(f"清理后响应内容（前500字符）: {cleaned_response[:500]}")
            # 检查是否有截断迹象
            if cleaned_response.endswith(',') or not cleaned_response.rstrip().endswith(']'):
                logger.warning("检测到可能的JSON截断（响应可能不完整）")
            return None
        
        # 验证和转换格式
        result = []
        for label in labels:
            if isinstance(label, int):
                if label in [0, 1]:
                    result.append(label)
                else:
                    logger.warning(f"无效标签值: {label}")
                    result.append(None)
            elif isinstance(label, str):
                # 尝试解析字符串
                parsed = parse_binary_label(label)
                result.append(parsed)
            else:
                result.append(None)
        
        return result if all(x is not None for x in result) else None
        
    except json.JSONDecodeError as e:
        logger.error(f"解析批量粗粒度标签失败: {e}, 响应: {response[:200]}")
        return None
    except Exception as e:
        logger.error(f"解析批量粗粒度标签出错: {e}")
        return None


def parse_batch_subtype_labels(response: str, batch_size: int) -> Optional[List[List[str]]]:
    """
    解析批量子标签
    
    Args:
        response: LLM 响应（JSON格式）
        batch_size: 批量大小
    
    Returns:
        子标签列表的列表，如果解析失败返回 None
    """
    # 检查 response 是否为 None 或空
    if response is None:
        logger.error("parse_batch_subtype_labels: response 为 None")
        return None
    if not isinstance(response, str):
        logger.error(f"parse_batch_subtype_labels: response 类型错误: {type(response)}")
        return None
    
    try:
        # 清理响应：移除代码块标记
        cleaned_response = response.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]  # 移除 ```json
        if cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]  # 移除 ```
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]  # 移除结尾的 ```
        cleaned_response = cleaned_response.strip()
        
        # 尝试解析JSON
        data = json.loads(cleaned_response)
        
        # 支持多种格式
        if isinstance(data, dict):
            labels = data.get('labels', data.get('label', []))
        elif isinstance(data, list):
            labels = data
        else:
            logger.warning(f"批量子标签格式不正确: {type(data)}")
            return None
        
        # 验证长度
        if len(labels) != batch_size:
            logger.warning(f"标签数量不匹配：期望 {batch_size}，得到 {len(labels)}")
            return None
        
        # 验证和转换格式
        valid_subtypes = ["porn", "politics", "abuse", "other"]
        result = []
        for label in labels:
            if isinstance(label, list):
                # 过滤有效标签
                filtered = [s for s in label if s in valid_subtypes]
                result.append(filtered if filtered else ["other"])
            elif isinstance(label, str):
                # 单个标签字符串，转换为列表
                if label in valid_subtypes:
                    result.append([label])
                else:
                    result.append(["other"])
            else:
                result.append(["other"])
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"解析批量子标签失败: {e}, 响应: {response[:200]}")
        return None
    except Exception as e:
        logger.error(f"解析批量子标签出错: {e}")
        return None


def parse_subtypes(response: str, valid_subtypes: list[str] = None) -> list[str]:
    """
    解析子标签（多标签）
    
    Args:
        response: LLM 响应
        valid_subtypes: 有效子标签列表
    
    Returns:
        子标签列表
    """
    if valid_subtypes is None:
        valid_subtypes = ["porn", "politics", "abuse", "other"]
    
    response_lower = response.lower()
    found_subtypes = []
    
    for subtype in valid_subtypes:
        if subtype in response_lower:
            found_subtypes.append(subtype)
    
    # 尝试 JSON 解析
    if not found_subtypes:
        try:
            data = json.loads(response)
            if isinstance(data, list):
                found_subtypes = [s for s in data if s in valid_subtypes]
            elif isinstance(data, dict):
                subtypes = data.get('subtypes', data.get('labels', []))
                if isinstance(subtypes, list):
                    found_subtypes = [s for s in subtypes if s in valid_subtypes]
        except json.JSONDecodeError:
            pass
    
    return found_subtypes if found_subtypes else ["other"]


# ============ 预定义的 Prompt 模板 ============

COARSE_LABEL_PROMPT = """请判断以下文本是否包含敏感内容（如色情、涉政、辱骂等）。

文本：
{text}

请只回答一个数字：
- 1 表示敏感
- 0 表示非敏感

回答："""

SUBTYPE_LABEL_PROMPT = """请判断以下敏感文本属于哪些类别（可多选）。

文本：
{text}

可选类别：
- porn: 色情内容
- politics: 涉政内容
- abuse: 辱骂/攻击性内容
- other: 其他敏感内容

请以 JSON 列表格式回答，例如：["porn", "abuse"]

回答："""

# ============ 批量处理 Prompt 模板 ============

COARSE_LABEL_BATCH_PROMPT = """请判断以下文本列表是否包含敏感内容（如色情、涉政、辱骂等）。

文本列表（JSON格式）：
{texts_json}

请以 JSON 对象格式返回标签，格式为：{{"labels": [1, 0, 1, 0, ...]}}
- 1 表示敏感
- 0 表示非敏感
- 数组长度必须与输入文本数量相同
- 数组顺序必须与输入文本顺序一致

只返回JSON，不要其他内容："""

SUBTYPE_LABEL_BATCH_PROMPT = """请判断以下敏感文本列表属于哪些类别（可多选）。

文本列表（JSON格式）：
{texts_json}

可选类别：
- porn: 色情内容
- politics: 涉政内容
- abuse: 辱骂/攻击性内容
- other: 其他敏感内容

请以 JSON 对象格式返回，格式为：{{"labels": [["porn", "abuse"], ["politics"], ["other"], ...]}}
- 每个元素是一个子标签列表（可多选）
- 数组长度必须与输入文本数量相同
- 数组顺序必须与输入文本顺序一致

只返回JSON，不要其他内容："""


# ============ 预定义的 LabelTask ============

COARSE_LABEL_TASK = LabelTask(
    name="coarse",
    prompt_template=COARSE_LABEL_PROMPT,
    parse_fn=parse_binary_label,
    valid_labels=[0, 1],
    batch_prompt_template=COARSE_LABEL_BATCH_PROMPT,
    batch_parse_fn=parse_batch_coarse_labels
)

SUBTYPE_LABEL_TASK = LabelTask(
    name="subtype",
    prompt_template=SUBTYPE_LABEL_PROMPT,
    parse_fn=parse_subtypes,
    valid_labels=["porn", "politics", "abuse", "other"],
    batch_prompt_template=SUBTYPE_LABEL_BATCH_PROMPT,
    batch_parse_fn=parse_batch_subtype_labels
)


if __name__ == "__main__":
    # 简单测试
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 打标工具")
    parser.add_argument("--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--prompt", default="你好，请介绍一下你自己。", help="测试 prompt")
    args = parser.parse_args()
    
    try:
        response = call_llm_backend(args.prompt)
        print(f"模型: gemma-3-27b-it")
        print(f"响应: {response}")
    except Exception as e:
        print(f"错误: {e}")

