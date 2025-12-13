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
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, List, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 加载 .env 文件（如果存在）
def load_env_file(env_path: str = ".env"):
    """从 .env 文件加载环境变量"""
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value and not os.getenv(key):
                        os.environ[key] = value

# 在导入时自动加载 .env 文件
load_env_file()

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
                    elif finish_reason == 'PROHIBITED_CONTENT':
                        raise ValueError("API 返回空响应，finish_reason: PROHIBITED_CONTENT")
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


# ============ 异步版本 ============

class RateLimiter:
    """令牌桶速率限制器，确保不超过API速率限制
    
    使用简单的时间戳检查方式，确保请求间隔至少 min_interval 秒
    """
    
    def __init__(self, max_rate: float = 30.0, time_window: float = 60.0):
        """
        Args:
            max_rate: 最大请求速率（每分钟请求数）
            time_window: 时间窗口（秒）
        """
        self.max_rate = max_rate
        self.time_window = time_window
        self.min_interval = time_window / max_rate  # 最小请求间隔（秒）
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """获取令牌，如果距离上次请求时间太短则等待"""
        request_start = time.time()
        logger.debug(f"[RateLimiter] 请求令牌，时间: {time.strftime('%H:%M:%S', time.localtime(request_start))}，距离上次: {request_start - self.last_request_time:.2f}秒")
        
        # 先不加锁快速检查（优化路径）
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed >= self.min_interval:
            # 可能可以立即执行，需要加锁确认
            async with self._lock:
                # 再次检查（防止并发竞态）
                now = time.time()
                elapsed = now - self.last_request_time
                if elapsed >= self.min_interval:
                    # 确认可以执行，更新时间
                    wait_time_total = now - request_start
                    self.last_request_time = now
                    if wait_time_total > 1.0:
                        logger.info(f"[RateLimiter] 快速路径，等待: {wait_time_total:.2f}秒")
                    return
        
        # 需要等待，计算等待时间并sleep
        async with self._lock:
            # 在锁内再次检查并计算等待时间
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed >= self.min_interval:
                # 在获取锁期间已经有其他请求更新了时间，可以立即执行
                self.last_request_time = now
                return
            else:
                # 计算需要等待的时间
                wait_time = self.min_interval - elapsed
        
        # 在锁外sleep，这样不会阻塞其他任务
        if wait_time > 0:
            logger.info(f"[RateLimiter] 需要等待 {wait_time:.2f}秒（距离上次请求: {elapsed:.2f}秒，最小间隔: {self.min_interval:.2f}秒）")
            await asyncio.sleep(wait_time)
        
        # sleep后再次获取锁并更新时间
        async with self._lock:
            # 确保不会因为并发导致时间倒退
            now = time.time()
            if now - self.last_request_time >= self.min_interval:
                self.last_request_time = now
            else:
                # 如果sleep后时间还不够，再次等待（理论上不应该发生）
                additional_wait = self.min_interval - (now - self.last_request_time)
                if additional_wait > 0:
                    logger.warning(f"[RateLimiter] 需要额外等待 {additional_wait:.2f}秒")
                    await asyncio.sleep(additional_wait)
                    self.last_request_time = time.time()
                else:
                    self.last_request_time = now
        
        total_wait = time.time() - request_start
        logger.info(f"[RateLimiter] 令牌获取完成，总等待: {total_wait:.2f}秒")


class AsyncGeminiBackend:
    """异步版本的 Google Gemini 后端"""
    
    def __init__(self, model: str = "gemma-3-27b-it", max_rate: float = 30.0):
        self.model = model
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 环境变量未设置")
        
        # 使用新版 SDK（同步版本）
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("请安装 google-genai: pip install google-genai")
        
        # 速率限制器
        self.rate_limiter = RateLimiter(max_rate=max_rate, time_window=60.0)
        # 线程池用于执行同步API调用
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def call_async(self, prompt: str) -> str:
        """异步调用 Gemini API"""
        total_start = time.time()
        logger.info(f"[call_async] 开始，时间: {time.strftime('%H:%M:%S', time.localtime(total_start))}")
        
        # 等待速率限制
        rate_limit_start = time.time()
        await self.rate_limiter.acquire()
        rate_limit_elapsed = time.time() - rate_limit_start
        logger.info(f"[call_async] 速率限制等待完成，耗时: {rate_limit_elapsed:.2f}秒")
        if rate_limit_elapsed > 5.0:
            logger.warning(f"⚠️ 速率限制等待时间过长: {rate_limit_elapsed:.2f}秒")
        
        # 在线程池中执行同步调用
        api_call_start = time.time()
        logger.info(f"[call_async] 准备执行API调用，时间: {time.strftime('%H:%M:%S', time.localtime(api_call_start))}")
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                self.executor,
                self._call_sync,
                prompt
            )
            api_call_elapsed = time.time() - api_call_start
            total_elapsed = time.time() - total_start
            
            logger.info(f"[call_async] API调用完成 - API耗时: {api_call_elapsed:.2f}秒，总耗时: {total_elapsed:.2f}秒（等待: {rate_limit_elapsed:.2f}秒）")
            
            if api_call_elapsed > 60.0:
                logger.warning(f"⚠️ API调用耗时过长: {api_call_elapsed:.2f}秒")
            elif api_call_elapsed > 30.0:
                logger.info(f"API调用耗时较长: {api_call_elapsed:.2f}秒")
            
            return response
        except Exception as e:
            logger.error(f"[call_async] 异常: {e}")
            # 重新抛出异常，让调用者处理
            raise
    
    def _call_sync(self, prompt: str) -> str:
        """同步调用 Gemini API（在线程池中执行）"""
        sync_start = time.time()
        logger.info(f"[_call_sync] 开始，时间: {time.strftime('%H:%M:%S', time.localtime(sync_start))}")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        sync_elapsed = time.time() - sync_start
        logger.info(f"[_call_sync] 完成，耗时: {sync_elapsed:.2f}秒")
        
        # 检查安全过滤器是否阻止了响应
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            # 检查是否有安全评分
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
                    elif finish_reason == 'PROHIBITED_CONTENT':
                        raise ValueError("API 返回空响应，finish_reason: PROHIBITED_CONTENT")
                    elif finish_reason == 'RECITATION':
                        raise ValueError("API 因版权原因阻止了响应")
                    else:
                        raise ValueError(f"API 返回空响应，finish_reason: {finish_reason}")
            raise ValueError("API 返回空响应（可能因内容违规被阻止）")
        
        if not response.text.strip():
            raise ValueError("API 返回空文本（可能因内容违规被阻止）")
        
        return response.text
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    @property
    def name(self) -> str:
        return f"async-gemini:{self.model}"


async def call_llm_backend_async(
    prompt: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0,
    backend: Optional[AsyncGeminiBackend] = None
) -> str:
    """
    异步版本的 LLM 后端调用接口
    
    Args:
        prompt: 提示词
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        backend: 可选的异步后端实例（如果为None则创建新实例）
    
    Returns:
        LLM 响应文本
    """
    if backend is None:
        backend = AsyncGeminiBackend()
    
    for attempt in range(max_retries):
        try:
            response = await backend.call_async(prompt)
            return response
        except ValueError as e:
            # 对于内容违规类错误，不重试（重试也没用）
            error_msg = str(e)
            if ("PROHIBITED_CONTENT" in error_msg or 
                "安全" in error_msg or 
                "违规" in error_msg or 
                "阻止" in error_msg or
                "SAFETY" in error_msg or
                "RECITATION" in error_msg):
                logger.warning(f"内容被API安全过滤器阻止，跳过重试: {e}")
                raise  # 直接抛出，不重试
            # 其他 ValueError 继续重试
            logger.warning(f"调用 {backend.name} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                raise RuntimeError(f"调用 {backend.name} 失败，已重试 {max_retries} 次: {e}")
        except Exception as e:
            logger.warning(f"调用 {backend.name} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                raise RuntimeError(f"调用 {backend.name} 失败，已重试 {max_retries} 次: {e}")


async def call_llm_backend_batch_async(
    prompts: List[str],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_concurrent: int = 5,
    backend: Optional[AsyncGeminiBackend] = None
) -> List[Tuple[int, str, Optional[Exception]]]:
    """
    批量异步调用 LLM，支持失败回退到逐条处理
    
    Args:
        prompts: 提示词列表
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        max_concurrent: 最大并发数（注意：实际速率仍受API限制控制）
        backend: 可选的异步后端实例
    
    Returns:
        结果列表，每个元素为 (索引, 响应文本, 异常)，如果成功则异常为None
    """
    if backend is None:
        backend = AsyncGeminiBackend()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def call_with_retry(idx: int, prompt: str) -> Tuple[int, str, Optional[Exception]]:
        """带重试的单个调用，单条API超过20秒则超时跳过"""
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    # 添加20秒超时限制
                    try:
                        response = await asyncio.wait_for(
                            backend.call_async(prompt),
                            timeout=20.0
                        )
                        return (idx, response, None)
                    except asyncio.TimeoutError:
                        logger.warning(f"提示词 {idx} API调用超时（>20秒），跳过本次调用")
                        return (idx, "", TimeoutError(f"API调用超时（>20秒）"))
                except ValueError as e:
                    # 对于内容违规类错误，不重试
                    error_msg = str(e)
                    if ("PROHIBITED_CONTENT" in error_msg or 
                        "安全" in error_msg or 
                        "违规" in error_msg or 
                        "阻止" in error_msg or
                        "SAFETY" in error_msg or
                        "RECITATION" in error_msg):
                        logger.warning(f"提示词 {idx} 被API安全过滤器阻止，跳过重试: {e}")
                        return (idx, "", e)  # 返回异常，让调用者决定如何处理
                    # 其他 ValueError 继续重试
                    logger.warning(f"调用提示词 {idx} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    else:
                        return (idx, "", e)
                except asyncio.TimeoutError as te:
                    # 超时异常已经在上面处理，这里不会到达，但如果到达则直接返回
                    logger.warning(f"提示词 {idx} API调用超时（>20秒），跳过本次调用")
                    return (idx, "", TimeoutError(f"API调用超时（>20秒）"))
                except Exception as e:
                    logger.warning(f"调用提示词 {idx} 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                    else:
                        return (idx, "", e)
            return (idx, "", RuntimeError(f"调用提示词 {idx} 失败，已重试 {max_retries} 次"))
    
    # 并发执行所有调用
    tasks = [call_with_retry(idx, prompt) for idx, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    
    return results


async def call_llm_backend_batch_with_fallback_async(
    batch_prompt: str,
    single_prompts: List[str],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    max_concurrent: int = 5,
    backend: Optional[AsyncGeminiBackend] = None
) -> Tuple[Optional[str], List[Tuple[int, str, Optional[Exception]]]]:
    """
    批量调用，如果批量失败则回退到逐条处理
    
    Args:
        batch_prompt: 批量处理的提示词（包含多条数据）
        single_prompts: 逐条处理的提示词列表（对应批量中的每条数据）
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）
        max_concurrent: 最大并发数（仅用于逐条处理回退时）
        backend: 可选的异步后端实例
    
    Returns:
        (批量响应, 逐条响应列表)
        - 如果批量成功：返回 (批量响应, [])
        - 如果批量失败：返回 (None, [(索引, 响应, 异常), ...])
    """
    if backend is None:
        backend = AsyncGeminiBackend()
    
    # 先尝试批量处理
    batch_start_time = time.time()
    try:
        batch_response = await call_llm_backend_async(
            batch_prompt, 
            max_retries=max_retries,
            retry_delay=retry_delay,
            backend=backend
        )
        batch_elapsed = time.time() - batch_start_time
        logger.info(f"批量处理成功，耗时: {batch_elapsed:.2f}秒")
        return (batch_response, [])
    except Exception as e:
        batch_elapsed = time.time() - batch_start_time
        error_msg = str(e)
        # 检查是否是安全策略阻止（这种情况下需要逐条处理）
        is_safety_blocked = (
            "PROHIBITED_CONTENT" in error_msg or 
            "安全" in error_msg or 
            "违规" in error_msg or 
            "阻止" in error_msg or
            "SAFETY" in error_msg
        )
        
        if is_safety_blocked:
            logger.warning(f"批量处理被安全策略阻止（耗时: {batch_elapsed:.2f}秒），回退到逐条处理: {e}")
        else:
            logger.warning(f"批量处理失败（耗时: {batch_elapsed:.2f}秒），回退到逐条处理: {e}")
        
        # 回退到逐条处理
        single_start_time = time.time()
        single_results = await call_llm_backend_batch_async(
            single_prompts,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_concurrent=max_concurrent,
            backend=backend
        )
        single_elapsed = time.time() - single_start_time
        logger.info(f"逐条处理完成，耗时: {single_elapsed:.2f}秒（{len(single_prompts)}条，平均每条: {single_elapsed/len(single_prompts):.2f}秒）")
        
        return (None, single_results)


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
        except ValueError as e:
            # 对于内容违规类错误，不重试（重试也没用）
            error_msg = str(e)
            if ("PROHIBITED_CONTENT" in error_msg or 
                "安全" in error_msg or 
                "违规" in error_msg or 
                "阻止" in error_msg or
                "SAFETY" in error_msg or
                "RECITATION" in error_msg):
                logger.warning(f"内容被API安全过滤器阻止，跳过重试: {e}")
                raise  # 直接抛出，不重试
            # 其他 ValueError 继续重试
            logger.warning(f"调用 gemma-3-27b-it 失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                raise RuntimeError(f"调用 gemma-3-27b-it 失败，已重试 {max_retries} 次: {e}")
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
                # 限制文本长度，避免超过API限制（每个文本最多1000字符）
                max_text_length = 1000
                texts = []
                for item in batch_items:
                    text = item.get(text_field, "")
                    if len(text) > max_text_length:
                        text = text[:max_text_length] + "...[截断]"
                    texts.append(text)
                
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
                        # 批量解析成功，通过id匹配拼接回原始数据
                        for i, (idx, item) in enumerate(batch):
                            # batch_labels 是 id 到 labels 的字典映射
                            if isinstance(batch_labels, dict):
                                # 通过id匹配（i是批量内的索引，对应发送时的id）
                                if i in batch_labels and batch_labels[i] is not None:
                                    item[f"{task.name}_label"] = batch_labels[i]
                                    item[f"{task.name}_response"] = response  # 保存完整响应
                                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                    f_out.flush()  # 每次写入后立即 flush
                                    stats["success"] += 1
                                else:
                                    logger.warning(f"第 {idx} 条数据（id={i}）标签解析失败")
                                    stats["failed"] += 1
                            # 兼容旧格式：列表格式
                            elif isinstance(batch_labels, list):
                                if i < len(batch_labels) and batch_labels[i] is not None:
                                    item[f"{task.name}_label"] = batch_labels[i]
                                    item[f"{task.name}_response"] = response
                                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                                    f_out.flush()
                                    stats["success"] += 1
                                else:
                                    logger.warning(f"第 {idx} 条数据标签解析失败")
                                    stats["failed"] += 1
                            else:
                                logger.warning(f"第 {idx} 条数据标签格式错误")
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


def parse_batch_subtype_labels(response: str, batch_size: int) -> Optional[Dict[int, List[str]]]:
    """
    解析批量子标签（返回id到labels的映射）
    
    Args:
        response: LLM 响应（JSON格式，包含id和labels）
        batch_size: 批量大小
    
    Returns:
        id到labels的字典映射，如果解析失败返回 None
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
        
        # 支持新格式：包含id和labels的数组
        if isinstance(data, list):
            # 新格式：[{"id": 0, "labels": ["porn"]}, ...]
            result_dict = {}
            valid_subtypes = ["porn", "politics", "abuse", "other"]
            
            for item in data:
                if isinstance(item, dict):
                    item_id = item.get('id')
                    labels = item.get('labels', [])
                    
                    if item_id is None:
                        logger.warning(f"返回项缺少id字段: {item}")
                        continue
                    
                    # 验证和转换标签格式
                    if isinstance(labels, list):
                        filtered = [s for s in labels if s in valid_subtypes]
                        result_dict[int(item_id)] = filtered if filtered else ["other"]
                    elif isinstance(labels, str):
                        if labels in valid_subtypes:
                            result_dict[int(item_id)] = [labels]
                        else:
                            result_dict[int(item_id)] = ["other"]
                    else:
                        result_dict[int(item_id)] = ["other"]
            
            # 验证是否所有id都存在
            if len(result_dict) != batch_size:
                logger.warning(f"标签数量不匹配：期望 {batch_size}，得到 {len(result_dict)}")
                # 尝试兼容旧格式
                return None
            
            return result_dict
        
        # 兼容旧格式：{"labels": [[...], ...]} 或 [[...], ...]
        elif isinstance(data, dict):
            labels = data.get('labels', data.get('label', []))
        elif isinstance(data, list) and len(data) > 0 and not isinstance(data[0], dict):
            # 旧格式：直接是标签列表
            labels = data
        else:
            logger.warning(f"批量子标签格式不正确: {type(data)}")
            return None
        
        # 旧格式处理：转换为id映射（按索引）
        if 'labels' in locals():
            valid_subtypes = ["porn", "politics", "abuse", "other"]
            result_dict = {}
            for i, label in enumerate(labels):
                if isinstance(label, list):
                    filtered = [s for s in label if s in valid_subtypes]
                    result_dict[i] = filtered if filtered else ["other"]
                elif isinstance(label, str):
                    if label in valid_subtypes:
                        result_dict[i] = [label]
                    else:
                        result_dict[i] = ["other"]
                else:
                    result_dict[i] = ["other"]
            
            if len(result_dict) != batch_size:
                logger.warning(f"标签数量不匹配：期望 {batch_size}，得到 {len(result_dict)}")
                return None
            
            return result_dict
        
        return None
        
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

文本列表（JSON格式，包含id和text）：
{texts_json}

可选类别：
- porn: 色情内容
- politics: 涉政内容
- abuse: 辱骂/攻击性内容
- other: 其他敏感内容

请以 JSON 数组格式返回，格式为：[{{"id": 0, "labels": ["porn", "abuse"]}}, {{"id": 1, "labels": ["politics"]}}, ...]
- 每个元素必须包含 "id" 字段（对应输入中的id）
- 每个元素必须包含 "labels" 字段（子标签列表，可多选）
- 数组长度必须与输入文本数量相同
- 返回中不要包含文本内容，只返回id和labels

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

