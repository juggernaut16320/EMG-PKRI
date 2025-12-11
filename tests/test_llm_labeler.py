"""
test_llm_labeler.py - llm_labeler 模块的单元测试
"""

import os
import sys
import json
import time
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from llm_labeler import (
    load_config,
    LabelTask,
    parse_binary_label,
    parse_subtypes,
    call_llm_backend,
    run_label_task,
    get_backend,
    GeminiBackend,
    COARSE_LABEL_TASK,
    SUBTYPE_LABEL_TASK,
    COARSE_LABEL_PROMPT,
)


class TestParseBinaryLabel:
    """测试二分类标签解析"""
    
    def test_parse_1(self):
        assert parse_binary_label("1") == 1
        
    def test_parse_0(self):
        assert parse_binary_label("0") == 0
    
    def test_parse_sensitive(self):
        assert parse_binary_label("sensitive") == 1
        assert parse_binary_label("Sensitive") == 1
        assert parse_binary_label("SENSITIVE") == 1
        
    def test_parse_non_sensitive(self):
        assert parse_binary_label("non-sensitive") == 0
        assert parse_binary_label("Non-Sensitive") == 0
        assert parse_binary_label("non_sensitive") == 0
        
    def test_parse_chinese_sensitive(self):
        assert parse_binary_label("敏感") == 1
        assert parse_binary_label("这是敏感内容") == 1
        
    def test_parse_chinese_non_sensitive(self):
        assert parse_binary_label("非敏感") == 0
        assert parse_binary_label("不敏感") == 0
        
    def test_parse_json_format(self):
        assert parse_binary_label('{"label": 1}') == 1
        assert parse_binary_label('{"label": 0}') == 0
        assert parse_binary_label('{"sensitive": 1}') == 1
        assert parse_binary_label('{"is_sensitive": 0}') == 0
        
    def test_parse_with_explanation(self):
        # 以数字开头的情况
        assert parse_binary_label("1，这是敏感内容因为...") == 1
        assert parse_binary_label("0，这不是敏感内容") == 0
        
    def test_parse_ambiguous_returns_first(self):
        # 当响应以特定标签开头时
        assert parse_binary_label("1 (sensitive)") == 1
        assert parse_binary_label("0 (non-sensitive)") == 0


class TestParseSubtypes:
    """测试子标签解析"""
    
    def test_parse_single_subtype(self):
        assert parse_subtypes("porn") == ["porn"]
        assert parse_subtypes("politics") == ["politics"]
        assert parse_subtypes("abuse") == ["abuse"]
        
    def test_parse_multiple_subtypes(self):
        result = parse_subtypes("这个文本包含 porn 和 abuse 内容")
        assert "porn" in result
        assert "abuse" in result
        
    def test_parse_json_list(self):
        assert parse_subtypes('["porn", "abuse"]') == ["porn", "abuse"]
        assert parse_subtypes('["politics"]') == ["politics"]
        
    def test_parse_json_dict(self):
        assert parse_subtypes('{"subtypes": ["porn", "abuse"]}') == ["porn", "abuse"]
        assert parse_subtypes('{"labels": ["politics"]}') == ["politics"]
        
    def test_parse_unknown_returns_other(self):
        assert parse_subtypes("unknown content") == ["other"]
        assert parse_subtypes("无法识别") == ["other"]
        
    def test_parse_custom_valid_subtypes(self):
        result = parse_subtypes("violence", valid_subtypes=["violence", "hate"])
        assert result == ["violence"]


class TestLabelTask:
    """测试 LabelTask 数据类"""
    
    def test_create_task(self):
        task = LabelTask(
            name="test",
            prompt_template="请分析：{text}",
            parse_fn=lambda x: x,
            valid_labels=[0, 1]
        )
        assert task.name == "test"
        assert "{text}" in task.prompt_template
        
    def test_coarse_label_task(self):
        assert COARSE_LABEL_TASK.name == "coarse"
        assert COARSE_LABEL_TASK.valid_labels == [0, 1]
        
    def test_subtype_label_task(self):
        assert SUBTYPE_LABEL_TASK.name == "subtype"
        assert "porn" in SUBTYPE_LABEL_TASK.valid_labels


class TestPromptTemplate:
    """测试 Prompt 模板"""
    
    def test_coarse_prompt_format(self):
        text = "这是一段测试文本"
        prompt = COARSE_LABEL_PROMPT.format(text=text)
        assert text in prompt
        assert "敏感" in prompt
        
    def test_prompt_has_required_elements(self):
        assert "{text}" in COARSE_LABEL_PROMPT
        assert "1" in COARSE_LABEL_PROMPT
        assert "0" in COARSE_LABEL_PROMPT


class TestGetBackend:
    """测试后端获取"""
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量才能测试 Gemini 后端初始化"
    )
    def test_get_gemini_backend(self):
        """测试获取 Gemini 后端（需要真实 API Key）"""
        backend = get_backend("gemma-3-27b-it")
        assert "gemini" in backend.name
        
    def test_get_invalid_backend(self):
        # 现在会自动使用默认模型 gemma-3-27b-it，而不是抛出错误
        backend = get_backend("invalid-backend")
        assert "gemma-3-27b-it" in backend.name


class TestCallLLMBackend:
    """测试 LLM 后端调用"""
    
    @patch("llm_labeler.get_backend")
    def test_call_success(self, mock_get_backend):
        mock_backend = Mock()
        mock_backend.call.return_value = "1"
        mock_get_backend.return_value = mock_backend
        
        result = call_llm_backend("test prompt")
        assert result == "1"
        
    @patch("llm_labeler.get_backend")
    def test_call_retry_on_failure(self, mock_get_backend):
        mock_backend = Mock()
        mock_backend.call.side_effect = [Exception("API Error"), "1"]
        mock_get_backend.return_value = mock_backend
        
        result = call_llm_backend("test prompt", max_retries=2, retry_delay=0.01)
        assert result == "1"
        assert mock_backend.call.call_count == 2
        
    @patch("llm_labeler.get_backend")
    def test_call_max_retries_exceeded(self, mock_get_backend):
        mock_backend = Mock()
        mock_backend.call.side_effect = Exception("API Error")
        mock_get_backend.return_value = mock_backend
        
        with pytest.raises(RuntimeError, match="已重试"):
            call_llm_backend("test prompt", max_retries=2, retry_delay=0.01)


class TestRunLabelTask:
    """测试运行标签任务"""
    
    @patch("llm_labeler.call_llm_backend")
    def test_run_task_basic(self, mock_call):
        mock_call.return_value = "1"
        
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write(json.dumps({"id": 1, "text": "测试文本1"}) + '\n')
            f.write(json.dumps({"id": 2, "text": "测试文本2"}) + '\n')
            input_path = f.name
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=COARSE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=False
            )
            
            assert stats["success"] == 2
            assert stats["failed"] == 0
            
            # 验证输出
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 2
                
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
            
    @patch("llm_labeler.call_llm_backend")
    def test_run_task_skip_existing(self, mock_call):
        mock_call.return_value = "1"
        
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write(json.dumps({"id": 1, "text": "测试文本1"}) + '\n')
            f.write(json.dumps({"id": 2, "text": "测试文本2"}) + '\n')
            input_path = f.name
        
        # 创建已有输出文件（已处理 id=1）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write(json.dumps({"id": 1, "text": "测试文本1", "coarse_label": 1}) + '\n')
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=COARSE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=True
            )
            
            assert stats["skipped"] == 1
            assert stats["success"] == 1
            
        finally:
            os.unlink(input_path)
            os.unlink(output_path)


class TestLoadConfig:
    """测试配置加载"""
    
    def test_load_config_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write("data_dir: ./data\nllm:\n  batch_size: 10")
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config["data_dir"] == "./data"
            assert config["llm"]["batch_size"] == 10
        finally:
            os.unlink(config_path)


class TestIntegration:
    """集成测试（需要真实 API Key）"""
    
    # API 请求间隔（秒），避免触发频率限制
    API_REQUEST_INTERVAL = 5
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_gemini_call(self):
        """真实调用 Gemini API 测试"""
        response = call_llm_backend("请只回答数字 1 或 0，不要其他内容。回答 1。")
        assert "1" in response or "0" in response
        time.sleep(self.API_REQUEST_INTERVAL)  # 请求间隔
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_coarse_label(self):
        """真实测试粗粒度标签打标"""
        # 测试明显敏感内容
        prompt = COARSE_LABEL_TASK.prompt_template.format(text="这是一段正常的天气预报。")
        response = call_llm_backend(prompt)
        label = parse_binary_label(response)
        assert label == 0, f"天气预报应该判定为非敏感，响应: {response}"
        time.sleep(self.API_REQUEST_INTERVAL)  # 请求间隔
        
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_subtype_label(self):
        """真实测试子类型标签打标"""
        from llm_labeler import SUBTYPE_LABEL_TASK
        prompt = SUBTYPE_LABEL_TASK.prompt_template.format(text="这是脏话测试文本，包含侮辱性语言。")
        response = call_llm_backend(prompt)
        subtypes = parse_subtypes(response)
        assert isinstance(subtypes, list), f"子类型应该是列表，响应: {response}"
        time.sleep(self.API_REQUEST_INTERVAL)  # 请求间隔


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

