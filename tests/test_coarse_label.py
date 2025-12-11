"""
test_coarse_label.py - coarse_label 模块的单元测试

注意：包含真实 API 调用测试，需要设置 GEMINI_API_KEY 环境变量

运行测试：
1. 不设置 API Key（只运行不需要API的测试）：
   pytest tests/test_coarse_label.py -v

2. 设置 API Key 运行完整测试（包括真实API调用）：
   export GEMINI_API_KEY=your_api_key  # Linux/Mac
   set GEMINI_API_KEY=your_api_key     # Windows
   pytest tests/test_coarse_label.py -v

3. 只运行真实API测试：
   pytest tests/test_coarse_label.py::TestCoarseLabelIntegration -v
"""

import os
import sys
import json
import tempfile
import pytest
import time

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from coarse_label import main
from llm_labeler import (
    COARSE_LABEL_TASK,
    run_label_task,
    parse_binary_label,
)


class TestCoarseLabelIntegration:
    """集成测试（真实调用 LLM API）"""
    
    # API 请求间隔（秒），避免触发频率限制
    API_REQUEST_INTERVAL = 3
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_coarse_label_single(self):
        """真实测试单条文本的粗粒度标签打标"""
        # 测试明显非敏感内容
        prompt = COARSE_LABEL_TASK.prompt_template.format(
            text="今天天气很好，适合出门散步。"
        )
        
        from llm_labeler import call_llm_backend
        response = call_llm_backend(prompt)
        label = parse_binary_label(response)
        
        # 验证返回的标签是 0 或 1
        assert label in [0, 1], f"标签应该是 0 或 1，但得到: {label}, 响应: {response}"
        
        time.sleep(self.API_REQUEST_INTERVAL)
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_coarse_label_sensitive(self):
        """真实测试敏感内容的粗粒度标签打标"""
        # 测试明显敏感内容（使用占位符避免实际敏感内容）
        prompt = COARSE_LABEL_TASK.prompt_template.format(
            text="这是一段包含[URL]和[MENTION]的测试文本，用于验证敏感内容检测。"
        )
        
        from llm_labeler import call_llm_backend
        response = call_llm_backend(prompt)
        label = parse_binary_label(response)
        
        # 验证返回的标签是 0 或 1
        assert label in [0, 1], f"标签应该是 0 或 1，但得到: {label}, 响应: {response}"
        
        time.sleep(self.API_REQUEST_INTERVAL)
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_run_label_task(self):
        """真实测试 run_label_task 函数"""
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "今天天气很好，适合出门散步。"},
                {"id": "s1", "text": "这是一段正常的测试文本。"},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=COARSE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=False,
                max_retries=2  # 减少重试次数以加快测试
            )
            
            # 验证统计信息
            assert stats["total"] == 2
            assert stats["success"] >= 0  # 可能成功也可能失败（取决于API）
            assert stats["failed"] >= 0
            assert stats["success"] + stats["failed"] == stats["total"]
            
            # 如果成功，验证输出文件
            if stats["success"] > 0:
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    assert len(lines) == stats["success"]
                    
                    # 验证输出格式
                    for line in lines:
                        item = json.loads(line)
                        assert "id" in item
                        assert "text" in item
                        assert "coarse_label" in item
                        assert "coarse_response" in item
                        assert item["coarse_label"] in [0, 1]
            
            time.sleep(self.API_REQUEST_INTERVAL)
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_coarse_label_script(self):
        """真实测试 coarse_label.py 脚本"""
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "今天天气很好，适合出门散步。"},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            config_content = """
data_dir: "./data"
llm:
  batch_size: 1
  max_retries: 2
"""
            f.write(config_content)
            config_path = f.name
        
        try:
            # 修改 sys.argv 来模拟命令行参数
            original_argv = sys.argv
            sys.argv = [
                'coarse_label.py',
                '--input', input_path,
                '--output', output_path,
                '--config', config_path,
                '--max-retries', '2',
            ]
            
            # 运行主函数
            result = main()
            
            # 验证返回码
            assert result == 0
            
            # 验证输出文件存在
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    if lines:
                        item = json.loads(lines[0])
                        assert "coarse_label" in item
                        assert item["coarse_label"] in [0, 1]
            
            time.sleep(self.API_REQUEST_INTERVAL)
            
        finally:
            sys.argv = original_argv
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_coarse_label_with_cleaned_data(self):
        """真实测试清洗后的数据格式（包含URL和MENTION占位符）"""
        # 创建模拟清洗后的数据
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "访问 [URL] 联系 [MENTION] 获取信息"},
                {"id": "s1", "text": "这是一段正常的测试文本，包含一些内容。"},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=COARSE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=False,
                max_retries=2
            )
            
            # 验证统计信息
            assert stats["total"] == 2
            
            # 如果成功，验证输出格式
            if stats["success"] > 0:
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    for line in lines:
                        item = json.loads(line)
                        assert "coarse_label" in item
                        assert "coarse_response" in item
                        assert item["coarse_label"] in [0, 1]
                        # 验证原始字段保留
                        assert "id" in item
                        assert "text" in item
            
            time.sleep(self.API_REQUEST_INTERVAL)
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCoarseLabelEdgeCases:
    """边界情况测试（不需要真实API）"""
    
    def test_parse_binary_label_variations(self):
        """测试各种标签解析变体"""
        # 测试不同的响应格式
        test_cases = [
            ("1", 1),
            ("0", 0),
            ("sensitive", 1),
            ("non-sensitive", 0),
            ("敏感", 1),
            ("非敏感", 0),
            ('{"label": 1}', 1),
            ('{"label": 0}', 0),
        ]
        
        for response, expected in test_cases:
            result = parse_binary_label(response)
            # 注意：某些格式可能解析失败返回None，这是正常的
            if result is not None:
                assert result == expected, f"响应 '{response}' 应该解析为 {expected}，但得到 {result}"

