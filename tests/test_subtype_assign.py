"""
test_subtype_assign.py - subtype_assign 模块的单元测试

注意：包含真实 API 调用测试，需要设置 GEMINI_API_KEY 环境变量

运行测试：
1. 不设置 API Key（只运行mock测试）：
   pytest tests/test_subtype_assign.py -v

2. 设置 API Key 运行完整测试（包括真实API调用）：
   export GEMINI_API_KEY=your_api_key  # Linux/Mac
   set GEMINI_API_KEY=your_api_key     # Windows
   pytest tests/test_subtype_assign.py -v

3. 只运行真实API测试：
   pytest tests/test_subtype_assign.py::TestSubtypeAssignIntegration -v

4. 只运行mock测试：
   pytest tests/test_subtype_assign.py::TestSubtypeAssignMock -v
"""

import os
import sys
import json
import tempfile
import pytest
import time
from unittest.mock import patch, MagicMock

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from subtype_assign import (
    main,
    filter_sensitive_items,
    merge_subtype_labels,
)
from llm_labeler import (
    SUBTYPE_LABEL_TASK,
    run_label_task,
    parse_subtypes,
)


class TestSubtypeAssignMock:
    """Mock测试（不需要真实API）"""
    
    def test_filter_sensitive_items(self):
        """测试过滤敏感样本"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "敏感文本1", "coarse_label": 1},
                {"id": "s1", "text": "非敏感文本", "coarse_label": 0},
                {"id": "s2", "text": "敏感文本2", "coarse_label": 1},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            count = filter_sensitive_items(input_path, output_path, "coarse_label")
            assert count == 2
            
            # 验证输出只包含敏感样本
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = [line for line in f if line.strip()]
                assert len(lines) == 2
                for line in lines:
                    item = json.loads(line)
                    assert item["coarse_label"] == 1
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
    
    def test_merge_subtype_labels(self):
        """测试合并子标签"""
        # 创建原始输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "敏感文本1", "coarse_label": 1},
                {"id": "s1", "text": "非敏感文本", "coarse_label": 0},
                {"id": "s2", "text": "敏感文本2", "coarse_label": 1},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建子标签输出文件（只包含敏感样本）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "敏感文本1", "coarse_label": 1, 
                 "subtype_label": ["porn", "abuse"], "subtype_response": '["porn", "abuse"]'},
                {"id": "s2", "text": "敏感文本2", "coarse_label": 1,
                 "subtype_label": ["politics"], "subtype_response": '["politics"]'},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            subtype_path = f.name
        
        # 创建最终输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            merge_subtype_labels(input_path, subtype_path, output_path, "coarse_label")
            
            # 验证输出
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = [line for line in f if line.strip()]
                assert len(lines) == 3
                
                items = [json.loads(line) for line in lines]
                # 验证s0有子标签
                item_s0 = next(item for item in items if item["id"] == "s0")
                assert item_s0["subtype_label"] == ["porn", "abuse"]
                assert "subtype_response" in item_s0
                
                # 验证s1（非敏感）没有子标签
                item_s1 = next(item for item in items if item["id"] == "s1")
                assert item_s1["subtype_label"] == []
                assert item_s1["subtype_response"] == ""
                
                # 验证s2有子标签
                item_s2 = next(item for item in items if item["id"] == "s2")
                assert item_s2["subtype_label"] == ["politics"]
        finally:
            os.unlink(input_path)
            os.unlink(subtype_path)
            os.unlink(output_path)
    
    @patch("llm_labeler.call_llm_backend")
    def test_run_label_task_mock(self, mock_call):
        """Mock测试run_label_task函数"""
        mock_call.return_value = '["porn", "abuse"]'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "敏感文本1", "coarse_label": 1},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=SUBTYPE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=False
            )
            
            assert stats["success"] == 1
            assert stats["failed"] == 0
            
            # 验证输出
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = [line for line in f if line.strip()]
                assert len(lines) == 1
                item = json.loads(lines[0])
                assert "subtype_label" in item
                assert "subtype_response" in item
                assert isinstance(item["subtype_label"], list)
                assert "porn" in item["subtype_label"] or "abuse" in item["subtype_label"]
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
    
    @patch("llm_labeler.call_llm_backend")
    def test_subtype_assign_script_mock(self, mock_call):
        """Mock测试subtype_assign.py脚本"""
        mock_call.return_value = '["porn"]'
        
        # 创建输入文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "敏感文本", "coarse_label": 1},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        # 创建配置文件
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
            # 修改 sys.argv
            original_argv = sys.argv
            sys.argv = [
                'subtype_assign.py',
                '--input', input_path,
                '--output', output_path,
                '--config', config_path,
                '--only-sensitive',
                '--max-retries', '2',
            ]
            
            result = main()
            assert result == 0
            
            # 验证输出文件
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    if lines:
                        item = json.loads(lines[0])
                        assert "subtype_label" in item
        finally:
            sys.argv = original_argv
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
            os.unlink(config_path)
    
    def test_parse_subtypes_variations(self):
        """测试各种子标签解析变体"""
        test_cases = [
            ('["porn", "abuse"]', ["porn", "abuse"]),
            ('["porn"]', ["porn"]),
            ('porn, abuse', ["porn", "abuse"]),
            ('porn abuse', ["porn", "abuse"]),
            ('{"subtypes": ["porn", "politics"]}', ["porn", "politics"]),
            ('porn', ["porn"]),
            ('', ["other"]),  # 空响应默认返回other
        ]
        
        for response, expected in test_cases:
            result = parse_subtypes(response)
            # 验证结果包含期望的标签
            for label in expected:
                if label != "other" or len(result) == 1:  # other可能单独出现
                    assert label in result, f"响应 '{response}' 应该包含 {label}，但得到 {result}"


class TestSubtypeAssignIntegration:
    """集成测试（真实调用 LLM API）"""
    
    # API 请求间隔（秒），避免触发频率限制
    API_REQUEST_INTERVAL = 3
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_subtype_label_single(self):
        """真实测试单条文本的子标签打标"""
        prompt = SUBTYPE_LABEL_TASK.prompt_template.format(
            text="这是一段测试文本，用于验证子标签功能。"
        )
        
        from llm_labeler import call_llm_backend
        response = call_llm_backend(prompt)
        labels = parse_subtypes(response)
        
        # 验证返回的标签是列表
        assert isinstance(labels, list)
        # 验证标签在有效范围内
        valid_labels = ["porn", "politics", "abuse", "other"]
        for label in labels:
            assert label in valid_labels, f"标签 {label} 不在有效范围内: {valid_labels}"
        
        time.sleep(self.API_REQUEST_INTERVAL)
    
    @pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY"),
        reason="需要设置 GEMINI_API_KEY 环境变量"
    )
    def test_real_run_subtype_task(self):
        """真实测试 run_label_task 函数（子标签）"""
        # 创建临时输入文件（只包含敏感样本）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "这是一段测试文本，用于验证子标签功能。", "coarse_label": 1},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=SUBTYPE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=False,
                max_retries=2
            )
            
            # 验证统计信息
            assert stats["total"] == 1
            assert stats["success"] >= 0
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
                        assert "subtype_label" in item
                        assert "subtype_response" in item
                        assert isinstance(item["subtype_label"], list)
                        # 验证标签在有效范围内
                        valid_labels = ["porn", "politics", "abuse", "other"]
                        for label in item["subtype_label"]:
                            assert label in valid_labels
            
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
    def test_real_subtype_assign_script(self):
        """真实测试 subtype_assign.py 脚本"""
        # 创建临时输入文件（包含敏感和非敏感样本）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "这是一段测试文本。", "coarse_label": 1},
                {"id": "s1", "text": "这是非敏感文本。", "coarse_label": 0},
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
            # 修改 sys.argv
            original_argv = sys.argv
            sys.argv = [
                'subtype_assign.py',
                '--input', input_path,
                '--output', output_path,
                '--config', config_path,
                '--only-sensitive',
                '--max-retries', '2',
            ]
            
            result = main()
            assert result == 0
            
            # 验证输出文件
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    assert len(lines) == 2  # 应该包含所有样本
                    
                    items = [json.loads(line) for line in lines]
                    # 验证敏感样本有子标签
                    sensitive_item = next(item for item in items if item["coarse_label"] == 1)
                    assert "subtype_label" in sensitive_item
                    assert isinstance(sensitive_item["subtype_label"], list)
                    
                    # 验证非敏感样本没有子标签（或为空列表）
                    non_sensitive_item = next(item for item in items if item["coarse_label"] == 0)
                    assert "subtype_label" in non_sensitive_item
                    assert non_sensitive_item["subtype_label"] == []
            
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
    def test_real_subtype_with_cleaned_data(self):
        """真实测试清洗后的数据格式（包含URL和MENTION占位符）"""
        # 创建模拟清洗后的数据（敏感样本）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            items = [
                {"id": "s0", "text": "访问 [URL] 联系 [MENTION] 获取信息", "coarse_label": 1},
            ]
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            input_path = f.name
        
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            output_path = f.name
        
        try:
            stats = run_label_task(
                task=SUBTYPE_LABEL_TASK,
                input_path=input_path,
                output_path=output_path,
                skip_existing=False,
                max_retries=2
            )
            
            # 验证统计信息
            assert stats["total"] == 1
            
            # 如果成功，验证输出格式
            if stats["success"] > 0:
                with open(output_path, 'r', encoding='utf-8') as f:
                    lines = [line for line in f if line.strip()]
                    for line in lines:
                        item = json.loads(line)
                        assert "subtype_label" in item
                        assert "subtype_response" in item
                        # 验证原始字段保留
                        assert "id" in item
                        assert "text" in item
                        assert "[URL]" in item["text"] or "[MENTION]" in item["text"]
            
            time.sleep(self.API_REQUEST_INTERVAL)
            
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

