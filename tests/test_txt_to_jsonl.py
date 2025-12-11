"""
test_txt_to_jsonl.py - txt_to_jsonl 模块的单元测试
"""

import os
import sys
import json
import tempfile
import shutil
import pytest
from pathlib import Path

# 添加 scripts 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from txt_to_jsonl import (
    get_max_id_from_jsonl,
    format_id,
    estimate_jsonl_size,
    process_txt_files,
    merge_jsonl_files,
)


class TestFormatId:
    """测试 ID 格式化"""
    
    def test_format_s_num(self):
        assert format_id(0) == "s0"
        assert format_id(1) == "s1"
        assert format_id(12345) == "s12345"
        assert format_id(999999) == "s999999"


class TestGetMaxIdFromJsonl:
    """测试从 JSONL 获取最大 ID"""
    
    def test_nonexistent_file(self):
        assert get_max_id_from_jsonl("/nonexistent/file.jsonl") == -1
    
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            assert get_max_id_from_jsonl(temp_path) == -1
        finally:
            os.unlink(temp_path)
    
    def test_single_record(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            record = {"id": "s0", "text": "test"}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            assert get_max_id_from_jsonl(temp_path) == 0
        finally:
            os.unlink(temp_path)
    
    def test_multiple_records(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            records = [
                {"id": "s0", "text": "test1"},
                {"id": "s1", "text": "test2"},
                {"id": "s5", "text": "test3"},
            ]
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            assert get_max_id_from_jsonl(temp_path) == 5
        finally:
            os.unlink(temp_path)
    
    def test_integer_id(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            record = {"id": 12345, "text": "test"}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            assert get_max_id_from_jsonl(temp_path) == 12345
        finally:
            os.unlink(temp_path)


class TestEstimateJsonlSize:
    """测试估算 JSONL 文件大小"""
    
    def test_nonexistent_file(self):
        assert estimate_jsonl_size("/nonexistent/file.jsonl") == 0
    
    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            assert estimate_jsonl_size(temp_path) == 0
        finally:
            os.unlink(temp_path)
    
    def test_multiple_records(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for i in range(5):
                record = {"id": f"s{i}", "text": f"test{i}"}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            temp_path = f.name
        
        try:
            assert estimate_jsonl_size(temp_path) == 5
        finally:
            os.unlink(temp_path)


class TestProcessTxtFiles:
    """测试处理 TXT 文件"""
    
    def test_nonexistent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "output.jsonl")
            stats = process_txt_files(
                unprocessed_dir=os.path.join(tmpdir, "nonexistent"),
                output_jsonl=output,
                start_id=12345
            )
            assert stats["total_txt_files"] == 0
            assert stats["processed"] == 0
    
    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unprocessed_dir = os.path.join(tmpdir, "unprocessed")
            os.makedirs(unprocessed_dir)
            output = os.path.join(tmpdir, "output.jsonl")
            stats = process_txt_files(
                unprocessed_dir=unprocessed_dir,
                output_jsonl=output,
                start_id=12345
            )
            assert stats["total_txt_files"] == 0
            assert stats["processed"] == 0
    
    def test_single_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unprocessed_dir = os.path.join(tmpdir, "unprocessed")
            os.makedirs(unprocessed_dir)
            output = os.path.join(tmpdir, "output.jsonl")
            
            # 创建测试 txt 文件
            txt_file = os.path.join(unprocessed_dir, "test1.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("这是测试文本内容")
            
            stats = process_txt_files(
                unprocessed_dir=unprocessed_dir,
                output_jsonl=output,
                start_id=0
            )
            
            assert stats["total_txt_files"] == 1
            assert stats["processed"] == 1
            assert stats["deleted"] == 1
            assert stats["start_id"] == 0
            assert stats["end_id"] == 0
            
            # 验证输出文件
            assert os.path.exists(output)
            with open(output, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
                record = json.loads(line)
                assert record["id"] == "s0"
                assert record["text"] == "这是测试文本内容"
                assert "source_file" not in record
            
            # 验证 txt 文件已删除
            assert not os.path.exists(txt_file)
    
    def test_multiple_txt_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            unprocessed_dir = os.path.join(tmpdir, "unprocessed")
            os.makedirs(unprocessed_dir)
            output = os.path.join(tmpdir, "output.jsonl")
            
            # 创建多个测试 txt 文件
            for i in range(3):
                txt_file = os.path.join(unprocessed_dir, f"test{i}.txt")
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"测试文本内容 {i}")
            
            stats = process_txt_files(
                unprocessed_dir=unprocessed_dir,
                output_jsonl=output,
                start_id=0
            )
            
            assert stats["total_txt_files"] == 3
            assert stats["processed"] == 3
            assert stats["deleted"] == 3
            assert stats["start_id"] == 0
            assert stats["end_id"] == 2
            
            # 验证输出文件
            assert os.path.exists(output)
            with open(output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 3
                for i, line in enumerate(lines):
                    record = json.loads(line.strip())
                    assert record["id"] == format_id(i)
                    assert record["text"] == f"测试文本内容 {i}"
    
    def test_incremental_processing(self):
        """测试增量处理：已存在 JSONL 文件时继续编号"""
        with tempfile.TemporaryDirectory() as tmpdir:
            unprocessed_dir = os.path.join(tmpdir, "unprocessed")
            os.makedirs(unprocessed_dir)
            output = os.path.join(tmpdir, "output.jsonl")
            
            # 先创建已存在的 JSONL 文件
            with open(output, 'w', encoding='utf-8') as f:
                for i in range(2):
                    record = {"id": format_id(i), "text": f"已有文本 {i}"}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # 创建新的 txt 文件
            txt_file = os.path.join(unprocessed_dir, "new_test.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("新的测试文本")
            
            stats = process_txt_files(
                unprocessed_dir=unprocessed_dir,
                output_jsonl=output,
                start_id=0
            )
            
            assert stats["processed"] == 1
            assert stats["start_id"] == 2  # 从 2 开始（0, 1 已存在）
            assert stats["end_id"] == 2
            
            # 验证输出文件包含所有记录
            with open(output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 3
                last_record = json.loads(lines[-1].strip())
                assert last_record["id"] == "s2"
                assert last_record["text"] == "新的测试文本"
    
    def test_empty_txt_file(self):
        """测试空 txt 文件处理"""
        with tempfile.TemporaryDirectory() as tmpdir:
            unprocessed_dir = os.path.join(tmpdir, "unprocessed")
            os.makedirs(unprocessed_dir)
            output = os.path.join(tmpdir, "output.jsonl")
            
            # 创建空 txt 文件
            txt_file = os.path.join(unprocessed_dir, "empty.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                pass  # 空文件
            
            stats = process_txt_files(
                unprocessed_dir=unprocessed_dir,
                output_jsonl=output,
                start_id=0
            )
            
            assert stats["total_txt_files"] == 1
            assert stats["processed"] == 0
            assert stats["failed"] == 1
            # 空文件不应该被删除
            assert os.path.exists(txt_file)


class TestMergeJsonlFiles:
    """测试合并 JSONL 文件"""
    
    def test_merge_new_to_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = os.path.join(tmpdir, "existing.jsonl")
            new = os.path.join(tmpdir, "new.jsonl")
            
            # 创建已存在的文件
            with open(existing, 'w', encoding='utf-8') as f:
                record = {"id": "s0", "text": "已有文本"}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # 创建新文件
            with open(new, 'w', encoding='utf-8') as f:
                record = {"id": "s1", "text": "新文本"}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            merge_jsonl_files(existing, new)
            
            # 验证合并结果
            assert os.path.exists(existing)
            assert not os.path.exists(new)
            with open(existing, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 2
    
    def test_merge_nonexistent_existing(self):
        """测试合并到不存在的文件（应该重命名）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = os.path.join(tmpdir, "existing.jsonl")
            new = os.path.join(tmpdir, "new.jsonl")
            
            # 只创建新文件
            with open(new, 'w', encoding='utf-8') as f:
                record = {"id": "s0", "text": "新文本"}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            merge_jsonl_files(existing, new)
            
            # 验证新文件被重命名
            assert os.path.exists(existing)
            assert not os.path.exists(new)
            with open(existing, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                assert len(lines) == 1

