"""测试 gemma-3-27b 模型是否能正常工作"""
import os
from google import genai

# 从环境变量读取 API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("错误: GEMINI_API_KEY 环境变量未设置")
    exit(1)

print(f"API Key 已设置: {api_key[:10]}...")

# 创建客户端
client = genai.Client(api_key=api_key)

# 测试 gemma-3-27b
print("\n测试 gemma-3-27b 模型...")
try:
    response = client.models.generate_content(
        model="gemma-3-27b",
        contents="你好，请简单介绍一下你自己。"
    )
    print("✅ 调用成功!")
    print(f"响应: {response.text}")
except Exception as e:
    print(f"❌ 调用失败: {e}")
    print(f"错误类型: {type(e).__name__}")

