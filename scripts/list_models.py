import os
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY", "AIzaSyBO15-oppfm7YH8Q7cppsCbX63Bh8HqYOw")
genai.configure(api_key=api_key)

print("可用的 Gemini 模型：")
for m in genai.list_models():
    if 'generateContent' in str(m.supported_generation_methods):
        print(f"  - {m.name}")



