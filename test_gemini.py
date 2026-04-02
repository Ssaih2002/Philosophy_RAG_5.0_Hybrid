import os
from google import genai


# 使用艾可云提供的本地 HTTP 代理端口（33210）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:33210"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:33210"

# 在这里填写你的 Gemini API Key 字符串
# 例子：API_KEY = "AIza......你的真实Key......"
API_KEY = "请输入你自己的Gemini API Key"

client = genai.Client(api_key=API_KEY)

print("Sending request...")
try:
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence.",
    )
    print("Received response:")
    print(resp.text)
except Exception as e:
    print("Request failed:", repr(e))