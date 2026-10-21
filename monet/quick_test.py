import httpx, json, time

url = "http://127.0.0.1:8080/v1/chat/completions"
payload = {
    "model": "qwen3",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 10,
    "stream": False
}

start = time.time()
try:
    resp = httpx.post(url, json=payload, timeout=30.0)
    print(f"状态码: {resp.status_code}")
    print(f"耗时: {time.time()-start:.2f}s")
    print(f"内容: {resp.json()}")
except Exception as e:
    print(f"错误: {e}")