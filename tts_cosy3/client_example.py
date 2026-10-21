# -*- coding: utf-8 -*-
"""机器人 LLM 进程调用示例。这个文件跑在 LLM 进程里，不要在这里加载 CosyVoice。

命令行：
  python -m tts_cosy3.client_example --text "你好，我是莫奈。" --speaker monet --out out.wav

curl（PCM 流，24kHz int16 单声道）：
  curl -N -X POST http://127.0.0.1:8810/v1/tts ^
    -H "Content-Type: application/json" ^
    -o reply.pcm ^
    -d "{\"text\":\"你好，我是莫奈。\",\"speaker_id\":\"monet\",\"format\":\"pcm\",\"stream\":true}"

curl（整段 wav）：
  curl -X POST http://127.0.0.1:8810/v1/tts ^
    -H "Content-Type: application/json" ^
    -o reply.wav ^
    -d "{\"text\":\"你好，我是莫奈。\",\"speaker_id\":\"monet\",\"format\":\"wav\",\"stream\":false}"

用本机参考音登记音色：
  curl -X POST http://127.0.0.1:8810/v1/speakers/clone ^
    -H "Content-Type: application/json" ^
    -d "{\"speaker_id\":\"monet\",\"prompt_wav\":\"D:\\\\projects\\\\digital_human\\\\voice\\\\monet\\\\monet-senlin_6to12s.wav\",\"prompt_text\":\"隆重登场的光影魔法森林。\"}"
"""
from __future__ import annotations

import argparse
import os
import time

import requests

DEFAULT_BASE = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8810")


def synthesize_pcm(text: str, speaker_id: str, base: str = DEFAULT_BASE,
                   timeout: float = 90, retries: int = 2) -> tuple[bytes, int]:
    """一句口播。服务若正在空闲重启，连接失败会短暂重试，不会在 LLM 进程里加载模型。"""
    url = base.rstrip("/") + "/v1/tts"
    body = {
        "text": text,
        "speaker_id": speaker_id,
        "format": "pcm",
        "stream": True,
    }
    last_err = None
    for attempt in range(retries + 1):
        buf = bytearray()
        odd = b""
        try:
            with requests.post(url, json=body, stream=True, timeout=(3, timeout)) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"TTS {resp.status_code}: {resp.text[:300]}")
                sr = int(resp.headers.get("X-Sample-Rate") or 24000)
                for chunk in resp.iter_content(4096):
                    if not chunk:
                        continue
                    if odd:
                        chunk = odd + chunk
                        odd = b""
                    if len(chunk) % 2:
                        odd = chunk[-1:]
                        chunk = chunk[:-1]
                    buf.extend(chunk)
            return bytes(buf), sr
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"TTS 进程不可用（可能正在空闲重启）: {last_err}") from last_err


def pcm_to_wav(pcm: bytes, sample_rate: int, path: str) -> None:
    import wave
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description="调用独立 CosyVoice3 TTS")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", default="monet")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    pcm, sr = synthesize_pcm(args.text, args.speaker, base=args.base)
    print(f"pcm={len(pcm)} bytes sr={sr} audio={len(pcm) / 2 / sr:.2f}s")
    if args.out:
        if args.out.lower().endswith(".wav"):
            pcm_to_wav(pcm, sr, args.out)
        else:
            with open(args.out, "wb") as fh:
                fh.write(pcm)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
