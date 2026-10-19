# -*- coding: utf-8 -*-
"""讲解进程里的 TTS 客户端。只发 HTTP，不加载 CosyVoice，不占一块模型显存。"""
from __future__ import annotations

import os
import time


def remote_base() -> str:
    return (os.environ.get("TTS_BASE_URL") or "http://127.0.0.1:8810").rstrip("/")


def use_remote_tts() -> bool:
    return (os.environ.get("TTS_BACKEND") or "").strip().lower() in {"http", "remote"}


class RemoteHandle:
    """占位对象。只提供 sample_rate，让讲解侧的 `if cosyvoice_model` 走得通。"""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = int(sample_rate or 24000)

    def __bool__(self):
        return True


def _sample_rate_from_health() -> int:
    try:
        import requests
        resp = requests.get(remote_base() + "/health", timeout=2)
        if resp.status_code == 200:
            return int(resp.json().get("sample_rate") or 24000)
    except Exception:
        pass
    return 24000


def activate_remote_tts() -> None:
    import tts.cosyvoice as cv
    sr = _sample_rate_from_health()
    cv.cosyvoice_model = RemoteHandle(sr)
    cv._tts_prompt_mode = "remote"
    cv._tts_support_stream = True
    cv._tts_use_spk_id = True
    cv._tts_last_error = ""
    print(f"🎤 TTS 使用独立进程 {remote_base()} sr={sr}（本进程不加载 CosyVoice）")


def speaker_id_of(persona, speaker: str = "") -> str:
    if persona is not None and getattr(persona, "id", None):
        return str(persona.id)
    return speaker or "monet"


def synthesize_pcm(text: str, speaker_id: str, cancel_ev=None, timeout: float = 90) -> tuple:
    """返回 (pcm_bytes, sample_rate)。连接失败重试两次，覆盖监督进程刚拉起的空档。"""
    import requests
    url = remote_base() + "/v1/tts"
    body = {"text": text, "speaker_id": speaker_id, "format": "pcm", "stream": True}
    last = None
    for attempt in range(3):
        buf = bytearray()
        odd = b""
        try:
            with requests.post(url, json=body, stream=True, timeout=(3, timeout)) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"TTS {resp.status_code}: {resp.text[:300]}")
                sr = int(resp.headers.get("X-Sample-Rate") or 24000)
                for chunk in resp.iter_content(4096):
                    if cancel_ev is not None and cancel_ev.is_set():
                        break
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
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"独立 TTS 不可用（可能正在空闲重启）: {last}") from last
