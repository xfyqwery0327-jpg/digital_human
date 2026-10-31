# -*- coding: utf-8 -*-
"""讲解进程里的 TTS 客户端。只发 HTTP，不加载 CosyVoice，不占一块模型显存。"""
from __future__ import annotations

import os
import time


def remote_base() -> str:
    return (os.environ.get("TTS_BASE_URL") or "http://127.0.0.1:8810").rstrip("/")


def qwen_service_url() -> str:
    """讲解员直连 qwen-tts-service。并行服务才走分发器。"""
    return (os.environ.get("TTS_QWEN_URL") or "http://127.0.0.1:8812").rstrip("/")


def use_qwen_service() -> bool:
    """Qwen 口播直连 8812。设 TTS_QWEN_REMOTE=0 才回到本进程加载权重。"""
    from tts.engine import use_qwen_tts
    if not use_qwen_tts():
        return False
    flag = (os.environ.get("TTS_QWEN_REMOTE") or "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


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


def activate_qwen_service() -> None:
    """LLM 答完后的口播直连 qwen-tts-service。本进程不加载 Qwen 权重。"""
    import tts.cosyvoice as cv
    base = qwen_service_url()
    sr = 24000
    try:
        import requests
        resp = requests.get(base + "/health", timeout=2)
        if resp.status_code == 200:
            payload = resp.json()
            sr = int(payload.get("sample_rate") or 24000)
            cv._tts_last_error = payload.get("tts_error") or ""
        else:
            cv._tts_last_error = "口播入口 %s 返回 %s" % (base, resp.status_code)
    except Exception as exc:
        cv._tts_last_error = "口播入口未就绪 %s: %s" % (base, exc)
    cv.cosyvoice_model = RemoteHandle(sr)
    cv._tts_prompt_mode = "faster-qwen3-tts"
    cv._tts_support_stream = True
    cv._tts_use_spk_id = True
    cv._tts_infer_mode = "faster-qwen3-tts"
    if not cv._tts_last_error:
        cv._tts_last_error = ""
    cv._tts_log("[TTS] llm 口播转 %s sr=%s" % (base, sr))
    print("🎤 LLM 口播调用 %s sr=%s" % (base, sr))


def speaker_id_of(persona, speaker: str = "") -> str:
    if persona is not None and getattr(persona, "id", None):
        return str(persona.id)
    return speaker or "monet"


def tts_payload(text: str, speaker_id: str = "monet") -> dict:
    """qwen-tts-service 的请求体。language 用 Chinese / English / Japanese / Korean。"""
    from rag.speech_lang import qwen_language_of
    sid = (speaker_id or "").strip() or "monet"
    return {
        "text": text,
        "speaker_id": sid,
        "format": "pcm",
        "stream": True,
        "language": qwen_language_of(text),
    }


def iter_pcm(text: str, speaker_id: str, cancel_ev=None, timeout: float = 90, base: str = "",
              xvec_only=None, prefill=None):
    """边收边吐 (pcm_chunk, sample_rate)，chunk 长度保证是偶数字节。

    请求体对齐 qwen-tts-service：speaker_id 为 kumarajiva、monet、fuhao。
    不传 xvec_only / prefill 时跟随服务当前模式，不在这里改服务模式。
    连接失败重试两次，覆盖监督进程刚拉起的空档；已经吐过数据就不再重试，避免重复念。
    """
    import requests
    url = (base or remote_base()).rstrip("/") + "/v1/tts"
    body = tts_payload(text, speaker_id)
    if xvec_only is not None:
        body["xvec_only"] = bool(xvec_only)
    if prefill is not None:
        body["prefill"] = bool(prefill)
    last = None
    for attempt in range(3):
        odd = b""
        yielded = False
        try:
            with requests.post(url, json=body, stream=True, timeout=(3, timeout)) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"TTS {resp.status_code}: {resp.text[:300]}")
                sr = int(resp.headers.get("X-Sample-Rate") or 24000)
                for chunk in resp.iter_content(4096):
                    if cancel_ev is not None and cancel_ev.is_set():
                        return
                    if not chunk:
                        continue
                    if odd:
                        chunk = odd + chunk
                        odd = b""
                    if len(chunk) % 2:
                        odd = chunk[-1:]
                        chunk = chunk[:-1]
                    if chunk:
                        yielded = True
                        yield chunk, sr
            return
        except (requests.ConnectionError, requests.Timeout) as exc:
            if yielded:
                raise RuntimeError(f"独立 TTS 中途断开: {exc}") from exc
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"独立 TTS 不可用（可能正在空闲重启）: {last}") from last


def synthesize_pcm(text: str, speaker_id: str, cancel_ev=None, timeout: float = 90, base: str = "") -> tuple:
    """返回 (pcm_bytes, sample_rate)。base 为空时走 CosyVoice 独立进程。"""
    buf = bytearray()
    sr = 24000
    for chunk, sr in iter_pcm(text, speaker_id, cancel_ev, timeout, base=base):
        buf.extend(chunk)
    return bytes(buf), sr
