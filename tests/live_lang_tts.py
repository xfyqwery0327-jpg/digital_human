# -*- coding: utf-8 -*-
"""现场看中英日韩：大模型口播，以及 qwen-tts-service 能不能出声。"""
import socket
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

from llm.client import clean_full_answer, llm_collect
from rag.speech_lang import LANGS, detect_lang, output_rule, prepare_spoken, qwen_language_of
from tts.remote import qwen_service_url, synthesize_pcm, tts_payload

ASK = "用一句话说明你此刻就在博博会的宽创展厅里。"
OUT = ROOT / "tests" / "_lang_tts"


def _port_open(url: str) -> bool:
    try:
        host, port = url.split("://", 1)[1].split("/", 1)[0].split(":")
        socket.create_connection((host, int(port)), 2).close()
        return True
    except Exception:
        return False


def _llm(code: str) -> str:
    rule = output_rule(code)
    messages = [{
        "role": "user",
        "content": "你是莫奈，人就在博博会现场的宽创展厅里。%s\n问题：%s\n/no_think" % (rule, ASK),
    }]
    raw = clean_full_answer(llm_collect(messages, max_tokens=180))
    spoken = prepare_spoken(raw, code)
    return raw, spoken


def _save_wav(path: Path, pcm: bytes, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)


def main():
    failed = 0
    for code in ("zh", "en", "ja", "ko"):
        label = LANGS[code]["label"]
        try:
            raw, spoken = _llm(code)
        except Exception as exc:
            failed += 1
            print("【%s 大模型】失败 %s" % (label, exc))
            continue
        got = detect_lang(spoken)
        print("【%s 大模型】检测=%s" % (label, got or "未认出"))
        print("  直接生成:", raw)
        print("  送去口播:", spoken)
        if got != code:
            failed += 1
            print("  语种不对")
        print()
    base = qwen_service_url()
    if not _port_open(base):
        print("qwen-tts-service 未开：%s" % base)
        print("请求体已经带 language，服务起来后才会出声。")
        return 1 if failed else 2
    for code in ("zh", "en", "ja", "ko"):
        text = LANGS[code]["confirm"]
        body = tts_payload(text, "monet")
        print("【%s TTS】language=%s" % (LANGS[code]["label"], body["language"]))
        try:
            pcm, rate = synthesize_pcm(text, "monet", timeout=60, base=base)
        except Exception as exc:
            failed += 1
            print("  合成失败:", exc)
            continue
        peak = 0
        if pcm:
            import array
            samples = array.array("h")
            samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
            peak = max(abs(n) for n in samples) if samples else 0
        sec = len(pcm) / 2.0 / float(rate or 24000)
        path = OUT / ("%s.wav" % code)
        _save_wav(path, pcm, rate or 24000)
        print("  时长=%.2fs 峰值=%s 文件=%s" % (sec, peak, path))
        if sec < 0.4 or peak < 800:
            failed += 1
            print("  音频太短或太轻")
        if qwen_language_of(text) != body["language"]:
            failed += 1
        print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
