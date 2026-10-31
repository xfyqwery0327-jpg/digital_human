# -*- coding: utf-8 -*-
"""长句：四国标点切句后，量 llm+tts 总回复、首句、句间间隔、RTF、平均峰值。"""
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))
os.environ.setdefault("QWEN_TTS_WORKER", "1")

import requests
from llm.client import extract_clean_delta, iter_llm_sse
from rag.speech_lang import LANGS, detect_lang, output_rule, prepare_spoken
from tts.speaktext import find_sentence_end, split_by_marks
from config import LLM_URL

ASK = "用五到六句讲你此刻在博博会宽创展厅里看到的光和颜色。每句都写完整。"


def _peak(pcm: bytes) -> int:
    if not pcm:
        return 0
    import array
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return max((abs(x) for x in samples), default=0)


def _synthesize(text: str, persona):
    """service 走 8812。否则本进程加载权重。"""
    if os.environ.get("BENCH_TTS") == "service":
        from tts.remote import qwen_service_url, synthesize_pcm
        return synthesize_pcm(text, "monet", timeout=180, base=qwen_service_url())
    from tts.qwen import synthesize_pcm
    return synthesize_pcm(text, persona)


def _pipeline(code: str, persona):
    """第一句一切出来就开始合成，后面的句子接着排。时钟从发出请求起算。"""
    box = queue.Queue()
    rule = output_rule(code)
    messages = [{
        "role": "user",
        "content": (
            "你是莫奈，人就在博博会现场的宽创展厅里。%s\n问题：%s\n/no_think" % (rule, ASK)
        ),
    }]

    def reader():
        payload = {"messages": messages, "temperature": 0.05, "max_tokens": 420, "stream": True}
        t0 = time.perf_counter()
        full_raw = ""
        clean_pos = 0
        buf = ""
        resp = None
        try:
            resp = requests.post(LLM_URL, json=payload, timeout=90, stream=True)
            resp.raise_for_status()
            box.put(("t0", t0))
            for line in iter_llm_sse(resp):
                if not line or not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0].get("delta", {})
                except Exception:
                    continue
                content = delta.get("content", "")
                if not content:
                    continue
                full_raw += content
                new_content, clean_pos = extract_clean_delta(full_raw, clean_pos)
                if not new_content:
                    continue
                buf += new_content
                while True:
                    pos = find_sentence_end(buf, code)
                    if pos < 0:
                        break
                    sent = buf[:pos + 1].strip()
                    buf = buf[pos + 1:]
                    sent = sent.replace("/no_think", "").strip()
                    if sent:
                        box.put(("sent", sent, time.perf_counter() - t0))
            tail = buf.replace("/no_think", "").strip()
            if tail:
                box.put(("sent", tail, time.perf_counter() - t0))
            box.put(("llm", time.perf_counter() - t0, full_raw))
        except Exception as exc:
            box.put(("err", exc))
        finally:
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass

    threading.Thread(target=reader, daemon=True).start()
    t0 = None
    raw = ""
    llm_done = None
    first_text = None
    rows = []
    while True:
        item = box.get()
        kind = item[0]
        if kind == "err":
            raise item[1]
        if kind == "t0":
            t0 = item[1]
            continue
        if kind == "llm":
            llm_done = item[1]
            raw = item[2]
            break
        text, ready = item[1], item[2]
        if first_text is None:
            first_text = ready
        spoken = prepare_spoken(text, code)
        synth_at = time.perf_counter()
        pcm, rate, peak, note = b"", 24000, 0, ""
        for attempt in range(2):
            try:
                pcm, rate = _synthesize(spoken, persona)
                peak = _peak(pcm)
                note = ""
                break
            except Exception as exc:
                pcm, rate, peak = b"", 24000, 0
                note = str(exc)
                if "quiet" not in note or attempt == 1:
                    print("  合成失败:", note)
                    break
                print("  静音重试:", spoken[:40])
        wall = time.perf_counter() - synth_at
        audio = (len(pcm) / 2.0 / float(rate)) if pcm and rate else 0.0
        rows.append({
            "text": spoken,
            "ready": ready,
            "wall": wall,
            "audio": audio,
            "peak": peak,
            "note": note,
            "done": time.perf_counter(),
        })
    return raw, rows, first_text or llm_done, llm_done, t0


def _report(code, raw, rows, first_text, llm_done, t0):
    label = LANGS[code]["label"]
    print("【%s】检测=%s 句数=%d 大模型收完=%.2fs 首句文本=%.2fs" % (
        label, detect_lang(raw) or "未认出", len(rows), llm_done, first_text))
    print("  原文:", raw.replace("\n", " ")[:240])
    if not rows:
        print("  没有句子")
        return
    play_end = None
    gaps = []
    peaks = []
    walls = 0.0
    audios = 0.0
    first_audio = None
    for i, row in enumerate(rows):
        peaks.append(row["peak"])
        walls += row["wall"]
        audios += row["audio"]
        ready_audio = row["done"] - t0
        if first_audio is None:
            first_audio = ready_audio
        if play_end is None:
            play_end = ready_audio + row["audio"]
        else:
            gap = ready_audio - play_end
            gaps.append(gap)
            play_end = max(play_end, ready_audio) + row["audio"]
        rtf = row["wall"] / row["audio"] if row["audio"] else 0.0
        print("  句%d 文本就绪=%.2fs 合成=%.2fs 音频=%.2fs rtf=%.2f 峰值=%d | %s" % (
            i + 1, row["ready"], row["wall"], row["audio"], rtf, row["peak"], row["text"][:80]))
    total = rows[-1]["done"] - t0
    avg_gap = sum(max(0.0, g) for g in gaps) / len(gaps) if gaps else 0.0
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    rtf = walls / audios if audios else 0.0
    print("  总回复=%.2fs 首句出声=%.2fs 句间间隔=%s 平均间隔=%.2fs RTF=%.2f 平均峰值=%d" % (
        total,
        first_audio,
        " ".join("%.2f" % g for g in gaps) or "-",
        avg_gap,
        rtf,
        int(avg_peak),
    ))
    print()


def main():
    from persona.registry import registry
    from persona.runtime import bind

    registry.load()
    persona = registry.get("monet")
    if persona is None:
        print("没有莫奈人设")
        return 1
    service = os.environ.get("BENCH_TTS") == "service"
    with bind(persona):
        if service:
            from tts.remote import qwen_service_url
            print("口播走 qwen-tts-service %s" % qwen_service_url())
        else:
            from tts.qwen import activate
            if not activate():
                print("本地语音没加载")
                return 1
        print("预热一句，不计入成绩")
        try:
            _synthesize("好的，接下来我用中文讲。", persona)
        except Exception as exc:
            print("预热跳过:", exc)
        codes = [item for item in sys.argv[1:] if item in LANGS] or ["zh", "en", "ja", "ko"]
        for code in codes:
            raw, rows, first_text, llm_done, t0 = _pipeline(code, persona)
            joined = "".join(row["text"] for row in rows)
            again = split_by_marks(joined, code)
            if len(again) != len(rows):
                print("【%s】流式切句和整段切句句数不同 流式=%d 整段=%d" % (
                    LANGS[code]["label"], len(rows), len(again)))
            _report(code, raw, rows, first_text, llm_done, t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
