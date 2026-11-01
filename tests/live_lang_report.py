# -*- coding: utf-8 -*-
"""多场景、多意图、三个人物、四国口播。文字进报告，语音另存 wav。"""
import json
import re
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server9"))

import requests
from config import MIN_SCORE_RERANK, MIN_SCORE_RRF
from llm.client import clean_full_answer, llm_collect
from llm.persona import chitchat_reply, system_prompt
from rag import index as rag
from rag.bobohui import merge_bobohui
from rag.dialogue import (
    RealtimeMemory, family_reply, is_noise_question, prepare_realtime, visit_fact_reply,
)
from rag.scope import is_other_exhibition_query, refuse_text
from rag.search import search, works_answer_hint
from rag.speech_lang import (
    LANGS, confirm_text, detect_lang, matches_lang, output_rule, prepare_spoken,
    qwen_language_of,
)
from rag.visit import visit_hint
from services.screen_hooks import booking_reply
from tts.remote import qwen_service_url, tts_payload

OUT = ROOT / "tests" / "reports" / "lang_scenes"
WAV = OUT / "wav"
JSONL = OUT / "rows.jsonl"
REPORT = OUT / "report.md"

PERSONAS = ("monet", "fuhao", "kumarajiva")
LANG_ORDER = ("zh", "en", "ja", "ko")
SWITCH_Q = {"zh": "用中文", "en": "切换英语", "ja": "换成日语", "ko": "说韩语"}
WORK_Q = {
    "monet": "介绍一下睡莲",
    "fuhao": "介绍一下妇好鸮尊",
    "kumarajiva": "介绍一下你翻译的金刚经",
}
NAME = {
    "zh": "博博会",
    "en": "China Museum Expo",
    "ja": "中国博物館エキスポ",
    "ko": "중국 박물관 엑스포",
}
NEED_NAME = {"weather", "news", "news_more", "visit", "bobohui", "booth"}
OLD = ("浪漫与星辰", "贵州省博物馆", "这场商代", "龟兹博物馆", "七十八元", "四十九元", "这场展有我的")
HAN = re.compile(r"[\u4e00-\u9fff]")
HALL = (
    "展会只介绍博博会，不要介绍原先的展，也不要报原先的门票价格。"
    "问到本人作品或经历时只讲事情本身，不要说这场展有我的作品。"
    "你人已经在博博会现场，就在宽创国际的展厅里。介绍时说我就在这里，不要说在那里，不要说我听说。\n"
    "用两到三句说完。\n"
)


def _peak(pcm: bytes) -> int:
    if not pcm:
        return 0
    import array
    samples = array.array("h")
    samples.frombytes(pcm[:len(pcm) - (len(pcm) % 2)])
    return max((abs(x) for x in samples), default=0)


def _wait_health(base: str) -> dict:
    last = ""
    for _ in range(90):
        try:
            resp = requests.get(base + "/health", timeout=2)
            if resp.status_code == 200:
                body = resp.json()
                if body.get("ok"):
                    return body
            last = "HTTP %s" % resp.status_code
        except Exception as exc:
            last = str(exc)
        time.sleep(2)
    raise RuntimeError("口播服务没就绪: %s" % last)


def _threshold() -> float:
    return MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF


def _lookup(persona, question):
    with __import__("persona.runtime", fromlist=["bind"]).bind(persona):
        try:
            hits = search(question, 4, None, None, persona)
        except Exception:
            hits = []
    return merge_bobohui(question, hits or [], _threshold())


def _llm(persona, question, code) -> str:
    hits = _lookup(persona, question)
    if not hits or float(hits[0].get("score") or 0) < _threshold():
        if not any(h.get("id") == "bobohui_2026_11" for h in hits):
            return refuse_text(persona)
    ref = "\n".join("- %s" % ((h.get("text") or "")[:220]) for h in hits[:2])
    user = "【参考资料】\n%s\n\n【问题】%s\n\n%s%s%s/no_think" % (
        ref, question, HALL, works_answer_hint(question, persona), visit_hint(question, persona) + output_rule(code),
    )
    messages = [
        {"role": "system", "content": system_prompt(persona)},
        {"role": "user", "content": user},
    ]
    with __import__("persona.runtime", fromlist=["bind"]).bind(persona):
        raw = llm_collect(messages, max_tokens=140)
    return (clean_full_answer(raw) or "").strip()


def _spoken(text: str, code: str) -> str:
    return prepare_spoken(text or "", code)


def _issues(row: dict) -> list:
    text = row.get("text") or ""
    code = row.get("lang") or "zh"
    found = []
    if row.get("error"):
        found.append("合成失败")
    if not text:
        found.append("没有口播")
        return found
    han_leak = (code == "en" and HAN.search(text) is not None) or (
        code == "ko" and len(HAN.findall(text)) >= 2)
    if han_leak:
        found.append("夹了汉字")
    elif not matches_lang(text, code):
        found.append("语种不是%s（检出%s）" % (LANGS[code]["label"], detect_lang(text) or "空"))
    if code == "ja" and "博博会" in text:
        found.append("日语里仍写着博博会")
    if "Bobohui" in text or "bobohui" in text.lower():
        found.append("用了旧译名 Bobohui")
    if row["scene"] in NEED_NAME and NAME[code] not in text:
        found.append("专名未写成%s" % NAME[code])
    bad = [word for word in OLD if word in text]
    if bad:
        found.append("旧展会残留：" + "、".join(bad))
    if "在那里" in text:
        found.append("说了在那里")
    if "我听说" in text:
        found.append("说了我听说")
    if row["scene"] in ("news", "news_more") and "http" in text.lower():
        found.append("新闻口播带了链接")
    if row.get("peak", 0) < 800:
        found.append("峰值过低")
    return found


def _synth(text: str, speaker: str, code: str, path: Path):
    body = tts_payload(text, speaker)
    body["language"] = qwen_language_of(text, code)
    url = qwen_service_url() + "/v1/tts"
    buf = bytearray()
    sr = 24000
    t0 = time.perf_counter()
    with requests.post(url, json=body, stream=True, timeout=(5, 180)) as resp:
        if resp.status_code >= 400:
            raise RuntimeError("TTS %s: %s" % (resp.status_code, resp.text[:200]))
        sr = int(resp.headers.get("X-Sample-Rate") or 24000)
        odd = b""
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
    wall = time.perf_counter() - t0
    pcm = bytes(buf)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm)
    audio = len(pcm) / 2 / sr if sr else 0
    return wall, audio, _peak(pcm), body["language"]


def _shared_fixed(scene: str, code: str, cache: dict) -> str:
    key = (scene, code)
    if key in cache:
        return cache[key]
    mem = cache.setdefault(("mem", code), RealtimeMemory())
    if scene == "switch":
        text = confirm_text(code)
    elif scene == "weather":
        text = (prepare_realtime("今天海口天气怎么样", mem, None).get("reply") or "").strip()
    elif scene == "news":
        text = cache.pop(("news_raw", code), "")
    elif scene == "news_more":
        text = (prepare_realtime("再来一条", mem, None).get("reply") or "").strip()
    elif scene == "visit":
        text = visit_fact_reply("博博会什么时候开，在哪里举办", None)
    elif scene == "ticket_price":
        text = visit_fact_reply("门票多少钱", None)
    elif scene == "ticket_book":
        text = booking_reply()
    else:
        text = ""
    cache[key] = text if scene == "switch" else _spoken(text, code)
    return cache[key]


def _persona_text(persona, scene: str, code: str) -> tuple:
    """返回 (中文或已按规则生成的原稿, 来源)。译成目标语在外面做。"""
    if scene == "hello":
        return chitchat_reply("你好", persona), "固定句"
    if scene == "noise":
        return refuse_text(persona) if is_noise_question("啊") else "", "固定句"
    if scene == "family":
        fact = family_reply("你的妻子是谁", persona=persona)
        return (fact or refuse_text(persona)), "固定句"
    if scene == "other_expo":
        q = "其他展会什么时候开"
        if is_other_exhibition_query(q, persona):
            return refuse_text(persona), "固定句"
        return _llm(persona, q, code), "大模型"
    if scene == "old_show":
        q = "浪漫与星辰什么时候开展"
        if is_other_exhibition_query(q, persona):
            return refuse_text(persona), "固定句"
        return _llm(persona, q, code), "大模型"
    if scene == "self":
        return _llm(persona, "请介绍一下你自己", code), "大模型"
    if scene == "work":
        return _llm(persona, WORK_Q[persona.id], code), "大模型"
    if scene == "bobohui":
        return _llm(persona, "介绍一下第十一届博博会", code), "大模型"
    if scene == "booth":
        return _llm(persona, "宽创展台是做什么的", code), "大模型"
    if scene == "guide":
        return _llm(persona, "带我看看这个展厅", code), "大模型"
    return "", ""


SCENES = (
    ("switch", "lang_switch", "切换语言", lambda p, code: SWITCH_Q[code]),
    ("hello", "chitchat", "打招呼", lambda p, code: "你好"),
    ("weather", "weather", "海口天气", lambda p, code: "今天海口天气怎么样"),
    ("news", "news", "新闻第一条", lambda p, code: "最近有什么新闻"),
    ("news_more", "news", "新闻再来一条", lambda p, code: "再来一条"),
    ("visit", "visit", "时间地点", lambda p, code: "博博会什么时候开，在哪里举办"),
    ("ticket_price", "visit", "问票价", lambda p, code: "门票多少钱"),
    ("ticket_book", "ticket_service", "预约门票", lambda p, code: "帮我预约门票"),
    ("guide", "guide", "展厅导览", lambda p, code: "带我看看这个展厅"),
    ("bobohui", "knowledge", "介绍博博会", lambda p, code: "介绍一下第十一届博博会"),
    ("booth", "knowledge", "宽创展台", lambda p, code: "宽创展台是做什么的"),
    ("self", "person", "自我介绍", lambda p, code: "请介绍一下你自己"),
    ("work", "exhibit", "代表作品", lambda p, code: WORK_Q[p.id]),
    ("family", "family", "问妻子", lambda p, code: "你的妻子是谁"),
    ("other_expo", "refuse", "其他展会", lambda p, code: "其他展会什么时候开"),
    ("old_show", "knowledge", "原先的展", lambda p, code: "浪漫与星辰什么时候开展"),
    ("noise", "refuse", "听不清", lambda p, code: "啊"),
)
SHARED = {"switch", "weather", "news", "news_more", "visit", "ticket_price", "ticket_book"}


def _write_report(rows: list, health: dict) -> None:
    bad = [row for row in rows if row.get("issues")]
    lines = [
        "# 四国语言多场景口播报告",
        "",
        "观众问题仍是中文。口播切到中文、英语、日语、韩语。语音走 qwen-tts-service `%s`，人物为莫奈、妇好、鸠摩罗什。" % qwen_service_url(),
        "",
        "- 服务：%s，语速 %s，xvec_only=%s，采样率 %s" % (
            health.get("engine") or "",
            health.get("speech_rate"),
            health.get("xvec_only"),
            health.get("sample_rate"),
        ),
        "- 博博会专名：英语 China Museum Expo，日语 中国博物館エキスポ，韩语 중국 박물관 엑스포",
        "- 新闻口播只保留钩子，不念链接。英语和韩语里的数字用阿拉伯数字。",
        "- 条数 %d，不合格 %d，语音目录 `wav/`" % (len(rows), len(bad)),
        "",
        "## 不合格",
        "",
    ]
    if not bad:
        lines.append("没有。")
    else:
        lines.append("| 人物 | 语言 | 场景 | 问题 |")
        lines.append("| --- | --- | --- | --- |")
        for row in bad:
            why = "；".join(row["issues"])
            lines.append("| %s | %s | %s | %s |" % (row["name"], row["lang_label"], row["title"], why))
    lines.extend(["", "## 总表", ""])
    lines.append("| 人物 | 语言 | 意图 | 场景 | 来源 | 检出 | 秒 | 峰值 | 语音 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        mark = "不合格" if row.get("issues") else "通过"
        lines.append("| %s | %s | %s | %s | %s | %s | %.1f | %s | [%s](%s) |" % (
            row["name"], row["lang_label"], row["intent"], row["title"],
            row["source"], detect_lang(row.get("text") or "") or "空",
            row.get("audio") or 0, row.get("peak") or 0, mark, row["rel"],
        ))
    lines.extend(["", "## 口播正文", ""])
    current = ""
    for row in rows:
        head = "%s · %s" % (row["name"], row["title"])
        if head != current:
            current = head
            lines.extend(["### %s" % head, "", "问题：%s" % row["question"], ""])
        lines.append("**%s**（%s，%s）" % (row["lang_label"], row["source"], "；".join(row["issues"]) or "通过"))
        lines.append("")
        lines.append(row.get("text") or "（空）")
        if row.get("news_url"):
            lines.append("")
            lines.append("前端参数 url：%s" % row["news_url"])
        lines.append("")
        lines.append("语音：[%s](%s)" % (row["rel"], row["rel"]))
        lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _done_keys() -> set:
    found = set()
    if not JSONL.exists():
        return found
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        wav = ROOT / "tests" / "reports" / "lang_scenes" / (row.get("rel") or "")
        if row.get("text") and wav.exists() and not row.get("error"):
            found.add(row.get("key"))
    return found


def main() -> int:
    from persona.registry import registry
    OUT.mkdir(parents=True, exist_ok=True)
    registry.load()
    rag.load_index()
    base = qwen_service_url()
    health = _wait_health(base)
    print("口播服务就绪", health.get("engine"), "xvec_only", health.get("xvec_only"))
    people = []
    for pid in PERSONAS:
        persona = registry.get(pid)
        if persona is None:
            print("缺少人设", pid)
            return 1
        people.append(persona)
    cache = {}
    text_cache = {}
    done = _done_keys()
    rows = []
    if done and JSONL.exists():
        seen = set()
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("key") in done and row["key"] not in seen:
                row["issues"] = _issues(row)
                rows.append(row)
                seen.add(row["key"])
    total = len(people) * len(SCENES) * len(LANG_ORDER)
    index = 0
    for persona in people:
        for scene, intent, title, question_of in SCENES:
            for code in LANG_ORDER:
                index += 1
                key = "%s|%s|%s" % (persona.id, scene, code)
                rel = "wav/%s/%s/%s.wav" % (persona.id, code, scene)
                if key in done:
                    print("[%d/%d] 已有 %s" % (index, total, key))
                    continue
                question = question_of(persona, code)
                source = "固定句"
                news_url = ""
                text = ""
                wall = audio = peak = 0
                lang_sent = ""
                error = ""
                try:
                    if scene in SHARED:
                        if scene == "news" and ("url", code) not in cache:
                            mem = cache.setdefault(("mem", code), RealtimeMemory())
                            turn = prepare_realtime("最近有什么新闻", mem, None)
                            cache[("news_raw", code)] = (turn.get("reply") or "").strip()
                            cache[("url", code)] = (turn.get("bundle") or {}).get("url") or ""
                        text = _shared_fixed(scene, code, cache)
                        if scene == "switch":
                            source = "固定确认"
                        elif code != "zh":
                            source = "翻译"
                        news_url = cache.get(("url", code), "") if scene in ("news", "news_more") else ""
                    else:
                        tkey = (persona.id, scene, code)
                        if tkey not in text_cache:
                            draft, origin = _persona_text(persona, scene, code)
                            spoken = draft if code == "zh" else _spoken(draft, code)
                            if origin == "大模型" and spoken == draft:
                                origin = "大模型"
                            elif spoken != draft:
                                origin = "翻译"
                            text_cache[tkey] = (spoken, origin)
                        text, source = text_cache[tkey]
                    if not (text or "").strip():
                        raise RuntimeError("大模型没有生成口播")
                    path = OUT / rel
                    wall, audio, peak, lang_sent = _synth(text, persona.id, code, path)
                except Exception as exc:
                    error = "%s: %s" % (type(exc).__name__, exc)
                    print("失败", key, error)
                row = {
                    "key": key,
                    "scene": scene,
                    "intent": intent,
                    "title": title,
                    "question": question,
                    "persona": persona.id,
                    "name": persona.display_name,
                    "lang": code,
                    "lang_label": LANGS[code]["label"],
                    "source": source,
                    "text": text if isinstance(text, str) else "",
                    "rel": rel,
                    "audio": round(audio, 2),
                    "wall": round(wall, 2),
                    "peak": int(peak),
                    "tts_language": lang_sent,
                    "news_url": news_url,
                    "error": error,
                }
                row["issues"] = _issues(row)
                with JSONL.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                print("[%d/%d] %s %s %s 句=%d 秒=%.1f 峰值=%d %s" % (
                    index, total, persona.display_name, LANGS[code]["label"], title,
                    len(row["text"]), audio, peak, "；".join(row["issues"]) or "通过",
                ))
                if index % 6 == 0:
                    _write_report(rows, health)
    _write_report(rows, health)
    bad = sum(1 for row in rows if row.get("issues"))
    print("报告", REPORT)
    print("条数", len(rows), "不合格", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
