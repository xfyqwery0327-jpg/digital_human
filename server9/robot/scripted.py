# -*- coding: utf-8 -*-
"""莫奈三问口播稿。

mode:
  1 固定文本，音频用预录 wav 转 PCM（默认）
  2 固定文本，音频由 CosyVoice 现合成
  3 不走固定稿，大模型写成 150 到 200 字，再由 CosyVoice 合成
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

ASK_MODE_WAV = 1
ASK_MODE_COSY = 2
ASK_MODE_LLM = 3
ASK_MODES = (ASK_MODE_WAV, ASK_MODE_COSY, ASK_MODE_LLM)

# 跟项目根走，部署到 C:\digital_human 时也能找到预合成 wav。
_AUDIO_DIR = Path(__file__).resolve().parents[2] / "voice" / "monet" / "senlin_cv3" / "synth" / "06_12"

_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"“”‘’'（）()【】\[\]《》·・]+")

# 顺序即匹配优先级：睡莲池先于「作品」，避免第三问被第二问吃掉。
_MONET_SCRIPTS = (
    {
        "id": "pond",
        "wav": "s3.wav",
        "text": (
            "我1900年画的《睡莲池》，现藏芝加哥艺术学院。吉维尼池塘里，我种下睡莲，本想画植物，"
            "却被水面变幻的光迷住，于是绕到侧面，不画日本桥，把水面和植被都收进画里。"
            "技法上，先用稀薄棕绿薄涂铺底，干后以厚笔触叠加。睡莲与叶用含铅白厚涂堆出，定住再加彩色；"
            "常一笔蘸多色，不调匀，让颜色自己说话。画布纬纱粗，我垂直涂色，颜料吃进纹理，水面便有了晃动波纹。"
        ),
    },
    {
        "id": "self",
        "wav": "s1.wav",
        "text": (
            "我是克劳德·莫奈，大家爱叫我“印象派之父”。我这一辈子，大概画了两千多幅油画、五百多件素描。"
            "其中《睡莲》我前后画了180多幅；另外还有《干草堆》25幅、《鲁昂大教堂》30多幅、《白杨树》24幅。"
            "我喜欢对同一个场景反复画，比如干草堆、教堂、伦敦的雾，就是为了捕捉不同时辰光线的变化。"
        ),
    },
    {
        "id": "works",
        "wav": "s2.wav",
        "text": (
            "我一生追逐光与瞬间。《印象·日出》捕捉勒阿弗尔港清晨的橙红与灰蓝，让“印象派”由此得名；"
            "而《睡莲》是我持续创作的一个庞大系列，其中最常被提及的一幅是《睡莲池》"
            "水面倒影与天空交融，几乎成了我晚年的象征。《干草堆》《白杨树》《鲁昂大教堂》用不同时辰的光让寻常景物变幻；"
            "《日本桥》《撑阳伞的女人》《圣拉扎尔火车站》则留住花园、风与蒸汽中的色彩。"
            "我画的不是物体，而是光在空气里的呼吸。"
        ),
    },
)


def _norm(text: str) -> str:
    return _PUNCT.sub("", (text or "").strip())


def _hit(script_id: str, query: str) -> bool:
    q = _norm(query)
    if not q:
        return False
    if script_id == "pond":
        if "睡莲池" in q:
            return True
        return "睡莲" in q and any(k in q for k in ("这幅", "这个作品", "这件", "介绍"))
    if script_id == "self":
        if any(k in q for k in ("你自己", "介绍自己", "介绍下自己", "自我介绍")):
            return True
        return "介绍" in q and "自己" in q
    if script_id == "works":
        return any(k in q for k in ("你的作品", "哪些作品", "什么作品", "介绍作品", "作品"))
    return False


def _match_item(persona_id: str, query: str) -> Optional[dict]:
    if (persona_id or "").strip() != "monet":
        return None
    for item in _MONET_SCRIPTS:
        if _hit(item["id"], query):
            return item
    return None


def match_fixed_script(persona_id: str, query: str) -> Optional[str]:
    """命中则返回口播全文；未命中返回 None，调用方继续走大模型。"""
    item = _match_item(persona_id, query)
    return item["text"] if item else None


def fixed_script_wav(persona_id: str, query: str) -> Optional[Path]:
    """命中讲稿时返回 06_12 预合成 wav；未命中返回 None。"""
    item = _match_item(persona_id, query)
    if not item:
        return None
    return _AUDIO_DIR / item["wav"]


def normalize_ask_mode(mode) -> int:
    """合法模式原样返回；缺省或无法解析时回到 1。"""
    try:
        n = int(mode)
    except (TypeError, ValueError):
        return ASK_MODE_WAV
    return n if n in ASK_MODES else ASK_MODE_WAV


def scripted_turn(persona_id: str, query: str, mode=ASK_MODE_WAV) -> Optional[dict]:
    """三段固定对话按模式出稿。未命中，或 mode=3，返回 None。

    命中时返回 {"text", "wav", "mode"}。mode=2 的 wav 为 None，改由 CosyVoice 合成。
    """
    mode = normalize_ask_mode(mode)
    if mode == ASK_MODE_LLM:
        return None
    item = _match_item(persona_id, query)
    if not item:
        return None
    wav = (_AUDIO_DIR / item["wav"]) if mode == ASK_MODE_WAV else None
    return {"text": item["text"], "wav": wav, "mode": mode}
