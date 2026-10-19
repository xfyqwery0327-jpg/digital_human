# -*- coding: utf-8 -*-
"""整句出声引擎（CosyVoice3 / Qwen3-TTS）的拼句节奏。

固定拼到 40 字不行：第一句播完时，第二段 40 字还没合成出来，照样断档。
这里按「客户端还剩几秒没播」和实测合成速度，决定下一次最多拼多少字：
缓冲少就只送一个小句，缓冲多再拼长，少算几遍参考音前缀。
"""
from __future__ import annotations

from typing import List, Optional, Tuple


class UnitPlanner:
    def __init__(self, max_chars: int, margin_sec: float = 0.3, alpha: float = 0.5):
        self.max_chars = max(0, int(max_chars))
        self.margin_sec = float(margin_sec)
        self.alpha = float(alpha)
        self.k: Optional[float] = None
        self.cps: Optional[float] = None

    def observe(self, chars: int, audio_sec: float, synth_sec: float) -> None:
        """记一次实测。k = 合成秒数 / 音频秒数（含参考音前缀开销），cps = 每秒念几个字。"""
        if chars <= 0 or audio_sec <= 0.2 or synth_sec <= 0:
            return
        k = synth_sec / audio_sec
        cps = chars / audio_sec
        a = self.alpha
        self.k = k if self.k is None else a * k + (1 - a) * self.k
        self.cps = cps if self.cps is None else a * cps + (1 - a) * self.cps

    def observe_round(self, measured: List[Tuple[int, float, float]],
                      batch_sec: Optional[float] = None) -> None:
        """measured：本轮每段 (字数, 音频秒数, 推送耗时)。

        batch_sec 给出时本轮是一次 batch 预取：耗时取决于最长那段，只按它记一次。
        """
        if not measured:
            return
        if batch_sec is None:
            for chars, audio_sec, synth_sec in measured:
                self.observe(chars, audio_sec, synth_sec)
            return
        chars, audio_sec, push_sec = max(measured, key=lambda m: m[0])
        self.observe(chars, audio_sec, batch_sec + push_sec)

    def char_cap(self, ahead_sec: float) -> int:
        """缓冲还剩 ahead_sec 秒时，下一段最多拼多少字才来得及。0 = 只送一个小句。"""
        if not self.max_chars or self.k is None or self.cps is None:
            return 0
        usable = float(ahead_sec) - self.margin_sec
        if usable <= 0:
            return 0
        return min(self.max_chars, int(usable / max(self.k, 0.05) * self.cps))

    def take(self, backlog: List[str], ahead_sec: float, max_units: int = 1) -> List[str]:
        """从 backlog 头部按顺序取走最多 max_units 段，每段由相邻小句拼成，不超过 char_cap。

        至少取一个小句，绝不让 GPU 空等更多文本。还没测过速度时只送一个小句，先出声。
        """
        cap = self.char_cap(ahead_sec)
        limit = max(1, int(max_units)) if self.k is not None else 1
        units: List[str] = []
        while backlog and len(units) < limit:
            # batch 耗时看最长那段：后面的段比第一段还长、又超出 cap，就留到下一轮。
            if units and len(backlog[0]) > max(cap, len(units[0])):
                break
            text = backlog.pop(0)
            while backlog and len(text) + len(backlog[0]) <= cap:
                text += backlog.pop(0)
            units.append(text)
        return units
