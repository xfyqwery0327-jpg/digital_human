# -*- coding: utf-8 -*-
"""整句出声引擎（CosyVoice3 / Qwen3-TTS）的拼句节奏。

Qwen3-TTS 每次 generate 都要重算参考音前缀，单句墙钟大约是音频的两倍多。
只送一句时，这句播完下一句还没出来。一次取满 max_units 段短句，墙钟约等于
最长那段，音频是各段相加，批的 RTF 才能低于 1，后面的空档才补得上。
拼句上限仍按客户端剩余缓冲收着，避免一段长到几十个字把整批墙钟拖死。
"""
from __future__ import annotations

from typing import List, Optional, Tuple


_CLAUSE_PUNCT = "，。！？；、,.!?"


class UnitPlanner:
    def __init__(self, max_chars: int, margin_sec: float = 0.3, alpha: float = 0.5,
                 split_chars: int = 0, split_marks: bool = False):
        self.max_chars = max(0, int(max_chars))
        self.margin_sec = float(margin_sec)
        self.alpha = float(alpha)
        self.split_chars = max(0, int(split_chars))
        self.split_marks = bool(split_marks)
        self.k: Optional[float] = None
        self.cps: Optional[float] = None

    def _split_clause(self, text: str) -> List[str]:
        """按书名号和标点切开。字数封顶只在明确打开 split_chars 时才用。"""
        if self.split_marks:
            from tts.speaktext import split_by_marks
            parts = split_by_marks(text)
            return parts or [text]
        limit = self.split_chars
        if limit <= 0 or len(text) <= limit:
            return [text]
        punct = text[-1] if text[-1] in _CLAUSE_PUNCT else ""
        body = text[:-1] if punct else text
        if len(body) <= limit:
            return [text]
        parts: List[str] = []
        while len(body) > limit:
            parts.append(body[:limit])
            body = body[limit:]
        if not body:
            parts[-1] += punct
        else:
            parts.append(body + punct)
        return parts

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

        batch_sec 给出时本轮是一次 batch 预取。墙钟只花一份，音频是各段相加，
        按整批来记 k。若只记最长那段，k 会停在 2 以上，后面每轮只敢送一两句。
        """
        if not measured:
            return
        if batch_sec is None:
            for chars, audio_sec, synth_sec in measured:
                self.observe(chars, audio_sec, synth_sec)
            return
        chars = sum(c for c, _, _ in measured)
        audio_sec = sum(a for _, a, _ in measured)
        push_sec = sum(s for _, _, s in measured)
        self.observe(chars, audio_sec, batch_sec + push_sec)

    def char_cap(self, ahead_sec: float) -> int:
        """缓冲还剩 ahead_sec 秒时，下一段最多拼多少字。0 = 不拼，原样逐段送。

        有效合成比 k 夹 0.5 下限：首句（前缀）是串行推送的，它的 observe 可能把
        k 记成≈0，导致 cap 暴涨、把整段答案一次塞进一个巨批——巨批合成 20s+ 期间
        音频协程被 await 卡死，客户端缓冲耗光 → 长静音/断档。夹住下限后保证
        "批量合成耗时 ≤ 客户端剩余缓冲"，合成始终跑在播放前面。
        """
        if not self.max_chars or self.k is None or self.cps is None:
            return 0
        k = max(self.k, 0.5)
        usable = float(ahead_sec) - self.margin_sec
        if usable <= 0:
            return 0
        return min(self.max_chars, int(usable / k * self.cps))

    def take(self, backlog: List[str], ahead_sec: float, max_units: int = 1) -> List[str]:
        """从 backlog 头部按顺序取走最多 max_units 段。相邻小句能放进 char_cap 才拼在一起。

        没测过速度时 cap 为 0，不拼句，但仍然取满 max_units。Qwen 的单句调用
        墙钟盖不住这句音频，首轮若只送一句，句间空档就是整句合成时间。
        下一段更长也要收进同一批：批的墙钟等于最长那段，多带上短句不增加墙钟，
        音频却变长。split_marks 时只在书名号右界和标点处切开，不从字中间断开。
        """
        cap = self.char_cap(ahead_sec)
        limit = max(1, int(max_units))
        units: List[str] = []
        while backlog and len(units) < limit:
            text = backlog.pop(0)
            pieces = self._split_clause(text)
            if len(pieces) > 1:
                backlog[:0] = pieces[1:]
                text = pieces[0]
            while backlog and cap > 0 and len(text) + len(backlog[0]) <= cap:
                text += backlog.pop(0)
            units.append(text)
        return units
