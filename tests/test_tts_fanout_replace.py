# -*- coding: utf-8 -*-
"""同一台机器重连 /ws/{persona}/tts 时，旧连接必须关掉，避免 ask 叠着播。"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from ws.hub import TtsFanout  # noqa: E402


class _Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port


class _Ws:
    def __init__(self, host: str, port: int):
        self.client = _Client(host, port)
        self.closed = False

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.code = code
        self.reason = reason

    async def send_json(self, obj):
        self.sent = obj


def test_same_host_reconnect_replaces_old_tts():
    fanout = TtsFanout()
    first = _Ws("192.168.1.104", 64758)
    second = _Ws("192.168.1.104", 65488)
    other = _Ws("10.0.7.233", 52091)
    fuhao = _Ws("192.168.1.104", 65235)
    old_turn = threading.Event()

    async def run():
        await fanout.register(first, asyncio.Lock(), {"old": old_turn}, persona_id="monet")
        await fanout.register(other, asyncio.Lock(), {}, persona_id="monet")
        await fanout.register(fuhao, asyncio.Lock(), {}, persona_id="fuhao")
        await fanout.register(second, asyncio.Lock(), {}, persona_id="monet")
        n = await fanout.send({"type": "tts_start", "text": "我一生追逐光与瞬间。"}, "monet")
        assert n == 2

    asyncio.run(run())
    assert old_turn.is_set()

    assert first.closed is True
    assert second.closed is False
    assert other.closed is False
    assert fuhao.closed is False
    assert fanout.client_count("monet") == 2
    assert fanout.client_count("fuhao") == 1
    assert fanout.client_count() == 3
