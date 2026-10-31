# -*- coding: utf-8 -*-
"""llama.cpp 的 text/event-stream 不带 charset 时，中文不能被当成 Latin-1。"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from requests import Response  # noqa: E402
from requests.structures import CaseInsensitiveDict  # noqa: E402
from requests.utils import get_encoding_from_headers  # noqa: E402

from llm.client import iter_llm_sse  # noqa: E402

ANSWER = "我是法国的印象派画家，以光影变化和色彩细腻著称。"


def _sse_body(text: str) -> bytes:
    payload = json.dumps(
        {"choices": [{"delta": {"content": text}}]},
        ensure_ascii=False,
    )
    return ("data: " + payload + "\n\n").encode("utf-8")


def _response(body: bytes) -> Response:
    resp = Response()
    resp.headers = CaseInsensitiveDict({"Content-Type": "text/event-stream"})
    resp.encoding = get_encoding_from_headers(resp.headers)
    resp.raw = BytesIO(body)
    return resp


def test_event_stream_header_is_latin1():
    enc = get_encoding_from_headers(
        CaseInsensitiveDict({"Content-Type": "text/event-stream"}))
    assert enc.lower() in ("iso-8859-1", "latin-1")
    json_enc = get_encoding_from_headers(
        CaseInsensitiveDict({"Content-Type": "application/json"}))
    assert json_enc.lower() == "utf-8"


def test_latin1_sse_breaks_chinese_and_utf8_reader_keeps_it():
    body = _sse_body(ANSWER)
    resp = _response(body)
    broken = body.decode(resp.encoding)
    assert ANSWER not in broken
    # UTF-8 多字节被拆成 C1 控制符时，字幕字体显示成菱形/问号。
    assert any(0x80 <= ord(ch) <= 0x9F for ch in broken)

    line = next(line for line in iter_llm_sse(resp) if line.startswith("data: "))
    got = json.loads(line[6:])["choices"][0]["delta"]["content"]
    assert got == ANSWER

    wire = json.dumps(
        {"type": "llm_answer", "text": got}, ensure_ascii=False).encode("utf-8")
    assert ANSWER.encode("utf-8") in wire
    start = json.dumps(
        {"type": "tts_start", "text": got}, ensure_ascii=False).encode("utf-8")
    assert ANSWER.encode("utf-8") in start


if __name__ == "__main__":
    test_event_stream_header_is_latin1()
    test_latin1_sse_breaks_chinese_and_utf8_reader_keeps_it()
    print("ok")
