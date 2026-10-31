# -*- coding: utf-8 -*-
"""Qwen 合成子进程。stdin 一行一个 JSON 任务，协议结果写回原始 stdout。

普通 print 改走 stderr，避免把加载日志写进协议管道。
"""
from __future__ import annotations

import os
import sys


def _bind_protocol():
    """协议走启动时的 stdout。之后 fd 1 改到 stderr。"""
    proto_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


def main() -> int:
    import base64
    import json
    from pathlib import Path
    from types import SimpleNamespace

    proto = _bind_protocol()
    os.environ["QWEN_TTS_WORKER"] = "1"
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    project = root.parent

    def emit(payload: dict) -> None:
        proto.write(json.dumps(payload, ensure_ascii=False) + "\n")
        proto.flush()

    try:
        from tts.qwen import activate, synthesize_pcm
    except Exception as e:
        emit({"ready": False, "error": "%s: %s" % (type(e).__name__, e)})
        return 1
    if not activate():
        from tts.qwen import last_error
        emit({"ready": False, "error": last_error() or "activate failed"})
        return 1
    emit({"ready": True, "lane": os.environ.get("QWEN_TTS_LANE") or ""})

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"id": "", "error": "bad json: %s" % e})
            continue
        job_id = str(job.get("id") or "")
        text = str(job.get("text") or "")
        wav = str(job.get("prompt_wav") or "")
        if wav and not Path(wav).is_file():
            alt = project / wav
            if alt.is_file():
                wav = str(alt)
        persona = SimpleNamespace(
            id=str(job.get("persona_id") or ""),
            tts=SimpleNamespace(
                prompt_wav=wav,
                prompt_text=str(job.get("prompt_text") or ""),
            ),
        )
        try:
            pcm, sr = synthesize_pcm(text, persona=persona)
        except Exception as e:
            emit({"id": job_id, "error": "%s: %s" % (type(e).__name__, e)})
            continue
        if not pcm:
            emit({"id": job_id, "error": "empty pcm"})
            continue
        emit({
            "id": job_id,
            "pcm": base64.b64encode(pcm).decode("ascii"),
            "sr": int(sr or 24000),
        })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
