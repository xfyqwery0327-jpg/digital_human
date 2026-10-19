# -*- coding: utf-8 -*-
"""启动入口：独立 FastAPI TTS 进程。

和机器人 LLM 拆开。本进程常驻 GPU，只在空闲、并且显存超过阈值时退出，
由 supervisor.py 拉起新进程。不要在这个进程里加载讲解模型。
"""
from __future__ import annotations

import os
import sys
import threading
import time

# 必须在 import torch 之前。expandable_segments 减轻句间碎片，不把权重搬到 CPU。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from tts_cosy3.engine import TTSEngine, log, pcm_to_wav_bytes, setup_log_file, wav_stream_header
from tts_cosy3.split_text import clamp_max_chars
from tts_cosy3.vram import (
    clear_restart_request,
    format_snapshot,
    idle_sec_required,
    over_limit,
    read_state,
    request_restart,
    restart_requested,
    snapshot,
    vram_limit_mb,
    write_state,
)

ENGINE = TTSEngine()
_STOP = threading.Event()


def _state_loop() -> None:
    """把忙闲和显存写给监督进程。空闲且被要求重启时退出，不打断正在说的那一句。"""
    while not _STOP.is_set():
        try:
            ENGINE.release_if_idle()
        except Exception as exc:
            log(f"[TTS] 空闲释放显存失败: {type(exc).__name__}: {exc}")
        snap = snapshot()
        write_state({
            "pid": os.getpid(),
            "ready": ENGINE.ready,
            "busy": ENGINE.busy,
            "last_request_end": ENGINE.last_request_end,
            "observed_mb": snap.get("observed_mb"),
            "weight_mb": ENGINE.weight_mb,
            "speakers": list(ENGINE.speakers),
            "ts": time.time(),
        })
        idle = time.time() - float(ENGINE.last_request_end or 0)
        supervised = os.environ.get("TTS_SUPERVISED", "") == "1"
        if (
            ENGINE.ready
            and not ENGINE.busy
            and restart_requested()
            and idle >= 2.0
        ):
            log("[TTS] 空闲，按重启请求退出，监督进程会拉起新进程")
            _STOP.set()
            os._exit(75)
        if (
            ENGINE.ready
            and not ENGINE.busy
            and over_limit(snap)
            and idle >= idle_sec_required()
        ):
            log(f"[TTS] 本进程显存 {snap.get('observed_mb')}MiB 超过 {vram_limit_mb()}MiB，标记空闲重启")
            request_restart("vram")
            if not supervised:
                log("[TTS] 无监督进程，空闲后原地重启")
                os.execv(sys.executable, [sys.executable, "-m", "tts_cosy3.tts_cosy3_server"])
        _STOP.wait(2.0)


def create_app():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from pydantic import BaseModel, Field

    app = FastAPI(title="CosyVoice3 TTS", version="1.0")

    class TTSRequest(BaseModel):
        text: str = Field(..., description="要合成的口播文本")
        speaker_id: str = Field("monet", description="已登记的音色")
        format: str = Field("pcm", description="pcm 或 wav")
        stream: bool = True
        max_chars: int = 36

    class ClonePathRequest(BaseModel):
        speaker_id: str
        prompt_wav: str
        prompt_text: str = ""

    @app.on_event("startup")
    def _startup():
        setup_log_file()
        clear_restart_request()
        log(
            f"[TTS] 启动 pid={os.getpid()} dtype={os.environ.get('TTS_DTYPE', 'fp16')} "
            f"int8={os.environ.get('TTS_INT8', '0')} trt={os.environ.get('COSYVOICE_LOAD_TRT', '0')} "
            f"limit={vram_limit_mb()}MiB"
        )
        try:
            ENGINE.load()
        except Exception as exc:
            log(f"[TTS] 加载失败: {type(exc).__name__}: {exc}")
            raise
        threading.Thread(target=_state_loop, name="tts-vram-watch", daemon=True).start()

    @app.get("/health")
    def health():
        snap = snapshot()
        return {
            "ok": ENGINE.ready,
            "busy": ENGINE.busy,
            "sample_rate": ENGINE.sample_rate,
            "speakers": list(ENGINE.speakers),
            "dtype": ENGINE.dtype_name,
            "int8": ENGINE.int8,
            "weight_mb": round(ENGINE.weight_mb, 1),
            "vram_limit_mb": vram_limit_mb(),
            "vram": snap,
        }

    @app.get("/v1/vram")
    def vram():
        snap = snapshot()
        log(format_snapshot("probe", snap))
        return snap

    @app.get("/v1/speakers")
    def speakers():
        return {"speakers": ENGINE.speakers}

    @app.post("/v1/speakers/clone")
    def clone_json(body: ClonePathRequest):
        if not ENGINE.ready:
            raise HTTPException(503, "模型还在加载")
        try:
            ENGINE.register_speaker(body.speaker_id, body.prompt_wav, body.prompt_text)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            log(f"[TTS] clone 失败: {type(exc).__name__}: {exc}")
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        return {"ok": True, "speaker_id": body.speaker_id}

    @app.post("/v1/speakers/clone-file")
    async def clone_file(
        speaker_id: str = Form(...),
        prompt_text: str = Form(""),
        prompt_wav: UploadFile = File(...),
    ):
        if not ENGINE.ready:
            raise HTTPException(503, "模型还在加载")
        from tts_cosy3.paths import speaker_cache_dir
        suffix = os.path.splitext(prompt_wav.filename or "prompt.wav")[1] or ".wav"
        dest = speaker_cache_dir() / f"{speaker_id}_upload{suffix}"
        data = await prompt_wav.read()
        if not data:
            raise HTTPException(400, "参考音为空")
        dest.write_bytes(data)
        try:
            ENGINE.register_speaker(speaker_id, str(dest), prompt_text)
        except Exception as exc:
            log(f"[TTS] clone-file 失败: {type(exc).__name__}: {exc}")
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        return {"ok": True, "speaker_id": speaker_id}

    def _pcm_stream(body: TTSRequest):
        max_chars = clamp_max_chars(body.max_chars)
        try:
            yield from ENGINE.synthesize(body.text, body.speaker_id, max_chars=max_chars)
        except KeyError as exc:
            log(f"[TTS] 音色不存在: {exc}")
            raise
        except Exception as exc:
            log(f"[TTS] 合成失败: {type(exc).__name__}: {exc}")
            raise

    @app.post("/v1/tts")
    def tts(body: TTSRequest):
        if not ENGINE.ready:
            raise HTTPException(503, "模型还在加载")
        text = (body.text or "").strip()
        if not text:
            raise HTTPException(400, "text 为空")
        fmt = (body.format or "pcm").lower()
        if fmt not in {"pcm", "wav"}:
            raise HTTPException(400, "format 只能是 pcm 或 wav")
        if body.speaker_id not in ENGINE.speakers:
            raise HTTPException(404, f"未知音色 {body.speaker_id}")
        headers = {
            "X-Sample-Rate": str(ENGINE.sample_rate),
            "X-Channels": "1",
            "X-Bit-Depth": "16",
            "X-Speaker": body.speaker_id,
        }
        if body.stream and fmt == "pcm":
            return StreamingResponse(_pcm_stream(body), media_type="audio/L16", headers=headers)
        if body.stream and fmt == "wav":
            def _wav():
                yield wav_stream_header(ENGINE.sample_rate)
                yield from _pcm_stream(body)
            return StreamingResponse(_wav(), media_type="audio/wav", headers=headers)
        try:
            pcm = b"".join(_pcm_stream(body))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
        if fmt == "wav":
            return Response(pcm_to_wav_bytes(pcm, ENGINE.sample_rate), media_type="audio/wav", headers=headers)
        return Response(pcm, media_type="audio/L16", headers=headers)

    @app.post("/admin/restart-when-idle")
    def restart_when_idle():
        request_restart("admin")
        return {"ok": True, "busy": ENGINE.busy, "note": "当前句说完且空闲后退出"}

    @app.exception_handler(KeyError)
    async def _key_error(_, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app


def main() -> None:
    import uvicorn
    host = os.environ.get("TTS_HOST", "127.0.0.1")
    port = int(os.environ.get("TTS_PORT", "8810"))
    # workers 必须是 1。多 worker 会把模型加载好几份，12GB 上和 LLM 共存会直接挤死。
    uvicorn.run(
        create_app(),
        host=host,
        port=port,
        workers=1,
        log_level=os.environ.get("TTS_LOG_LEVEL", "info"),
        timeout_keep_alive=30,
    )


if __name__ == "__main__":
    main()
