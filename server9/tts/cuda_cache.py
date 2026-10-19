# -*- coding: utf-8 -*-
"""把 TTS 用完的空闲 CUDA 缓存还给驱动。

一轮问答结束时调用。句与句之间只在整卡剩余不足 2GB 时才调用，
避免下一批还要同样大小的块时反复分配。
"""
from __future__ import annotations

_SPARE_BYTES = 64 * 1024 * 1024
_TIGHT_FREE_BYTES = 2 * 1024 * 1024 * 1024


def _log(msg: str) -> None:
    try:
        import tts.cosyvoice as cv
        cv._tts_log(msg)
    except Exception:
        print(msg)


def _reset_qwen_talker() -> None:
    try:
        from tts import qwen
    except Exception:
        return
    model = getattr(qwen, "_model", None)
    if model is None:
        return
    qwen._reset_talker_cache(model)


def release_cuda_cache(reason: str = "", tight: bool = False) -> None:
    """tight=False：任务结束，有空闲缓存就还。tight=True：只在整卡快满时还。"""
    import gc
    gc.collect()
    _reset_qwen_talker()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info()
    except Exception:
        free_b, total_b = 0, 0
    alloc = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    active = alloc
    try:
        active = int(torch.cuda.memory_stats().get("active_bytes.all.current", alloc))
    except Exception:
        pass
    spare = reserved - alloc
    nearly_full = 0 < free_b < _TIGHT_FREE_BYTES
    should = spare >= _SPARE_BYTES and (nearly_full if tight else True)
    freed = False
    if should:
        torch.cuda.empty_cache()
        freed = True
        gc.collect()
    alloc2 = int(torch.cuda.memory_allocated())
    reserved2 = int(torch.cuda.memory_reserved())
    active2 = alloc2
    try:
        active2 = int(torch.cuda.memory_stats().get("active_bytes.all.current", alloc2))
    except Exception:
        pass
    if tight and not nearly_full:
        return
    tag = "release" if freed else "hold"
    _log(
        "[TTS] cuda cache %s reason=%s tight=%s free=%.0f/%.0fMiB "
        "active=%.0f->%.0fMiB alloc=%.0f->%.0fMiB reserved=%.0f->%.0fMiB"
        % (
            tag,
            reason or "-",
            int(bool(tight)),
            free_b / 1048576.0,
            total_b / 1048576.0,
            active / 1048576.0, active2 / 1048576.0,
            alloc / 1048576.0, alloc2 / 1048576.0,
            reserved / 1048576.0, reserved2 / 1048576.0,
        )
    )
