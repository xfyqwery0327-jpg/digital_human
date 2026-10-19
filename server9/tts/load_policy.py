# -*- coding: utf-8 -*-
"""TTS 加载决策（无 GPU / 无 CosyVoice 依赖，便于单测）。

线上合成走 embedding-only。探测只负责注册音色并缓存向量；
inference_zero_shot 失败、warmup 失败、Tensor 不能当布尔值，都不得把已加载模型卸掉。
"""
from __future__ import annotations


def first_tensor(*vals):
    """取第一个非 None。禁止对 Tensor 做 `or` / `if tensor`。"""
    for v in vals:
        if v is not None:
            return v
    return None


def register_succeeded(value) -> bool:
    """add_zero_shot_spk 成功：不是 False / None。返回 Tensor 也算成功。"""
    return value is not False and value is not None


def seed_speaker_mode(*, register_result, extract_ok: bool) -> str:
    """先 add_zero_shot_spk，失败再 embedding-only 抽取。不跑 zero-shot 合成。"""
    if register_succeeded(register_result):
        return "path"
    if extract_ok:
        return "embedding"
    return ""


def resolve_prompt_mode(registered_mode: str, model_loaded: bool) -> str:
    """模型已起来时不得停留 unknown。"""
    if registered_mode in ("path", "tensor", "embedding"):
        return registered_mode
    if model_loaded:
        return "embedding"
    return "unknown"


def tts_health_ready(model, emb_cache) -> bool:
    """/health tts：CosyVoice3() 成功且至少缓存了一个人设 embedding。"""
    return model is not None and bool(emb_cache)


def apply_probe_decision(model, registered_mode: str, emb_cache, warmup_error=None):
    """CosyVoice3() 已经成功之后：探测/预热失败也保留模型。"""
    keep = model is not None
    return {
        "keep_model": keep,
        "unload": False,
        "prompt_mode": resolve_prompt_mode(registered_mode, keep),
        "health_tts": tts_health_ready(model if keep else None, emb_cache),
        "warmup_error": warmup_error or "",
    }
