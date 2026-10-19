# -*- coding: utf-8 -*-
"""显存管理。

CosyVoice3 的 KV 不是模块上的 buffer，而是 Qwen2LM.inference_wrapper 里的局部变量
`cache`（transformers 的 DynamicCache / past_key_values）。官方 CosyVoice3Model
没有 reset_kv_cache。这里把方法绑到模型上，并在每句结束后丢掉仍被引用的缓存。

调用时机：一句的 generator 已经耗尽或 close 之后。句中调用会拆掉正在用的
hift_cache_dict，下一跳直接 KeyError 或爆音。
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from typing import Any

from tts_cosy3.paths import restart_request_path, state_path

_CREATE_NO_WINDOW = 0x08000000
_SESSION_DICTS = (
    "tts_speech_token_dict",
    "llm_end_dict",
    "mel_overlap_dict",
    "flow_cache_dict",
    "hift_cache_dict",
)


def vram_limit_mb() -> int:
    raw = os.environ.get("TTS_VRAM_LIMIT_MB", "4096")
    try:
        return max(512, int(raw))
    except ValueError:
        return 4096


def idle_sec_required() -> float:
    raw = os.environ.get("TTS_IDLE_SEC", "10")
    try:
        return max(2.0, float(raw))
    except ValueError:
        return 10.0


def stuck_sec() -> float:
    """单句卡死超过这个时间，监督进程可以在空档之外强制重启。正常短句远小于它。"""
    raw = os.environ.get("TTS_STUCK_SEC", "180")
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 180.0


def _smi_lines(query: str) -> list:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}" if query.startswith("memory") else f"--query-compute-apps={query}",
        "--format=csv,noheader,nounits",
    ]
    # query-gpu 和 query-compute-apps 不能混在一条。调用方分开传。
    try:
        flags = _CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def query_gpu() -> dict:
    """整卡占用。4070 上这里包含旁边的 LLM 进程。"""
    lines = []
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        flags = _CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=4, check=False, creationflags=flags,
        )
        lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    except (OSError, subprocess.TimeoutExpired):
        lines = []
    used = total = free = -1
    if lines:
        parts = [p.strip() for p in lines[0].split(",")]
        try:
            used, total, free = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])))
        except (ValueError, IndexError):
            pass
    return {"gpu_used_mb": used, "gpu_total_mb": total, "gpu_free_mb": free}


def query_process_mb(pid: int) -> int:
    """nvidia-smi 里这个 PID 的显存。看不到时返回 -1，改由 torch reserved 兜底。"""
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        flags = _CREATE_NO_WINDOW if os.name == "nt" else 0
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=4, check=False, creationflags=flags,
        )
        text = proc.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return -1
    total = 0
    seen = False
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
            mem = int(float(parts[1]))
        except ValueError:
            continue
        if row_pid == int(pid):
            seen = True
            total += mem
    return total if seen else -1


def torch_reserved_mb() -> int:
    try:
        import torch
    except ImportError:
        return -1
    if not torch.cuda.is_available():
        return -1
    try:
        return int(torch.cuda.memory_reserved(0) / (1024 * 1024))
    except Exception:
        return -1


def torch_allocated_mb() -> int:
    try:
        import torch
    except ImportError:
        return -1
    if not torch.cuda.is_available():
        return -1
    try:
        return int(torch.cuda.memory_allocated(0) / (1024 * 1024))
    except Exception:
        return -1


def snapshot(pid: int = None) -> dict:
    pid = os.getpid() if pid is None else int(pid)
    gpu = query_gpu()
    proc = query_process_mb(pid)
    reserved = torch_reserved_mb()
    allocated = torch_allocated_mb()
    # 阈值看本进程。nvidia-smi 有采样延迟，取它和 torch reserved 里更大的那个。
    observed = proc
    if reserved >= 0 and (observed < 0 or reserved > observed):
        observed = reserved
    return {
        "pid": pid,
        "proc_mb": proc,
        "reserved_mb": reserved,
        "allocated_mb": allocated,
        "observed_mb": observed,
        **gpu,
    }


def format_snapshot(phase: str, snap: dict) -> str:
    return (
        f"[VRAM] phase={phase} pid={snap.get('pid')} "
        f"smi_proc={snap.get('proc_mb')}MiB reserved={snap.get('reserved_mb')}MiB "
        f"allocated={snap.get('allocated_mb')}MiB "
        f"gpu_used={snap.get('gpu_used_mb')}MiB gpu_free={snap.get('gpu_free_mb')}MiB "
        f"gpu_total={snap.get('gpu_total_mb')}MiB"
    )


def _clear_cache_attr(module) -> None:
    for attr in ("past_key_values", "_past_key_values", "kv_cache"):
        if not hasattr(module, attr):
            continue
        try:
            setattr(module, attr, None)
        except Exception:
            pass
    cache = getattr(module, "cache", None)
    if cache is None:
        return
    name = type(cache).__name__
    if "Cache" in name:
        try:
            setattr(module, "cache", None)
        except Exception:
            pass


def _walk(module, visit, seen: set) -> None:
    if module is None or id(module) in seen:
        return
    seen.add(id(module))
    visit(module)
    children = getattr(module, "children", None)
    if not callable(children):
        return
    try:
        kids = list(children())
    except Exception:
        return
    for kid in kids:
        _walk(kid, visit, seen)


def reset_kv_cache(model) -> None:
    """绑到 CosyVoice3Model 上的方法。一句结束之后调用。

    做的事：
    - 清掉 tts / flow / hift 的会话字典（CosyVoice3 的 hift mel 是整段拼接，不清会涨）
    - 把流式改过的 token_hop_len 拉回 25
    - 丢掉模块上残留的 DynamicCache / past_key_values
    不做的事：不释放仍被未关闭 generator 引用的局部 cache。调用前必须先 close。
    """
    if model is None:
        return
    for name in _SESSION_DICTS:
        bucket = getattr(model, name, None)
        if isinstance(bucket, dict):
            bucket.clear()
    if hasattr(model, "token_hop_len"):
        try:
            model.token_hop_len = 25
        except Exception:
            pass
    llm = getattr(model, "llm", None)
    queue = getattr(llm, "vllm_output_queue", None)
    if isinstance(queue, dict):
        queue.clear()
    _walk(model, _clear_cache_attr, set())
    _walk(llm, _clear_cache_attr, set())


def install_reset_kv_cache(model) -> None:
    """官方类没有这个方法。绑上之后，业务代码才能写 model.reset_kv_cache()。"""
    import types
    model.reset_kv_cache = types.MethodType(reset_kv_cache, model)


def release_cuda(reason: str = "") -> None:
    """先断引用，再 collect，再把缓存分配器还给驱动。空引用时 empty_cache 几乎无效。"""
    del reason
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()


def finish_sentence(model) -> None:
    """一句 TTS 的唯一收尾。顺序不能反：先 reset，再 gc，再 empty_cache。"""
    if model is not None and hasattr(model, "reset_kv_cache"):
        model.reset_kv_cache()
    else:
        reset_kv_cache(model)
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def write_state(payload: dict) -> None:
    path = state_path()
    tmp = path.with_suffix(".json.tmp")
    data = json.dumps(payload, ensure_ascii=False)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def read_state() -> dict:
    path = state_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def request_restart(reason: str) -> None:
    path = restart_request_path()
    path.write_text(f"{time.time():.3f} {reason}\n", encoding="utf-8")


def restart_requested() -> bool:
    return restart_request_path().is_file()


def clear_restart_request() -> None:
    path = restart_request_path()
    try:
        path.unlink()
    except OSError:
        pass


def over_limit(snap: dict) -> bool:
    observed = int(snap.get("observed_mb") or -1)
    return observed >= vram_limit_mb()
