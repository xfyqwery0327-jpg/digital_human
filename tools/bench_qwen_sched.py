# -*- coding: utf-8 -*-
"""单进程动态 batch 的离线压测。默认不加载模型，不连 8848/8850/8080。

统计：单路延迟、吞吐、batch 大小、原生 EOS、silence_guard_force_eos、静音重算。
码本计数来自脚本里喂进 session.guard 的假 token，用来核对埋点。
线上真实合成看 logs/tts-load.log 里的「qwen sched」行；当前模型接口数不到逐 token，
那些 force_eos 在真推理里会一直是 0，直到 main_code 能从 generate 外面拿到。

用法（仓库根目录）：
  D:\\anaconda\\envs\\digital_human_server\\python.exe tools\\bench_qwen_sched.py

真 GPU 不要用本脚本直接打模型。开服务时保持：
  QWEN_TTS_SCHED=batch
  QWEN_TTS_BATCH_MAX=3
  QWEN_TTS_BATCH_WAIT_MS=8
然后看 logs/tts-load.log。对比多进程时才设 QWEN_TTS_SCHED=procs。
"""
from __future__ import annotations

import array
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER9 = ROOT / "server9"
if str(SERVER9) not in sys.path:
    sys.path.insert(0, str(SERVER9))

from tts.qwen_batch import (  # noqa: E402
    QwenBatchScheduler,
    SchedStats,
    SilenceGuard,
)


class _Persona:
    id = "bench"

    class tts:
        prompt_wav = "bench.wav"
        prompt_text = "ref"


def _loud() -> bytes:
    return array.array("h", [3000] * 80).tobytes()


def _simulate_guard(book: SchedStats) -> None:
    """假 main_code。不是模型采样结果。"""
    native = SilenceGuard((1,), max_cont=3, stats_obj=book)
    native.feed([5, 5, 9], eos_id=9)
    forced = SilenceGuard((1,), max_cont=3, stats_obj=book)
    forced.feed([5, 1, 1, 1, 1], eos_id=9)
    other = SilenceGuard((1,), max_cont=3, stats_obj=book)
    assert other.count == 0


def main() -> None:
    book = SchedStats()
    calls = []

    def generate(batch, cancel_ev):
        del cancel_ev
        calls.append(len(batch))
        time.sleep(0.01)
        return [(_loud(), 24000) for _ in batch]

    sched = QwenBatchScheduler(
        generate_fn=generate,
        max_batch=3,
        wait_ms=30,
        silence_codes=(1,),
        max_cont=3,
        stats_obj=book,
        enable_cache=False,
    )
    sched.ensure_started()
    try:
        futs = [
            sched.submit("第%s句。" % i, "spk", _Persona(), None)
            for i in range(6)
        ]
        for fut in futs:
            fut.result(timeout=5)
    finally:
        sched.close()
    _simulate_guard(book)
    snap = book.snapshot()
    print("gpu=skipped simulated_tokens=yes")
    print("batches=%s" % calls)
    print("completed=%s p50_ms=%.1f p95_ms=%.1f tput=%.2f/s" % (
        snap["completed"], snap["p50_ms"], snap["p95_ms"], snap["throughput"]))
    print("batch_hist=%s native_eos=%s silence_guard_force_eos=%s pcm_quiet_retry=%s token_guard=%s" % (
        snap["batch_hist"], snap["native_eos"], snap["silence_guard_force_eos"],
        snap["pcm_quiet_retry"], "on" if snap["token_steps"] else "off"))
    if any(n > 3 for n in calls):
        raise SystemExit("batch exceeded 3")


if __name__ == "__main__":
    main()
