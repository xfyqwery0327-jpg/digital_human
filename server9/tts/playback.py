# -*- coding: utf-8 -*-
"""本机播放。普通对话走浏览器喇叭；API 必须自己出声，且不能打到 Digital Output。"""
from __future__ import annotations
import io
import os
import sys
import time
import wave
import tempfile
from typing import List, Optional, Tuple
import queue
import threading

from config import ROBOT_PLAY_TAIL_SEC, TTS_SAMPLE_RATE

_DIGITAL = ("digital", "spdif", "hdmi", "optical", "display", "nvidia")
_MAPPER = ("mapper", "primary sound", "主声音")


class PcmPlayClock:
    """按前端排队播放估算收尾时间。

    合成是整句瞬间倒完（pace=False），末句 PCM 到达时前面的已经播完，
    末句自己的时长还没开始。用「总 PCM / 开讲墙钟」会算出 remain<=0，
    立刻 tts_end，Unity 把末句掐掉。
    """

    def __init__(self, sample_rate: int = 0):
        self.sr = float(sample_rate or TTS_SAMPLE_RATE or 24000)
        self.deadline = 0.0
        self.nbytes = 0

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.nbytes += len(chunk)
        now = time.time()
        if self.deadline < now:
            self.deadline = now
        self.deadline += len(chunk) / 2.0 / self.sr

    def remain(self, tail: float = None) -> float:
        extra = float(ROBOT_PLAY_TAIL_SEC if tail is None else tail) or 0.0
        return max(0.0, self.deadline + extra - time.time())

    @property
    def duration(self) -> float:
        if self.nbytes <= 0 or self.sr <= 0:
            return 0.0
        return self.nbytes / 2.0 / self.sr


def wav_duration_sec(wav_bytes: bytes) -> float:
    if not wav_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            sr = w.getframerate()
            if not sr:
                return 0.0
            return w.getnframes() / float(sr)
    except Exception as e:
        print(f"[ROBOT] 读取 WAV 时长失败: {e}")
        return 0.0


def _is_digital(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _DIGITAL)


def _is_mapper(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _MAPPER)


def _device_mark(name: str) -> str:
    if _is_digital(name):
        return " [数字输出，展厅听不到]"
    if _is_mapper(name):
        return " [系统默认映射，会指向数字口]"
    if "speaker" in (name or "").lower() or "扬声器" in (name or ""):
        return " [模拟音箱，将选用]"
    return ""


def wave_out_device_names() -> List[str]:
    names = _wasapi_output_names()
    if names:
        return names
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class WAVEOUTCAPS(ctypes.Structure):
            _fields_ = [
                ("wMid", wintypes.WORD),
                ("wPid", wintypes.WORD),
                ("vDriverVersion", wintypes.UINT),
                ("szPname", wintypes.WCHAR * 32),
                ("dwFormats", wintypes.DWORD),
                ("wChannels", wintypes.WORD),
                ("wReserved1", wintypes.WORD),
                ("dwSupport", wintypes.DWORD),
            ]

        winmm = ctypes.windll.winmm
        n = int(winmm.waveOutGetNumDevs())
        names = []
        caps = WAVEOUTCAPS()
        for i in range(n):
            if winmm.waveOutGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
                names.append(caps.szPname)
        return names
    except Exception as e:
        return [f"(枚举失败: {e})"]


def _wasapi_output_names() -> List[str]:
    try:
        import sounddevice as sd
        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                name = d.get("name") or ""
                out.append(f"{i}:{name}{_device_mark(name)}")
        return out
    except Exception:
        pass
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        out = []
        try:
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                if int(d.get("maxOutputChannels") or 0) > 0:
                    name = d.get("name") or ""
                    out.append(f"{i}:{name}{_device_mark(name)}")
        finally:
            p.terminate()
        return out
    except Exception:
        return []


def _wav_to_int16(wav_bytes: bytes) -> Tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    if sw != 2:
        raise RuntimeError(f"不支持的 sample width={sw}")
    import numpy as np
    audio = np.frombuffer(pcm, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch)
        audio = audio.mean(axis=1).astype(np.int16)
        ch = 1
    return audio.tobytes(), int(sr), ch


def _pick_analog_index(devices: List[Tuple[int, str]]) -> Tuple[Optional[int], str]:
    """优先 Speakers/扬声器，跳过 Digital 和 Sound Mapper（Mapper 仍指向默认数字口）。"""
    prefer = os.environ.get("ROBOT_PLAY_DEVICE", "speakers").strip().lower()
    scored = []
    for i, n in devices:
        if _is_digital(n) or _is_mapper(n):
            continue
        nl = n.lower()
        score = 10
        if prefer and prefer in nl:
            score = 100
        elif "speaker" in nl or "扬声器" in nl:
            score = 50
        elif "headphone" in nl or "headset" in nl or "耳机" in nl:
            score = 20
        scored.append((score, i, n))
    if not scored:
        return None, ""
    scored.sort(key=lambda x: (-x[0], x[1]))
    _, idx, name = scored[0]
    return idx, name


def _play_sounddevice(wav_bytes: bytes, duration: float) -> str:
    import numpy as np
    import sounddevice as sd
    pcm, sr, ch = _wav_to_int16(wav_bytes)
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 1:
        audio = audio.reshape(-1, 1)
    cands = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0:
            cands.append((i, d.get("name") or ""))
    idx, name = _pick_analog_index(cands)
    if idx is None:
        raise RuntimeError("没有可用的模拟输出设备")
    print(f"[ROBOT] WASAPI/sounddevice 播放 device={idx} {name}")
    sd.play(audio, sr, device=idx, blocking=True)
    return name


def _play_pyaudio(wav_bytes: bytes, duration: float) -> str:
    import pyaudio
    pcm, sr, ch = _wav_to_int16(wav_bytes)
    p = pyaudio.PyAudio()
    try:
        cands = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if int(d.get("maxOutputChannels") or 0) > 0:
                cands.append((i, d.get("name") or ""))
        idx, name = _pick_analog_index(cands)
        if idx is None:
            raise RuntimeError("没有可用的模拟输出设备")
        print(f"[ROBOT] PyAudio 播放 device={idx} {name}")
        stream = p.open(
            format=pyaudio.paInt16,
            channels=ch,
            rate=sr,
            output=True,
            output_device_index=idx,
        )
        try:
            stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
        return name
    finally:
        p.terminate()


def play_wav_until_done(wav_bytes: bytes) -> Tuple[bool, float, str]:
    """播放到 Speakers 模拟口，播完再返回。不落盘。
    返回 (played_ok, duration_sec, device_name)。"""
    duration = wav_duration_sec(wav_bytes)
    if duration <= 0:
        return False, 0.0, ""
    wait_total = duration + max(0.0, ROBOT_PLAY_TAIL_SEC)
    t0 = time.time()
    played = False
    used = ""
    devices = wave_out_device_names()
    print(f"[ROBOT] 播放设备列表: {devices or '(未枚举到)'}")
    print(f"[ROBOT] 开始播放 duration={duration:.2f}s，将优先选 Speakers 而不是 Sound Mapper/Digital")

    for name, fn in (
        ("sounddevice", _play_sounddevice),
        ("pyaudio", _play_pyaudio),
    ):
        try:
            used = fn(wav_bytes, duration) or ""
            played = True
            print(f"[ROBOT] {name} 播放完成 device={used}")
            break
        except Exception as e:
            print(f"[ROBOT] {name} 不可用: {e}")

    if not played:
        print("[ROBOT] 没有 WASAPI 播放库，winsound 会打到系统默认数字口，展厅听不到")
        tmp_path = None
        try:
            if sys.platform == "win32":
                import winsound
                fd, tmp_path = tempfile.mkstemp(prefix="monet_tts_", suffix=".wav")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    f.write(wav_bytes)
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                played = True
                used = "winsound-default"
        except Exception as e:
            print(f"[ROBOT] winsound 也失败: {e}")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    remain = wait_total - (time.time() - t0)
    if remain > 0.05:
        time.sleep(remain)
    print(f"[ROBOT] 本机播放结束 played={played} device={used} 实际 {time.time() - t0:.2f}s")
    return played, duration, used


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> bytes:
    if not pcm:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


class AnalogPcmPlayer:
    """边收 PCM 边打到 Speakers，不走 Sound Mapper / Digital Output。"""

    def __init__(self):
        self.device_name = ""
        self.ok = False
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
        self._th: Optional[threading.Thread] = None
        self._err = None
        self._sr = 24000
        self._ch = 1
        self._idx = None
        self._ready = threading.Event()

    def start(self, sample_rate: int, channels: int = 1) -> bool:
        try:
            import sounddevice as sd
        except Exception as e:
            print(f"[ROBOT] AnalogPcmPlayer sounddevice 不可用: {e}")
            return False
        cands = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                cands.append((i, d.get("name") or ""))
        idx, name = _pick_analog_index(cands)
        if idx is None:
            print("[ROBOT] AnalogPcmPlayer 没有可用模拟输出")
            return False
        self.device_name = name
        self._sr = sample_rate
        self._ch = channels
        self._idx = idx
        self.ok = True
        self._ready.clear()
        self._th = threading.Thread(target=self._run, name="robot-pcm-play", daemon=True)
        self._th.start()
        if not self._ready.wait(timeout=5):
            print("[ROBOT] AnalogPcmPlayer 打开声卡超时")
            self.ok = False
            return False
        if not self.ok:
            return False
        print(f"[ROBOT] 模拟口实时播放 device={idx} {name} sr={sample_rate}")
        return True

    def feed(self, pcm: bytes):
        """禁止阻塞事件循环：队列满则丢最旧块。"""
        if not self.ok or not pcm:
            return
        try:
            self._q.put_nowait(pcm)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(pcm)
            except queue.Full:
                pass

    def finish(self, timeout: float = 90) -> bool:
        if not self.ok:
            return False
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(None)
            except Exception:
                pass
        wait = min(max(5.0, timeout), 90.0)
        if self._th:
            self._th.join(timeout=wait)
        if ROBOT_PLAY_TAIL_SEC > 0:
            time.sleep(ROBOT_PLAY_TAIL_SEC)
        return self._err is None

    def _run(self):
        import numpy as np
        try:
            import sounddevice as sd
            stream = sd.OutputStream(
                samplerate=self._sr, channels=self._ch,
                dtype="int16", device=self._idx)
            stream.start()
            self._ready.set()
            try:
                while True:
                    item = self._q.get()
                    if item is None:
                        break
                    audio = np.frombuffer(item, dtype=np.int16)
                    if self._ch == 1:
                        audio = audio.reshape(-1, 1)
                    stream.write(audio)
            finally:
                stream.stop()
                stream.close()
        except Exception as e:
            self._err = e
            self.ok = False
            print(f"[ROBOT] AnalogPcmPlayer 失败: {e}")
            self._ready.set()

