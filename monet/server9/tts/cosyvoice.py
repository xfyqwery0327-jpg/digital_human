# -*- coding: utf-8 -*-
"""CosyVoice 加载、探测、合成。"""
from __future__ import annotations
import os
import sys
import time
import io
import base64
import hashlib
import threading
import wave
import tempfile
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
from scipy.io import wavfile

from config import *

cosyvoice_model = None
tts_executor = None
audio_cache = None
_tts_prompt_arg = None
_tts_prompt_mode = "unknown"
_tts_support_stream = True
_tts_use_spk_id = False
_infer_sem = threading.Semaphore(1)

class AudioCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size
        self._lock = threading.Lock()

    def _get_key(self, text: str, speaker: str, fmt: str = "pcm") -> str:
        return hashlib.md5(f"{fmt}|{text}|{speaker}".encode()).hexdigest()

    def get(self, text: str, speaker: str, fmt: str = "pcm"):
        key = self._get_key(text, speaker, fmt)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, text: str, speaker: str, audio_data, fmt: str = "pcm"):
        key = self._get_key(text, speaker, fmt)
        with self._lock:
            self.cache[key] = audio_data
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

def _tts_infer(text: str, block: bool = True,
               cancel_ev: Optional[threading.Event] = None):
    """统一推理入口。
    block=True + cancel_ev → 可中断阻塞：每 50ms 检查一次 cancel_ev，
                             取消时返回 None 而不是永久挂起
    block=True             → 普通阻塞（/ws/chat、POST /tts）
    block=False            → GPU 忙时立即返回 None（已无调用方使用此路径）
    """
    if block:
        if cancel_ev is not None:
            acquired = False
            while not cancel_ev.is_set():
                acquired = _infer_sem.acquire(blocking=True, timeout=0.05)
                if acquired:
                    break
            if not acquired:   # 被取消，未拿到锁
                return None
        else:
            _infer_sem.acquire(blocking=True)
    else:
        if not _infer_sem.acquire(blocking=False):
            return None

    if cosyvoice_model is None:
        _infer_sem.release()
        raise RuntimeError("CosyVoice 未加载")

    kwargs = {"stream": True} if _tts_support_stream else {}
    try:
        if _tts_use_spk_id:
            raw_gen = cosyvoice_model.inference_zero_shot(
                text, '', '', zero_shot_spk_id=TTS_SPK_ID, speed=1.2, **kwargs)  # 👈 加这里
        elif _tts_prompt_arg is None:
            raise RuntimeError("参考音频未就绪")
        else:
            raw_gen = cosyvoice_model.inference_zero_shot(
                text, PROMPT_TEXT, _tts_prompt_arg, speed=1.2, **kwargs)  # 👈 加这里
    except Exception:
        _infer_sem.release()
        raise

    def _guarded():
        try:
            yield from raw_gen
        finally:
            _infer_sem.release()

    return _guarded()


def load_cosyvoice():
    global cosyvoice_model, tts_executor, audio_cache
    global _tts_prompt_arg, _tts_prompt_mode, _tts_support_stream, _tts_use_spk_id

    import torch

    if not ENABLE_TTS:
        print("⚠️ TTS 已禁用")
        return
    try:
        print("🔍 检查 CosyVoice 依赖...")
        try:
            import hyperpyyaml  # noqa: F401
            print("   ✅ hyperpyyaml 已安装")
        except ImportError:
            print("   ❌ hyperpyyaml 未安装")
            print("\n📦 解决方案:\n   pip install hyperpyyaml onnxruntime-gpu modelscope\n")
            return

        cosyvoice_path = Path(COSYVOICE_PATH)
        model_path = Path(COSYVOICE_MODEL)
        if not cosyvoice_path.exists():
            print(f"   ❌ CosyVoice 路径不存在: {COSYVOICE_PATH}")
            return
        if not model_path.exists():
            print(f"   ❌ 模型路径不存在: {COSYVOICE_MODEL}")
            return
        if not Path(PROMPT_WAV_PATH).exists():
            print(f"   ❌ 参考音频不存在: {PROMPT_WAV_PATH}")
            return

        required_files = ["cosyvoice2.yaml", "flow.pt", "hift.pt", "llm.pt",
                          "campplus.onnx", "speech_tokenizer_v2.onnx"]
        missing = [f for f in required_files if not (model_path / f).exists()]
        if missing:
            print(f"❌ 缺少模型文件: {missing}")
            return
        print("   ✅ 路径与模型文件齐全")

        sys.path.insert(0, COSYVOICE_PATH)
        sys.path.insert(0, str(Path(COSYVOICE_PATH) / "third_party" / "Matcha-TTS"))

        # 清理旧模块缓存
        for module_name in [k for k in sys.modules if 'cosyvoice' in k.lower()]:
            del sys.modules[module_name]

        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        print("✅ CosyVoice 模块导入成功")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️ CUDA 不可用，将使用 CPU（会很慢）")

        import _locale
        _locale._getdefaultlocale = lambda *args: ['en_US', 'utf8']

        print("📦 加载 CosyVoice2 模型...")
        import os as _os
        _devnull = open(_os.devnull, 'w', encoding='utf-8', errors='ignore')
        _old_stdout = sys.stdout
        sys.stdout = _devnull
        try:
            cosyvoice_model = CosyVoice2(COSYVOICE_MODEL,load_jit=False,fp16=True)
        finally:
            sys.stdout = _old_stdout
            _devnull.close()
        print("✅ 模型加载完成")

        if device == 'cuda':
            print("🚀 将 PyTorch 模块移至 GPU...")
            success_count = 0
            for module_name in ['llm', 'flow', 'hift']:
                if hasattr(cosyvoice_model.model, module_name):
                    module = getattr(cosyvoice_model.model, module_name)
                    if module is not None and isinstance(module, torch.nn.Module):
                        try:
                            module.to(device)
                            param = next(module.parameters(), None)
                            if param is not None and param.device.type == 'cuda':
                                success_count += 1
                                print(f"{module_name}: device={param.device}, dtype={param.dtype}")

                        except Exception as e:
                            print(f"{module_name} 移动失败: {e}")
                if hasattr(cosyvoice_model.model, 'fp16'):
                   cosyvoice_model.model.fp16 = True
                if hasattr(cosyvoice_model.model, 'device'):
                   cosyvoice_model.model.device = torch.device(device)
            print(f"{'所有' if success_count == 3 else f'⚠️ 仅 {success_count}/3 个'} PyTorch 模块在 GPU 上")

            import onnxruntime as ort
            cuda_provider = ("CUDAExecutionProvider", {"device_id": 0})
            fname_map = {'campplus_session': 'campplus.onnx',
                         'speech_tokenizer_session': 'speech_tokenizer_v2.onnx'}
            if hasattr(cosyvoice_model, 'frontend'):
                fe = cosyvoice_model.frontend
                for attr, fname in fname_map.items():
                    sess = getattr(fe, attr, None)
                    if sess is None:
                        continue
                    if 'CUDAExecutionProvider' in sess.get_providers():
                        continue
                    onnx_path = getattr(sess, '_model_path', None)
                    if not onnx_path:
                        candidate = Path(COSYVOICE_MODEL) / fname
                        onnx_path = str(candidate) if candidate.exists() else None
                    if onnx_path:
                        try:
                            new_sess = ort.InferenceSession(
                                onnx_path, providers=[cuda_provider, "CPUExecutionProvider"])
                            setattr(fe, attr, new_sess)
                            print(f"   ✅ {attr} → CUDAExecutionProvider")
                        except Exception as e:
                            print(f"   ⚠️ {attr} ONNX GPU 迁移失败: {e}")

        # ============ v8.4.1: 探测参考音频传参方式 ============
        print("📦 探测参考音频传参方式...")

        candidates = [("path", PROMPT_WAV_PATH)]
        try:
            candidates.append(("tensor", load_wav(PROMPT_WAV_PATH, 16000)))
        except Exception as e:
            print(f"   ⚠️ load_wav 失败，仅尝试路径模式: {e}")

        def _silent_infer(*args, **kwargs):
            """静默跑一小段合成，返回 (chunks, err)"""
            chunks = []
            _dn = open(_os.devnull, 'w', encoding='utf-8', errors='ignore')
            _so = sys.stdout
            sys.stdout = _dn
            try:
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=(device == 'cuda'),
                                            dtype=torch.float16):
                        for out in cosyvoice_model.inference_zero_shot(*args, **kwargs):
                            chunks.append(out['tts_speech'] if isinstance(out, dict) else out)
                return (chunks or None), None
            except Exception as e:
                return None, e
            finally:
                sys.stdout = _so
                _dn.close()

        probe_text = '测试。'
        for mode, arg in candidates:
            for use_stream in (True, False):
                kw = {"stream": True} if use_stream else {}
                chunks, err = _silent_infer(probe_text, PROMPT_TEXT, arg, **kw)
                if chunks:
                    _tts_prompt_mode = mode
                    _tts_prompt_arg = arg
                    _tts_support_stream = use_stream
                    break
                if isinstance(err, TypeError) and 'stream' in str(err):
                    print("   ⚠️ 该版本不支持 stream 参数，降级为整句合成后分块")
                    continue
                print(f"   ⚠️ {mode} 模式不可用: {type(err).__name__}: {str(err)[:90]}")
                break
            if _tts_prompt_arg is not None:
                break

        if _tts_prompt_arg is None:
            print("❌ 参考音频两种传参方式均失败，TTS 不可用")
            cosyvoice_model = None
            return
        print(f"   ✅ 传参模式={_tts_prompt_mode} | stream={_tts_support_stream}")

        # 尝试注册音色：成功可省去每句重复处理参考音频，注册后必须实推验证
        try:
            if cosyvoice_model.add_zero_shot_spk(PROMPT_TEXT, _tts_prompt_arg, TTS_SPK_ID):
                kw = {"stream": True} if _tts_support_stream else {}
                chunks, err = _silent_infer(probe_text, '', '',
                                            zero_shot_spk_id=TTS_SPK_ID, **kw)
                if chunks:
                    _tts_use_spk_id = True
                    print(f"✅ 音色已注册并验证: spk_id='{TTS_SPK_ID}'")
                else:
                    print(f"⚠️ 注册后推理失败({type(err).__name__}: {str(err)[:80]})，"
                          f"回退为逐句传参考音频")
            else:
                print("⚠️ add_zero_shot_spk 返回 False，回退为逐句传参考音频")
        except AttributeError:
            print("⚠️ 当前版本无 add_zero_shot_spk，回退为逐句传参考音频")
        except Exception as e:
            print(f"⚠️ 音色注册失败: {str(e)[:100]}，回退为逐句传参考音频")

        # 预热（走与线上完全一致的 _tts_infer 路径）
        print("🔥 预热 CosyVoice...")
        t0 = time.time()
        warmup_chunks = []
        _devnull2 = open(_os.devnull, 'w', encoding='utf-8', errors='ignore')
        _old_stdout2 = sys.stdout
        sys.stdout = _devnull2
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=(device == 'cuda'),
                                        dtype=torch.float16):
                    for output in _tts_infer('印象派的魅力。'):
                        warmup_chunks.append(
                            output['tts_speech'] if isinstance(output, dict) else output)
        finally:
            sys.stdout = _old_stdout2
            _devnull2.close()
        warmup_time = time.time() - t0
        print(f"✅ 预热完成，耗时 {warmup_time:.2f}s，共 {len(warmup_chunks)} 段")

        if device == 'cuda' and warmup_chunks:
            audio_duration = sum(c.shape[-1] for c in warmup_chunks) / TTS_SAMPLE_RATE
            actual_rtf = warmup_time / audio_duration if audio_duration > 0 else 999
            if actual_rtf < 0.5:
                print(f"✅ GPU 加速完美（RTF={actual_rtf:.2f}，音频 {audio_duration:.2f}s）")
            elif actual_rtf < 2.5:
                print(f"✅ GPU 已生效（RTF={actual_rtf:.2f}，音频 {audio_duration:.2f}s）")
            else:
                print(f"❌ GPU 未生效（RTF={actual_rtf:.2f}），请检查 ONNX Runtime GPU 安装")

        tts_executor = ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)
        if ENABLE_AUDIO_CACHE:
            audio_cache = AudioCache(max_size=AUDIO_CACHE_SIZE)
        print("✅ CosyVoice 加载成功")

    except Exception as e:
        print(f"\n❌ CosyVoice 加载失败: {e}")
        import traceback
        traceback.print_exc()
        cosyvoice_model = None

def synthesize_speech_pcm(text: str, speaker: str = "中文男",
                          cancel_ev: Optional[threading.Event] = None) -> Tuple[bytes, int]:
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")
    if audio_cache:
        cached = audio_cache.get(text, speaker, fmt='pcm')
        if cached:
            return cached

    import torch
    segs = []
    sr = TTS_SAMPLE_RATE
    t0 = time.time()

    gen = _tts_infer(text, block=True)   # 阻塞等 GPU，不会返回 None
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(),
                                    dtype=torch.float16):
                for output in gen:
                    if isinstance(output, dict):
                        sr = output.get('sample_rate', TTS_SAMPLE_RATE)
                        seg = output['tts_speech'].cpu().float().numpy().squeeze()
                    else:
                        seg = output.cpu().float().numpy().squeeze()
                    segs.append(seg)
                    if cancel_ev is not None and cancel_ev.is_set():
                        break
    finally:
        gen.close()   # 无论 break/异常，确保信号量立即释放

    if not segs or (cancel_ev is not None and cancel_ev.is_set()):
        return b"", sr

    audio_data = np.concatenate(segs) if len(segs) > 1 else segs[0]
    duration = len(audio_data) / sr
    rtf = (time.time() - t0) / duration if duration > 0 else 0
    print(f"⏱️ TTS '{text[:10]}...' | 音频 {duration:.2f}s | 耗时 {time.time()-t0:.2f}s | RTF={rtf:.2f}")

    if audio_data.dtype != np.int16:
        audio_data = (audio_data * 32767).astype(np.int16)
    pcm_bytes = audio_data.tobytes()
    if audio_cache:
        audio_cache.put(text, speaker, (pcm_bytes, sr), fmt='pcm')
    return pcm_bytes, sr

async def synthesize_speech_streaming(websocket, safe_send, text: str, speaker: str,
                                      session_id: str, seq: int,
                                      cancel_ev: threading.Event,
                                      subtitle_text: str = None,
                                      subtitle_index: int = None,
                                      on_pcm: Optional[Callable[[bytes], None]] = None
                                      ) -> Tuple[int, bool]:
    """流式合成并分块发送。取消判断统一走 cancel_ev（不受会话字典 pop 影响）。
    subtitle_text/subtitle_index：若提供，在首个 PCM chunk 发出前先发 tts_sentence，
    实现字幕与声音严格同步。on_pcm：每块原始 PCM 的同步回调（本机播放/组 WAV）。"""
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")

    async def emit_chunk(chunk: bytes):
        nonlocal seq
        if on_pcm:
            on_pcm(chunk)
        await safe_send({"type": "tts_chunk", "session_id": session_id,
                         "seq": seq, "data": base64.b64encode(chunk).decode()})
        seq += 1

    # 缓存命中（pcm）。必须按音频时长匀速推，不能 20ms 内倒完 30s PCM：
    # Unity 还没进播放态就会先收到 tts_end，听起来像 tts_start 没推流成功。
    if audio_cache:
        cached = audio_cache.get(text, speaker, fmt='pcm')
        if cached:
            pcm_bytes, sr = cached
            sr = sr or TTS_SAMPLE_RATE
            duration = len(pcm_bytes) / 2.0 / float(sr) if pcm_bytes else 0.0
            print(f"🔁 TTS 缓存命中 '{text[:15]}...' | {duration:.2f}s，按播放节奏推流")
            if subtitle_text is not None:
                await safe_send({"type": "tts_sentence", "session_id": session_id,
                                 "index": subtitle_index, "text": subtitle_text})
            offset = 0
            while offset < len(pcm_bytes):
                if cancel_ev.is_set():
                    return seq, True
                chunk = pcm_bytes[offset:offset + TTS_CHUNK_SIZE]
                await emit_chunk(chunk)
                offset += TTS_CHUNK_SIZE
                chunk_sec = len(chunk) / 2.0 / float(sr)
                await asyncio.sleep(max(0.01, chunk_sec * 0.85))
            return seq, False

    loop = asyncio.get_running_loop()
    seg_queue: asyncio.Queue = asyncio.Queue()
    t0 = time.time()
    first_chunk_time = None
    all_segs = []
    subtitle_sent = False

    def _run_inference():
        gen = _tts_infer(text, block=True, cancel_ev=cancel_ev)
        if gen is None:
            loop.call_soon_threadsafe(seg_queue.put_nowait, None)
            return
        import torch
        try:
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(),
                                        dtype=torch.float16):
                    for output in gen:
                        if cancel_ev.is_set():
                            break
                        loop.call_soon_threadsafe(seg_queue.put_nowait, output)
        except Exception as e:
            loop.call_soon_threadsafe(seg_queue.put_nowait, e)
        finally:
            gen.close()
            loop.call_soon_threadsafe(seg_queue.put_nowait, None)

    threading.Thread(target=_run_inference, daemon=True).start()

    sr = TTS_SAMPLE_RATE
    cancelled = False

    while True:
        item = await seg_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            print(f"⚠️ 推理线程异常: {item}")
            cancelled = True
            break
        if cancel_ev.is_set():
            cancelled = True
            break

        output = item
        if isinstance(output, dict):
            sr = output.get('sample_rate', TTS_SAMPLE_RATE)
            seg = output['tts_speech'].cpu().float().numpy().squeeze()
        else:
            seg = output.cpu().float().numpy().squeeze()
        all_segs.append(seg)

        seg_pcm = (seg * 32767).astype(np.int16) if seg.dtype != np.int16 else seg
        pcm_bytes = seg_pcm.tobytes()

        offset = 0
        while offset < len(pcm_bytes):
            if cancel_ev.is_set():
                cancelled = True
                break
            chunk = pcm_bytes[offset:offset + TTS_CHUNK_SIZE]

            # 首包：先发字幕，再发音频
            if first_chunk_time is None:
                first_chunk_time = time.time() - t0
                if subtitle_text is not None and not subtitle_sent:
                    await safe_send({"type": "tts_sentence",
                                     "session_id": session_id,
                                     "index": subtitle_index,
                                     "text": subtitle_text})
                    subtitle_sent = True
                print(f"🎤 首包延迟: {first_chunk_time:.2f}s | '{text[:15]}...'")

            await emit_chunk(chunk)
            offset += TTS_CHUNK_SIZE
            await asyncio.sleep(0)

        if cancelled:
            break

    if all_segs and not cancelled:
        audio_data = np.concatenate(all_segs) if len(all_segs) > 1 else all_segs[0]
        duration = len(audio_data) / sr
        rtf = (time.time() - t0) / duration if duration > 0 else 0
        if audio_cache and duration > 0:
            full_pcm = (audio_data * 32767).astype(np.int16).tobytes() \
                if audio_data.dtype != np.int16 else audio_data.tobytes()
            audio_cache.put(text, speaker, (full_pcm, sr), fmt='pcm')
        print(f"✅ 流式合成: '{text[:15]}...' | {duration:.2f}s | RTF={rtf:.2f}")

    return seq, cancelled

async def push_sentence_pcm(websocket, safe_send, sentence, speaker,
                            session_id, seq, cancel_ev,
                            subtitle_text: str = None,
                            subtitle_index: int = None,
                            on_pcm: Optional[Callable[[bytes], None]] = None) -> Tuple[int, bool]:
    return await synthesize_speech_streaming(
        websocket, safe_send, sentence, speaker, session_id, seq, cancel_ev,
        subtitle_text=subtitle_text, subtitle_index=subtitle_index, on_pcm=on_pcm)

def synthesize_speech(text: str, speaker: str = "中文男") -> bytes:
    """WAV 合成（须在 tts_executor 内调用）"""
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")
    if audio_cache:
        cached = audio_cache.get(text, speaker, fmt='wav')
        if cached:
            return cached
    pcm_bytes, sr = synthesize_speech_pcm(text, speaker)
    buffer = io.BytesIO()
    wavfile.write(buffer, sr, np.frombuffer(pcm_bytes, dtype=np.int16))
    buffer.seek(0)
    audio_bytes = buffer.read()
    if audio_cache:
        audio_cache.put(text, speaker, audio_bytes, fmt='wav')
    return audio_bytes

def text_to_base64_audio(text: str, speaker: str = "中文男") -> str:
    return base64.b64encode(synthesize_speech(text, speaker)).decode('utf-8')
