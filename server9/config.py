# -*- coding: utf-8 -*-
"""路径、阈值、机器人回调。必须最先导入（DLL / 离线 HuggingFace）。"""
import os
import sys
import logging
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

def _first_existing(*candidates) -> str:
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            return str(p)
    return str(candidates[-1]) if candidates else ""


def _first_dir(*candidates) -> str:
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir():
            return str(p)
    return ""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
PERSONAS_DIR = PROJECT_ROOT / "personas"
VOICE_DIR = PROJECT_ROOT / "voice"
VOICES_DIR = VOICE_DIR  # 兼容旧名；实际按 voice/{persona_id}/ 分目录
KB_DIR = PROJECT_ROOT / "kb"
DEFAULT_PERSONA_ID = os.environ.get("MONET_PERSONA", "monet")

_conda = os.environ.get("CONDA_PREFIX", "")
_cuda_bin = os.environ.get("CUDA_BIN") or _first_dir(
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
)
_cudnn_bin = os.environ.get("CUDNN_BIN") or _first_dir(
    Path(_conda) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin" if _conda else "",
    Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
    r"C:\Users\BroadMesse\miniconda3\envs\robot\Lib\site-packages\nvidia\cudnn\bin",
)

for _d in (_cuda_bin, _cudnn_bin):
    if not _d:
        continue
    try:
        os.add_dll_directory(_d)
    except (FileNotFoundError, OSError) as _e:
        try:
            print(f"⚠️ DLL 目录不可用: {_d} ({_e})，如 TTS/ONNX 报错请先检查此路径")
        except UnicodeEncodeError:
            print(f"[WARN] DLL not found: {_d} ({_e})")

_path_prefix = [d for d in (_cuda_bin, _cudnn_bin) if d]
if _path_prefix:
    os.environ["PATH"] = ";".join(_path_prefix) + ";" + os.environ.get("PATH", "")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def _looks_like_embed_model(path) -> bool:
    p = Path(path) if path else Path()
    if not p.is_dir():
        return False
    return any((p / name).exists() for name in (
        "modules.json", "config.json", "pytorch_model.bin", "model.safetensors"))


def _resolve_embed_dir(*candidates) -> str:
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if _looks_like_embed_model(p):
            return str(p)
        if p.is_dir():
            snaps = sorted(p.glob("snapshots/*"))
            for snap in snaps:
                if _looks_like_embed_model(snap):
                    return str(snap)
    return ""


_GTE_SNAP = "snapshots/71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
_GTE_HUB = Path("models--thenlper--gte-base-zh") / _GTE_SNAP
_MS_GTE = Path.home() / ".cache" / "modelscope" / "hub"
EMBED_MODEL = os.environ.get("EMBED_MODEL") or _resolve_embed_dir(
    PROJECT_ROOT / "models" / "gte-base-zh",
    PROJECT_ROOT / "models" / "thenlper-gte-base-zh",
    Path.home() / ".cache" / "huggingface" / "hub" / _GTE_HUB,
    Path(r"C:\Users\BroadMesse\.cache\huggingface\hub") / _GTE_HUB,
    _MS_GTE / "AI-ModelScope" / "gte-base-zh",
    _MS_GTE / "models" / "AI-ModelScope" / "gte-base-zh",
)
RERANK_MODEL = None
DICT_PATH = PROJECT_ROOT / "dict" / "jieba_dict.txt"

# CosyVoice 目录是指向 Robot_Intelligent_Voice_System\CosyVoice 的联接，代码和权重都在那边。
_ROBOT_COSYVOICE = Path(r"E:\projects\Robot_Intelligent_Voice_System\CosyVoice")
_CV2_DIRNAME = "CosyVoice2-0.5B"
_CV3_DIRNAME = "Fun-CosyVoice3-0.5B"


def _cosyvoice_choice() -> str:
    """last-start.txt 的 cosyvoice= 优先，否则 cosyvoice.model。2 用 CosyVoice2，其余用 3。"""
    token = ""
    last = PROJECT_ROOT / "deploy" / "last-start.txt"
    try:
        if last.is_file():
            for line in last.read_text(encoding="utf-8", errors="ignore").splitlines():
                row = line.strip()
                if row.lower().startswith("cosyvoice="):
                    token = row.split("=", 1)[1].strip().lower()
                    break
    except OSError:
        token = ""
    if not token:
        marker = PROJECT_ROOT / "deploy" / "cosyvoice.model"
        try:
            text = marker.read_text(encoding="utf-8") if marker.is_file() else ""
        except OSError:
            text = ""
        token = (text.strip().split() or [""])[0].lower()
    if token in {"2", "cv2", "cosyvoice2", "2-0.5b", "cosyvoice2-0.5b"}:
        return "2"
    return "3"


_CV_PRIMARY = _CV2_DIRNAME if _cosyvoice_choice() == "2" else _CV3_DIRNAME
_CV_SECONDARY = _CV3_DIRNAME if _CV_PRIMARY == _CV2_DIRNAME else _CV2_DIRNAME
COSYVOICE_PATH = os.environ.get("COSYVOICE_PATH") or _first_existing(
    PROJECT_ROOT / "CosyVoice",
    _ROBOT_COSYVOICE,
    r"C:\ro\CosyVoice",
)
COSYVOICE_MODEL = os.environ.get("COSYVOICE_MODEL") or _first_existing(
    Path(COSYVOICE_PATH) / "pretrained_models" / _CV_PRIMARY if COSYVOICE_PATH else "",
    _ROBOT_COSYVOICE / "pretrained_models" / _CV_PRIMARY,
    Path(COSYVOICE_PATH) / "pretrained_models" / _CV_SECONDARY if COSYVOICE_PATH else "",
    _ROBOT_COSYVOICE / "pretrained_models" / _CV_SECONDARY,
)
COSYVOICE3_MODEL = COSYVOICE_MODEL
ENABLE_TTS = True
TTS_SAMPLE_RATE = 24000
TTS_MAX_WORKERS = 1
ENABLE_AUDIO_CACHE = True
AUDIO_CACHE_SIZE = 100


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# CosyVoice3 一次只能整句出声，每次调用都要重算参考音前缀。
# 按逗号一句一调，固定开销会把句间空档拉长。这里把相邻小句拼到这个字数再合成。
# 0 = 回到逐小句合成（CosyVoice2 流式不受影响，始终逐小句）。
TTS_MERGE_CHARS = max(0, _env_int("TTS_MERGE_CHARS", 40))
# Qwen3-TTS 相邻小句最多拼到这个字数。0 = 不拼句。环境变量 QWEN_TTS_MERGE_CHARS 可覆盖。
QWEN_TTS_MERGE_CHARS = max(0, _env_int("QWEN_TTS_MERGE_CHARS", 5))
# 0 = 不按字数硬切。Qwen 只在标点和书名号右界断开。环境变量可改回正数做字数硬切。
QWEN_TTS_SPLIT_CHARS = max(0, _env_int("QWEN_TTS_SPLIT_CHARS", 0))
# 整句合成的头尾静音只留这么多毫秒，句与句之间不再叠两段留白。0 = 不裁。
TTS_TRIM_HEAD_MS = max(0, _env_int("TTS_TRIM_HEAD_MS", 40))
TTS_TRIM_TAIL_MS = max(0, _env_int("TTS_TRIM_TAIL_MS", 160))
# Qwen3-TTS 一次 generate 并行合成的段数。12GB 卡上 8 段会把空闲显存打到 2GB 以下，
# 批结束触发 empty_cache，下一批还要重新分配。6 段足够把 RTF 压到 1 以下。
QWEN_TTS_BATCH = max(1, _env_int("QWEN_TTS_BATCH", 6))
# 游客问答边生成边送句。凑满这么多句，或最多再等这么多毫秒，再开算。
# 避免第一句单独合成、后面的句子进不了这一批。0 毫秒 = 不等。
QWEN_TTS_COALESCE_UNITS = max(1, _env_int("QWEN_TTS_COALESCE_UNITS", 4))
QWEN_TTS_COALESCE_MS = max(0, _env_int("QWEN_TTS_COALESCE_MS", 800))
# 首句等到句号、问号、叹号，或最多再等这么多毫秒。标点还没到也可以先出，避免一直不开声。
QWEN_TTS_FIRST_MS = max(0, _env_int("QWEN_TTS_FIRST_MS", 3000))
# 1 = CosyVoice 的 Flow 走 TensorRT（需装 tensorrt，首次会编译 plan 几分钟）。失败自动退回 PyTorch。
COSYVOICE_LOAD_TRT = os.environ.get("COSYVOICE_LOAD_TRT", "0").strip().lower() in {"1", "true", "yes"}

# 每片 PCM 字节数。Unity 会把整段 JSON（含 base64）打进日志，单片过大或全 0
# 会变成超长 AAAA... 撑爆 LogBlueprintUserMessages。
TTS_CHUNK_SIZE = 4096
TTS_CHUNK_MAX = 8192
TTS_SERVICE_NAME = "persona-tts-stream"
TTS_SPK_ID = "monet"  # 仅作莫奈缺省；运行时用 voice/{id}/ 与 YAML

PROMPT_WAV_PATH = _first_existing(
    VOICE_DIR / "monet" / "yinpin_16k.wav",
    VOICE_DIR / "monet" / "cross_lingual_prompt.wav",
    VOICE_DIR / "monet" / "yinpin.mp3",
    r"C:\ro\CosyVoice\asset\cross_lingual_prompt.wav",
)
PROMPT_TEXT = '我梦见了我们年轻的时候，当我醒来时，我终于明白了自己一生都在寻找什么。'
MONET_LEGACY_PROMPT_WAV = r'C:\ro\CosyVoice\asset\cross_lingual_prompt.wav'

# memory API 为空壳实现且 8080 端口与 llama.cpp 冲突，默认关闭；
# 待独立 memory 服务就绪后再开启并改端口
ENABLE_MEMORY = False
MEMORY_API_BASE = "http://localhost:8081/memory"
MEMORY_MAX_TURNS = 10
CONTEXT_WINDOW = 3

SESSION_TIMEOUT = 30 * 60
SESSION_CLEANUP_INTERVAL = 5 * 60
SESSION_PERSISTENCE = True
SESSION_STORAGE_PATH = PROJECT_ROOT / "sessions"

ENABLE_NER = True
NER_TYPES = ["nr", "nz", "ns"]

ENABLE_SENTIMENT = True
SENTIMENT_KEYWORDS = {
    "positive": ["好", "喜欢", "棒", "有意思", "精彩", "不错", "太美了", "震撼", "赞"],
    "negative": ["无聊", "没意思", "不好", "差", "看不懂", "不清楚", "糟糕"],
    "neutral": ["知道了", "明白", "好的", "谢谢", "嗯"]
}

ENABLE_RECOMMENDATION = True
RECOMMENDATION_THRESHOLD = 3
RECOMMENDATION_COOLDOWN = 5

ENABLE_INTENT = True
INTENT_CONFIDENCE_THRESHOLD = {
    "command": 0.4,
    "chitchat": 0.5,
    "knowledge": 0.0
}

# 机器人互动（可用环境变量覆盖，便于 Windows 现场改 IP 而不改代码）
ROBOT_MAX_ROUNDS = int(os.environ.get("ROBOT_MAX_ROUNDS", "3"))
ROBOT_IDLE_TIMEOUT = int(os.environ.get("ROBOT_IDLE_TIMEOUT", "120"))
ROBOT_API_TOKEN = os.environ.get("ROBOT_API_TOKEN", "")
# ask 受理后立刻 HTTP 202。需要改成 200 可设 ROBOT_ACCEPTED_STATUS=200
ROBOT_ACCEPTED_STATUS = int(os.environ.get("ROBOT_ACCEPTED_STATUS", "202"))
ROBOT_PLAY_TAIL_SEC = float(os.environ.get("ROBOT_PLAY_TAIL_SEC", "0.6"))
# 欢迎/告别/拒答一律来自 personas/*.yaml，禁止再用全局莫奈文案做别人的兜底
ROBOT_HELLO_TEXTS = []
ROBOT_BYE_TEXTS = []

VEC_TOPK = 50
BM25_TOPK = 50
RRF_K = 60
RERANK_CAND = 20
COMPRESS_MAX = 500
MIN_SCORE_RRF = 0.015
MIN_SCORE_RERANK = 0.50

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
PERSONA = "莫奈"  # 仅文档/旧字段；运行时以 personas/ 为准
EXHIBITION_CATEGORY = []  # 禁止全局类目；检索只认当前人设 categories


HOST = os.environ.get("MONET_HOST", "0.0.0.0")
PORT = int(os.environ.get("MONET_PORT", "8848"))
REFUSE_TEXT = "这个我还不太清楚。我只讲解眼前这场展。"
LOG_DIR = PROJECT_ROOT / "logs"
ROBOT_LOG_LEVEL = os.environ.get("ROBOT_LOG_LEVEL", "DEBUG").upper()
sys.path.insert(0, COSYVOICE_PATH)


def runtime_log(filename: str, msg: str) -> None:
    """Write ASCII-safe runtime logs even when the scheduled-task console is Hidden."""
    from datetime import datetime
    line = "[{}] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / filename, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


class _SuppressTooShort(logging.Filter):
    def filter(self, record):
        return "too short than prompt text" not in record.getMessage()

for h in logging.getLogger().handlers:
    h.addFilter(_SuppressTooShort())
logging.getLogger().addFilter(_SuppressTooShort())
