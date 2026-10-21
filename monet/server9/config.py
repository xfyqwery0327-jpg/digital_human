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

_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
_cudnn_bin = r"C:\Users\BroadMesse\miniconda3\envs\robot\Lib\site-packages\nvidia\cudnn\bin"

for _d in (_cuda_bin, _cudnn_bin):
    try:
        os.add_dll_directory(_d)
    except (FileNotFoundError, OSError) as _e:
        try:
            print(f"⚠️ DLL 目录不可用: {_d} ({_e})，如 TTS/ONNX 报错请先检查此路径")
        except UnicodeEncodeError:
            print(f"[WARN] DLL not found: {_d} ({_e})")

os.environ["PATH"] = _cuda_bin + ";" + _cudnn_bin + ";" + os.environ.get("PATH", "")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "index"
EMBED_MODEL = r"C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh\snapshots\71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
RERANK_MODEL = None
DICT_PATH = PROJECT_ROOT / "dict" / "jieba_dict.txt"

COSYVOICE_PATH = r"C:\ro\CosyVoice"
COSYVOICE_MODEL = r"C:\ro\CosyVoice\pretrained_models\CosyVoice2-0.5B"
ENABLE_TTS = True
TTS_SAMPLE_RATE = 24000
TTS_MAX_WORKERS = 1
ENABLE_AUDIO_CACHE = True
AUDIO_CACHE_SIZE = 100

TTS_CHUNK_SIZE = 8192
TTS_SERVICE_NAME = "monet-tts-stream"
TTS_SPK_ID = "monet"

PROMPT_WAV_PATH = r'C:\ro\CosyVoice\asset\cross_lingual_prompt.wav'
PROMPT_TEXT = '我梦见了我们年轻的时候，当我醒来时，我终于明白了自己一生都在寻找什么。'

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
ROBOT_PLAY_TAIL_SEC = float(os.environ.get("ROBOT_PLAY_TAIL_SEC", "0.25"))
ROBOT_HELLO_TEXTS = [
    "欢迎来到我的世界。我是克洛德·莫奈，今天由我陪你看看光和颜色的故事。",
    "你来了。我是莫奈，吉维尼花园的主人。想先看睡莲，还是《日出·印象》？",
    "欢迎。光线每时每刻都在变化，这正是我想捕捉的魔法。你想听哪一段？",
]
ROBOT_BYE_TEXTS = [
    "很高兴今天和你聊天。再见，愿你把这里的光带走。",
    "我们就先讲到这里。再见，欢迎下次再来我的花园。",
    "再见。有机会再来，我还想给你讲讲干草堆上的颜色。",
]

VEC_TOPK = 50
BM25_TOPK = 50
RRF_K = 60
RERANK_CAND = 20
COMPRESS_MAX = 500
MIN_SCORE_RRF = 0.015
MIN_SCORE_RERANK = 0.50

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
PERSONA = "莫奈"
EXHIBITION_CATEGORY = ["exhibition_monet"]


HOST = os.environ.get("MONET_HOST", "0.0.0.0")
PORT = int(os.environ.get("MONET_PORT", "8848"))
REFUSE_TEXT = "这个我还不太清楚。我只讲解眼前这场展，你可以问我睡莲、《日出·印象》，或展厅里的其他展品。"
LOG_DIR = PROJECT_ROOT / "logs"
ROBOT_LOG_LEVEL = os.environ.get("ROBOT_LOG_LEVEL", "DEBUG").upper()
sys.path.insert(0, COSYVOICE_PATH)


class _SuppressTooShort(logging.Filter):
    def filter(self, record):
        return "too short than prompt text" not in record.getMessage()

for h in logging.getLogger().handlers:
    h.addFilter(_SuppressTooShort())
logging.getLogger().addFilter(_SuppressTooShort())
