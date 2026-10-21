#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
server.py v6.4 - 流式音频 + 意图识别完整版
修复内容:
1. 彻底修复 <think> 标签导致 full_answer 偏移量计算错误(核心 bug)
2. 修复句子切分逻辑：用 while+find 替代 for+split+break，避免混合标点漏切
3. 合并两条 system message，避免部分后端不支持多 system 角色
4. 去掉 stop 参数（该参数让推理模型在输出第一个 token 时就停止）
5. 新增 OPEN_TAG_RE 用于检测未闭合的 <think> 前缀，防止把思维过程
   中尚未闭合的内容当成正文输出
"""
import sys
import json
import pickle
import random
import re
import asyncio
import base64
import io
import hashlib
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

import faiss
import jieba
import jieba.posseg as pseg
import requests
from rank_bm25 import BM25Okapi
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from scipy.io import wavfile

# ============================================================
# 配置
# ============================================================

INDEX_DIR = Path("index")
EMBED_MODEL = r"C:\Users\BroadMesse\.cache\huggingface\hub\models--thenlper--gte-base-zh\snapshots\71ab7947d6fac5b64aa299e6e40e6c2b2e85976c"
RERANK_MODEL = None
DICT_PATH = Path("dict/jieba_dict.txt")

COSYVOICE_PATH = r"C:\ro\CosyVoice"
COSYVOICE_MODEL = r"C:\ro\CosyVoice\pretrained_models\CosyVoice-300M-SFT"
ENABLE_TTS = True
TTS_SAMPLE_RATE = 22050
TTS_MAX_WORKERS = 2
ENABLE_AUDIO_CACHE = True
AUDIO_CACHE_SIZE = 100

ENABLE_MEMORY = True
MEMORY_API_BASE = "http://localhost:8080/memory"
MEMORY_MAX_TURNS = 10
CONTEXT_WINDOW = 3

SESSION_TIMEOUT = 30 * 60
SESSION_CLEANUP_INTERVAL = 5 * 60
SESSION_PERSISTENCE = True
SESSION_STORAGE_PATH = Path("sessions")

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
    "command": 0.5,
    "chitchat": 0.6,
    "knowledge": 0.0
}

VEC_TOPK = 50
BM25_TOPK = 50
RRF_K = 60
RERANK_CAND = 20
COMPRESS_MAX = 500

MIN_SCORE_RRF = 0.020
MIN_SCORE_RERANK = 0.50

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
PERSONA = "莫奈"

SYSTEM_PROMPT = """你是{persona},正在展厅里为观众做讲解。回答规则:
1. 只依据【参考资料】作答;资料里没有的信息一律回答"这个我还不太清楚",绝不自行推测或补充。
2. 口语化、短句,整段不超过 120 字,因为要念出来给观众听。
3. 带点趣味性:可以用比喻、反问、或者抛出一个小悬念,让观众觉得有意思。
4. 当观众问到展品、特展、活动相关的问题时,如果【参考资料】里有票务信息(价格、优惠、购票方式),自然地提一句,但别生硬推销。
5. 不要出现"根据参考资料""文档中提到"这类字样,直接说内容。
6. 如果用户提到"它""那个""刚才说的"等指代词,结合【对话历史】和【实体上下文】理解指代对象。
7. 严格禁止输出任何 <think>、</think>、<reasoning>、</reasoning> 等 XML 标签,不要输出思维过程、推理步骤、内心独白。你的回答必须是直接面向观众的讲解内容,就像在现场对话一样。
"""

# 匹配已完整闭合的 think/reasoning 块
THINK_RE = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)

# 匹配尚未闭合的 think/reasoning 开头标签（用于截断未关闭的思维块）
OPEN_TAG_RE = re.compile(r"<(?:think|reasoning)\b", re.IGNORECASE)

SENT_END = "。！？；!?;"

# ============================================================
# 全局变量
# ============================================================

embedder = None
index = None
items = []
faiss_map = []
bm25 = None
reranker = None
intent_vecs = None
cosyvoice_model = None
tts_executor = None
audio_cache = None

active_sessions: Dict[str, 'SessionState'] = {}
cleanup_task: Optional[asyncio.Task] = None

HAS_RERANKER = False
HAS_INTENT = False

# ============================================================
# 数据类
# ============================================================

@dataclass
class Entity:
    text: str
    type: str
    mention_count: int = 1
    last_mention_turn: int = 0

    def to_dict(self):
        return asdict(self)

@dataclass
class Sentiment:
    label: str
    score: float
    keywords: List[str]

    def to_dict(self):
        return asdict(self)

class ConversationTurn:
    def __init__(self, question: str, answer: str, sources: List[str],
                 timestamp: str, sentiment: Optional[Sentiment] = None,
                 entities: Optional[List[Entity]] = None,
                 intent: Optional[str] = None):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.timestamp = timestamp
        self.sentiment = sentiment
        self.entities = entities or []
        self.intent = intent

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "timestamp": self.timestamp,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "entities": [e.to_dict() for e in self.entities],
            "intent": self.intent
        }

class SessionState:
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.history: List[ConversationTurn] = []
        self.preferences: Dict = {}
        self.user_profile: Dict = {}
        self.created_at = time.time()
        self.last_active = time.time()
        self.entity_memory: Dict[str, Entity] = {}
        self.last_recommendation_turn = -999
        self.recommendation_count = 0
        self.sentiment_history: List[Sentiment] = []

    def add_turn(self, question: str, answer: str, sources: List[str],
                 sentiment: Optional[Sentiment] = None,
                 entities: Optional[List[Entity]] = None,
                 intent: Optional[str] = None):
        turn = ConversationTurn(
            question=question,
            answer=answer,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            sentiment=sentiment,
            entities=entities,
            intent=intent
        )
        self.history.append(turn)
        self.last_active = time.time()

        if sentiment:
            self.sentiment_history.append(sentiment)
            if len(self.sentiment_history) > 10:
                self.sentiment_history = self.sentiment_history[-10:]

        if entities:
            turn_idx = len(self.history) - 1
            for entity in entities:
                if entity.text in self.entity_memory:
                    self.entity_memory[entity.text].mention_count += 1
                    self.entity_memory[entity.text].last_mention_turn = turn_idx
                else:
                    entity.last_mention_turn = turn_idx
                    self.entity_memory[entity.text] = entity

        if len(self.history) > MEMORY_MAX_TURNS:
            self.history = self.history[-MEMORY_MAX_TURNS:]

    def get_context_window(self, max_turns: int = CONTEXT_WINDOW) -> List[ConversationTurn]:
        return self.history[-max_turns:]

    def get_last_sources(self) -> List[str]:
        if self.history:
            return self.history[-1].sources
        return []

    def get_recent_entities(self, max_turns: int = 3) -> List[Entity]:
        current_turn = len(self.history) - 1
        recent = []
        for entity in self.entity_memory.values():
            if current_turn - entity.last_mention_turn <= max_turns:
                recent.append(entity)
        return sorted(recent, key=lambda e: e.last_mention_turn, reverse=True)

    def get_overall_sentiment(self) -> str:
        if not self.sentiment_history:
            return "neutral"
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for s in self.sentiment_history:
            counts[s.label] += 1
        return max(counts, key=counts.get)

    def should_recommend(self) -> bool:
        if not ENABLE_RECOMMENDATION:
            return False
        turns_since_last = len(self.history) - self.last_recommendation_turn
        return (len(self.history) >= RECOMMENDATION_THRESHOLD and
                turns_since_last >= RECOMMENDATION_COOLDOWN)

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "history": [t.to_dict() for t in self.history],
            "preferences": self.preferences,
            "user_profile": self.user_profile,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "entity_memory": {k: v.to_dict() for k, v in self.entity_memory.items()},
            "last_recommendation_turn": self.last_recommendation_turn,
            "recommendation_count": self.recommendation_count,
            "sentiment_history": [s.to_dict() for s in self.sentiment_history]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionState':
        session = cls(data["user_id"], data["session_id"])
        session.preferences = data.get("preferences", {})
        session.user_profile = data.get("user_profile", {})
        session.created_at = data.get("created_at", time.time())
        session.last_active = data.get("last_active", time.time())
        session.last_recommendation_turn = data.get("last_recommendation_turn", -999)
        session.recommendation_count = data.get("recommendation_count", 0)

        for turn_data in data.get("history", []):
            sentiment = None
            if turn_data.get("sentiment"):
                sentiment = Sentiment(**turn_data["sentiment"])

            entities = []
            for e_data in turn_data.get("entities", []):
                entities.append(Entity(**e_data))

            turn = ConversationTurn(
                question=turn_data["question"],
                answer=turn_data["answer"],
                sources=turn_data["sources"],
                timestamp=turn_data["timestamp"],
                sentiment=sentiment,
                entities=entities,
                intent=turn_data.get("intent")
            )
            session.history.append(turn)

        for text, e_data in data.get("entity_memory", {}).items():
            session.entity_memory[text] = Entity(**e_data)

        for s_data in data.get("sentiment_history", []):
            session.sentiment_history.append(Sentiment(**s_data))

        return session

# ============================================================
# 持久化
# ============================================================

class SessionPersistence:
    @staticmethod
    def save_session(session: SessionState):
        if not SESSION_PERSISTENCE:
            return
        try:
            SESSION_STORAGE_PATH.mkdir(exist_ok=True)
            file_path = SESSION_STORAGE_PATH / f"{session.session_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存会话失败: {e}")

    @staticmethod
    def load_session(session_id: str) -> Optional[SessionState]:
        if not SESSION_PERSISTENCE:
            return None
        try:
            file_path = SESSION_STORAGE_PATH / f"{session_id}.json"
            if not file_path.exists():
                return None
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SessionState.from_dict(data)
        except Exception:
            return None

    @staticmethod
    def delete_session(session_id: str):
        if not SESSION_PERSISTENCE:
            return
        try:
            file_path = SESSION_STORAGE_PATH / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    @staticmethod
    def list_sessions() -> List[str]:
        if not SESSION_PERSISTENCE:
            return []
        try:
            if not SESSION_STORAGE_PATH.exists():
                return []
            return [f.stem for f in SESSION_STORAGE_PATH.glob("*.json")]
        except Exception:
            return []

persistence = SessionPersistence()

# ============================================================
# 音频缓存
# ============================================================

class AudioCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def _get_key(self, text: str, speaker: str) -> str:
        return hashlib.md5(f"{text}|{speaker}".encode()).hexdigest()

    def get(self, text: str, speaker: str):
        key = self._get_key(text, speaker)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, text: str, speaker: str, audio_bytes: bytes):
        key = self._get_key(text, speaker)
        self.cache[key] = audio_bytes
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

# ============================================================
# 意图识别
# ============================================================

class IntentClassifier:
    @staticmethod
    def classify(query: str) -> Tuple[str, float]:
        if not ENABLE_INTENT or not HAS_INTENT or not intent_vecs:
            return "knowledge", 1.0

        q_vec = embedder.encode([query], normalize_embeddings=True)[0]

        similarities = {}
        for label, intent_vec in intent_vecs.items():
            sim = float(np.dot(q_vec, intent_vec))
            similarities[label] = sim

        intent = max(similarities, key=similarities.get)
        confidence = similarities[intent]

        max_sim = max(similarities.values())
        min_sim = min(similarities.values())
        if max_sim > min_sim:
            confidence = (confidence - min_sim) / (max_sim - min_sim)

        print(f"🎯 意图识别: {intent} (置信度={confidence:.3f})")
        return intent, confidence

    @staticmethod
    def should_route_to(intent: str, confidence: float) -> bool:
        threshold = INTENT_CONFIDENCE_THRESHOLD.get(intent, 0.0)
        return confidence >= threshold

intent_classifier = IntentClassifier()

# ============================================================
# NER + 共指消解
# ============================================================

class NERSystem:
    @staticmethod
    def extract_entities(text: str) -> List[Entity]:
        if not ENABLE_NER:
            return []
        entities = []
        words = pseg.cut(text)
        for word, flag in words:
            if flag in NER_TYPES and len(word) > 1:
                entities.append(Entity(text=word, type=flag))
        return entities

    @staticmethod
    def resolve_coreference(question: str, session: SessionState) -> Tuple[str, List[Entity]]:
        if not session.history:
            return question, []

        pronouns = {
            "它": ["nr", "nz"],
            "他": ["nr"],
            "她": ["nr"],
            "那个": ["nr", "nz", "ns"],
            "这个": ["nr", "nz", "ns"],
            "那件": ["nz"],
            "这件": ["nz"],
            "那里": ["ns"],
            "那边": ["ns"]
        }

        resolved = question
        matched_entities = []

        for pronoun, allowed_types in pronouns.items():
            if pronoun not in question:
                continue

            recent_entities = session.get_recent_entities(max_turns=2)
            for entity in recent_entities:
                if entity.type in allowed_types:
                    resolved = resolved.replace(pronoun, entity.text, 1)
                    matched_entities.append(entity)
                    print(f"🔗 共指消解: '{pronoun}' → '{entity.text}'")
                    break

        if resolved == question and any(p in question for p in pronouns.keys()):
            last_sources = session.get_last_sources()
            if last_sources:
                last_item = next((it for it in items if it["id"] == last_sources[0]), None)
                if last_item and last_item.get("title"):
                    title = last_item["title"]
                    for pronoun in pronouns.keys():
                        if pronoun in resolved:
                            resolved = resolved.replace(pronoun, title, 1)
                            matched_entities.append(Entity(text=title, type="item"))
                            print(f"🔗 共指消解(展品): '{pronoun}' → '{title}'")
                            break

        return resolved, matched_entities

ner_system = NERSystem()

# ============================================================
# 情感分析
# ============================================================

class SentimentAnalyzer:
    @staticmethod
    def analyze(text: str) -> Sentiment:
        if not ENABLE_SENTIMENT:
            return Sentiment(label="neutral", score=0.5, keywords=[])

        text_lower = text.lower()
        scores = {"positive": 0, "negative": 0, "neutral": 0}
        matched_keywords = []

        for label, keywords in SENTIMENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[label] += 1
                    matched_keywords.append(kw)

        if scores["positive"] > scores["negative"]:
            label = "positive"
            score = min(0.5 + scores["positive"] * 0.1, 1.0)
        elif scores["negative"] > scores["positive"]:
            label = "negative"
            score = min(0.5 + scores["negative"] * 0.1, 1.0)
        else:
            label = "neutral"
            score = 0.5

        return Sentiment(label=label, score=score, keywords=matched_keywords)

sentiment_analyzer = SentimentAnalyzer()

# ============================================================
# 推荐系统
# ============================================================

class RecommendationEngine:
    @staticmethod
    def generate_recommendations(session: SessionState, top_k: int = 3) -> List[Dict]:
        if not ENABLE_RECOMMENDATION:
            return []

        try:
            visited_categories = set()
            visited_items = set()

            for turn in session.history:
                for source_id in turn.sources:
                    visited_items.add(source_id)
                    item = next((it for it in items if it["id"] == source_id), None)
                    if item:
                        visited_categories.add(item["category"])

            preferred_categories = session.preferences.get("favorite_categories", [])
            target_categories = list(set(preferred_categories) | visited_categories)

            if not target_categories:
                target_categories = list(set(it["category"] for it in items))

            candidates = []
            for item in items:
                if (item["id"] not in visited_items and
                        item["category"] in target_categories):
                    candidates.append(item)

            if len(candidates) > top_k:
                recommendations = random.sample(candidates, top_k)
            else:
                recommendations = candidates

            return [
                {
                    "id": r["id"],
                    "title": r.get("title", ""),
                    "category": r["category"],
                    "reason": f"您似乎对{r['category']}感兴趣"
                }
                for r in recommendations
            ]

        except Exception as e:
            print(f"⚠️ 推荐生成失败: {e}")
            return []

recommendation_engine = RecommendationEngine()

# ============================================================
# 记忆系统
# ============================================================

class MemorySystem:
    @staticmethod
    async def load_user_preferences(user_id: str) -> Dict:
        if not ENABLE_MEMORY:
            return {}
        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{MEMORY_API_BASE}/search",
                    json={
                        "queries": [f"user {user_id} preferences"],
                        "layers": ["preference"],
                        "topK": {"preferences": 5}
                    },
                    timeout=5
                )
            )
            if resp.status_code == 200:
                data = resp.json()
                prefs = data.get("preferences", [])
                return {
                    "favorite_categories": [p.get("memoryCategory") for p in prefs if p.get("memoryCategory")]
                }
        except Exception:
            pass
        return {}

    @staticmethod
    async def load_user_profile(user_id: str) -> Dict:
        if not ENABLE_MEMORY:
            return {}
        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{MEMORY_API_BASE}/search",
                    json={
                        "queries": [f"user {user_id}"],
                        "layers": ["identity"],
                        "topK": {"identities": 3}
                    },
                    timeout=5
                )
            )
            if resp.status_code == 200:
                data = resp.json()
                identities = data.get("identities", [])
                if identities:
                    return {
                        "role": identities[0].get("withIdentity", {}).get("role"),
                        "interests": identities[0].get("tags", [])
                    }
        except Exception:
            pass
        return {}

    @staticmethod
    async def save_conversation_turn(session: SessionState, turn: ConversationTurn):
        pass

    @staticmethod
    async def save_session_summary(session: SessionState):
        pass

memory_system = MemorySystem()

# ============================================================
# 会话清理
# ============================================================

async def cleanup_inactive_sessions():
    while True:
        try:
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
            now = time.time()
            expired_sessions = []

            for session_id, session in list(active_sessions.items()):
                if now - session.last_active > SESSION_TIMEOUT:
                    expired_sessions.append(session_id)

            for session_id in expired_sessions:
                session = active_sessions[session_id]
                if ENABLE_MEMORY:
                    await memory_system.save_session_summary(session)
                persistence.save_session(session)
                del active_sessions[session_id]

            if expired_sessions:
                print(f"✅ 清理了 {len(expired_sessions)} 个过期会话")
        except Exception as e:
            print(f"⚠️ 清理任务出错: {e}")

# ============================================================
# 加载模型
# ============================================================

def load_jieba_dict():
    try:
        if DICT_PATH.exists():
            jieba.load_userdict(str(DICT_PATH))
            print(f"📖 已加载专名词典: {DICT_PATH}")
    except Exception as e:
        print(f"⚠️ 词典加载失败: {e}")

def load_cosyvoice():
    global cosyvoice_model, tts_executor, audio_cache

    if not ENABLE_TTS:
        print("⚠️ TTS 已禁用")
        return

    try:
        print("🔍 检查 CosyVoice 依赖...")
        try:
            import hyperpyyaml
            print("   ✅ hyperpyyaml 已安装")
        except ImportError:
            print("   ❌ hyperpyyaml 未安装")
            print("\n📦 解决方案:")
            print("   pip install hyperpyyaml onnxruntime modelscope\n")
            return

        print("🔍 检查 whisper 包...")
        try:
            import whisper
            whisper_file = whisper.__file__
            print(f"   ✅ whisper 位置: {whisper_file}")

            if whisper_file.endswith("whisper.py"):
                print("   ⚠️ 检测到单文件 whisper.py,可能与 openai-whisper 冲突")
                print("\n📦 解决方案:")
                print("   pip uninstall whisper -y")
                print("   pip install openai-whisper\n")
                return
        except ImportError:
            print("   ⚠️ whisper 未安装,将在导入 CosyVoice 时自动检查")

        print(f"🔍 检查路径...")
        cosyvoice_path = Path(COSYVOICE_PATH)
        model_path = Path(COSYVOICE_MODEL)

        if not cosyvoice_path.exists():
            print(f"   ❌ CosyVoice 路径不存在: {COSYVOICE_PATH}")
            return
        print(f"   ✅ CosyVoice 路径: {COSYVOICE_PATH}")

        if not model_path.exists():
            print(f"   ❌ 模型路径不存在: {COSYVOICE_MODEL}")
            return
        print(f"   ✅ 模型路径: {COSYVOICE_MODEL}")

        required_files = [
            "cosyvoice.yaml",
            "flow.pt",
            "hift.pt",
            "llm.pt",
            "campplus.onnx",
            "speech_tokenizer_v1.onnx"
        ]

        print("🔍 检查模型文件...")
        missing_files = []
        for file in required_files:
            file_path = model_path / file
            if not file_path.exists():
                missing_files.append(file)
                print(f"   ❌ 缺少文件: {file}")
            else:
                print(f"   ✅ {file}")

        if missing_files:
            print(f"\n❌ 缺少 {len(missing_files)} 个必需文件")
            return

        sys.path.insert(0, COSYVOICE_PATH)
        sys.path.insert(0, str(Path(COSYVOICE_PATH) / "third_party" / "Matcha-TTS"))
        print("✅ 已添加 sys.path")

        print("🎤 导入 CosyVoice 模块...")
        from cosyvoice.cli.cosyvoice import CosyVoice
        print("✅ CosyVoice 模块导入成功")

        print(f"🎤 加载 CosyVoice 模型: {COSYVOICE_MODEL}")
        cosyvoice_model = CosyVoice(COSYVOICE_MODEL)
        print("✅ CosyVoice 模型加载成功")

        print("🔥 预热 CosyVoice...")
        for _ in cosyvoice_model.inference_sft(tts_text="你好", spk_id="中文女"):
            pass
        print("✅ 预热完成")

        tts_executor = ThreadPoolExecutor(max_workers=TTS_MAX_WORKERS)

        if ENABLE_AUDIO_CACHE:
            audio_cache = AudioCache(max_size=AUDIO_CACHE_SIZE)

        print("✅ CosyVoice 加载成功")
        print(f"   可用说话人: {cosyvoice_model.list_available_spks()}")

    except ImportError as e:
        error_str = str(e)
        print(f"\n❌ 导入错误: {e}")
        print(f"   错误类型: {type(e).__name__}")

        if "whisper" in error_str.lower():
            print("\n🔍 检测到 whisper 相关错误!")
            print("📦 可能的解决方案:")
            print("   1. pip uninstall whisper -y")
            print("   2. pip install openai-whisper")
            print("   3. 或者删除错误的 whisper.py 文件\n")
        else:
            print("\n📦 可能的解决方案:")
            print("   1. pip install hyperpyyaml")
            print("   2. pip install onnxruntime modelscope")

        import traceback
        traceback.print_exc()

    except TypeError as e:
        error_str = str(e)
        print(f"\n❌ TypeError: {e}")
        print(f"   错误类型: {type(e).__name__}")

        if "NoneType" in error_str and "iterable" in error_str:
            print("\n🔍 检测到 whisper libc 错误!")
            print("   这通常是因为安装了错误的 whisper 包")
            print("\n📦 解决方案:")
            print("   1. pip uninstall whisper -y")
            print("   2. pip install openai-whisper")
            print("   3. 重新运行 python server.py\n")

        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"\n❌ CosyVoice 加载失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        print("\n🔍 完整错误堆栈:")
        import traceback
        traceback.print_exc()

        print("\n📋 诊断信息:")
        print(f"   Python 版本: {sys.version}")
        print(f"   CosyVoice 路径: {COSYVOICE_PATH}")
        print(f"   模型路径: {COSYVOICE_MODEL}")
        print(f"   sys.path 前 3 项: {sys.path[:3]}")

def load_index():
    global embedder, index, items, faiss_map, bm25, reranker, HAS_RERANKER, intent_vecs, HAS_INTENT
    load_jieba_dict()
    print("🧮 加载 embedding 模型:", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"📦 加载索引: {INDEX_DIR}")
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    items = json.loads((INDEX_DIR / "items.json").read_text(encoding="utf-8"))
    faiss_map = json.loads((INDEX_DIR / "faiss_map.json").read_text(encoding="utf-8"))
    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        docs = pickle.load(f)["docs"]
        bm25 = BM25Okapi(docs)
    print(f"✅ FAISS {index.ntotal} 向量 | items {len(items)} | BM25 {len(docs)} 篇")

    if RERANK_MODEL:
        try:
            from sentence_transformers import CrossEncoder
            reranker = CrossEncoder(RERANK_MODEL)
            HAS_RERANKER = True
            print(f"✅ reranker: {RERANK_MODEL}")
        except Exception as e:
            print(f"⚠️ reranker 加载失败: {e}")

    try:
        INTENT_EXAMPLES = {
            "command": [
                "声音大一点", "音量调大", "大声点", "小声点", "声音小一点",
                "暂停", "别说了", "停", "继续", "再说一遍", "重复一遍"
            ],
            "chitchat": [
                "你好", "嗨", "您好", "你是谁", "你叫什么", "谢谢", "再见",
                "今天天气怎么样", "你会什么", "聊聊", "讲个笑话"
            ],
            "knowledge": [
                "妇好是谁", "司母辛鼎有多重", "商朝的玉从哪里来",
                "这个展品是什么", "介绍一下这件文物"
            ],
        }
        intent_vecs = {}
        for label, exs in INTENT_EXAMPLES.items():
            v = embedder.encode(exs, normalize_embeddings=True).mean(axis=0)
            intent_vecs[label] = v / np.linalg.norm(v)
        HAS_INTENT = True
        print(f"✅ 意图识别: {list(intent_vecs.keys())}")
    except Exception as e:
        print(f"⚠️ 意图识别初始化失败: {e}")

    print("🔥 预热 embedding 模型...")
    embedder.encode(["warmup"], normalize_embeddings=True)
    print("✅ 预热完成")

    load_cosyvoice()

# ============================================================
# TTS 合成
# ============================================================

def synthesize_speech(text: str, speaker: str = "中文女") -> bytes:
    if not cosyvoice_model:
        raise RuntimeError("CosyVoice 未加载")

    if audio_cache:
        cached = audio_cache.get(text, speaker)
        if cached:
            return cached

    try:
        output_generator = cosyvoice_model.inference_sft(
            tts_text=text,
            spk_id=speaker
        )

        audio_data = None
        sr = TTS_SAMPLE_RATE

        for output in output_generator:
            if isinstance(output, dict):
                sr = output.get('sample_rate', TTS_SAMPLE_RATE)
                audio_tensor = output['tts_speech']
                audio_data = audio_tensor.cpu().numpy().squeeze()
            else:
                audio_data = output.cpu().numpy().squeeze()
            break

        if audio_data is None:
            raise RuntimeError("TTS 生成失败:无输出")

        buffer = io.BytesIO()

        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        wavfile.write(buffer, sr, audio_data)
        buffer.seek(0)
        audio_bytes = buffer.read()

        if audio_cache:
            audio_cache.put(text, speaker, audio_bytes)

        return audio_bytes

    except Exception as e:
        print(f"⚠️ TTS 合成失败: {e}")
        raise

def text_to_base64_audio(text: str, speaker: str = "中文女") -> str:
    audio_bytes = synthesize_speech(text, speaker)
    return base64.b64encode(audio_bytes).decode('utf-8')

# ============================================================
# 检索
# ============================================================

def search(query: str, top_k: int, category: str | None, session: Optional[SessionState] = None):
    if session:
        query, _ = ner_system.resolve_coreference(query, session)

    q_vec = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_vec, VEC_TOPK)
    vec_hits = {}
    for rank, row in enumerate(I[0]):
        if row == -1:
            continue
        item_idx = faiss_map[row]
        it = items[item_idx]
        if category and it["category"] != category:
            continue
        if item_idx not in vec_hits:
            vec_hits[item_idx] = rank

    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    ranked_idx = sorted(
        (i for i, it in enumerate(items) if (not category or it["category"] == category) and scores[i] > 0),
        key=lambda i: -scores[i]
    )[:BM25_TOPK]
    bm25_hits = {}
    for rank, item_idx in enumerate(ranked_idx):
        if item_idx not in bm25_hits:
            bm25_hits[item_idx] = rank

    rrf = {}
    recall_src = {}
    for item_idx, rank in vec_hits.items():
        rrf[item_idx] = rrf.get(item_idx, 0.0) + 1.0 / (RRF_K + rank)
        recall_src.setdefault(item_idx, set()).add("vector")
    for item_idx, rank in bm25_hits.items():
        rrf[item_idx] = rrf.get(item_idx, 0.0) + 1.0 / (RRF_K + rank)
        recall_src.setdefault(item_idx, set()).add("bm25")

    seen = set()
    fused = []
    for item_idx, score in sorted(rrf.items(), key=lambda x: -x[1]):
        key = items[item_idx].get("dedup_key") or items[item_idx]["id"]
        if key in seen:
            continue
        seen.add(key)
        fused.append((item_idx, score))

    cand = fused[:RERANK_CAND]
    if HAS_RERANKER and reranker:
        pairs = [(query, items[i]["text"][:COMPRESS_MAX]) for i, _ in cand]
        r_scores = reranker.predict(pairs)
        probs = 1.0 / (1.0 + np.exp(-np.array(r_scores)))
        ranked = sorted(zip(cand, probs), key=lambda x: -x[1])
        results = []
        for (item_idx, _rrf), prob in ranked:
            it = items[item_idx]
            results.append(_fmt(it, item_idx, recall_src, prob, "rerank"))
        return results

    results = []
    for item_idx, score in fused[:top_k]:
        it = items[item_idx]
        results.append(_fmt(it, item_idx, recall_src, score, None))
    return results

def _fmt(it, item_idx, recall_src, score, mode):
    src = recall_src.get(item_idx, set())
    if mode == "rerank":
        recall = "rerank"
    elif len(src) == 2:
        recall = "both"
    elif "bm25" in src:
        recall = "bm25"
    else:
        recall = "vector"
    return {
        "id": it["id"],
        "text": it["text"],
        "source": it["source"],
        "category": it["category"],
        "title": it.get("title", ""),
        "score": round(float(score), 4),
        "recall": recall
    }

# ============================================================
# 意图处理器
# ============================================================

async def handle_command(websocket: WebSocket, question: str, session: SessionState, speaker: str):
    commands = {
        "暂停": "好的,已暂停。需要继续时告诉我。",
        "停": "好的,已暂停。",
        "声音大": "好的,我会说大声一点。",
        "大声": "好的,我会说大声一点。",
        "声音小": "好的,我会轻声一些。",
        "小声": "好的,我会轻声一些。",
        "再说一遍": session.history[-1].answer if session.history else "还没有对话记录哦。",
        "重复": session.history[-1].answer if session.history else "还没有对话记录哦。"
    }

    reply = None
    for keyword, response in commands.items():
        if keyword in question:
            reply = response
            break

    if not reply:
        reply = "不好意思,我没理解这个指令。"

    await websocket.send_json({"type": "text", "content": reply})

    if cosyvoice_model:
        try:
            loop = asyncio.get_running_loop()
            audio_b64 = await loop.run_in_executor(tts_executor, text_to_base64_audio, reply, speaker)
            await websocket.send_json({"type": "audio", "content": audio_b64, "sentence": reply, "index": 0})
        except Exception as e:
            print(f"⚠️ TTS 失败: {e}")

    session.add_turn(question, reply, [], None, [], "command")
    persistence.save_session(session)

    await websocket.send_json({"type": "done", "intent": "command", "refused": False, "sources": []})

async def handle_chitchat(websocket: WebSocket, question: str, session: SessionState, speaker: str):
    responses = {
        "你好": "你好!我是莫奈,很高兴为你讲解展品。",
        "嗨": "嗨!有什么想了解的展品吗?",
        "你是谁": f"我是{PERSONA},正在这个展厅为大家做讲解。",
        "你叫什么": f"我叫{PERSONA}。",
        "谢谢": "不客气!",
        "再见": "再见!欢迎下次再来。"
    }

    reply = "很高兴和你聊天!有什么想了解的展品吗?"
    for keyword, resp in responses.items():
        if keyword in question:
            reply = resp
            break

    await websocket.send_json({"type": "text", "content": reply})

    if cosyvoice_model:
        try:
            loop = asyncio.get_running_loop()
            audio_b64 = await loop.run_in_executor(tts_executor, text_to_base64_audio, reply, speaker)
            await websocket.send_json({"type": "audio", "content": audio_b64, "sentence": reply, "index": 0})
        except Exception as e:
            print(f"⚠️ TTS 失败: {e}")

    session.add_turn(question, reply, [], None, [], "chitchat")
    persistence.save_session(session)

    await websocket.send_json({"type": "done", "intent": "chitchat", "refused": False, "sources": []})

# ============================================================
# 流式文本辅助：切句并发送
# ============================================================

async def flush_sentences(
    websocket: WebSocket,
    buffer: str,
    sentence_index: int,
    enable_audio: bool,
    speaker: str,
    loop: asyncio.AbstractEventLoop,
) -> Tuple[str, int]:
    """
    从 buffer 中切出所有完整句子（以 SENT_END 结尾），逐句发送文本和音频。
    返回 (剩余 buffer, 更新后的 sentence_index)。
    """
    while True:
        # 找最早出现的句末符位置
        earliest = -1
        for ch in SENT_END:
            pos = buffer.find(ch)
            if pos != -1 and (earliest == -1 or pos < earliest):
                earliest = pos

        if earliest == -1:
            break  # 没有完整句子，等待更多内容

        sentence = buffer[: earliest + 1].strip()
        buffer = buffer[earliest + 1 :]

        if not sentence:
            continue

        await websocket.send_json({"type": "text", "content": sentence})

        if enable_audio and cosyvoice_model:
            try:
                audio_b64 = await loop.run_in_executor(
                    tts_executor, text_to_base64_audio, sentence, speaker
                )
                await websocket.send_json({
                    "type": "audio",
                    "content": audio_b64,
                    "sentence": sentence,
                    "index": sentence_index,
                })
            except Exception as e:
                print(f"⚠️ TTS 失败: {e}")

        sentence_index += 1

    return buffer, sentence_index

# ============================================================
# 生命周期管理
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cleanup_task

    print("\n" + "=" * 60)
    print("🚀 正在启动服务...")
    print("=" * 60)

    load_index()

    cleanup_task = asyncio.create_task(cleanup_inactive_sessions())
    print(f"⏰ 启动会话清理任务(间隔 {SESSION_CLEANUP_INTERVAL}s,超时 {SESSION_TIMEOUT}s)")

    if SESSION_PERSISTENCE:
        persisted_sessions = persistence.list_sessions()
        print(f"📂 发现 {len(persisted_sessions)} 个持久化会话")

    print("✅ 启动完成\n")

    yield

    print("\n" + "=" * 60)
    print("🛑 正在关闭服务...")
    print("=" * 60)

    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    print(f"💾 保存 {len(active_sessions)} 个活跃会话...")
    for session in active_sessions.values():
        try:
            persistence.save_session(session)
            if ENABLE_MEMORY:
                await memory_system.save_session_summary(session)
        except Exception as e:
            print(f"⚠️ 保存会话 {session.session_id} 失败: {e}")

    print("✅ 会话保存完成")
    print("=" * 60)

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="RAG + LLM + Streaming TTS + Intent Recognition",
    lifespan=lifespan
)

# ============================================================
# WebSocket 对话接口
# ============================================================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()

    session: Optional[SessionState] = None
    session_id = None

    try:
        init_data = await websocket.receive_json()
        user_id = init_data.get("user_id", f"anonymous_{id(websocket)}")
        session_id = init_data.get("session_id")

        if session_id:
            if session_id in active_sessions:
                session = active_sessions[session_id]
            else:
                session = persistence.load_session(session_id)
                if session:
                    active_sessions[session_id] = session

        if not session:
            session_id = f"session_{int(time.time())}_{id(websocket)}"
            session = SessionState(user_id, session_id)
            active_sessions[session_id] = session

            if ENABLE_MEMORY:
                session.preferences = await memory_system.load_user_preferences(user_id)
                session.user_profile = await memory_system.load_user_profile(user_id)

        await websocket.send_json({
            "type": "init",
            "session_id": session_id,
            "user_id": user_id,
            "preferences": session.preferences,
            "history_count": len(session.history),
            "resumed": len(session.history) > 0
        })

        while True:
            data = await websocket.receive_json()
            question = data.get("question", "")
            category = data.get("category")
            top_k = data.get("top_k", 2)
            enable_audio = data.get("enable_audio", True)
            speaker = data.get("speaker", "中文男")

            if not question:
                await websocket.send_json({"type": "error", "content": "问题不能为空"})
                continue

            intent, confidence = intent_classifier.classify(question)
            print(f"🎯 意图: {intent} (置信度={confidence:.3f})")

            if intent == "command" and intent_classifier.should_route_to("command", confidence):
                await handle_command(websocket, question, session, speaker)
                continue

            elif intent == "chitchat" and intent_classifier.should_route_to("chitchat", confidence):
                await handle_chitchat(websocket, question, session, speaker)
                continue

            sentiment = sentiment_analyzer.analyze(question)
            entities = ner_system.extract_entities(question)

            if not category and session.preferences.get("favorite_categories"):
                category = session.preferences["favorite_categories"][0]

            hits = search(question, top_k, category, session)

            if not hits or hits[0]["score"] < (MIN_SCORE_RERANK if HAS_RERANKER else MIN_SCORE_RRF):
                refuse_text = "这个我还不太清楚,你可以问我展厅里其他的展品。"

                await websocket.send_json({"type": "text", "content": refuse_text})

                if enable_audio and cosyvoice_model:
                    try:
                        loop = asyncio.get_running_loop()
                        audio_b64 = await loop.run_in_executor(
                            tts_executor, text_to_base64_audio, refuse_text, speaker
                        )
                        await websocket.send_json({
                            "type": "audio",
                            "content": audio_b64,
                            "sentence": refuse_text,
                            "index": 0
                        })
                    except Exception as e:
                        print(f"⚠️ TTS 失败: {e}")

                session.add_turn(question, refuse_text, [], sentiment, entities, "knowledge")
                persistence.save_session(session)

                await websocket.send_json({
                    "type": "done",
                    "refused": True,
                    "intent": "knowledge",
                    "sources": []
                })
                continue

            contexts = [h["text"][:COMPRESS_MAX] for h in hits]
            ref = "\n".join(f"- {c}" for c in contexts)

            history_text = ""
            for turn in session.get_context_window():
                history_text += f"Q: {turn.question}\nA: {turn.answer}\n\n"

            recent_entities = session.get_recent_entities()
            entity_context = "、".join([e.text for e in recent_entities[:5]])

            # 合并成单条 system message，避免部分后端不支持多 system 角色
            system_content = (
                "You are in direct response mode. Never output<think>tags or show any reasoning process.\n\n"
                + SYSTEM_PROMPT.format(persona=PERSONA)
            )
            messages = [{"role": "system", "content": system_content}]

            for turn in session.get_context_window():
                messages.append({"role": "user", "content": turn.question})
                messages.append({"role": "assistant", "content": turn.answer})

            user_prompt = f"【参考资料】\n{ref}\n\n【问题】{question}"
            if history_text:
                user_prompt = f"【对话历史】\n{history_text}\n" + user_prompt
            if entity_context:
                user_prompt = f"【实体上下文】最近提到: {entity_context}\n\n" + user_prompt

            messages.append({"role": "user", "content": user_prompt})

            try:
                resp = requests.post(
                    LLM_URL,
                    json={
                        "messages": messages,
                        "temperature": 0.3,
                        "max_tokens": 300,
                        "stream": True,
                        # 不设 stop——推理模型首 token 是 <think>，设 stop 会立即终止生成
                    },
                    stream=True,
                    timeout=60
                )
                resp.raise_for_status()
            except Exception as e:
                error_text = "抱歉,我现在有点不舒服。"
                await websocket.send_json({"type": "error", "content": str(e)})
                await websocket.send_json({"type": "text", "content": error_text})
                await websocket.send_json({"type": "done", "refused": False, "sources": []})
                continue

            # ── 流式处理核心（修复版）────────────────────────────
            # 关键设计：
            #   full_raw_text  : 原始流式拼接，包含 <think>...</think>
            #   clean_sent_len : 已处理干净文本的字节偏移，独立于 full_raw_text
            #                    避免原始文本长度 != 干净文本长度导致的偏移计算错误
            #   OPEN_TAG_RE    : 检测尚未闭合的 <think>，截断防泄漏
            # ─────────────────────────────────────────────────────
            sentence_buffer = ""
            sentence_index = 0
            full_raw_text = ""
            clean_sent_len = 0  # 已推入 sentence_buffer 的干净文本长度
            loop = asyncio.get_running_loop()

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break

                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if not content:
                        continue

                    full_raw_text += content

                    # 第一步：剥掉所有已完整闭合的 think/reasoning 块
                    clean_text = THINK_RE.sub("", full_raw_text)

                    # 第二步：如果还有未闭合的开标签，只处理它之前的内容
                    m = OPEN_TAG_RE.search(clean_text)
                    safe_end = m.start() if m else len(clean_text)

                    # 第三步：取出尚未处理的新增干净内容
                    new_content = clean_text[clean_sent_len:safe_end]
                    if not new_content:
                        continue

                    clean_sent_len += len(new_content)
                    sentence_buffer += new_content

                    # 第四步：切句并发送
                    sentence_buffer, sentence_index = await flush_sentences(
                        websocket, sentence_buffer, sentence_index,
                        enable_audio, speaker, loop
                    )

                except Exception as e:
                    print(f"⚠️ 流式解析失败: {e}")
                    continue

            # 流结束后处理剩余 buffer（无句末符的尾句）
            if sentence_buffer.strip():
                sentence = sentence_buffer.strip()
                await websocket.send_json({"type": "text", "content": sentence})

                if enable_audio and cosyvoice_model:
                    try:
                        audio_b64 = await loop.run_in_executor(
                            tts_executor, text_to_base64_audio, sentence, speaker
                        )
                        await websocket.send_json({
                            "type": "audio",
                            "content": audio_b64,
                            "sentence": sentence,
                            "index": sentence_index,
                        })
                    except Exception as e:
                        print(f"⚠️ TTS 失败: {e}")

            # 流结束后统一生成 full_answer（保证干净）
            full_answer = THINK_RE.sub("", full_raw_text).strip()

            source_ids = [h["id"] for h in hits]
            session.add_turn(question, full_answer, source_ids, sentiment, entities, "knowledge")
            persistence.save_session(session)

            if ENABLE_MEMORY:
                last_turn = session.history[-1]
                asyncio.create_task(memory_system.save_conversation_turn(session, last_turn))

            recommendations = []
            if session.should_recommend():
                recommendations = recommendation_engine.generate_recommendations(session, top_k=3)
                session.last_recommendation_turn = len(session.history) - 1

                if recommendations:
                    rec_text = (
                        "顺便说一句,"
                        + "、".join([r["title"] for r in recommendations[:2]])
                        + " 也很有意思。"
                    )
                    await websocket.send_json({
                        "type": "recommendation",
                        "content": rec_text,
                        "items": recommendations
                    })

            await websocket.send_json({
                "type": "done",
                "refused": False,
                "intent": "knowledge",
                "sources": source_ids,
                "suggestions": recommendations,
                "sentiment": sentiment.to_dict()
            })

    except WebSocketDisconnect:
        print(f"🔌 WebSocket 断开: {session_id}")
        if session:
            persistence.save_session(session)

    except Exception as e:
        print(f"❌ WebSocket 错误: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# HTTP 接口
# ============================================================

class SearchReq(BaseModel):
    query: str
    top_k: int = 2
    category: str | None = None

@app.post("/search")
def api_search(req: SearchReq):
    results = search(req.query, req.top_k, req.category)

    if HAS_RERANKER:
        kept = [r for r in results if r["score"] >= MIN_SCORE_RERANK]
    else:
        kept = [r for r in results if r["score"] >= MIN_SCORE_RRF]

    return {"query": req.query, "results": kept}

@app.post("/tts")
def api_tts(req: dict):
    text = req.get("text", "")
    speaker = req.get("speaker", "中文男")

    if not cosyvoice_model:
        return {"error": "TTS 未启用"}

    if not text:
        return {"error": "text 不能为空"}

    try:
        audio_b64 = text_to_base64_audio(text, speaker)
        return {"audio": audio_b64, "format": "wav"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def api_health():
    return {
        "status": "healthy",
        "chunks": len(items),
        "vectors": index.ntotal if index else 0,
        "tts": bool(cosyvoice_model),
        "intent": HAS_INTENT,
        "sessions": len(active_sessions)
    }

# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 启动 RAG + LLM + 流式 TTS + 意图识别 服务 v6.4")
    print("=" * 60)
    print(f"📍 服务地址: http://0.0.0.0:8848")
    print(f"📍 WebSocket: ws://0.0.0.0:8848/ws/chat")
    print("\n核心修复:")
    print("  ✅ 去掉 stop 参数，推理模型不再被截断")
    print("  ✅ clean_sent_len 独立偏移，彻底修复<think>长度污染")
    print("  ✅ OPEN_TAG_RE 截断未闭合思维块，防正文泄漏")
    print("  ✅ flush_sentences 统一切句逻辑，混合标点不再漏切")
    print("  ✅ 合并 system message，兼容不支持多 system 角色的后端")
    print("  ✅ full_answer 在流结束后统一生成，保证干净")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8848, log_level="info")