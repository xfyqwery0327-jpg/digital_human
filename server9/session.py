# -*- coding: utf-8 -*-
"""会话状态、落盘、过期清理。"""
from __future__ import annotations
import json
import time
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict

from config import (
    MEMORY_MAX_TURNS, CONTEXT_WINDOW, ENABLE_RECOMMENDATION,
    RECOMMENDATION_THRESHOLD, RECOMMENDATION_COOLDOWN,
    SESSION_PERSISTENCE, SESSION_STORAGE_PATH,
    SESSION_CLEANUP_INTERVAL, SESSION_TIMEOUT, ENABLE_MEMORY,
)

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
    def __init__(self, question: str, answer: str, sources: List[str], timestamp: str,
                 sentiment: Optional[Sentiment] = None, entities: Optional[List[Entity]] = None,
                 intent: Optional[str] = None):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.timestamp = timestamp
        self.sentiment = sentiment
        self.entities = entities or []
        self.intent = intent

    def to_dict(self):
        return {"question": self.question, "answer": self.answer, "sources": self.sources,
                "timestamp": self.timestamp,
                "sentiment": self.sentiment.to_dict() if self.sentiment else None,
                "entities": [e.to_dict() for e in self.entities], "intent": self.intent}

class SessionState:
    def __init__(self, user_id: str, session_id: str, persona_id: str = ""):
        self.user_id = user_id
        self.session_id = session_id
        self.persona_id = persona_id or ""
        self.history: List[ConversationTurn] = []
        self.preferences: Dict = {}
        self.user_profile: Dict = {}
        self.created_at = time.time()
        self.last_active = time.time()
        self.entity_memory: Dict[str, Entity] = {}
        self.last_recommendation_turn = -999
        self.recommendation_count = 0
        self.sentiment_history: List[Sentiment] = []
        self.hook_used = set()

    def add_turn(self, question, answer, sources, sentiment=None, entities=None, intent=None):
        turn = ConversationTurn(question=question, answer=answer, sources=sources,
                                timestamp=datetime.now().isoformat(),
                                sentiment=sentiment, entities=entities, intent=intent)
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
        return (len(self.history) >= RECOMMENDATION_THRESHOLD
                and turns_since_last >= RECOMMENDATION_COOLDOWN)

    def to_dict(self) -> Dict:
        return {"user_id": self.user_id, "session_id": self.session_id,
                "persona_id": self.persona_id,
                "history": [t.to_dict() for t in self.history],
                "preferences": self.preferences, "user_profile": self.user_profile,
                "created_at": self.created_at, "last_active": self.last_active,
                "entity_memory": {k: v.to_dict() for k, v in self.entity_memory.items()},
                "last_recommendation_turn": self.last_recommendation_turn,
                "recommendation_count": self.recommendation_count,
                "sentiment_history": [s.to_dict() for s in self.sentiment_history],
                "hook_used": list(self.hook_used)}

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionState':
        session = cls(data["user_id"], data["session_id"], data.get("persona_id", ""))
        session.preferences = data.get("preferences", {})
        session.user_profile = data.get("user_profile", {})
        session.created_at = data.get("created_at", time.time())
        session.last_active = data.get("last_active", time.time())
        session.last_recommendation_turn = data.get("last_recommendation_turn", -999)
        session.recommendation_count = data.get("recommendation_count", 0)
        session.hook_used = set(data.get("hook_used", []))
        for turn_data in data.get("history", []):
            sentiment = Sentiment(**turn_data["sentiment"]) if turn_data.get("sentiment") else None
            entities = [Entity(**e) for e in turn_data.get("entities", [])]
            session.history.append(ConversationTurn(
                question=turn_data["question"], answer=turn_data["answer"],
                sources=turn_data["sources"], timestamp=turn_data["timestamp"],
                sentiment=sentiment, entities=entities, intent=turn_data.get("intent")))
        for text, e_data in data.get("entity_memory", {}).items():
            session.entity_memory[text] = Entity(**e_data)
        for s_data in data.get("sentiment_history", []):
            session.sentiment_history.append(Sentiment(**s_data))
        return session

class SessionPersistence:
    @staticmethod
    def save_session(session: SessionState):
        if not SESSION_PERSISTENCE:
            return
        try:
            SESSION_STORAGE_PATH.mkdir(exist_ok=True)
            sub = (session.persona_id or "monet").strip() or "monet"
            dest_dir = SESSION_STORAGE_PATH / sub
            dest_dir.mkdir(exist_ok=True)
            file_path = dest_dir / f"{session.session_id}.json"
            tmp_path = file_path.with_suffix(".tmp")
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
            tmp_path.replace(file_path)
        except Exception as e:
            print(f"⚠️ 保存会话失败: {e}")

    @staticmethod
    def load_session(session_id: str) -> Optional[SessionState]:
        if not SESSION_PERSISTENCE:
            return None
        try:
            file_path = SESSION_STORAGE_PATH / f"{session_id}.json"
            if not file_path.exists():
                for sub in SESSION_STORAGE_PATH.glob("*"):
                    if sub.is_dir():
                        cand = sub / f"{session_id}.json"
                        if cand.exists():
                            file_path = cand
                            break
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
            else:
                for sub in SESSION_STORAGE_PATH.glob("*"):
                    if sub.is_dir():
                        cand = sub / f"{session_id}.json"
                        if cand.exists():
                            cand.unlink()
                            break
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


active_sessions: Dict[str, SessionState] = {}
cleanup_task = None
robot_idle_task = None


async def cleanup_inactive_sessions():
    from rag.intent import memory_system
    while True:
        try:
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
            now = time.time()
            expired = [sid for sid, s in list(active_sessions.items())
                       if now - s.last_active > SESSION_TIMEOUT]
            for session_id in expired:
                session = active_sessions[session_id]
                if ENABLE_MEMORY:
                    await memory_system.save_session_summary(session)
                persistence.save_session(session)
                del active_sessions[session_id]
            if expired:
                print(f"✅ 清理了 {len(expired)} 个过期会话")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"⚠️ 清理任务出错: {e}")
