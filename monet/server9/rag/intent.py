# -*- coding: utf-8 -*-
"""意图 / NER / 情感 / 推荐 / 记忆。"""
from __future__ import annotations
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import jieba.posseg as pseg
import requests
import asyncio

from config import *
from session import Entity, Sentiment, SessionState, ConversationTurn
from rag import index as rag

class IntentClassifier:
    @staticmethod
    def classify(query: str) -> Tuple[str, float, float]:
        """返回 (intent, raw_score, normalized_confidence)。同步函数，调用方负责放 executor。"""
        if not ENABLE_INTENT or not rag.HAS_INTENT or not rag.intent_vecs:
            return "knowledge", 1.0, 1.0
        q_vec = rag.embedder.encode([query], normalize_embeddings=True)[0]
        raw_scores = {}
        for label, intent_vec in rag.intent_vecs.items():
            raw_scores[label] = float(np.dot(q_vec, intent_vec))
        intent = max(raw_scores, key=raw_scores.get)
        max_raw = raw_scores[intent]
        values = list(raw_scores.values())
        min_v, max_v = min(values), max(values)
        conf = (max_raw - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        print(f"🎯 意图识别: {intent} (原始={max_raw:.3f}, 归一化={conf:.3f})")
        return intent, max_raw, conf

    @staticmethod
    def should_route_to(intent: str, raw_score: float) -> bool:
        threshold = INTENT_CONFIDENCE_THRESHOLD.get(intent, 0.0)
        return raw_score >= threshold

intent_classifier = IntentClassifier()

class NERSystem:
    @staticmethod
    def extract_entities(text: str) -> List[Entity]:
        if not ENABLE_NER:
            return []
        entities = []
        for word, flag in pseg.cut(text):
            if flag in NER_TYPES and len(word) > 1:
                entities.append(Entity(text=word, type=flag))
        return entities

    @staticmethod
    def resolve_coreference(question: str, session: SessionState) -> Tuple[str, List[Entity]]:
        if not session.history:
            return question, []
        pronouns = {"它": ["nr", "nz"], "他": ["nr"], "她": ["nr"],
                    "那个": ["nr", "nz", "ns"], "这个": ["nr", "nz", "ns"],
                    "那件": ["nz"], "这件": ["nz"], "那里": ["ns"], "那边": ["ns"]}
        resolved = question
        matched_entities = []
        for pronoun, allowed_types in pronouns.items():
            if pronoun not in question:
                continue
            for entity in session.get_recent_entities(max_turns=2):
                if entity.type in allowed_types:
                    resolved = resolved.replace(pronoun, entity.text, 1)
                    matched_entities.append(entity)
                    print(f"🔗 共指消解: '{pronoun}' → '{entity.text}'")
                    break
        if resolved == question and any(p in question for p in pronouns):
            last_sources = session.get_last_sources()
            if last_sources:
                last_item = next((it for it in rag.items if it["id"] == last_sources[0]), None)
                if last_item and last_item.get("title"):
                    title = last_item["title"]
                    for pronoun in pronouns:
                        if pronoun in resolved:
                            resolved = resolved.replace(pronoun, title, 1)
                            matched_entities.append(Entity(text=title, type="item"))
                            print(f"🔗 共指消解(展品): '{pronoun}' → '{title}'")
                            break
        return resolved, matched_entities

ner_system = NERSystem()

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
            label, score = "positive", min(0.5 + scores["positive"] * 0.1, 1.0)
        elif scores["negative"] > scores["positive"]:
            label, score = "negative", min(0.5 + scores["negative"] * 0.1, 1.0)
        else:
            label, score = "neutral", 0.5
        return Sentiment(label=label, score=score, keywords=matched_keywords)

sentiment_analyzer = SentimentAnalyzer()

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
                    item = next((it for it in rag.items if it["id"] == source_id), None)
                    if item:
                        visited_categories.add(item["category"])
            preferred_categories = session.preferences.get("favorite_categories", [])
            hall = set(EXHIBITION_CATEGORY)
            target_categories = list((set(preferred_categories) | visited_categories) & hall)
            if not target_categories:
                target_categories = list(hall)
            candidates = [it for it in rag.items
                          if it["id"] not in visited_items
                          and it["category"] in target_categories
                          and it.get("title")]
            recommendations = random.sample(candidates, top_k) if len(candidates) > top_k else candidates
            return [{"id": r["id"], "title": r["title"], "category": r["category"],
                     "reason": f"您似乎对{r['category']}感兴趣"} for r in recommendations]
        except Exception as e:
            print(f"⚠️ 推荐生成失败: {e}")
            return []

recommendation_engine = RecommendationEngine()

class MemorySystem:
    @staticmethod
    async def load_user_preferences(user_id: str) -> Dict:
        if not ENABLE_MEMORY:
            return {}
        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: requests.post(
                f"{MEMORY_API_BASE}/search",
                json={"queries": [f"user {user_id} preferences"], "layers": ["preference"],
                      "topK": {"preferences": 5}}, timeout=5))
            if resp.status_code == 200:
                prefs = resp.json().get("preferences", [])
                return {"favorite_categories": [p.get("memoryCategory") for p in prefs
                                                if p.get("memoryCategory")]}
        except Exception:
            pass
        return {}

    @staticmethod
    async def load_user_profile(user_id: str) -> Dict:
        if not ENABLE_MEMORY:
            return {}
        try:
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: requests.post(
                f"{MEMORY_API_BASE}/search",
                json={"queries": [f"user {user_id}"], "layers": ["identity"],
                      "topK": {"identities": 3}}, timeout=5))
            if resp.status_code == 200:
                identities = resp.json().get("identities", [])
                if identities:
                    return {"role": identities[0].get("withIdentity", {}).get("role"),
                            "interests": identities[0].get("tags", [])}
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
