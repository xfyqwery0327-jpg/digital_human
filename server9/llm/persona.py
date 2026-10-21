# -*- coding: utf-8 -*-
"""人设 Prompt、钩子、票务、闲聊。全部来自当前请求的 PersonaConfig。"""
from __future__ import annotations
import random
from session import SessionState
from persona.models import PersonaConfig


def _p(persona: PersonaConfig = None) -> PersonaConfig:
    if persona is not None:
        return persona
    from persona.runtime import active
    return active()


def _active_hooks(persona: PersonaConfig = None):
    return dict(_p(persona).hooks or {})


def _active_ticket(persona: PersonaConfig = None):
    return dict(_p(persona).ticket_info or {})


def system_prompt(persona: PersonaConfig = None) -> str:
    return _p(persona).system_prompt or ""


def pick_hello(persona: PersonaConfig = None) -> str:
    texts = [t for t in (_p(persona).hello_texts or []) if t]
    return random.choice(texts) if texts else f"我是{_p(persona).display_name}。"


def pick_bye(persona: PersonaConfig = None) -> str:
    texts = [t for t in (_p(persona).bye_texts or []) if t]
    return random.choice(texts) if texts else "今日就讲到这里。再见。"


def chitchat_reply(question: str, persona: PersonaConfig = None) -> str:
    p = _p(persona)
    mapping = dict(p.chitchat or {})
    q = (question or "").strip()
    keys = [k for k in mapping.keys() if k and not str(k).startswith("_")]
    for keyword in sorted(keys, key=len, reverse=True):
        if keyword in q:
            return mapping[keyword]
    if any(x in q for x in ("你好", "您好", "嗨", "哈喽")):
        return pick_hello(p)
    if any(x in q for x in ("再见", "拜拜")):
        return pick_bye(p)
    return mapping.get("_default") or f"我是{p.display_name}。有想了解的展品，直接问我。"


def is_greeting(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if q in ("你好", "您好", "嗨", "哈喽", "hello", "Hello", "欢迎"):
        return True
    return q.startswith("你好") or q.startswith("您好")


class HookGenerator:
    @staticmethod
    def should_add_hook(session: SessionState) -> bool:
        if len(session.history) == 0:
            return True
        if len(session.history) >= 3 and len(session.history) % 3 == 0:
            return True
        if session.get_overall_sentiment() == "negative":
            return True
        return False

    @staticmethod
    def select_hook(session: SessionState, hook_type: str = None,
                    persona: PersonaConfig = None) -> str:
        hooks = _active_hooks(persona)
        if not hooks:
            return ""
        if not hook_type:
            if len(session.history) == 0:
                hook_type = "knowledge"
            elif len(session.history) <= 2:
                hook_type = random.choice(["knowledge", "technique"])
            elif len(session.history) <= 5:
                hook_type = random.choice(["interactive", "technique"])
            else:
                hook_type = random.choice(["deep", "interactive"])
        available = [h for h in hooks.get(hook_type, []) if h not in session.hook_used]
        if not available:
            session.hook_used.clear()
            available = hooks.get(hook_type, [])
        if available:
            hook = random.choice(available)
            session.hook_used.add(hook)
            return hook
        return ""

hook_generator = HookGenerator()

class TicketPromoter:
    @staticmethod
    def should_promote(question: str, session: SessionState,
                       persona: PersonaConfig = None) -> bool:
        info = _active_ticket(persona)
        if not info:
            return False
        ticket_keywords = ["门票", "价格", "多少钱", "买票", "购票", "特展", "展览", "票"]
        if any(kw in question for kw in ticket_keywords):
            return True
        positive_keywords = ["喜欢", "不错", "有意思", "想看", "感兴趣"]
        if any(kw in question for kw in positive_keywords) and len(session.history) >= 2:
            return True
        if len(session.history) >= 5 and session.get_overall_sentiment() == "positive":
            turns_since_promo = 999
            for i, turn in enumerate(session.history):
                if any(v and v in turn.answer for v in info.values()):
                    turns_since_promo = len(session.history) - i - 1
            if turns_since_promo >= 5:
                return True
        return False

    @staticmethod
    def get_ticket_info(question: str, persona: PersonaConfig = None) -> str:
        info = _active_ticket(persona)
        if "价格" in question or "多少钱" in question:
            return info.get("price") or ""
        elif "展期" in question or "什么时候" in question or "到期" in question:
            return info.get("duration") or ""
        elif "优惠" in question or "免费" in question:
            return info.get("discount") or ""
        elif "在哪" in question or "位置" in question:
            return info.get("location") or ""
        else:
            choices = [info.get("price"), info.get("duration"), info.get("promo")]
            choices = [c for c in choices if c]
            return random.choice(choices) if choices else ""

    @staticmethod
    def embed_naturally(base_answer: str, ticket_info: str) -> str:
        connector = random.choice(["对了，", "顺便说一句，", "另外，", ""])
        return f"{base_answer} {connector}{ticket_info}"

ticket_promoter = TicketPromoter()
