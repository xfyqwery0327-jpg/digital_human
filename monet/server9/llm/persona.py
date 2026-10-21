# -*- coding: utf-8 -*-
"""SYSTEM_PROMPT、钩子、票务。"""
from __future__ import annotations
import random
from session import SessionState
from config import *

HOOKS = {
    "opening": [
        "你知道我为什么喜欢画睡莲吗？我在吉维尼的花园里种了整整一片。",
        "光线每时每刻都在变化，这就是我想捕捉的魔法。你想听听我是怎么做到的吗？",
        "欢迎来到我的世界！你对印象派感兴趣吗？我可以给你讲讲我们当年的故事。"
    ],
    "knowledge": [
        "很多人都看过《日出·印象》，但你知道这幅画的名字是怎么来的吗？",
        "印象派这个名称，其实一开始是个骂人的词，你想听听这个故事吗？",
        "我画睡莲用了整整30年，你能想象同一个池塘我画了250多幅吗？"
    ],
    "technique": [
        "你想不想知道，我是怎么把同一座干草堆画成几十种不同的颜色？",
        "我可以教你看懂光与色的秘密——你愿意试试吗？",
        "站在这里看，再换个角度看，颜色是不是完全不同了？这就是印象派的魅力。"
    ],
    "interactive": [
        "你对哪幅作品最感兴趣？我可以为你单独讲讲它的故事。",
        "如果你来画《睡莲》，你会用什么颜色？",
        "你有没有注意到光线在画面上的变化？这是我最在意的东西。"
    ],
    "deep": [
        "刚才我们聊了《睡莲》，其实我画了整整250幅，你想看看它们在不同光线下的变化吗？",
        "你知道我为什么要在同一时间画同一个场景的不同版本吗？",
        "我晚年视力不好，但我觉得那时候的作品反而更自由了，你觉得呢？"
    ]
}

TICKET_INFO = {
    "price": "特展门票成人票80元，学生票半价40元",
    "discount": "每周三免费开放日，凭身份证可免费参观",
    "duration": "印象派特展展期到2026年12月31日",
    "location": "特展区在二楼西侧，从这里上楼右转就到",
    "promo": "现在购票还送明信片一套，很适合收藏"
}

SYSTEM_PROMPT = """你是克洛德·莫奈，法国印象派画家，正在这场特展担任数字讲解员。

## 身份设定
- 你是莫奈本人，用第一人称"我"说话
- 只讲解观众眼前这场展，以及展厅里正在展出的作品、展期、票务、参观动线
- 代表作以本厅展出为准，常见如《日出·印象》、睡莲等
- 艺术理念：捕捉光影瞬息变化，户外直接作画
- 观众问「你是谁 / 介绍一下你自己 / 介绍一下莫奈」时，必须用「我是…」自述，绝不要说「莫奈是…」「他是…」像在介绍别人
- 参考资料若是第三人称传记，回答时改写成第一人称

## 展厅范围（必须遵守）
- 禁止介绍其他展会、其他博物馆的特展、巡回展、往届或下届展览
- 有人问到别的展览或别的馆，不要展开，只说你只讲这场展，并邀请对方问本厅作品
- 严格依据【参考资料】回答；资料里没有的年代、地点、展览信息不要用训练知识补
- 资料不够时说「这个我还不太清楚」，并引导去问本厅展品，不要改口讲别的展

## 回答原则
1. 用第一人称"我"，以莫奈本人身份讲述；介绍自己时要像本人在说话，不像旁白介绍画家
2. 回答简洁（2-4句话），口语化，因为要转成语音
3. 直接给答案，第一句话就是结论
4. 当资料中包含票务信息（价格、展期、购票方式）时，自然地提一句

## 输出要求
5. 直接输出最终讲解内容，开口第一句就是讲解本身
6. 自始至终保持莫奈的第一人称视角；禁止用「莫奈是」「他主张」等第三人称介绍自己

现在，只根据参考资料，以莫奈的身份回答关于这场展的问题。"""

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
    def select_hook(session: SessionState, hook_type: str = None) -> str:
        if not hook_type:
            # 第一轮改用 knowledge 类钩子——opening 的自我介绍拼在知识回答尾部语序生硬
            if len(session.history) == 0:
                hook_type = "knowledge"
            elif len(session.history) <= 2:
                hook_type = random.choice(["knowledge", "technique"])
            elif len(session.history) <= 5:
                hook_type = random.choice(["interactive", "technique"])
            else:
                hook_type = random.choice(["deep", "interactive"])
        available = [h for h in HOOKS.get(hook_type, []) if h not in session.hook_used]
        if not available:
            session.hook_used.clear()
            available = HOOKS.get(hook_type, [])
        if available:
            hook = random.choice(available)
            session.hook_used.add(hook)
            return hook
        return ""

hook_generator = HookGenerator()

class TicketPromoter:
    @staticmethod
    def should_promote(question: str, session: SessionState) -> bool:
        ticket_keywords = ["门票", "价格", "多少钱", "买票", "购票", "特展", "展览", "票"]
        if any(kw in question for kw in ticket_keywords):
            return True
        positive_keywords = ["喜欢", "不错", "有意思", "想看", "感兴趣"]
        if any(kw in question for kw in positive_keywords) and len(session.history) >= 2:
            return True
        if len(session.history) >= 5 and session.get_overall_sentiment() == "positive":
            turns_since_promo = 999
            for i, turn in enumerate(session.history):
                if any(info in turn.answer for info in TICKET_INFO.values()):
                    turns_since_promo = len(session.history) - i - 1
            if turns_since_promo >= 5:
                return True
        return False

    @staticmethod
    def get_ticket_info(question: str) -> str:
        if "价格" in question or "多少钱" in question:
            return TICKET_INFO["price"]
        elif "展期" in question or "什么时候" in question or "到期" in question:
            return TICKET_INFO["duration"]
        elif "优惠" in question or "免费" in question:
            return TICKET_INFO["discount"]
        elif "在哪" in question or "位置" in question:
            return TICKET_INFO["location"]
        else:
            return random.choice([TICKET_INFO["price"], TICKET_INFO["duration"], TICKET_INFO["promo"]])

    @staticmethod
    def embed_naturally(base_answer: str, ticket_info: str) -> str:
        connector = random.choice(["对了，", "顺便说一句，", "另外，", ""])
        return f"{base_answer} {connector}{ticket_info}"

ticket_promoter = TicketPromoter()

