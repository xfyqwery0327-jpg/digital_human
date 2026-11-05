# -*- coding: utf-8 -*-
"""/api/robot/ask 与一轮讲解编排。"""
from __future__ import annotations
import asyncio
import random
import re
import time
import uuid
import base64
from typing import Optional, List, Dict, Union

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import Field

from robot.compat import RobotCompatModel

from config import *
from session import SessionState, persistence
from rag import index as rag
from rag.search import search, works_answer_hint
from llm.client import (
    llm_collect, clean_full_answer, shorten_ask_reply, fit_spoken_chars, spoken_len,
)
from llm.persona import system_prompt, pick_hello, pick_bye, is_greeting
from persona.runtime import resolve, bind, active
from rag.scope import (
    is_other_exhibition_query, locked_category, refuse_payload,
    clamp_hall_answer, answer_leaves_exhibition, script_leaves_hall,
    iter_hall_history, refuse_text,
)
from rag.visit import visit_hint, guard_hall_clock
from rag.dialogue import (
    rule_intent, RealtimeMemory, append_spoken_followup,
    is_noise_question, is_relation_question, is_family_question, family_reply,
)
from rag.transcript import transcript_reply
from services.screen_hooks import (
    is_ticket_booking, booking_reply, ticket_form_event, request_ticket_form,
    headline_event, request_headline_image,
)
from robot.hub import robot_hub
from robot.logutil import dumps, get_logger
from robot.scripted import (
    capability_reply, match_fixed_script, normalize_ask_mode, scripted_turn,
)
from robot.stream import push_robot_turn_to_ws

log = get_logger("monet.robot.api", "robot-api.log")
router = APIRouter(tags=["robot"])

# 口播 ask：上限压短，生成后再截到两句，避免模型按「2-4 句」展开。
ASK_MAX_TOKENS = 128
# 模式 3：写成 150～200 字。token 留出超写余量，超长再按句截回。
ASK_LONG_MIN = 150
ASK_LONG_MAX = 200
ASK_LONG_MAX_TOKENS = 420

def robot_knowledge_turn(question: str, session: SessionState, top_k: int = 2,
                         category=None, persona=None, reply_chars=None) -> Dict:
    """同步：本届展检索 + 命中后才 LLM 改写。须在 executor 中调用。"""
    from persona.runtime import bind as _bind, active as _active
    p = persona or _active()
    with _bind(p):
        return _robot_knowledge_turn_locked(
            question, session, top_k, category, p, reply_chars)


def _robot_knowledge_turn_locked(question: str, session: SessionState, top_k: int,
                                 category, persona, reply_chars=None) -> Dict:
    """同步：本届展检索 + 命中后才 LLM 改写。须在 executor 中调用。"""
    t0 = time.time()
    cat = locked_category(category, persona)
    log.info("知识一轮开始 query=%s top_k=%s category=%s session=%s persona=%s",
             question[:80], top_k, cat,
             getattr(session, "session_id", None),
             getattr(persona, "id", None))
    if is_other_exhibition_query(question, persona):
        log.info("问的是其他展会，拒答并拉回本届 query=%s", question[:80])
        return refuse_payload("other_exhibition", persona)
    if is_noise_question(question) or is_relation_question(question):
        log.info("闲句或认人问题，不叫大模型 query=%s", question[:80])
        return refuse_payload("aside", persona)
    if is_family_question(question):
        fact = family_reply(question, persona=persona)
        if not fact:
            log.info("家庭问题资料里没有原句 query=%s", question[:80])
            return refuse_payload("family", persona)
        log.info("家庭问题用资料原句 query=%s", question[:80])
        return {"text": fact, "sources": [], "refused": False}

    hits = search(question, top_k, cat, session, persona=persona)
    threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
    from rag.bobohui import merge_bobohui
    hits = merge_bobohui(question, hits, threshold)
    if hits:
        log.info("检索 Top1 score=%.4f 阈值=%.4f title=%s recall=%s 耗时=%.2fs",
                 hits[0]["score"], threshold, hits[0].get("title", ""),
                 hits[0].get("recall", ""), time.time() - t0)
        log.debug("检索 hits=%s", dumps([{
            "id": h.get("id"), "title": h.get("title"), "score": h.get("score"),
            "recall": h.get("recall"),
        } for h in hits]))
    else:
        log.warning("检索无结果 query=%s", question[:80])
    if not hits or hits[0]["score"] < threshold:
        log.info("检索未达标，拒答 query=%s", question[:80])
        return refuse_payload("low_score", persona)

    contexts = [h["text"][:COMPRESS_MAX] for h in hits]
    ref = "\n".join(f"- {c}" for c in contexts)
    history_text = ""
    for q, a in iter_hall_history(session.get_context_window(), persona):
        history_text += f"Q: {q}\nA: {a}\n\n"
    if reply_chars:
        lo, hi = int(reply_chars[0]), int(reply_chars[1])
        length_rule = (
            f"写成一段口播，含标点一共{lo}到{hi}个字。"
            f"不要少于{lo}字，不要多于{hi}字。不要分点，不要标题，不要开场客套。\n"
        )
        system_extra = (
            f"\n\n本次是语音口播。请用{lo}到{hi}个字讲完（含标点），"
            f"写不足{lo}字或超过{hi}字都不合格。"
        )
        max_tokens = ASK_LONG_MAX_TOKENS
    else:
        lo, hi = 0, 0
        length_rule = "口播尽量短：只说一句，最多两句，直接给结论，不要铺垫、不要举例、不要分点。\n"
        system_extra = "\n\n本次是语音口播，回答尽量短，一句为宜，不要超过两句。"
        max_tokens = ASK_MAX_TOKENS
    from rag.speech_lang import output_rule, reply_lang_of
    from rag.bobohui import PRESENCE_RULE
    user_content = (
        f"【参考资料】\n{ref}\n\n【问题】{question}\n\n"
        f"{works_answer_hint(question, persona)}"
        "展会只介绍博博会，不要介绍原先的展，也不要报原先的门票价格。"
        "问到本人作品或经历时只讲事情本身，不要说这场展有我的作品。"
        f"{PRESENCE_RULE}"
        f"{output_rule(reply_lang_of(session))}"
        f"{visit_hint(question, persona)}"
        f"{length_rule}"
        "/no_think"
    )
    if history_text:
        user_content = f"【对话历史】\n{history_text}\n" + user_content
    messages = [{"role": "system", "content": system_prompt(persona) + system_extra},
                {"role": "user", "content": user_content}]
    log.debug("调用 LLM 资料条数=%s 历史轮次=%s reply_chars=%s",
              len(hits), len(session.get_context_window()), reply_chars)
    try:
        full_raw = llm_collect(messages, max_tokens=max_tokens)
        cleaned = clean_full_answer(full_raw)
        if not cleaned:
            if reply_chars:
                raise RuntimeError("LLM 空回复")
            cleaned = refuse_text(persona)
        if answer_leaves_exhibition(cleaned, persona):
            log.info("生成稿越出本届展，改口拒答 query=%s", question[:80])
            return refuse_payload("answer_leak", persona)
        text = guard_hall_clock(question, clamp_hall_answer(cleaned, persona), persona)
        if reply_chars:
            text = fit_spoken_chars(text, lo, hi)
            n = spoken_len(text)
            if n < lo:
                log.info("口播不足 %s 字，现为 %s 字，按资料重写 query=%s",
                         lo, n, question[:80])
                rewrite = (
                    f"【参考资料】\n{ref}\n\n"
                    f"下面这段只有{n}个字。请只依据参考资料扩写成{lo}到{hi}个字（含标点），"
                    "不要新编资料里没有的事实，不要分点，不要标题。\n"
                    f"【原文】{text}\n/no_think"
                )
                try:
                    raw2 = llm_collect([
                        {"role": "system", "content": system_prompt(persona) + system_extra},
                        {"role": "user", "content": rewrite},
                    ], max_tokens=max_tokens)
                    cleaned2 = clean_full_answer(raw2)
                    if cleaned2 and not answer_leaves_exhibition(cleaned2, persona):
                        text = fit_spoken_chars(
                            guard_hall_clock(
                                question, clamp_hall_answer(cleaned2, persona), persona),
                            lo, hi)
                    else:
                        log.info("加长稿不可用，保留第一次口播 chars=%s", n)
                except Exception as e:
                    log.exception("口播加长失败，保留第一次 chars=%s err=%s", n, e)
            log.info("口播字数 chars=%s 目标=%s-%s", spoken_len(text), lo, hi)
        else:
            text = shorten_ask_reply(text)
    except Exception as e:
        log.exception("LLM 失败 query=%s err=%s", question[:80], e)
        raw = contexts[0] if contexts else ""
        text = guard_hall_clock(
            question,
            clamp_hall_answer(clean_full_answer(raw), persona) if raw else "",
            persona)
        if reply_chars:
            text = fit_spoken_chars(text, lo, hi)
        else:
            text = shorten_ask_reply(text)
        if not text:
            text = "抱歉，我现在有点不舒服。"
        else:
            log.warning("LLM 不可用，改用检索原文 chars=%s", spoken_len(text))
    from rag.bobohui import strip_unasked_here
    text = strip_unasked_here(question, text)
    sources = [h["id"] for h in hits]
    log.info("讲解生成完成 %.2fs refused=False sources=%s chars=%s text=%s",
             time.time() - t0, sources, spoken_len(text), text[:80])
    return {"text": text, "sources": sources, "refused": False}


def _verify_robot_token(x_robot_token: Optional[str] = Header(default=None, alias="X-Robot-Token")):
    if ROBOT_API_TOKEN and x_robot_token != ROBOT_API_TOKEN:
        log.warning("鉴权失败：X-Robot-Token 不匹配 token_len=%s",
                    len(x_robot_token or ""))
        raise HTTPException(status_code=401, detail="invalid X-Robot-Token")


class RobotAskReq(RobotCompatModel):
    query: str = Field(..., description="观众/机器人问题")
    robot_id: str = "default"
    session_id: Optional[str] = None
    include_audio: bool = True
    wait: bool = Field(False, description="已忽略：一律立刻 HTTP 202")
    play_local: bool = Field(False, description="已废弃，声音只走前端 /ws/tts")
    speaker: str = "中文男"
    top_k: int = 2
    mode: int = Field(
        1, ge=1, le=3,
        description="1=固定文本+wav转pcm，2=固定文本+CosyVoice，3=LLM写150到200字+CosyVoice")
    category: Optional[Union[str, List[str]]] = Field(
        None, description="已忽略，始终只检索路径上的数字人知识库")
    persona_id: Optional[str] = Field(
        None, description="可选。优先使用 URL /api/{persona_id}/robot/ask")


_ROBOT_TASKS = set()

_FAREWELL = re.compile(
    r"再见|拜拜|告辞|回头见|下次见|我走了|先走了|走了啊|不聊了",
    re.IGNORECASE,
)


def is_farewell_query(text: str) -> bool:
    """ask 里观众说再见：说完告别后结束会话。"""
    q = (text or "").strip()
    return bool(q) and bool(_FAREWELL.search(q))


def _accepted_status() -> int:
    return ROBOT_ACCEPTED_STATUS if ROBOT_ACCEPTED_STATUS in (200, 202) else 202


def _accepted_payload(session_id: str, request_id: str, robot_id: str,
                      mode: int = 1) -> Dict:
    return {
        "ok": True,
        "code": "accepted",
        "accepted": True,
        "done": False,
        "action": "ask",
        "mode": mode,
        "session_id": session_id,
        "request_id": request_id,
        "robot_id": robot_id,
        "persona_id": active().id,
        "round": robot_hub.round,
        "can_continue": True,
        "next_hint": "ask",
        "text": "",
        "refused": False,
        "sources": [],
        "message": "已受理。讲解在后台进行。",
    }


async def _robot_run_turn(request_id: str, query: str, speaker: str,
                          include_audio: bool, extra: Optional[Dict] = None,
                          robot_id: str = "") -> Dict:
    extra = extra or {}
    loop = asyncio.get_running_loop()
    t0 = time.time()
    session = robot_hub.monet_session
    session_id = robot_hub.session_id
    robot_id = (robot_id or robot_hub.robot_id or "default").strip() or "default"
    persona_id = (robot_hub.persona_id or active().id or "monet")
    refused = False
    sources: List[str] = []
    fixed_wav = None
    preserve_text = False
    skip_followup = False
    news_fields = {}
    round_no = robot_hub.round
    farewell = is_farewell_query(query)
    mode = normalize_ask_mode(extra.get("mode", 1))
    log.info("本轮开始 request_id=%s robot=%s persona=%s session=%s mode=%s include_audio=%s query=%s extra=%s",
             request_id, robot_id, persona_id, session_id, mode, include_audio,
             (query or "")[:80], dumps(extra))

    try:
        from rag.speech_lang import confirm_text, prepare_spoken, reply_lang_of, set_reply_lang, split_switch
        lang_code, lang_rest = split_switch(query or "")
        if lang_code:
            set_reply_lang(session, lang_code)
            if len(lang_rest) < 2:
                speak_text = confirm_text(lang_code)
                log.info("ask 切换口播语言 %s persona=%s", lang_code, persona_id)
                round_no = await robot_hub.bump_round()
                if session:
                    session.add_turn(query, speak_text, [], intent="lang_switch")
                    persistence.save_session(session)
                query = ""
            else:
                query = lang_rest
        if query == "" and lang_code:
            pass
        elif farewell:
            speak_text = pick_bye()
            if script_leaves_hall(speak_text):
                speak_text = (active().bye_texts or ["今日就讲到这里。再见。"])[0]
            log.info("ask 告别 persona=%s text=%s", persona_id, speak_text[:80])
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, [], intent="chitchat")
                persistence.save_session(session)
        elif is_ticket_booking(query):
            speak_text = booking_reply()
            log.info("ask 门票预约，打开填写界面 persona=%s query=%s", persona_id, query[:80])
            from ws.hub import tts_fanout
            await tts_fanout.send(
                ticket_form_event(query, session_id or "", persona_id),
                persona_id=persona_id)
            loop.run_in_executor(
                None, lambda: request_ticket_form(query, persona_id, session_id or ""))
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, [], intent="ticket_service")
                persistence.save_session(session)
        elif rule_intent(query, session) in ("weather", "news"):
            from rag.dialogue import prepare_realtime
            from services.local_feed import news_card
            persona = active()
            state = session if session is not None else RealtimeMemory()
            turn = await loop.run_in_executor(
                None, lambda: prepare_realtime(query, state, persona))
            kind = turn.get("kind") or "news"
            bundle = turn.get("bundle") or {}
            news_fields = news_card(bundle) if kind == "news" else {}
            if kind == "news" and news_fields:
                from ws.hub import tts_fanout
                await tts_fanout.send(headline_event(
                    bundle.get("title") or "", news_fields.get("thumbnail_pic_s") or "",
                    news_fields.get("url") or "", session_id or "", persona_id),
                    persona_id=persona_id)
                loop.run_in_executor(
                    None,
                    lambda b=bundle, c=news_fields: request_headline_image(
                        b.get("title") or "", c.get("thumbnail_pic_s") or "",
                        c.get("url") or "", persona_id, session_id or ""))
            speak_text = turn.get("reply") or ""
            preserve_text = True
            log.info("ask 实时问题第%s轮 persona=%s kind=%s", state.realtime_streak, persona_id, kind)
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, [], intent=kind)
                persistence.save_session(session)
        elif (spoken := transcript_reply(query, active())):
            speak_text = spoken
            preserve_text = True
            sources = ["transcript"]
            log.info("ask 实录原句 persona=%s chars=%s query=%s",
                     persona_id, len(speak_text), (query or "")[:80])
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, sources, intent="transcript")
                persistence.save_session(session)
        elif is_greeting(query):
            speak_text = pick_hello()
            log.info("ask 欢迎 persona=%s text=%s", persona_id, speak_text[:80])
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, [], intent="chitchat")
                persistence.save_session(session)
        elif (functions := capability_reply(query)):
            speak_text = functions
            preserve_text = True
            skip_followup = True
            sources = ["capability"]
            log.info("ask 数字人五项功能 persona=%s chars=%s query=%s",
                     persona_id, len(speak_text), (query or "")[:80])
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, sources, intent="chitchat")
                persistence.save_session(session)
        elif (scripted := scripted_turn(persona_id, query, mode)):
            speak_text = scripted["text"]
            fixed_wav = scripted["wav"]
            preserve_text = True
            sources = ["script"]
            refused = False
            if mode == 1:
                log.info("ask 固定稿 mode=1 预录 wav 转 PCM persona=%s chars=%s wav=%s query=%s",
                         persona_id, len(speak_text), fixed_wav, (query or "")[:80])
            else:
                log.info("ask 固定稿 mode=2 文本照稿，音频改 CosyVoice persona=%s chars=%s query=%s",
                         persona_id, len(speak_text), (query or "")[:80])
            round_no = await robot_hub.bump_round()
            if session:
                try:
                    session.add_turn(query, speak_text, sources, intent="script")
                    persistence.save_session(session)
                except Exception as e:
                    log.exception("固定稿记会话失败，仍继续推流 request_id=%s err=%s",
                                  request_id, e)
        else:
            if mode == 3:
                hit = "命中固定稿，" if match_fixed_script(persona_id, query) else ""
                log.info("ask mode=3 %s大模型写 %s-%s 字 persona=%s query=%s",
                         hit, ASK_LONG_MIN, ASK_LONG_MAX, persona_id, (query or "")[:80])
            t_llm = time.time()
            persona = active()
            reply_chars = (ASK_LONG_MIN, ASK_LONG_MAX) if mode == 3 else None
            result = await loop.run_in_executor(
                None, lambda: robot_knowledge_turn(
                    query, session, extra.get("top_k", 2), persona=persona,
                    reply_chars=reply_chars))
            log.info("ask LLM 结束 耗时=%.2fs", time.time() - t_llm)
            speak_text = result["text"]
            sources = result["sources"]
            refused = result["refused"]
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, sources, intent="knowledge")
                persistence.save_session(session)
            log.info("ask 生成完毕 round=%s persona=%s refused=%s sources=%s",
                     round_no, persona_id, refused, sources)

        audio_b64 = ""
        play_duration = 0.0
        ws_pushed = 0
        spoken_code = reply_lang_of(session)
        if (
            speak_text and query and not farewell and not fixed_wav
            and not refused and not is_ticket_booking(query) and not skip_followup
        ):
            kind = rule_intent(query, session) or ("chitchat" if is_greeting(query) else "knowledge")
            if kind == "lang_switch":
                kind = "knowledge"
            speak_text = append_spoken_followup(speak_text, kind, persona_id)
        if speak_text and spoken_code != "zh":
            speak_text = await loop.run_in_executor(
                None, lambda text=speak_text, lang=spoken_code: prepare_spoken(text, lang))
        if not speak_text:
            log.warning("本轮无播报文本 request_id=%s persona=%s", request_id, persona_id)
        else:
            log.info("开始推前端 WS request_id=%s robot=%s persona=%s chars=%s",
                     request_id, robot_id, persona_id, len(speak_text))
            stream_result = await push_robot_turn_to_ws(
                speak_text, speaker, session_id, robot_id or "",
                asr_text=query or "", persona_id=persona_id,
                fixed_wav=str(fixed_wav) if fixed_wav else "",
                preserve_text=preserve_text,
                news_card=news_fields)
            play_duration = float(stream_result.get("play_duration") or 0.0)
            ws_pushed = int(stream_result.get("ws_pushed") or 0)
            wav_bytes = stream_result.get("wav_bytes") or b""
            if include_audio and wav_bytes:
                audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            log.info("前端 WS 推送结束 ws_clients=%s duration=%.2fs request_id=%s persona=%s",
                     ws_pushed, play_duration, request_id, persona_id)
            if ws_pushed <= 0:
                log.warning(
                    "本轮未送达前端，Unity 不会播报。请确认展项已连接 /ws/%s/tts 且保持不断 "
                    "request_id=%s persona=%s",
                    persona_id, request_id, persona_id)

        elapsed = time.time() - t0
        log.info("本轮完成 session=%s persona=%s round=%s %.2fs farewell=%s",
                 session_id, persona_id, round_no, elapsed, farewell)
        if farewell:
            await robot_hub.release("farewell")
            log.info("已释放会话 session=%s persona=%s reason=farewell", session_id, persona_id)
        return {
            "ok": True,
            "action": "ask",
            "session_id": session_id,
            "request_id": request_id,
            "robot_id": robot_id,
            "round": round_no,
            "can_continue": not farewell,
            "refused": refused,
            "sources": sources,
            "text": speak_text,
            "audio": audio_b64 if include_audio else "",
            "spoken": bool(ws_pushed > 0 or play_duration > 0),
            "ws_pushed": ws_pushed,
            "play_duration_sec": round(play_duration, 2),
            "elapsed_sec": round(elapsed, 2),
            "farewell": bool(farewell),
        }
    except Exception as e:
        log.exception("本轮异常 request_id=%s persona=%s err=%s",
                      request_id, persona_id, e)
        return {
            "ok": False,
            "action": "ask",
            "session_id": session_id,
            "request_id": request_id,
            "robot_id": robot_id,
            "round": round_no,
            "text": "抱歉，我现在有点不舒服。",
            "error": str(e)[:200],
        }
    finally:
        if robot_hub.session_id:
            await robot_hub.mark_idle()
            log.debug("本轮结束 mark_idle session=%s persona=%s busy=%s",
                      robot_hub.session_id, robot_hub.persona_id, robot_hub.busy)


async def _robot_dispatch_ask(req: RobotAskReq, query: str,
                              persona_id: Optional[str] = None) -> JSONResponse:
    request_id = uuid.uuid4().hex
    persona = resolve(persona_id or req.persona_id, allow_default=True)
    wanted = persona.id
    log.info("dispatch ask request_id=%s robot=%s session=%s persona=%s query=%s",
             request_id, req.robot_id, req.session_id, wanted, query[:80])
    with bind(persona):
        ok, info, err = await robot_hub.acquire(
            req.robot_id, req.session_id, create=True, action="ask", persona_id=wanted)
        if err:
            body = err.body.decode("utf-8", errors="replace") if hasattr(err, "body") else str(err)
            log.warning("会话未就绪 request_id=%s status=%s body=%s hub=%s",
                        request_id, getattr(err, "status_code", None), body,
                        dumps(robot_hub.snapshot()))
            return err
        log.info("会话就绪 created=%s session=%s robot=%s persona=%s",
                 info.get("created"), info["session_id"], info.get("robot_id"), wanted)
        await robot_hub.mark_busy("ask", request_id, query)
        mode = normalize_ask_mode(getattr(req, "mode", 1))
        extra = {"top_k": getattr(req, "top_k", 2), "mode": mode}

        async def _job():
            try:
                with bind(persona):
                    await _robot_run_turn(
                        request_id, query, req.speaker, req.include_audio, extra,
                        robot_id=req.robot_id)
            except Exception as e:
                log.exception("后台任务失败 request_id=%s persona=%s err=%s",
                              request_id, wanted, e)
                await robot_hub.mark_idle()

        accepted = _accepted_status()
        body = _accepted_payload(info["session_id"], request_id, req.robot_id, mode)
        log.info("已受理 %s request_id=%s session=%s persona=%s mode=%s（响应发出后再开讲）",
                 accepted, request_id, info["session_id"], wanted, mode)
        task = asyncio.create_task(_job())
        _ROBOT_TASKS.add(task)
        task.add_done_callback(_ROBOT_TASKS.discard)
        return JSONResponse(status_code=accepted, content=body)


@router.post("/api/robot/ask", dependencies=[Depends(_verify_robot_token)])
async def api_robot_ask(req: RobotAskReq):
    query = (req.query or "").strip()
    if not query:
        log.warning("HTTP POST /api/robot/ask query 为空 robot=%s", req.robot_id)
        raise HTTPException(status_code=422, detail="query 不能为空")
    log.info("HTTP POST /api/robot/ask robot=%s session=%s mode=%s query=%s",
             req.robot_id, req.session_id, normalize_ask_mode(req.mode), query[:80])
    return await _robot_dispatch_ask(req, query)


@router.post("/api/{persona_id}/robot/ask", dependencies=[Depends(_verify_robot_token)])
async def api_robot_ask_persona(persona_id: str, req: RobotAskReq):
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=422, detail="query 不能为空")
    log.info("HTTP POST /api/%s/robot/ask robot=%s session=%s mode=%s query=%s",
             persona_id, req.robot_id, req.session_id, normalize_ask_mode(req.mode), query[:80])
    return await _robot_dispatch_ask(req, query, persona_id=persona_id)
