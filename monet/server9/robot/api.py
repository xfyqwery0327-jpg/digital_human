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
from starlette.background import BackgroundTask
from pydantic import Field

from robot.compat import RobotCompatModel

from config import *
from session import SessionState, persistence
from rag import index as rag
from rag.search import search
from llm.client import llm_collect, clean_full_answer
from llm.persona import SYSTEM_PROMPT
from rag.scope import (
    is_other_exhibition_query, locked_category, refuse_payload,
    clamp_hall_answer, answer_leaves_exhibition, script_leaves_hall,
    iter_hall_history,
)
from robot.hub import robot_hub
from robot.logutil import dumps, get_logger
from robot.stream import push_robot_turn_to_ws

log = get_logger("monet.robot.api", "robot-api.log")
router = APIRouter(tags=["robot"])

def robot_knowledge_turn(question: str, session: SessionState, top_k: int = 2,
                         category=None) -> Dict:
    """同步：本届展检索 + 命中后才 LLM 改写。须在 executor 中调用。"""
    t0 = time.time()
    cat = locked_category(category)
    log.info("知识一轮开始 query=%s top_k=%s category=%s session=%s",
             question[:80], top_k, cat,
             getattr(session, "session_id", None))
    if is_other_exhibition_query(question):
        log.info("问的是其他展会，拒答并拉回本届 query=%s", question[:80])
        return refuse_payload("other_exhibition")

    hits = search(question, top_k, cat, session)
    threshold = MIN_SCORE_RERANK if rag.HAS_RERANKER else MIN_SCORE_RRF
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
        return refuse_payload("low_score")

    contexts = [h["text"][:COMPRESS_MAX] for h in hits]
    ref = "\n".join(f"- {c}" for c in contexts)
    history_text = ""
    for q, a in iter_hall_history(session.get_context_window()):
        history_text += f"Q: {q}\nA: {a}\n\n"
    user_content = (
        f"【参考资料】\n{ref}\n\n【问题】{question}\n\n"
        "只根据参考资料讲这场展，不要提及其他展览。\n/no_think"
    )
    if history_text:
        user_content = f"【对话历史】\n{history_text}\n" + user_content
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}]
    log.debug("调用 LLM 资料条数=%s 历史轮次=%s", len(hits), len(session.get_context_window()))
    try:
        full_raw = llm_collect(messages)
        cleaned = clean_full_answer(full_raw) or REFUSE_TEXT
        if answer_leaves_exhibition(cleaned):
            log.info("生成稿越出本届展，改口拒答 query=%s", question[:80])
            return refuse_payload("answer_leak")
        text = clamp_hall_answer(cleaned)
    except Exception as e:
        log.exception("LLM 失败 query=%s err=%s", question[:80], e)
        text = "抱歉，我现在有点不舒服。"
    sources = [h["id"] for h in hits]
    log.info("讲解生成完成 %.2fs refused=False sources=%s text=%s",
             time.time() - t0, sources, text[:80])
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
    category: Optional[Union[str, List[str]]] = Field(
        None, description="已忽略，始终只检索本届展 exhibition_monet")


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


def _accepted_payload(session_id: str, request_id: str, robot_id: str) -> Dict:
    return {
        "ok": True,
        "code": "accepted",
        "accepted": True,
        "done": False,
        "action": "ask",
        "session_id": session_id,
        "request_id": request_id,
        "robot_id": robot_id,
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
    refused = False
    sources: List[str] = []
    round_no = robot_hub.round
    farewell = is_farewell_query(query)
    log.info("本轮开始 request_id=%s robot=%s session=%s include_audio=%s query=%s extra=%s",
             request_id, robot_id, session_id, include_audio,
             (query or "")[:80], dumps(extra))

    try:
        if farewell:
            speak_text = random.choice(ROBOT_BYE_TEXTS)
            if script_leaves_hall(speak_text):
                speak_text = ROBOT_BYE_TEXTS[0]
            log.info("ask 告别 text=%s", speak_text[:80])
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, [], intent="chitchat")
                persistence.save_session(session)
        else:
            t_llm = time.time()
            result = await loop.run_in_executor(
                None, lambda: robot_knowledge_turn(
                    query, session, extra.get("top_k", 2)))
            log.info("ask LLM 结束 耗时=%.2fs", time.time() - t_llm)
            speak_text = result["text"]
            sources = result["sources"]
            refused = result["refused"]
            round_no = await robot_hub.bump_round()
            if session:
                session.add_turn(query, speak_text, sources, intent="knowledge")
                persistence.save_session(session)
            log.info("ask 生成完毕 round=%s refused=%s sources=%s", round_no, refused, sources)

        audio_b64 = ""
        play_duration = 0.0
        ws_pushed = 0
        if not speak_text:
            log.warning("本轮无播报文本 request_id=%s", request_id)
        else:
            log.info("开始推前端 WS request_id=%s robot=%s chars=%s",
                     request_id, robot_id, len(speak_text))
            stream_result = await push_robot_turn_to_ws(
                speak_text, speaker, session_id, robot_id or "",
                asr_text=query or "")
            play_duration = float(stream_result.get("play_duration") or 0.0)
            ws_pushed = int(stream_result.get("ws_pushed") or 0)
            wav_bytes = stream_result.get("wav_bytes") or b""
            if include_audio and wav_bytes:
                audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
            log.info("前端 WS 推送结束 ws_clients=%s duration=%.2fs request_id=%s",
                     ws_pushed, play_duration, request_id)
            if ws_pushed <= 0:
                log.warning("本轮未送达前端，Unity 不会播报。请确认展项已连接 /ws/tts 且保持不断 request_id=%s",
                            request_id)

        elapsed = time.time() - t0
        log.info("本轮完成 session=%s round=%s %.2fs farewell=%s",
                 session_id, round_no, elapsed, farewell)
        if farewell:
            await robot_hub.release("farewell")
            log.info("已释放会话 session=%s reason=farewell", session_id)
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
        log.exception("本轮异常 request_id=%s err=%s", request_id, e)
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
            log.debug("本轮结束 mark_idle session=%s busy=%s", robot_hub.session_id, robot_hub.busy)


async def _robot_dispatch_ask(req: RobotAskReq, query: str) -> JSONResponse:
    request_id = uuid.uuid4().hex
    log.info("dispatch ask request_id=%s robot=%s session=%s query=%s",
             request_id, req.robot_id, req.session_id, query[:80])
    ok, info, err = await robot_hub.acquire(
        req.robot_id, req.session_id, create=True, action="ask")
    if err:
        body = err.body.decode("utf-8", errors="replace") if hasattr(err, "body") else str(err)
        log.warning("会话未就绪 request_id=%s status=%s body=%s hub=%s",
                    request_id, getattr(err, "status_code", None), body,
                    dumps(robot_hub.snapshot()))
        return err
    log.info("会话就绪 created=%s session=%s robot=%s",
             info.get("created"), info["session_id"], info.get("robot_id"))
    await robot_hub.mark_busy("ask", request_id, query)
    extra = {"top_k": getattr(req, "top_k", 2)}

    async def _job():
        try:
            await _robot_run_turn(
                request_id, query, req.speaker, req.include_audio, extra,
                robot_id=req.robot_id)
        except Exception as e:
            log.exception("后台任务失败 request_id=%s err=%s", request_id, e)
            await robot_hub.mark_idle()

    accepted = _accepted_status()
    body = _accepted_payload(info["session_id"], request_id, req.robot_id)
    log.info("已受理 %s request_id=%s session=%s（响应发出后再开讲）",
             accepted, request_id, info["session_id"])
    return JSONResponse(
        status_code=accepted,
        content=body,
        background=BackgroundTask(_job),
    )


@router.post("/api/robot/ask", dependencies=[Depends(_verify_robot_token)])
async def api_robot_ask(req: RobotAskReq):
    query = (req.query or "").strip()
    if not query:
        log.warning("HTTP POST /api/robot/ask query 为空 robot=%s", req.robot_id)
        raise HTTPException(status_code=422, detail="query 不能为空")
    log.info("HTTP POST /api/robot/ask robot=%s session=%s query=%s",
             req.robot_id, req.session_id, query[:80])
    return await _robot_dispatch_ask(req, query)
