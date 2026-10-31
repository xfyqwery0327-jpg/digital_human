# -*- coding: utf-8 -*-
"""天气默认海口，问到地名再查那一座城市。新闻走聚合头条，缓存到次日早上六点。"""
from __future__ import annotations
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Optional

import requests

from config import (
    DEFAULT_WEATHER_CITY, JUHE_NEWS_CONTENT_URL, JUHE_NEWS_KEY, JUHE_NEWS_LIST_URL,
    runtime_log,
)
from tts.speaktext import to_spoken_zh

_LOCK = threading.Lock()
_PLACE = {"at": 0.0, "value": None}
_FACTS = {}
_PLACE_TTL = 1800
_FACT_TTL = 600
_TIMEOUT = 4
_SH = timezone(timedelta(hours=8))
_NEWS = {"until": 0.0, "items": [], "cursor": 0}
_WEATHER = {}
_RECENT_NEWS = []
_RECENT_NEWS_TTL = 30 * 60
_CITY_FILLER = re.compile(
    r"请问|帮我|查一下|查询|一下|怎么样|如何|怎样|多少|会不会|是否|"
    r"今天|今日|现在|当前|天气|气温|温度|下雨|降雨|降温|升温|预报|情况|"
    r"这边|当地|这里|那里|吗|呢|啊|呀|吧|的"
)

_UA = {"User-Agent": "Mozilla/5.0"}


def clean_city(name: str) -> str:
    text = (name or "").strip()
    if text.endswith("市") and len(text) > 2:
        return text[:-1]
    return text


def weather_line(city: str, code, low, high) -> str:
    """一天一句。缺气温或城市时返回空，避免口播里出现编造的数字。"""
    city = clean_city(city)
    if not city or low is None or high is None:
        return ""
    try:
        lo = int(round(float(low)))
        hi = int(round(float(high)))
    except (TypeError, ValueError):
        return ""
    if lo > hi:
        lo, hi = hi, lo
    cond = _condition(code)
    body = "%s今天%s，%s到%s摄氏度。" % (city, cond, lo, hi) if cond else (
        "%s今天%s到%s摄氏度。" % (city, lo, hi))
    return to_spoken_zh(body)


def news_line(city: str, title: str) -> str:
    """只留一条当地标题，后面由讲解把话拉回本厅。"""
    city = clean_city(city)
    title = _trim_title(title)
    if not city or not title:
        return ""
    return to_spoken_zh("%s这边有一条消息，%s" % (city, title))


def headline_line(title: str) -> str:
    """头条只念标题。图片地址不进这句话。"""
    title = _trim_title(title)
    if not title:
        return ""
    return to_spoken_zh("今天的头条是，%s" % title)


def headline_from_item(item: dict) -> dict:
    """从一条新闻里取出可念的标题和原文 jpg。没有 jpg 时 image_url 为空。"""
    empty = {"text": "", "title": "", "image_url": "", "link": ""}
    if not isinstance(item, dict):
        return empty
    raw_title = item.get("title") or ""
    title = _trim_title(raw_title).rstrip("。")
    image = item.get("img")
    url = ""
    if isinstance(image, dict):
        url = image.get("u") or image.get("url") or ""
    elif isinstance(image, str):
        url = image
    url = (url or "").replace("\\/", "/").strip()
    if ".jpg" not in url.lower():
        url = ""
    link = (item.get("url") or item.get("wapurl") or "").strip()
    return {
        "text": headline_line(raw_title),
        "title": title,
        "image_url": url,
        "link": link,
    }


def feed_sentence(kind: str, question: str = "", avoid_keys=None) -> str:
    """只要口播句子。头条图片用 feed_bundle。"""
    return (feed_bundle(kind, question, avoid_keys).get("text") or "")


def news_card(bundle: dict) -> dict:
    """列表接口里的 url、thumbnail_pic_s。没有图或链接时返回空。"""
    if not isinstance(bundle, dict):
        return {}
    url = (bundle.get("url") or bundle.get("link") or "").strip()
    pic = (bundle.get("thumbnail_pic_s") or bundle.get("image_url") or "").strip()
    if not url and not pic:
        return {}
    return {"url": url, "thumbnail_pic_s": pic}


def feed_bundle(kind: str, question: str = "", avoid_keys=None) -> dict:
    """weather 带口播。news 另带标题、列表 url 和 thumbnail_pic_s。失败时文本为空。"""
    empty = {
        "text": "", "title": "", "image_url": "", "link": "",
        "url": "", "thumbnail_pic_s": "", "uniquekey": "",
    }
    kind = "news" if kind == "news" else "weather"
    try:
        if kind == "news":
            bundle = _pick_news(avoid_keys)
        else:
            bundle = _weather_bundle(question)
        if not isinstance(bundle, dict):
            bundle = dict(empty, text=bundle or "")
        runtime_log("feed.log", "kind=%s ok=%s image=%s key=%s" % (
            kind, bool(bundle.get("text")), bool(bundle.get("thumbnail_pic_s")),
            bundle.get("uniquekey") or bundle.get("title") or ""))
        return bundle
    except Exception as exc:
        runtime_log("feed.log", "kind=%s err=%s" % (kind, exc))
        return empty


def next_beijing_clock(hour: int, now: datetime = None) -> datetime:
    """下一次北京时间的整点。已经过了今天这个点，就取明天。"""
    now = now or datetime.now(_SH)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_SH)
    else:
        now = now.astimezone(_SH)
    point = now.replace(hour=int(hour), minute=0, second=0, microsecond=0)
    if now >= point:
        point += timedelta(days=1)
    return point


def extract_weather_city(question: str) -> str:
    """从问句里取出地名。没有地名时返回空，调用方再用默认城市。"""
    text = _CITY_FILLER.sub("", question or "")
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    text = re.sub(r"会$", "", text)
    for suffix in ("特别行政区", "自治区", "市", "省"):
        if text.endswith(suffix) and len(text) > len(suffix) + 1:
            text = text[:-len(suffix)]
            break
    if not text or len(text) < 2 or len(text) > 8:
        return ""
    return text


_LINK = re.compile(
    r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%\-]+|www\.[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%\-]+",
    re.I,
)


def strip_spoken_links(text: str) -> str:
    """链接只留给前端参数，口播里不念网址。"""
    return _LINK.sub("", text or "")


def plain_news(html: str) -> str:
    text = re.sub(r"(?is)<script[\s\S]*?</script>|<style[\s\S]*?</style>", "", html or "")
    text = re.sub(r"(?is)<[^>]+>", "", text)
    for src, dst in (("&nbsp;", ""), ("&amp;", "与"), ("&lt;", ""), ("&gt;", ""), ("&quot;", "")):
        text = text.replace(src, dst)
    text = strip_spoken_links(text)
    return re.sub(r"\s+", "", text)


def anchor_script(title: str, content: str, lo: int = 200, hi: int = 250) -> str:
    """口播用聊天的口气概括到 lo 到 hi 个字。网址不进这句话。"""
    title = strip_spoken_links(title or "")
    title = re.sub(r"\s+", "", title.replace("——", "，").replace("—", "，"))
    title = title.strip("，,。！？")
    body = to_spoken_zh(plain_news(content))
    lead = "那现在为你播报下今日头条新闻。"
    if title:
        lead += to_spoken_zh(title) + "。"
    parts = [p.strip() for p in re.split(r"(?<=[。！？])", body) if p.strip()]
    text = lead
    for part in parts:
        piece = part if part[-1] in "。！？" else part + "。"
        if piece and piece in text:
            continue
        if len(text) >= lo and len(text) + len(piece) > hi:
            break
        text += piece
        if len(text) >= hi:
            break
    for pad in (
            "这条先说到这里。",
            "后面的情况，公开报道还会接着说。",
            "您要是还想听，可以再让我播一条。",
    ):
        if len(text) >= lo:
            break
        if len(text) + len(pad) <= hi:
            text += pad
    if len(text) > hi:
        cut = text[:hi]
        pos = max(cut.rfind(mark) for mark in ("。", "！", "？"))
        if pos >= lo - 1:
            text = cut[:pos + 1]
        else:
            text = cut[:hi - 1] + "。"
    if text and text[-1] not in "。！？":
        if len(text) < hi:
            text += "。"
        else:
            text = text[:hi - 1] + "。"
    return text


def select_news(items, avoid, cursor: int = 0):
    """跳过短时间里已经播过的 uniquekey。全都播过才允许再从头挑。"""
    rows = [it for it in items or [] if isinstance(it, dict) and (it.get("plain") or it.get("title"))]
    fresh = [it for it in rows if (it.get("uniquekey") or "") not in set(avoid or [])]
    pool = fresh or rows
    if not pool:
        return None, cursor
    idx = int(cursor or 0) % len(pool)
    return pool[idx], int(cursor or 0) + 1


def _recent_news_keys(now: float = None):
    now = time.time() if now is None else now
    with _LOCK:
        kept = [(key, ts) for key, ts in _RECENT_NEWS if now - ts < _RECENT_NEWS_TTL]
        _RECENT_NEWS[:] = kept
        return {key for key, _ts in kept}


def _note_recent_news(key: str) -> None:
    if not key:
        return
    now = time.time()
    with _LOCK:
        _RECENT_NEWS.append((key, now))


def _pick_news(avoid_keys) -> dict:
    empty = {
        "text": "", "title": "", "image_url": "", "link": "",
        "url": "", "thumbnail_pic_s": "", "uniquekey": "",
    }
    items = _news_items()
    avoid = set(avoid_keys or []) | _recent_news_keys()
    with _LOCK:
        cursor = int(_NEWS.get("cursor") or 0)
    item, cursor = select_news(items, avoid, cursor)
    with _LOCK:
        _NEWS["cursor"] = cursor
    if not item:
        return empty
    _note_recent_news(item.get("uniquekey") or "")
    pic = item.get("thumbnail_pic_s") or ""
    url = item.get("url") or ""
    return {
        "text": anchor_script(item.get("title") or "", item.get("plain") or ""),
        "title": item.get("title") or "",
        "image_url": pic,
        "link": url,
        "url": url,
        "thumbnail_pic_s": pic,
        "uniquekey": item.get("uniquekey") or "",
    }


def _news_items():
    now = time.time()
    with _LOCK:
        if _NEWS["items"] and now < float(_NEWS["until"] or 0):
            return list(_NEWS["items"])
    rows = _fetch_news_rows()
    if not rows:
        return []
    until = next_beijing_clock(6).timestamp()
    with _LOCK:
        _NEWS["items"] = rows
        _NEWS["until"] = until
        _NEWS["cursor"] = 0
    runtime_log("feed.log", "news cache n=%s until=%s" % (len(rows), int(until)))
    return rows


def _fetch_news_rows():
    resp = requests.get(
        JUHE_NEWS_LIST_URL,
        params={
            "type": "top",
            "key": JUHE_NEWS_KEY,
            "page": 1,
            "page_size": 30,
            "is_filter": 1,
        },
        timeout=8, headers=_UA)
    resp.raise_for_status()
    payload = resp.json() or {}
    if int(payload.get("error_code") or 0) != 0:
        runtime_log("feed.log", "news list reason=%s" % (payload.get("reason") or ""))
        return []
    data = ((payload.get("result") or {}).get("data") or [])[:30]
    if not data:
        return []

    def one(item):
        key = (item.get("uniquekey") or "").strip()
        detail = _fetch_news_detail(key) if key else ""
        return {
            "uniquekey": key,
            "title": (item.get("title") or "").strip(),
            "url": (item.get("url") or "").strip(),
            "thumbnail_pic_s": (item.get("thumbnail_pic_s") or "").strip(),
            "plain": detail,
        }

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, data))


def _fetch_news_detail(uniquekey: str) -> str:
    try:
        resp = requests.get(
            JUHE_NEWS_CONTENT_URL,
            params={"uniquekey": uniquekey, "key": JUHE_NEWS_KEY},
            timeout=8, headers=_UA)
        resp.raise_for_status()
        payload = resp.json() or {}
        if int(payload.get("error_code") or 0) != 0:
            return ""
        return plain_news((payload.get("result") or {}).get("content") or "")
    except Exception as exc:
        runtime_log("feed.log", "news detail err=%s" % exc)
        return ""


def _weather_bundle(question: str) -> dict:
    empty = {
        "text": "", "title": "", "image_url": "", "link": "",
        "url": "", "thumbnail_pic_s": "", "uniquekey": "",
    }
    city = extract_weather_city(question) or clean_city(DEFAULT_WEATHER_CITY) or "海口"
    now = time.time()
    with _LOCK:
        cached = _WEATHER.get(city)
        if cached and now < float(cached.get("until") or 0):
            return dict(cached["bundle"])
    place = {"city": city}
    lat, lon = _geocode(city)
    place["lat"] = lat
    place["lon"] = lon
    bundle = dict(empty, text=_weather(place))
    if bundle.get("text"):
        until = next_beijing_clock(0).timestamp()
        with _LOCK:
            _WEATHER[city] = {"until": until, "bundle": dict(bundle)}
        runtime_log("feed.log", "weather city=%s until=%s" % (city, int(until)))
    else:
        runtime_log("feed.log", "weather city=%s empty" % city)
    return bundle


def _condition(code) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return ""
    if code == 0:
        return "晴"
    if code in (1, 2):
        return "多云"
    if code == 3:
        return "阴"
    if code in (45, 48):
        return "有雾"
    if code in (51, 53, 55, 56, 57):
        return "有小雨"
    if code in (61, 63, 66, 80, 81):
        return "有雨"
    if code in (65, 67, 82):
        return "有大雨"
    if code in (71, 73, 75, 77, 85, 86):
        return "有雪"
    if code in (95, 96, 99):
        return "有雷雨"
    return ""


def _trim_title(title: str) -> str:
    raw = re.sub(r"\s+", "", (title or "").replace("——", "，").replace("—", "，"))
    clipped = raw.endswith("...") or raw.endswith("…")
    text = raw.replace("...", "").replace("…", "").strip("，,、-|— ")
    if clipped:
        cut = max(text.rfind(mark) for mark in ("。", "！", "？", "，", "——"))
        if cut >= 6:
            text = text[:cut]
    text = text.strip("，,、-|— ")
    if len(text) > 36:
        cut = max(text.rfind(mark, 0, 36) for mark in ("。", "！", "？", "，", "——"))
        text = text[:cut] if cut >= 8 else text[:36]
        text = text.strip("，,、-|— ")
    if len(text) < 6:
        return ""
    if text[-1] not in "。！？":
        text += "。"
    return text


def _cached_fact(kind: str, city: str):
    with _LOCK:
        item = _FACTS.get((kind, city))
        if item and time.time() - item["at"] < _FACT_TTL:
            return item["text"]
    return None


def _store_fact(kind: str, city: str, text: str) -> None:
    with _LOCK:
        _FACTS[(kind, city)] = {"at": time.time(), "text": text}


def _place() -> Optional[dict]:
    forced = clean_city(os.environ.get("REALTIME_CITY", ""))
    with _LOCK:
        if not forced and _PLACE["value"] and time.time() - _PLACE["at"] < _PLACE_TTL:
            return _PLACE["value"]
    city = forced or _city_from_ip()
    if not city:
        return None
    lat, lon = _geocode(city)
    value = {"city": city, "lat": lat, "lon": lon}
    with _LOCK:
        _PLACE["at"] = time.time()
        _PLACE["value"] = value
    runtime_log("feed.log", "place city=%s geo=%s" % (city, lat is not None))
    return value


def _city_from_ip() -> str:
    city = _pconline_city() or _ipip_city()
    return clean_city(city)


def _pconline_city() -> str:
    try:
        resp = requests.get(
            "https://whois.pconline.com.cn/ipJson.jsp?json=true",
            timeout=_TIMEOUT, headers=_UA)
        text = resp.content.decode("gbk", errors="replace")
        matched = re.search(r"\{.*\}", text, re.S)
        if not matched:
            return ""
        data = json.loads(matched.group(0))
        return (data.get("city") or data.get("pro") or "").strip()
    except Exception as exc:
        runtime_log("feed.log", "pconline err=%s" % exc)
        return ""


def _ipip_city() -> str:
    try:
        resp = requests.get("https://myip.ipip.net", timeout=_TIMEOUT, headers=_UA)
        text = resp.content.decode("utf-8", errors="replace")
        matched = re.search(r"来自于：\s*(\S+)\s+(\S+)\s+(\S+)", text)
        if not matched:
            return ""
        province, city = matched.group(2), matched.group(3)
        if city in ("电信", "联通", "移动", "铁通", "教育网"):
            return province
        return city
    except Exception as exc:
        runtime_log("feed.log", "ipip err=%s" % exc)
        return ""


def _geocode(city: str):
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 5, "language": "zh", "format": "json"},
            timeout=_TIMEOUT, headers=_UA)
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
        picked = None
        for item in results:
            if str(item.get("country_code") or "").upper() == "CN":
                picked = item
                break
        if picked is None and results:
            picked = results[0]
        if not picked:
            return None, None
        return float(picked["latitude"]), float(picked["longitude"])
    except Exception as exc:
        runtime_log("feed.log", "geocode city=%s err=%s" % (city, exc))
        return None, None


def _weather(place: dict) -> str:
    if place.get("lat") is None or place.get("lon") is None:
        return ""
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["lat"],
            "longitude": place["lon"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "forecast_days": 1,
            "timezone": "Asia/Shanghai",
        },
        timeout=_TIMEOUT, headers=_UA)
    resp.raise_for_status()
    daily = (resp.json() or {}).get("daily") or {}
    codes = daily.get("weather_code") or [None]
    highs = daily.get("temperature_2m_max") or [None]
    lows = daily.get("temperature_2m_min") or [None]
    return weather_line(place["city"], codes[0], lows[0], highs[0])


def _news(place: dict) -> dict:
    """取一条带原文 jpg 的头条。念的是标题，图交给屏幕。"""
    empty = {"text": "", "title": "", "image_url": "", "link": ""}
    fallback = empty
    for lid in ("2510", "2511"):
        resp = requests.get(
            "https://feed.mix.sina.com.cn/api/roll/get",
            params={"pageid": "153", "lid": lid, "num": "8", "page": "1"},
            timeout=_TIMEOUT, headers=_UA)
        resp.raise_for_status()
        items = ((resp.json() or {}).get("result") or {}).get("data") or []
        for item in items:
            pack = headline_from_item(item)
            if not pack.get("text"):
                continue
            if pack.get("image_url"):
                return pack
            if not fallback.get("text"):
                fallback = pack
    return fallback
