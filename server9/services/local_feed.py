# -*- coding: utf-8 -*-
"""按本机出口 IP 查城市，再取一句天气或一条当地新闻。失败返回空串，不编数字。"""
from __future__ import annotations
import json
import os
import re
import threading
import time
from typing import Optional

import requests

from config import runtime_log
from tts.speaktext import to_spoken_zh

_LOCK = threading.Lock()
_PLACE = {"at": 0.0, "value": None}
_FACTS = {}
_PLACE_TTL = 1800
_FACT_TTL = 600
_TIMEOUT = 4

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


def feed_sentence(kind: str) -> str:
    """只要口播句子。头条图片用 feed_bundle。"""
    return (feed_bundle(kind).get("text") or "")


def feed_bundle(kind: str) -> dict:
    """weather 只有 text。news 另带标题、原文 jpg 和链接。失败时文本为空。"""
    empty = {"text": "", "title": "", "image_url": "", "link": ""}
    kind = "news" if kind == "news" else "weather"
    try:
        place = _place() or {}
        city = place.get("city") or ""
        if kind == "weather" and not city:
            runtime_log("feed.log", "locate empty kind=%s" % kind)
            return empty
        cached = _cached_fact(kind, city or "headline")
        if isinstance(cached, dict):
            return cached
        bundle = _news(place) if kind == "news" else {
            "text": _weather(place), "title": "", "image_url": "", "link": "",
        }
        if not isinstance(bundle, dict):
            bundle = {"text": bundle or "", "title": "", "image_url": "", "link": ""}
        _store_fact(kind, city or "headline", bundle)
        runtime_log("feed.log", "kind=%s city=%s ok=%s image=%s" % (
            kind, city or "headline", bool(bundle.get("text")), bool(bundle.get("image_url"))))
        return bundle
    except Exception as exc:
        runtime_log("feed.log", "kind=%s err=%s" % (kind, exc))
        return empty


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
