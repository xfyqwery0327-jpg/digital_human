# -*- coding: utf-8 -*-
"""从 personas/ 加载人设 YAML。音色只从 voice/{persona_id}/ 解析，不串别人的参考音。"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

from config import DEFAULT_PERSONA_ID, INDEX_DIR, PERSONAS_DIR, PROJECT_ROOT, VOICE_DIR
from persona.models import PersonaConfig, PersonaTts


def _yaml_load(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _as_str_list(raw) -> List[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw if str(x).strip()]


def _resolve_index_dir(persona_id: str, configured: str) -> Path:
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if (p / "faiss.index").exists():
            return p
    dedicated = INDEX_DIR / persona_id
    if (dedicated / "faiss.index").exists():
        return dedicated
    return dedicated


def _persona_voice_dir(persona_id: str) -> Path:
    d = VOICE_DIR / persona_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_voice(persona_id: str, raw: str) -> str:
    """只在本人人设目录（及显式绝对路径）里找参考音，禁止落到别人的 voice 目录。"""
    vdir = _persona_voice_dir(persona_id)
    candidates: List[Path] = []
    if raw:
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(PROJECT_ROOT / raw)
            candidates.append(vdir / Path(raw).name)
    if raw:
        name = Path(raw).name
        stem = Path(raw).stem
        for n in (name, f"{stem}.wav", f"{stem}_16k.wav", f"{stem}.mp3"):
            candidates.append(vdir / n)
    else:
        # 未配置文件名时才扫目录；跳过派生 clip，避免当 YAML 源二次裁切
        for ext in ("*.wav", "*.mp3"):
            for hit in sorted(vdir.glob(ext)):
                if "_male" in hit.stem and hit.stem.endswith("s"):
                    continue
                candidates.append(hit)
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.exists() and cand.is_file():
                # 禁止误用另一人设目录
                try:
                    cand.resolve().relative_to((VOICE_DIR / persona_id).resolve())
                except (ValueError, OSError):
                    if cand.is_absolute() and VOICE_DIR.resolve() not in cand.resolve().parents:
                        return str(cand)  # 人设外的绝对路径（如 CosyVoice 资源）仅当 YAML 写死才允许
                    continue
                return str(cand)
        except OSError:
            continue
        if cand.exists() and cand.is_file():
            return str(cand)
    return str(vdir / Path(raw).name) if raw else ""


def _seed_legacy_voice(cfg: PersonaConfig) -> None:
    """YAML 已指向存在的参考音时不要覆盖。只在缺文件时回填旧位置。"""
    if cfg.tts.prompt_wav and Path(cfg.tts.prompt_wav).is_file():
        return
    from shutil import copy2
    dest_dir = _persona_voice_dir(cfg.id)
    if cfg.id == "monet":
        from config import MONET_LEGACY_PROMPT_WAV
        dest = dest_dir / "cross_lingual_prompt.wav"
        prefer = [
            dest_dir / "monet-senlin_6to12s.wav",
            dest_dir / "monet-senlin.mp3",
            dest_dir / "monet_demo_LIJ.mp3",
            dest_dir / "yinpin_16k.wav",
            dest,
            dest_dir / "yinpin.mp3",
            Path(MONET_LEGACY_PROMPT_WAV),
        ]
        src = Path(MONET_LEGACY_PROMPT_WAV)
        if src.exists() and src.is_file() and not dest.exists():
            copy2(src, dest)
            print(f"🎤 已将莫奈参考音拷入 {dest}")
        for cand in prefer:
            if cand.exists() and cand.is_file():
                cfg.tts.prompt_wav = str(cand)
                return
        return
    if cfg.id == "fuhao":
        name = "jimeng-2026-09-11-5129.mp3"
        dest = dest_dir / name
        for src in (
            PROJECT_ROOT / "voices" / name,
            PROJECT_ROOT / "voice" / name,
            dest_dir / name,
        ):
            if src.exists() and src.is_file():
                if src.resolve() != dest.resolve() and not dest.exists():
                    copy2(src, dest)
                    print(f"🎤 已将妇好参考音拷入 {dest}")
                cfg.tts.prompt_wav = str(dest if dest.exists() else src)
                return


def _parse_pairs(raw) -> List[tuple]:
    out = []
    for item in raw or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict):
            out.append((str(item.get("wrong", "")), str(item.get("right", ""))))
    return out


def _parse_persona(data: dict, fallback_id: str) -> PersonaConfig:
    pid = str(data.get("id") or fallback_id).strip()
    tts_raw = data.get("tts") or {}
    wav = _resolve_voice(pid, str(tts_raw.get("prompt_wav") or ""))
    tts = PersonaTts(
        spk_id=str(tts_raw.get("spk_id") or pid),
        prompt_wav=wav,
        prompt_text=str(tts_raw.get("prompt_text") or ""),
    )
    cats = data.get("categories") or [f"exhibition_{pid}"]
    if isinstance(cats, str):
        cats = [cats]
    chitchat = data.get("chitchat") or {}
    if not isinstance(chitchat, dict):
        chitchat = {}
    intent = data.get("intent_examples") or {}
    if not isinstance(intent, dict):
        intent = {}
    refuse = str(data.get("refuse_text") or "").strip()
    if not refuse:
        refuse = "这个我还不太清楚。我只讲解眼前这场展。"
    return PersonaConfig(
        id=pid,
        display_name=str(data.get("display_name") or pid),
        enabled=bool(data.get("enabled", True)),
        system_prompt=str(data.get("system_prompt") or ""),
        index_dir=str(_resolve_index_dir(pid, str(data.get("index_dir") or ""))),
        categories=list(cats),
        tts=tts,
        hello_texts=_as_str_list(data.get("hello_texts")),
        bye_texts=_as_str_list(data.get("bye_texts")),
        refuse_text=refuse,
        hooks=dict(data.get("hooks") or {}),
        ticket_info=dict(data.get("ticket_info") or {}),
        chitchat={str(k): str(v) for k, v in chitchat.items()},
        persona_fix=_parse_pairs(data.get("persona_fix")),
        persona_patterns=_parse_pairs(data.get("persona_patterns")),
        other_exhibition_patterns=_as_str_list(data.get("other_exhibition_patterns")),
        leak_patterns=_as_str_list(data.get("leak_patterns")),
        forbidden_terms=_as_str_list(data.get("forbidden_terms")),
        cross_persona_patterns=_as_str_list(data.get("cross_persona_patterns")),
        intent_examples={str(k): _as_str_list(v) for k, v in intent.items()},
        warmup_text=str(data.get("warmup_text") or "测试。"),
        opening_strip_patterns=_as_str_list(data.get("opening_strip_patterns")),
        realtime_first=str((data.get("realtime") or {}).get("first") or ""),
        realtime_again=str((data.get("realtime") or {}).get("again") or ""),
        realtime_hooks=_as_str_list((data.get("realtime") or {}).get("hooks")),
        exhibit_aliases={
            str(k): str(v) for k, v in (data.get("exhibit_aliases") or {}).items()
        } if isinstance(data.get("exhibit_aliases") or {}, dict) else {},
    )


class PersonaRegistry:
    def __init__(self):
        self.default_id = DEFAULT_PERSONA_ID
        self._items: Dict[str, PersonaConfig] = {}

    def load(self) -> None:
        self._items.clear()
        PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        manifest = PERSONAS_DIR / "registry.yaml"
        names: List[str] = []
        if manifest.exists():
            data = _yaml_load(manifest)
            self.default_id = str(data.get("default_persona_id") or DEFAULT_PERSONA_ID)
            names = [str(x) for x in (data.get("personas") or [])]
        if not names:
            names = [p.stem for p in PERSONAS_DIR.glob("*.yaml") if p.name != "registry.yaml"]
        for name in names:
            _persona_voice_dir(name)
            path = PERSONAS_DIR / f"{name}.yaml"
            if path.exists():
                try:
                    cfg = _parse_persona(_yaml_load(path), name)
                    _seed_legacy_voice(cfg)
                    self._items[cfg.id] = cfg
                    print(f"👤 已加载人设 {cfg.id} ({cfg.display_name}) "
                          f"知识={'就绪' if cfg.knowledge_ready() else '未建索引'} "
                          f"音色={'就绪' if cfg.voice_ready() else '缺参考音'} "
                          f"voice={cfg.tts.prompt_wav}")
                except Exception as e:
                    print(f"⚠️ 人设 {name} 加载失败: {e}")
        if self.default_id not in self._items and self._items:
            self.default_id = next(iter(self._items))

    def get(self, persona_id: str) -> Optional[PersonaConfig]:
        return self._items.get((persona_id or "").strip())

    def require(self, persona_id: str) -> PersonaConfig:
        cfg = self.get(persona_id)
        if not cfg:
            raise KeyError(persona_id)
        return cfg

    def all(self) -> List[PersonaConfig]:
        return list(self._items.values())

    def enabled(self) -> List[PersonaConfig]:
        return [p for p in self._items.values() if p.enabled]

    def endpoint_hint(self, channel: str = "tts") -> str:
        ids = [p.id for p in self.enabled()] or [self.default_id or "monet"]
        joined = " 或 ".join(f"/ws/{i}/{channel}" for i in ids)
        return f"请连接 {joined}"


registry = PersonaRegistry()
