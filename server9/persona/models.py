# -*- coding: utf-8 -*-
"""数字人人设配置。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PersonaTts:
    spk_id: str
    prompt_wav: str
    prompt_text: str


@dataclass
class PersonaConfig:
    id: str
    display_name: str
    enabled: bool = True
    system_prompt: str = ""
    index_dir: str = ""
    categories: List[str] = field(default_factory=list)
    tts: PersonaTts = field(default_factory=lambda: PersonaTts("", "", ""))
    hello_texts: List[str] = field(default_factory=list)
    bye_texts: List[str] = field(default_factory=list)
    refuse_text: str = "这个我还不太清楚。我只讲解眼前这场展。"
    hooks: Dict[str, List[str]] = field(default_factory=dict)
    ticket_info: Dict[str, str] = field(default_factory=dict)
    chitchat: Dict[str, str] = field(default_factory=dict)
    persona_fix: List[Tuple[str, str]] = field(default_factory=list)
    persona_patterns: List[Tuple[str, str]] = field(default_factory=list)
    other_exhibition_patterns: List[str] = field(default_factory=list)
    leak_patterns: List[str] = field(default_factory=list)
    forbidden_terms: List[str] = field(default_factory=list)
    cross_persona_patterns: List[str] = field(default_factory=list)
    intent_examples: Dict[str, List[str]] = field(default_factory=dict)
    warmup_text: str = "测试。"
    opening_strip_patterns: List[str] = field(default_factory=list)
    realtime_first: str = ""
    realtime_again: str = ""
    realtime_hooks: List[str] = field(default_factory=list)
    exhibit_aliases: Dict[str, str] = field(default_factory=dict)

    @property
    def index_path(self) -> Path:
        return Path(self.index_dir)

    def knowledge_ready(self) -> bool:
        d = self.index_path
        return all((d / name).exists() for name in (
            "faiss.index", "items.json", "faiss_map.json", "bm25.pkl"))

    def voice_ready(self) -> bool:
        p = Path(self.tts.prompt_wav) if self.tts.prompt_wav else None
        return bool(p and p.exists())

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "knowledge_ready": self.knowledge_ready(),
            "voice_ready": self.voice_ready(),
            "categories": list(self.categories),
        }
