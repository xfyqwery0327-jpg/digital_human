#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""莫奈数字讲解员入口。实现已拆到 server9/，仍用: python server8.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "server9"))

from app import app, run  # noqa: E402

if __name__ == "__main__":
    run()
