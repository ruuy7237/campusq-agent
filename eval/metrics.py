"""评测用的指标函数，各项目的 eval 脚本共用。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold = set(gold)
    return 1.0 if gold & set(ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], gold: Iterable[str]) -> float:
    gold = set(gold)
    for i, doc in enumerate(ranked, start=1):
        if doc in gold:
            return 1.0 / i
    return 0.0


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def dump(path: str | Path, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


