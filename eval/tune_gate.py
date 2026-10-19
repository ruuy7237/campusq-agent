"""证据闸门阈值调参：打印可答/不可答两类问题的覆盖率分布，据此定阈值。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.ingest import load  # noqa: E402
from app.qa import _coverage, _strict_tokens  # noqa: E402


def main() -> None:
    store = load()
    rows = [json.loads(l) for l in (PROJECT / "data" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pos, neg = [], []
    for g in rows:
        hits = store.search(g["q"], top_k=5)
        q = _strict_tokens(g["q"])
        best = max((_coverage(q, h["text"] + h["title"]) for h in hits), default=0.0)
        (pos if g["answerable"] else neg).append((round(best, 3), g["q"]))

    print("可答 top5 覆盖率（升序）：")
    for c, q in sorted(pos)[:6]:
        print(f"  {c:<6} {q}")
    print("不可答 top5 覆盖率（降序）：")
    for c, q in sorted(neg, reverse=True):
        print(f"  {c:<6} {q}")

    for th in (0.24, 0.28, 0.30, 0.32, 0.34, 0.36):
        tp = sum(1 for c, _ in pos if c >= th)
        fp = sum(1 for c, _ in neg if c >= th)
        print(f"阈值 {th}: 可答放行 {tp}/{len(pos)}，不可答误放 {fp}/{len(neg)}")


if __name__ == "__main__":
    main()
