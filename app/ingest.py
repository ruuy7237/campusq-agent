"""把 data/corpus 下的课程资料切分、向量化、落盘成可检索索引。

用法：
    python -m app.ingest            # 重建索引
    python -m app.ingest --stats    # 只看统计
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.store import HybridStore  # noqa: E402
from core.text import iter_docs  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT / "data" / "corpus"
INDEX = PROJECT / "data" / "index"


def build(chunk_size: int = 420, overlap: int = 90) -> HybridStore:
    files = sorted(CORPUS.glob("*.md"))
    if not files:
        raise SystemExit(f"未找到语料：{CORPUS}")
    store = HybridStore()
    total = 0
    for source, title, text in iter_docs(files):
        n = store.add_document(Path(source).stem, title, source, text, chunk_size, overlap)
        total += n
        print(f"  {source:<40} {n:>3} 块")
    store.build()
    store.save(INDEX)
    print(f"[ingest] {len(files)} 篇文档 / {total} 个分块 / 向量后端 {store.embedder.backend}")
    return store


def load() -> HybridStore:
    if not (INDEX / "chunks.jsonl").exists():
        return build()
    return HybridStore.load(INDEX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    if args.stats:
        store = load()
        avg = sum(len(c.text) for c in store.chunks) / max(len(store.chunks), 1)
        print(f"分块数 {len(store.chunks)}，平均长度 {avg:.0f} 字")
        return
    build()


if __name__ == "__main__":
    main()
