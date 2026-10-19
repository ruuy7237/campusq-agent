"""问答主流程：检索 → 覆盖率闸门 → 带引用生成。

闸门放在生成之前。检索不到依据就直接拒答，比让模型答完再校验省钱，也不容易出现编造的答案。
LLM 和抽取式两条链路返回同样的结构，没有 key 的时候自动降级，保证换台机器也能演示。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.llm import LLM  # noqa: E402
from core.store import HybridStore  # noqa: E402
from core.text import split_sentences, tokenize  # noqa: E402

GATE = 0.28          # 证据闸门：查询词在最佳分块中的覆盖率下限（由 eval/tune_gate.py 调参得出）
MAX_CONTEXT_CHARS = 1800
REFUSE_TEXT = "资料中未找到相关内容，建议换一种问法，或补充这门课的资料后再问。"

SYSTEM_PROMPT = """你是课程资料的问答助手。规则：
1. 只能依据【资料】作答，不得使用资料以外的知识；
2. 每句结论后用 [编号] 标注来源，例如 [1]；
3. 资料不足以回答时，只输出三个字：无依据；
4. 用中文作答，先给结论再给依据，不超过 200 字。"""


def _strict_tokens(text: str) -> set[str]:
    """只保留词与二字组，剔掉单字——单字重合会把「学校」「学习」这类无关问题误判为命中。"""
    return {t for t in tokenize(text) if len(t) >= 2}


def _coverage(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    toks = _strict_tokens(text)
    return len(query_tokens & toks) / len(query_tokens)


class CampusQA:
    def __init__(self, store: HybridStore | None = None, llm: LLM | None = None) -> None:
        if store is None:
            from app.ingest import load
            store = load()
        self.store = store
        self.llm = llm or LLM()

    # ---------- 对外接口 ----------
    def ask(self, question: str, top_k: int = 4) -> dict:
        started = time.perf_counter()
        hits = self.store.search(question, top_k=top_k)
        q_tokens = _strict_tokens(question)
        for h in hits:
            h["coverage"] = _coverage(q_tokens, h["text"] + h["title"])

        best = max((h["coverage"] for h in hits), default=0.0)
        if not hits or best < GATE:
            return self._refuse(started, best)

        context = self._build_context(hits)
        answer, mode = self._generate(question, context)
        if answer is None:
            return self._refuse(started, best)

        used = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
        cites = [h for i, h in enumerate(hits, 1) if i in set(used)] or hits[:2]
        return {
            "answer": answer,
            "refused": False,
            "mode": mode,
            "confidence": round(0.6 * best + 0.4 * min(hits[0]["rerank"] * 1.6, 1.0), 3),
            "citations": [
                {"n": i, "doc": h["doc_id"], "title": h["title"],
                 "chunk": h["cid"], "snippet": h["text"][:120]}
                for i, h in enumerate(hits, 1)
                if h in cites
            ],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": self.llm.usage.as_dict(),
        }

    # ---------- 内部 ----------
    def _build_context(self, hits: list[dict]) -> str:
        parts, used = [], 0
        for i, h in enumerate(hits, 1):
            block = f"[{i}]《{h['title']}》\n{h['text']}"
            if used + len(block) > MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)

    def _generate(self, question: str, context: str):
        reply = self.llm.chat(SYSTEM_PROMPT, f"【资料】\n{context}\n\n【问题】{question}",
                              temperature=0.1, max_tokens=500)
        if reply:
            if reply.strip() in {"无依据", "无依据。", "无依据."} or "无依据" in reply[:8]:
                return None, "llm"
            return reply, "llm"
        return self._extractive(question, context), "extractive"

    def _extractive(self, question: str, context: str) -> str:
        """无 LLM 时的兜底：按与问题的词面重合度挑句子，保留来源编号。"""
        q = set(tokenize(question))
        picked: list[tuple[int, str]] = []
        cur = 1
        for line in context.splitlines():
            m = re.match(r"\[(\d+)\]", line)
            if m:
                cur = int(m.group(1))
                continue
            if not line.strip():
                continue
            for sent in split_sentences(line):
                if len(sent) < 8:
                    continue
                score = len(q & set(tokenize(sent))) / (len(q) or 1)
                if score >= 0.12:
                    picked.append((cur, sent))
        if not picked:
            return None
        seen, out = set(), []
        for _, sent in sorted(picked, key=lambda x: -len(q & set(tokenize(x[1]))))[:3]:
            if sent in seen:
                continue
            seen.add(sent)
            out.append(sent)
        idx = picked[0][0]
        return "；".join(s.rstrip("。") for s in out)[:220] + f"。[{idx}]"

    def _refuse(self, started: float, cov: float) -> dict:
        return {
            "answer": REFUSE_TEXT,
            "refused": True,
            "mode": "gate",
            "confidence": round(max(cov, 0.0), 3),
            "citations": [],
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "usage": self.llm.usage.as_dict(),
        }
