"""CampusQ 离线评测：检索命中、一次解决率、拒答准确率、延迟与成本。

    python eval/run_eval.py                # 默认走 LLM（若配置了 key）
    python eval/run_eval.py --no-llm       # 强制抽取式，做对照组
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from app.ingest import load  # noqa: E402
from app.qa import CampusQA  # noqa: E402
from core.llm import LLM  # noqa: E402
from metrics import dump, hit_at_k, mrr, pct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    store = load()
    llm = LLM()
    if args.no_llm:
        llm.api_key = None
    qa = CampusQA(store, llm)

    golden = [json.loads(l) for l in (PROJECT / "data" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    answerable = [g for g in golden if g["answerable"]]
    unanswerable = [g for g in golden if not g["answerable"]]

    h1 = h3 = 0.0
    rr = 0.0
    solved = 0
    covs, lat = [], []

    for g in answerable:
        hits = store.search(g["q"], top_k=args.top_k)
        ranked = [h["doc_id"] for h in hits]
        h1 += hit_at_k(ranked, g["docs"], 1)
        h3 += hit_at_k(ranked, g["docs"], 3)
        rr += mrr(ranked, g["docs"])

        res = qa.ask(g["q"], top_k=args.top_k)
        evidence = res["answer"] + " " + " ".join(c["snippet"] for c in res["citations"])
        cov = sum(1 for k in g["must"] if k in evidence) / len(g["must"])
        covs.append(cov)
        lat.append(res["latency_ms"])
        if not res["refused"] and cov >= 0.5:
            solved += 1

    refused_ok = 0
    for g in unanswerable:
        res = qa.ask(g["q"], top_k=args.top_k)
        lat.append(res["latency_ms"])
        if res["refused"]:
            refused_ok += 1

    n = len(answerable) or 1
    result = {
        "dataset": {"answerable": len(answerable), "unanswerable": len(unanswerable),
                    "chunks": len(store.chunks), "embedder": store.embedder.backend,
                    "mode": "extractive" if args.no_llm or not llm.is_live else "llm"},
        "retrieval": {"hit@1": round(h1 / n, 4), "hit@3": round(h3 / n, 4), "mrr": round(rr / n, 4)},
        "qa": {
            "one_shot_solve_rate": round(solved / n, 4),
            "keyword_coverage_avg": round(sum(covs) / n, 4),
            "refusal_accuracy": round(refused_ok / (len(unanswerable) or 1), 4),
        },
        "latency_ms": {"p50": round(statistics.median(lat), 1),
                       "p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 1)},
        "llm_usage": llm.usage.as_dict(),
    }
    dump(PROJECT / "data" / "metrics.json", result)

    print(f"\n=== CampusQ 评测（{result['dataset']['mode']} 模式）===")
    print(f"评测集 {len(answerable)} 条可答 + {len(unanswerable)} 条不可答，索引 {len(store.chunks)} 块")
    print(f"检索 hit@1 {pct(result['retrieval']['hit@1'])} | hit@3 {pct(result['retrieval']['hit@3'])} | MRR {result['retrieval']['mrr']}")
    print(f"一次解决率 {pct(result['qa']['one_shot_solve_rate'])} | 要点覆盖 {pct(result['qa']['keyword_coverage_avg'])} | 拒答准确 {pct(result['qa']['refusal_accuracy'])}")
    print(f"延迟 p50 {result['latency_ms']['p50']}ms / p95 {result['latency_ms']['p95']}ms | tokens {llm.usage.total_tokens}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
