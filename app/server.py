"""CampusQ HTTP 服务。

    uvicorn app.server:app --reload --port 8011
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingest import load  # noqa: E402
from app.planner import build_plan  # noqa: E402
from app.qa import CampusQA  # noqa: E402

app = FastAPI(title="CampusQ", description="校内课程问答与复习规划", version="1.0.0")
_qa: CampusQA | None = None


def qa() -> CampusQA:
    global _qa
    if _qa is None:
        _qa = CampusQA(load())
    return _qa


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=200)
    top_k: int = Field(default=4, ge=1, le=8)


class PlanIn(BaseModel):
    exam_date: str = Field(description="YYYY-MM-DD")
    weak_topics: list[str] = Field(default_factory=list)
    hours_per_day: float = 2.0


@app.get("/health")
def health() -> dict:
    return {"ok": True, "live_llm": qa().llm.is_live,
            "chunks": len(qa().store.chunks), "embedder": qa().store.embedder.backend}


@app.post("/ask")
def ask(body: AskIn) -> dict:
    return qa().ask(body.question, body.top_k)


@app.post("/plan")
def plan(body: PlanIn) -> dict:
    return build_plan(body.exam_date, body.weak_topics, body.hours_per_day).as_dict()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html><meta charset="utf-8"><title>CampusQ</title>
<style>body{font-family:system-ui;max-width:720px;margin:40px auto;padding:0 16px;color:#222}
input,button{font-size:15px;padding:8px}input{width:70%}#a{margin-top:20px;line-height:1.7}
.c{color:#666;font-size:13px;margin-top:8px}</style>
<h2>CampusQ · 课程问答</h2>
<p>只回答语料内的问题，找不到依据会明确告知。</p>
<input id="q" placeholder="例如：TCP 为什么需要三次握手" />
<button onclick="go()">提问</button>
<div id="a"></div>
<script>
async function go(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  document.getElementById('a').textContent='检索中…';
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({question:q})});
  const d=await r.json();
  document.getElementById('a').innerHTML =
    '<b>'+d.answer+'</b><div class="c">'+
    (d.refused?'':d.citations.map(c=>'['+c.n+']《'+c.title+'》').join(' ')+
    ' · 置信度 '+d.confidence+' · '+d.latency_ms+'ms')+'</div>';
}
</script>"""
