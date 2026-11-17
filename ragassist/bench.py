"""Retrieval benchmark: does each retriever put a gold section near the top?

Metrics over the answerable questions in eval/questions.yaml:
  Recall@k  share of questions with at least one gold section in the top k results
  MRR@10    mean of 1/rank of the first gold section (0 if not in the top 10)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from . import app
from .llm import make_llm
from .retrievers import build_retriever


def load_questions(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def first_gold_rank(sources: list[str], gold: list[str]) -> int | None:
    for rank, s in enumerate(sources, 1):
        if s in gold:
            return rank
    return None


def evaluate_retriever(retriever, questions: list[dict], depth: int = 10) -> dict:
    ranks, misses = [], []
    start = time.perf_counter()
    for q in questions:
        sources = [h.chunk.source for h in retriever.search(q["question"], depth)]
        rank = first_gold_rank(sources, q["gold"])
        ranks.append(rank)
        if rank is None or rank > 5:
            misses.append({"id": q["id"], "rank": rank, "top3": sources[:3]})
    n = len(questions)
    ms = (time.perf_counter() - start) * 1000 / max(1, n)
    return {
        "retriever": retriever.name,
        "questions": n,
        "recall@1": sum(1 for r in ranks if r and r <= 1) / n,
        "recall@3": sum(1 for r in ranks if r and r <= 3) / n,
        "recall@5": sum(1 for r in ranks if r and r <= 5) / n,
        "mrr@10": sum(1 / r for r in ranks if r) / n,
        "ms_per_query": ms,
        "misses": misses,
    }


def markdown_table(rows: list[dict]) -> str:
    lines = ["| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@10 | ms/query |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['retriever']} | {r['recall@1']:.1%} | {r['recall@3']:.1%} | {r['recall@5']:.1%} "
                     f"| {r['mrr@10']:.3f} | {r['ms_per_query']:.0f} |")
    return "\n".join(lines)


def run_benchmark(questions_path: str, retriever_names: list[str], out_md: str | None, out_json: str | None) -> int:
    questions = load_questions(questions_path)["answerable"]
    chunks = app.chunks()
    rows = []
    for name in retriever_names:
        name = name.strip()
        needs_dense = name.split("+")[0] in ("dense", "hybrid")
        llm = make_llm("claude-haiku-5-5", offline_ok=False) if name.endswith("+rerank") else None
        print(f"Building {name}...", flush=True)
        retriever = build_retriever(name, chunks, llm=llm, embedder=app.embedder() if needs_dense else None)
        rows.append(evaluate_retriever(retriever, questions))
        rows[-1]["retriever"] = name
    table = markdown_table(rows)
    print()
    print(table)
    for r in rows:
        if r["misses"]:
            print(f"\n{r['retriever']}: {len(r['misses'])} questions without a gold section in the top 5")
            for m in r["misses"]:
                print(f"  {m['id']}: gold rank {m['rank']}, top 3 {m['top3']}")
    if out_md:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text(f"# Retrieval benchmark\n\n{len(questions)} questions, {len(chunks)} chunks.\n\n"
                                f"{table}\n", encoding="utf-8")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0
