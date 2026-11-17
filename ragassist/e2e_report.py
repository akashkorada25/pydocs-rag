"""Summarise an EvalForge run of the RAG suite: one row per system, one column per check."""
from __future__ import annotations

import statistics
from pathlib import Path

COLUMNS = [
    ("correct", "Correct", "answerable"),
    ("faithful", "Grounded in sources", "answerable"),
    ("key-facts", "Key facts", "answerable"),
    ("cited", "Cites sources", "answerable"),
    ("retrieved-gold", "Retrieved right section", "answerable"),
    ("declines", "Declines out-of-scope", "unanswerable"),
]


def summarise(results) -> list[dict]:
    """results: EvalForge Result objects for one run."""
    by_system: dict[str, list] = {}
    for r in results:
        by_system.setdefault(r.provider_id, []).append(r)
    rows = []
    for system, rs in by_system.items():
        row = {"system": system, "errors": sum(1 for r in rs if r.error)}
        for check_name, _label, tag in COLUMNS:
            outcomes = [c["passed"] for r in rs if tag in r.tags for c in r.checks
                        if (c.get("name") or c["type"]) == check_name]
            if not outcomes:  # this check never ran for this system
                row[check_name] = None
                continue
            errored = sum(1 for r in rs if tag in r.tags and r.error)  # errors count as failures
            row[check_name] = sum(outcomes) / (len(outcomes) + errored)
        latencies = [r.latency_ms for r in rs if not r.error]
        row["p50_latency_ms"] = statistics.median(latencies) if latencies else 0.0
        row["cost_per_question"] = sum(r.cost_usd for r in rs) / len(rs)
        row["judge_cost"] = sum(r.judge_cost_usd for r in rs)
        rows.append(row)
    order = ["closed-book", "bm25", "dense", "hybrid", "dense+rerank", "hybrid+rerank"]
    rows.sort(key=lambda x: order.index(x["system"]) if x["system"] in order else 99)
    return rows


def markdown(rows: list[dict]) -> str:
    head = "| System | " + " | ".join(label for _, label, _ in COLUMNS) + " | Median latency | Cost / question |"
    sep = "|---" * (len(COLUMNS) + 3) + "|"
    lines = [head, sep]
    for r in rows:
        cells = ["n/a" if r[name] is None else f"{r[name]:.0%}" for name, _, _ in COLUMNS]
        lines.append(f"| {r['system']} | " + " | ".join(cells) +
                     f" | {r['p50_latency_ms'] / 1000:.1f} s | ${r['cost_per_question']:.4f} |")
    return "\n".join(lines)


def run(db: str, run_id: str, out: str | None) -> int:
    from evalforge.storage import Store

    store = Store(db)
    run = store.get_run(run_id)
    rows = summarise(store.load_results(run["id"]))
    table = markdown(rows)
    judge = sum(r["judge_cost"] for r in rows)
    text = (f"# End-to-end evaluation\n\nEvalForge run `{run['id']}` ({run['created_at']}). "
            f"Judge cost for the whole run: ${judge:.2f}.\n\n{table}\n")
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    return 0
