"""Command line:

  pydocs-rag ask "How do I sort by two keys?" [--retriever hybrid] [--k 5] [--json]
  pydocs-rag serve [--port 8000]
  pydocs-rag bench [--retrievers bm25,tfidf,dense,hybrid] [--out results/retrieval.md]
  pydocs-rag ingest [--raw data/raw]
  pydocs-rag e2e-report [RUN_ID] [--db .evalforge/results.db]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import app


def cmd_ask(args) -> int:
    a = app.get_pipeline(args.retriever, args.k, args.model).answer(args.question)
    if args.json:
        print(json.dumps(a.to_dict(), indent=2))
        return 0
    print(a.text)
    if a.passages:
        print("\nSources:")
        for s in a.sources():
            mark = "*" if s["cited"] else " "
            print(f" {mark}[{s['n']}] {s['source']}\n      {s['url']}")
    print(f"\n({a.retriever}, {a.model}, {a.latency_ms:.0f} ms, ${a.cost_usd:.5f})")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.host, args.port)
    return 0


def cmd_ingest(args) -> int:
    from .ingest import ingest

    chunks = ingest(args.raw, app.DEFAULT_CHUNKS, max_words=args.max_words)
    print(f"Wrote {len(chunks)} chunks to {app.DEFAULT_CHUNKS}")
    return 0


def cmd_bench(args) -> int:
    from .bench import run_benchmark

    return run_benchmark(args.questions, args.retrievers.split(","), args.out, args.json_out)


def cmd_e2e_report(args) -> int:
    from .e2e_report import run

    return run(args.db, args.run_id, args.out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="pydocs-rag", description="Ask questions about the Python docs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="Answer one question")
    a.add_argument("question")
    a.add_argument("--retriever", default="auto")
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--model", default="claude-haiku-5-5")
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_ask)

    s = sub.add_parser("serve", help="Run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    i = sub.add_parser("ingest", help="Rebuild the chunk index from the reST sources")
    i.add_argument("--raw", default=str(Path("data/raw")))
    i.add_argument("--max-words", type=int, default=220)
    i.set_defaults(fn=cmd_ingest)

    b = sub.add_parser("bench", help="Retrieval benchmark: recall@k and MRR per retriever")
    b.add_argument("--questions", default="eval/questions.yaml")
    b.add_argument("--retrievers", default="bm25,tfidf,hybrid-lexical")
    b.add_argument("--out", default="results/retrieval.md")
    b.add_argument("--json-out", default="results/retrieval.json")
    b.set_defaults(fn=cmd_bench)

    e = sub.add_parser("e2e-report", help="Per-check table for an EvalForge run of eval/rag_suite.yaml")
    e.add_argument("run_id", nargs="?", default="latest")
    e.add_argument("--db", default=".evalforge/results.db")
    e.add_argument("--out", default="results/end_to_end.md")
    e.set_defaults(fn=cmd_e2e_report)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, ImportError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
