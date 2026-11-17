"""Builds and caches pipelines, and exposes the function EvalForge calls."""
from __future__ import annotations

import threading
from pathlib import Path

from .ingest import Chunk, load_chunks
from .llm import make_llm
from .pipeline import RAG, format_passages
from .retrievers import CachingEmbedder, FastEmbedEmbedder, build_retriever, dense_available

DEFAULT_CHUNKS = Path(__file__).parent / "data" / "chunks.jsonl"

_lock = threading.Lock()
_chunks: list[Chunk] | None = None
_embedder: CachingEmbedder | None = None
_pipelines: dict[tuple, RAG] = {}


def chunks() -> list[Chunk]:
    global _chunks
    if _chunks is None:
        _chunks = load_chunks(DEFAULT_CHUNKS)
    return _chunks


def embedder() -> CachingEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = CachingEmbedder(FastEmbedEmbedder())
    return _embedder


def default_retriever() -> str:
    return "hybrid" if dense_available() else "hybrid-lexical"


def get_pipeline(retriever: str = "auto", k: int = 5, model: str = "claude-haiku-5-5",
                 rerank_model: str = "claude-haiku-5-5", allow_offline: bool = True) -> RAG:
    """retriever: bm25 | tfidf | dense | hybrid | hybrid-lexical [+rerank] | none (closed-book) | auto."""
    retriever = default_retriever() if retriever == "auto" else retriever
    key = (retriever, k, model, rerank_model, allow_offline)
    with _lock:
        if key not in _pipelines:
            llm = make_llm(model, offline_ok=allow_offline)
            if retriever == "none":
                _pipelines[key] = RAG(None, llm, k)
            else:
                needs_dense = retriever.split("+")[0] in ("dense", "hybrid")
                rerank_llm = make_llm(rerank_model, offline_ok=allow_offline) if retriever.endswith("+rerank") else None
                r = build_retriever(retriever, chunks(), llm=rerank_llm, embedder=embedder() if needs_dense else None)
                _pipelines[key] = RAG(r, llm, k)
        return _pipelines[key]


def evalforge_answer(prompt: str, system: str | None = None, retriever: str = "auto", k: int = 5,
                     model: str = "claude-haiku-5-5", rerank_model: str = "claude-haiku-5-5",
                     allow_offline: bool = False) -> dict:
    """Entry point for EvalForge's python provider (model: "ragassist.app:evalforge_answer").

    allow_offline is False by default so an evaluation never silently scores the offline
    fallback when the API key is missing.
    """
    a = get_pipeline(retriever, k, model, rerank_model, allow_offline).answer(prompt)
    return {
        "text": a.text,
        "input_tokens": a.input_tokens,
        "output_tokens": a.output_tokens,
        "cost_usd": a.cost_usd,
        "metadata": {
            "retriever": a.retriever,
            "sources": [c.source for c in a.passages],
            "cited": [a.passages[n - 1].source for n in a.cited],
            "context": format_passages(a.passages) if a.passages else "(no documents: closed-book answer)",
        },
    }
