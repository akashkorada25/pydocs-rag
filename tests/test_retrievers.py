import hashlib

import numpy as np
import pytest

from ragassist.ingest import Chunk
from ragassist.llm import Reply
from ragassist.retrievers import (BM25Retriever, CachingEmbedder, DenseRetriever, Embedder, HybridRetriever,
                                  LLMReranker, TfidfRetriever, build_retriever, stem, tokenize)


def make_chunks():
    texts = {
        "sets": "Use set() to create an empty set; curly braces create an empty dictionary.",
        "queues": "Lists are slow as queues; use collections.deque for fast appends and pops at both ends.",
        "venv": "Create a virtual environment with python -m venv tutorial-env and activate it.",
        "sorting": "Sorts are stable: records with equal keys keep their original order.",
        "logging": "The default level is WARNING, so info messages are not shown.",
    }
    return [Chunk(f"doc/{k}#0", f"doc/{k}", k.title(), k.title(), k, f"https://x/{k}", t) for k, t in texts.items()]


class HashEmbedder(Embedder):
    """Deterministic fake embedding: bag of hashed tokens, so similar words give similar vectors."""

    def _vec(self, text):
        v = np.zeros(64)
        for t in tokenize(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 64] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def test_tokenize_and_stem():
    assert stem("sorting") == "sort" and stem("sorted") == "sort" and stem("queues") == "queu"
    assert stem("__init__") == "__init__"
    assert "the" not in tokenize("the venv") and "venv" in tokenize("the venv")


@pytest.mark.parametrize("cls", [BM25Retriever, TfidfRetriever])
def test_lexical_retrievers_find_the_right_chunk(cls):
    r = cls(make_chunks())
    assert r.search("how to make an empty set", 1)[0].chunk.doc == "doc/sets"
    assert r.search("default logging level warning", 1)[0].chunk.doc == "doc/logging"
    assert len(r.search("venv", 10)) == 5


def test_bm25_unknown_words_score_zero():
    assert not BM25Retriever(make_chunks()).scores("zzzz qqqq").any()


def test_dense_and_hybrid_with_fake_embedder():
    chunks = make_chunks()
    emb = CachingEmbedder(HashEmbedder())
    dense = DenseRetriever(chunks, emb)
    assert dense.search("virtual environment venv", 1)[0].chunk.doc == "doc/venv"
    hybrid = HybridRetriever(chunks, [BM25Retriever(chunks), dense])
    assert hybrid.name == "hybrid(bm25+dense)"
    assert hybrid.search("deque queue", 1)[0].chunk.doc == "doc/queues"
    # the corpus is embedded once even when a second retriever uses the same embedder
    DenseRetriever(chunks, emb)
    assert len(emb._docs) == 1


class FixedLLM:
    def __init__(self, text):
        self.text = text

    def complete(self, system, user, max_tokens=1024):
        return Reply(self.text, 100, 10, 0.001)


def test_reranker_reorders_and_falls_back():
    chunks = make_chunks()
    base = BM25Retriever(chunks)
    order = [h.chunk.doc for h in base.search("set", 5)]
    # Give the last candidate the top score
    scores = [0] * len(order)
    scores[-1] = 3
    rr = LLMReranker(chunks, base, FixedLLM('{"scores": %s}' % scores), candidates=5)
    assert rr.search("set", 2)[0].chunk.doc == order[-1]
    assert rr.last_usage["cost_usd"] == 0.001
    broken = LLMReranker(chunks, base, FixedLLM("not json"), candidates=5)
    assert [h.chunk.doc for h in broken.search("set", 3)] == order[:3]


def test_build_retriever_names():
    chunks = make_chunks()
    assert build_retriever("bm25", chunks).name == "bm25"
    assert build_retriever("hybrid-lexical", chunks).name == "hybrid(bm25+tfidf)"
    assert build_retriever("hybrid", chunks, embedder=HashEmbedder()).name == "hybrid(bm25+dense)"
    assert build_retriever("tfidf+rerank", chunks, llm=FixedLLM("{}")).name == "tfidf+rerank"
    with pytest.raises(ValueError):
        build_retriever("magic", chunks)
    with pytest.raises(ValueError):
        build_retriever("bm25+rerank", chunks)
