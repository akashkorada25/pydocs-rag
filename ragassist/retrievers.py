"""Retrieval methods: BM25, TF-IDF, dense embeddings, hybrid (reciprocal rank fusion) and LLM re-ranking.

Every retriever takes the list of chunks at construction time and answers
search(query, k) -> list[Hit], best first.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .ingest import Chunk

STOPWORDS = set("""a an the and or but of to in on at for with by from is are was were be been being it its this
that these those as i you he she they we our their his her them my your me us do does did not no so if then than
has have had will would can could should may might what which who whom how why when where there here into about
also just only very any some such each other more most own same too out up down over under again further once all
both few nor off s t don use using used get""".split())


def stem(token: str) -> str:
    """A deliberately small stemmer: enough to match 'sorting'/'sorted'/'sorts' to 'sort'."""
    if token.startswith("__") or len(token) <= 4:
        return token
    for suffix in ("ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            return base + "y" if suffix in ("ies", "ied") else base
    return token


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z_][a-z0-9_]*|\d+", text.lower())
    return [stem(t) for t in tokens if t not in STOPWORDS]


@dataclass
class Hit:
    index: int      # position in the chunk list
    score: float
    chunk: Chunk


class Retriever:
    name = "base"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def scores(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s = self.scores(query)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k] if k < len(s) else np.arange(len(s))
        top = top[np.argsort(-s[top], kind="stable")]
        return [Hit(int(i), float(s[i]), self.chunks[i]) for i in top]


class BM25Retriever(Retriever):
    """Okapi BM25 over the chunk text, the classic keyword-search baseline."""

    name = "bm25"

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vectorizer.fit_transform([c.search_text for c in chunks]).tocsc().astype(np.float64)
        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        # Precompute the BM25 weight of every (doc, term) pair once
        norm = k1 * (1 - b + b * doc_len / doc_len.mean())
        tf = tf.tocoo()
        weights = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        from scipy.sparse import csc_matrix

        self.weights = csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        self.vocab = self.vectorizer.vocabulary_

    def scores(self, query):
        cols = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not cols:
            return np.zeros(len(self.chunks))
        return np.asarray(self.weights[:, cols] @ self.idf[cols]).ravel()


class TfidfRetriever(Retriever):
    """TF-IDF vectors with cosine similarity."""

    name = "tfidf"

    def __init__(self, chunks):
        super().__init__(chunks)
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                          sublinear_tf=True, ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform([c.search_text for c in chunks])

    def scores(self, query):
        q = self.vectorizer.transform([query])
        return np.asarray((self.matrix @ q.T).todense()).ravel()


class Embedder:
    """Interface for embedding models. Vectors should be L2-normalised."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    """Local embedding model through fastembed (ONNX, runs on CPU, no API key)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ImportError("Dense retrieval needs fastembed: pip install 'pydocs-rag[dense]'") from e
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name)

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)

    def embed_documents(self, texts):
        vecs = self.model.passage_embed(texts) if hasattr(self.model, "passage_embed") else self.model.embed(texts)
        return self._normalise(np.array(list(vecs), dtype=np.float32))

    def embed_query(self, text):
        vecs = self.model.query_embed(text) if hasattr(self.model, "query_embed") else self.model.embed([text])
        return self._normalise(np.array(list(vecs), dtype=np.float32))[0]


class CachingEmbedder(Embedder):
    """Wraps an embedder so the corpus is embedded once, even if several retrievers use it."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self._docs: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def embed_documents(self, texts):
        key = hash(tuple(texts))
        with self._lock:
            if key not in self._docs:
                self._docs[key] = self.inner.embed_documents(texts)
            return self._docs[key]

    def embed_query(self, text):
        return self.inner.embed_query(text)


class DenseRetriever(Retriever):
    """Semantic search: cosine similarity between embedding vectors."""

    name = "dense"

    def __init__(self, chunks, embedder: Embedder | None = None):
        super().__init__(chunks)
        self.embedder = embedder or FastEmbedEmbedder()
        self.matrix = self.embedder.embed_documents([c.search_text for c in chunks])

    def scores(self, query):
        return self.matrix @ self.embedder.embed_query(query)


class HybridRetriever(Retriever):
    """Reciprocal rank fusion: combines rankings, so scores on different scales need no tuning."""

    def __init__(self, chunks, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 50):
        super().__init__(chunks)
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def scores(self, query):
        fused = np.zeros(len(self.chunks))
        for r in self.retrievers:
            for rank, hit in enumerate(r.search(query, self.depth)):
                fused[hit.index] += 1.0 / (self.rrf_k + rank + 1)
        return fused


RERANK_SYSTEM = """You rank documentation passages by how well they answer a question.
Score each passage: 3 = directly answers it, 2 = closely related and partly answers it,
1 = same topic but does not answer it, 0 = unrelated.
Reply with JSON only: {"scores": [<one integer per passage, in order>]}"""


class LLMReranker(Retriever):
    """Takes the base retriever's top candidates and asks an LLM to re-order them."""

    def __init__(self, chunks, base: Retriever, llm, candidates: int = 15):
        super().__init__(chunks)
        self.base, self.llm, self.candidates = base, llm, candidates
        self.name = f"{base.name}+rerank"
        self._local = threading.local()

    @property
    def last_usage(self) -> dict:
        return getattr(self._local, "usage", {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})

    def search(self, query, k=5):
        hits = self.base.search(query, self.candidates)
        passages = "\n\n".join(f"[{i + 1}] {h.chunk.search_text[:1200]}" for i, h in enumerate(hits))
        reply = self.llm.complete(RERANK_SYSTEM, f"Question: {query}\n\nPassages:\n{passages}", max_tokens=300)
        self._local.usage = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                             "cost_usd": reply.cost_usd}
        try:
            raw = json.loads(re.search(r"\{.*\}", reply.text, re.S).group(0))["scores"]
            llm_scores = [float(x) for x in raw][: len(hits)]
        except (AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return hits[:k]  # fall back to the base ranking if the reply is unusable
        llm_scores += [0.0] * (len(hits) - len(llm_scores))
        # Sort by LLM score, keep the base order for ties
        order = sorted(range(len(hits)), key=lambda i: (-llm_scores[i], i))
        return [Hit(hits[i].index, llm_scores[i], hits[i].chunk) for i in order[:k]]

    def scores(self, query):  # pragma: no cover - search is overridden
        raise NotImplementedError


def build_retriever(name: str, chunks: list[Chunk], llm=None, embedder: Embedder | None = None) -> Retriever:
    """bm25 | tfidf | dense | hybrid (bm25+dense) | hybrid-lexical (bm25+tfidf), optionally with '+rerank'."""
    rerank = name.endswith("+rerank")
    base_name = name[: -len("+rerank")] if rerank else name
    if base_name == "bm25":
        r = BM25Retriever(chunks)
    elif base_name == "tfidf":
        r = TfidfRetriever(chunks)
    elif base_name == "dense":
        r = DenseRetriever(chunks, embedder)
    elif base_name == "hybrid":
        r = HybridRetriever(chunks, [BM25Retriever(chunks), DenseRetriever(chunks, embedder)])
    elif base_name == "hybrid-lexical":
        r = HybridRetriever(chunks, [BM25Retriever(chunks), TfidfRetriever(chunks)])
    else:
        raise ValueError(f"Unknown retriever '{name}'. Use bm25, tfidf, dense, hybrid or hybrid-lexical, "
                         "optionally with '+rerank'.")
    if rerank:
        if llm is None:
            raise ValueError("Re-ranking needs an LLM")
        r = LLMReranker(chunks, r, llm)
    return r


def dense_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False
