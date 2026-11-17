"""The RAG pipeline: retrieve passages, ask the model to answer from them with citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .ingest import Chunk
from .retrievers import LLMReranker, Retriever

RAG_SYSTEM = """You answer questions about Python using only the numbered passages from the official
Python documentation that are given to you.

- Cite the passages that support each statement with their numbers in brackets, like [1] or [2][3].
  Use one number per bracket.
- If the passages do not contain the answer, say that the provided documentation does not cover it.
  Do not answer from your own knowledge.
- Be concise: at most about 150 words. Include a short code example only if it helps and the
  passages support it."""

CLOSED_BOOK_SYSTEM = """You answer questions about Python. Be concise: at most about 150 words.
If you are not sure of the answer, say so."""


@dataclass
class Answer:
    question: str
    text: str
    passages: list[Chunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)  # 1-based passage numbers cited in the text
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    retriever: str = ""
    model: str = ""

    def sources(self) -> list[dict]:
        return [{"n": i + 1, "source": c.source, "url": c.url, "cited": (i + 1) in self.cited,
                 "doc_title": c.doc_title, "section": c.section} for i, c in enumerate(self.passages)]

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.text, "sources": self.sources(),
                "retriever": self.retriever, "model": self.model,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                          "cost_usd": round(self.cost_usd, 6), "latency_ms": round(self.latency_ms)}}


def format_passages(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i + 1}] ({c.doc} - {c.section})\n{c.text}" for i, c in enumerate(chunks))


def cited_numbers(text: str, n_passages: int) -> list[int]:
    """Passage numbers cited as [n], ignoring anything inside code (where [1, 2] is just a list)."""
    prose = re.sub(r"```.*?```|`[^`\n]+`", " ", text, flags=re.S)
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", prose)}
    return sorted(n for n in nums if 1 <= n <= n_passages)


class RAG:
    def __init__(self, retriever: Retriever | None, llm, k: int = 5):
        """retriever=None gives the closed-book baseline: the model answers with no documents."""
        self.retriever, self.llm, self.k = retriever, llm, k

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        in_tok = out_tok = 0
        cost = 0.0
        if self.retriever is None:
            reply = self.llm.complete(CLOSED_BOOK_SYSTEM, question)
            passages: list[Chunk] = []
        else:
            hits = self.retriever.search(question, self.k)
            if isinstance(self.retriever, LLMReranker):
                usage = self.retriever.last_usage
                in_tok, out_tok, cost = usage["input_tokens"], usage["output_tokens"], usage["cost_usd"]
            passages = [h.chunk for h in hits]
            user = f"Passages:\n\n{format_passages(passages)}\n\nQuestion: {question}"
            reply = self.llm.complete(RAG_SYSTEM, user)
        return Answer(
            question=question,
            text=reply.text.strip(),
            passages=passages,
            cited=cited_numbers(reply.text, len(passages)),
            input_tokens=in_tok + reply.input_tokens,
            output_tokens=out_tok + reply.output_tokens,
            cost_usd=cost + reply.cost_usd,
            latency_ms=(time.perf_counter() - start) * 1000,
            retriever=self.retriever.name if self.retriever else "closed-book",
            model=getattr(self.llm, "model", ""),
        )
