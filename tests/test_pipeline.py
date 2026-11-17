import pytest

from ragassist import app
from ragassist.ingest import Chunk
from ragassist.llm import LLMError, OfflineLLM, Reply
from ragassist.pipeline import RAG, cited_numbers, format_passages
from ragassist.retrievers import BM25Retriever


class RecordingLLM:
    model = "recording"

    def __init__(self, reply="Use set() [1]. Braces make a dict [1][2]."):
        self.reply, self.calls = reply, []

    def complete(self, system, user, max_tokens=1024):
        self.calls.append((system, user))
        return Reply(self.reply, 500, 20, 0.0002)


def chunks():
    return [
        Chunk("d/sets#0", "d/sets", "Data Structures", "Sets", "tut-sets", "https://x/sets",
              "Use set() to create an empty set, not {}."),
        Chunk("d/dict#0", "d/dict", "Data Structures", "Dictionaries", "", "https://x/dict",
              "Curly braces create an empty dictionary."),
    ]


def test_cited_numbers():
    assert cited_numbers("a [1] b [2][3] c [4] d [9]", 4) == [1, 2, 3, 4]
    assert cited_numbers("prints `[1, 2]` and ```\n[3]\n``` [2]", 4) == [2]
    assert cited_numbers("no citations", 3) == []


def test_rag_prompt_contains_numbered_passages_and_answer_tracks_citations():
    llm = RecordingLLM()
    rag = RAG(BM25Retriever(chunks()), llm, k=2)
    a = rag.answer("How do I make an empty set?")
    system, user = llm.calls[0]
    assert "only the numbered passages" in system
    assert "[1] (d/sets - Sets)" in user and user.endswith("Question: How do I make an empty set?")
    assert a.cited == [1, 2] and a.cost_usd == 0.0002
    d = a.to_dict()
    assert d["sources"][0]["url"] == "https://x/sets" and d["sources"][0]["cited"]
    assert d["retriever"] == "bm25"


def test_closed_book_sends_no_passages():
    llm = RecordingLLM("A set is unordered.")
    a = RAG(None, llm).answer("What is a set?")
    assert "Passages" not in llm.calls[0][1]
    assert a.passages == [] and a.retriever == "closed-book"


def test_offline_llm_quotes_first_passage_with_citation():
    user = f"Passages:\n\n{format_passages(chunks())}\n\nQuestion: empty set?"
    reply = OfflineLLM().complete("system", user)
    assert reply.text.startswith("Use set()") and reply.text.endswith("[1]")


def test_evalforge_answer_refuses_to_run_offline_by_default(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        app.evalforge_answer("How do I sort?", retriever="bm25")
    out = app.evalforge_answer("How do I sort by a key?", retriever="bm25", allow_offline=True)
    assert out["text"] and out["metadata"]["sources"]
    assert "[1]" in out["metadata"]["context"]
    closed = app.evalforge_answer("What is Python?", retriever="none", allow_offline=True)
    assert closed["metadata"]["sources"] == []
