"""The evaluation data must stay valid as the corpus changes."""
import sys
from pathlib import Path

import pytest

from ragassist import app
from ragassist.bench import evaluate_retriever, first_gold_rank, load_questions
from ragassist.e2e_report import markdown, summarise
from ragassist.retrievers import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = load_questions(ROOT / "eval" / "questions.yaml")


def test_question_ids_unique_and_complete():
    ids = [q["id"] for q in QUESTIONS["answerable"] + QUESTIONS["unanswerable"]]
    assert len(ids) == len(set(ids))
    for q in QUESTIONS["answerable"]:
        assert q["question"] and q["gold"] and q["must_include"] and q["reference"]


def test_gold_sections_exist_and_contain_the_key_facts():
    chunks = app.chunks()
    sources = {c.source for c in chunks}
    for q in QUESTIONS["answerable"]:
        assert set(q["gold"]) <= sources, (q["id"], set(q["gold"]) - sources)
        gold_text = " ".join(c.text for c in chunks if c.source in q["gold"]).lower()
        for term in q["must_include"]:
            assert term.lower() in gold_text, (q["id"], term)


def test_first_gold_rank():
    assert first_gold_rank(["a", "b", "c"], ["c", "x"]) == 3
    assert first_gold_rank(["a"], ["b"]) is None


def test_bm25_benchmark_regression_guard():
    row = evaluate_retriever(BM25Retriever(app.chunks()), QUESTIONS["answerable"])
    assert row["questions"] == len(QUESTIONS["answerable"])
    assert row["recall@5"] >= 0.7 and row["mrr@10"] >= 0.5


def test_generated_suites_are_valid_evalforge_suites(tmp_path):
    evalforge = pytest.importorskip("evalforge.config")
    sys.path.insert(0, str(ROOT / "eval"))
    import build_suite
    import yaml

    for offline in (True, False):
        path = tmp_path / f"suite_{offline}.yaml"
        path.write_text(yaml.safe_dump(build_suite.build(offline)))
        suite = evalforge.load_suite(path)
        assert len(suite.cases) == len(QUESTIONS["answerable"]) + len(QUESTIONS["unanswerable"])
        assert [p.id for p in suite.providers] == ["closed-book", "bm25", "hybrid", "hybrid+rerank"]


class FakeResult:
    def __init__(self, provider, tags, checks, error=None, cost=0.001, latency=900.0):
        self.provider_id, self.tags, self.checks, self.error = provider, tags, checks, error
        self.cost_usd, self.judge_cost_usd, self.latency_ms = cost, 0.01, latency


def test_e2e_summary_per_check():
    results = [
        FakeResult("bm25", ["answerable"], [{"type": "llm_judge", "name": "correct", "passed": True},
                                            {"type": "regex", "name": "cited", "passed": True}]),
        FakeResult("bm25", ["answerable"], [{"type": "llm_judge", "name": "correct", "passed": False},
                                            {"type": "regex", "name": "cited", "passed": True}]),
        FakeResult("bm25", ["answerable"], [], error="boom"),
        FakeResult("bm25", ["unanswerable"], [{"type": "llm_judge", "name": "declines", "passed": True}]),
    ]
    row = summarise(results)[0]
    assert row["correct"] == pytest.approx(1 / 3)  # errors count as failures
    assert row["cited"] == pytest.approx(2 / 3)
    assert row["declines"] == 1.0 and row["faithful"] is None
    assert "| bm25 | 33% | n/a |" in markdown([row])
