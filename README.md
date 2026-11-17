# pydocs-rag: Ask the Python Docs

A retrieval-augmented (RAG) assistant that answers Python questions **only from the official Python documentation**, cites the exact sections it used, and says so when the docs don't cover a question. Every design choice is measured with [EvalForge](https://github.com/akashkorada25/evalforge).

![Screenshot](docs/screenshot.png)

## Why this project

Most RAG demos stop at "it answers questions." This one asks **how well, and which design choices matter**:

- Does retrieval actually beat asking the model with no documents?
- Keyword search, embeddings, or both? Is LLM re-ranking worth the extra call?
- Are the answers **grounded** in the retrieved passages, or does the model add things?
- Does it **decline** questions the documentation doesn't cover, instead of making something up?

## Results

<!-- RESULTS:START -->
Results are produced by the GitHub Actions workflows in `.github/workflows/` and recorded here.
<!-- RESULTS:END -->

## How it works

```mermaid
flowchart LR
    A[Python docs<br/>reST sources] --> B[Clean + split by section<br/>727 chunks]
    B --> C{Retriever}
    C -->|BM25| D[Top passages]
    C -->|Embeddings| D
    C -->|Hybrid: rank fusion| D
    D --> E[Optional:<br/>Claude re-ranks top 15]
    E --> F[Claude Haiku 5.5 answers<br/>from numbered passages]
    F --> G[Answer with citations<br/>linked to docs.python.org]
```

1. **Ingestion** (`ragassist/ingest.py`): parses the reStructuredText sources of the Python tutorial, FAQ and six HOWTO guides; removes markup; splits by section heading; and packs paragraphs into ~220-word chunks with overlap. Each chunk keeps its page, section and a link to that exact section on docs.python.org.
2. **Retrieval** (`ragassist/retrievers.py`):
   - **BM25**: classic keyword search, implemented with sparse matrices.
   - **TF-IDF**: word and word-pair vectors with cosine similarity.
   - **Dense**: `BAAI/bge-small-en-v1.5` embeddings through fastembed (runs locally on CPU, no API).
   - **Hybrid**: reciprocal rank fusion of BM25 and dense rankings (no score tuning needed).
   - **Re-ranking**: Claude Haiku scores the top 15 candidates for relevance and re-orders them.
3. **Generation** (`ragassist/pipeline.py`): Claude Haiku 5.5 answers from numbered passages, cites them as `[n]`, and is told to say when the passages don't contain the answer. A **closed-book** mode (no documents) is the baseline.
4. **App** (`ragassist/server.py`): a small web chat and JSON API using only the standard library.

## How it's evaluated

**`eval/questions.yaml`** has 38 questions:

- **32 answerable questions**, written the way people actually ask ("I used an empty list as a default parameter and items keep piling up"), not copied from section titles. Each has the gold section(s) that answer it, key terms a correct answer must include, and a reference answer. A test checks that every gold section exists and contains those terms, so the labels can't silently drift from the corpus.
- **6 out-of-scope questions** (Flask on AWS, pandas, PyTorch...) that the assistant should decline.

Two layers of evaluation:

| Layer | What it measures | Cost |
|---|---|---|
| Retrieval benchmark (`pydocs-rag bench`) | Recall@1/3/5 and MRR: does each retriever rank a gold section near the top? | Free (re-ranking rows: ~$0.02) |
| End-to-end (EvalForge, `eval/rag_suite.yaml`) | For each system: **correct** (Claude Sonnet judge vs. reference), **grounded** (judge sees the retrieved passages), **key facts**, **cites sources**, **retrieved the right section**, **declines** out-of-scope questions | ~$2-3 per run |

The EvalForge suite is generated from `questions.yaml` by `eval/build_suite.py`, so there is one source of truth. It uses EvalForge's `python` provider to call the whole RAG pipeline, and its context-aware judge to check faithfulness.

## Quick start

```bash
git clone https://github.com/akashkorada25/pydocs-rag && cd pydocs-rag
pip install -e ".[dense]"            # drop [dense] for keyword search only

export ANTHROPIC_API_KEY=...         # optional: without it, the app quotes the best passage
pydocs-rag serve                     # open http://127.0.0.1:8000
pydocs-rag ask "How do I sort by two keys?"
```

```bash
pydocs-rag bench --retrievers bm25,tfidf,dense,hybrid           # retrieval benchmark (free)

pip install -e ".[dense,eval]"
python eval/build_suite.py
evalforge run eval/rag_suite.yaml --report report.html          # end-to-end (needs the API key)
pydocs-rag e2e-report                                           # per-check table
evalforge failures latest                                       # why cases failed
```

The API: `POST /api/ask` with `{"question": "...", "retriever": "hybrid", "k": 5}` returns the answer, sources with links, and token usage and cost. Links like `/?q=How%20do%20I%20sort%3F` open the page with that question answered.

## Project structure

```
ragassist/
  ingest.py       reST parsing, cleaning, section-aware chunking
  retrievers.py   BM25, TF-IDF, dense, hybrid (RRF), LLM re-ranking
  llm.py          Anthropic client (standard library) and offline fallback
  pipeline.py     retrieval + grounded generation with citations
  app.py          pipeline cache and the EvalForge entry point
  server.py       web app and JSON API
  bench.py        retrieval benchmark
  e2e_report.py   per-check summary of an EvalForge run
  static/         chat page
eval/
  questions.yaml  38 questions with gold sections and reference answers
  build_suite.py  generates the EvalForge suite
data/raw/         Python documentation sources (PSF License)
tests/            pytest: ingestion, retrievers, pipeline, server, eval data
```

## Design decisions

- **Section-aware chunks with titles.** Each chunk is indexed with its page and section titles, and FAQ sections are themselves questions, which helps both keyword and semantic matching.
- **Reciprocal rank fusion** for hybrid search, because BM25 and cosine scores are on different scales and fusing ranks needs no weights.
- **A cheap model for answers, a stronger one for judging.** Haiku 5.5 answers at a fraction of a cent per question; Sonnet 5.5 grades.
- **Evaluations never silently use the offline fallback.** The EvalForge entry point refuses to run without an API key unless told to.

## Limitations

- 38 questions is enough to see large differences between systems, not small ones. Treat differences of a few points as noise.
- An LLM judge can be wrong. EvalForge's `calibrate` command can measure its agreement with human labels.
- The corpus is a snapshot of the docs on CPython's main branch (see `data/LICENSE-PSF.txt` for the commit), linked to docs.python.org/3, so a few sections may differ from the version you use.

## License

Code: MIT. Python documentation in `data/raw/` and `ragassist/data/`: © Python Software Foundation, used under the PSF License Version 2 (see `data/LICENSE-PSF.txt`).
