# Retrieval benchmark

32 answerable questions, 727 chunks. GitHub Actions run 44135192700 (Nov 17, 2026).

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| bm25 | 40.6% | 71.9% | 78.1% | 0.559 | 0 |
| tfidf | 43.8% | 62.5% | 81.2% | 0.582 | 1 |
| hybrid-lexical | 50.0% | 75.0% | 81.2% | 0.616 | 1 |
| dense | 71.9% | 90.6% | 90.6% | 0.797 | 8 |
| hybrid | 68.8% | 81.2% | 84.4% | 0.769 | 7 |
| bm25+rerank | 59.4% | 78.1% | 81.2% | 0.686 | 1134 |
| dense+rerank | 71.9% | 90.6% | 90.6% | 0.807 | 1076 |
| hybrid+rerank | 71.9% | 84.4% | 87.5% | 0.796 | 1139 |
