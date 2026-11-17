# End-to-end evaluation

EvalForge run `20261117-173005-a8aa7f` (2026-11-17T17:30:05+00:00). Judge cost for the whole run: $1.35.

| System | Correct | Grounded in sources | Key facts | Cites sources | Retrieved right section | Declines out-of-scope | Median latency | Cost / question |
|---|---|---|---|---|---|---|---|---|
| closed-book | 100% | 0% | 100% | 12% | 0% | 0% | 1.5 s | $0.0001 |
| bm25 | 81% | 94% | 91% | 100% | 78% | 100% | 1.5 s | $0.0004 |
| dense | 91% | 100% | 100% | 100% | 91% | 100% | 1.6 s | $0.0003 |
| hybrid | 91% | 100% | 94% | 100% | 84% | 100% | 1.6 s | $0.0004 |
| dense+rerank | 91% | 100% | 97% | 100% | 91% | 100% | 2.6 s | $0.0010 |
