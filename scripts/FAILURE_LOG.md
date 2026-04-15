# Failure Log

Track failed questions and fixes during iteration.

| Question | Category | Retrieval OK? | Problem | Fix Applied | Fixed? |
|---|---|---|---|---|---|
| In Blackstone TOC, where does Item 8 begin? | factual | Partially | Off-by-one page in answer | Added hybrid BM25+vector retrieval and source-aware rerank | In progress |
| Which had higher 2025 net sales: iPhone vs Americas? | cross-reference | Partially | Wrong Americas value selected from broader table context | Added cross-reference mode with larger candidate pool and source-constrained ranking | In progress |
| Capital return question (buybacks + dividends + quarterly rate) | cross-reference | Yes | Mixed source noise in final answer | Added stronger prompt guardrail and cross-reference retrieval mode | Improved |
| Query with poor grammar: "what jpmorgan apple results" | ambiguous/cross-ref | No (before) | Sparse/noisy query wording | Added deterministic query expansion + hybrid retrieval + source hints | Re-test required |
