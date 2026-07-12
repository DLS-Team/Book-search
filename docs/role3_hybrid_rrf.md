# Role 3 — Hybrid BM25 + Dense RRF

## Decision

Hybrid retrieval uses Reciprocal Rank Fusion (RRF) over the BM25 and FAISS
Flat result lists:

```text
RRF(chapter) = sum(1 / (60 + source_rank))
```

Both sources have equal weight. The raw BM25 score and dense cosine similarity
are retained for diagnostics but do not affect the fused order. Their scales
are unrelated, so raw addition is invalid without calibration. Normalized
weighted fusion would introduce a normalization method, tuned weights, and
query-dependent score-distribution sensitivity. RRF is the stable initial
baseline because it depends only on ranks.

## Merge, deduplication, and ranking

Each retriever supplies five times the requested final result count. Candidates
are merged by the canonical string form of `chapter_id`. Duplicate IDs inside
one source contribute only at their first list position; a chapter found by
both sources receives both reciprocal-rank contributions. Results are sorted
by RRF score, source agreement, best source rank, and `chapter_id`, then
truncated to `top_k`.

## Role 4 handoff

```python
from hybrid_search import search_hybrid_rrf

results = search_hybrid_rrf(query, top_k=5)
```

The function returns dictionaries containing `chapter_id`, final `score`,
final `rank`, per-source ranks and scores, and `retrieval_sources`. BM25
metadata is preserved when available. Dense-only candidates can be resolved
through Role 4's corpus/refinement layer.

With two sources and `rrf_k=60`, the maximum score is approximately `0.0328`.
It must not be compared with BM25 or cosine thresholds. In particular, Role 4
must calibrate the hybrid quality gate independently; the current placeholder
threshold `2.0` is outside the RRF score range.
