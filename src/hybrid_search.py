"""Hybrid retrieval using Reciprocal Rank Fusion (RRF).

Task 3.3 combines the ranked candidate lists produced by BM25 and dense
retrieval.  Their raw scores are intentionally not combined: BM25 relevance
scores and dense cosine similarities have unrelated scales.  RRF uses only a
candidate's position in each list::

    rrf_score(document) = sum(1 / (rrf_k + rank(document, source)))

Candidates are merged and deduplicated by ``chapter_id``.  A chapter can
contribute at most once per source, and ranks are derived from list order.
The public ``search_hybrid_rrf`` function is the interface consumed by Role 4.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BM25_INDEX_DIR = PROJECT_ROOT / "indexes" / "bm25"
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 5


def _run_bm25_search(index_dir: Path, query: str, top_k: int) -> list[dict[str, Any]]:
    """Load the Role 1 search function lazily to keep fusion independently testable."""
    from bm25_search import search

    return search(index_dir=index_dir, query=query, top_k=top_k)


def _run_dense_search(query: str, top_k: int) -> Sequence[Any]:
    """Load the Role 2 search function lazily to keep fusion independently testable."""
    from faiss_search import search

    return search(query=query, top_k=top_k)


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _canonical_chapter_id(item: Any) -> str | None:
    value = _field(item, "chapter_id")
    if value is None:
        return None
    chapter_id = str(value).strip()
    return chapter_id or None


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_rrf_k(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("rrf_k must be a positive number")
    return float(value)


def reciprocal_rank_fusion(
    bm25_results: Sequence[Any],
    dense_results: Sequence[Any],
    top_k: int,
    *,
    rrf_k: float = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Merge BM25 and dense candidates into one deterministic ranked list.

    ``bm25_results`` normally contains dictionaries and ``dense_results``
    normally contains ``DenseResult`` objects, but either input may contain
    mappings or objects exposing ``chapter_id`` and ``score`` attributes.
    Original scores are retained for diagnostics and never affect ranking.
    """
    top_k = _validate_positive_int(top_k, "top_k")
    rrf_k = _validate_rrf_k(rrf_k)

    candidates: dict[str, dict[str, Any]] = {}

    for source, results in (("bm25", bm25_results), ("dense", dense_results)):
        seen_in_source: set[str] = set()

        for source_rank, item in enumerate(results, start=1):
            chapter_id = _canonical_chapter_id(item)
            if chapter_id is None or chapter_id in seen_in_source:
                continue
            seen_in_source.add(chapter_id)

            candidate = candidates.setdefault(
                chapter_id,
                {
                    "chapter_id": chapter_id,
                    "score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                    "bm25_score": None,
                    "dense_score": None,
                    "retrieval_sources": [],
                },
            )

            # BM25 supplies the rich metadata used by downstream presentation.
            # DenseResult intentionally contains only ID, score, and rank.
            if source == "bm25" and isinstance(item, Mapping):
                reserved = {
                    "chapter_id",
                    "score",
                    "rank",
                    "bm25_rank",
                    "dense_rank",
                    "bm25_score",
                    "dense_score",
                    "retrieval_sources",
                }
                candidate.update({key: value for key, value in item.items() if key not in reserved})

            candidate["score"] += 1.0 / (rrf_k + source_rank)
            candidate[f"{source}_rank"] = source_rank
            candidate[f"{source}_score"] = _field(item, "score")
            candidate["retrieval_sources"].append(source)

    def sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
        source_ranks = [
            rank
            for rank in (candidate["bm25_rank"], candidate["dense_rank"])
            if rank is not None
        ]
        best_source_rank = min(source_ranks)
        return (
            -candidate["score"],
            -len(candidate["retrieval_sources"]),
            best_source_rank,
            candidate["chapter_id"],
        )

    ranked = sorted(candidates.values(), key=sort_key)[:top_k]
    for final_rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = final_rank

    return ranked


def search_hybrid_rrf(
    query: str,
    top_k: int = 5,
    *,
    candidate_k: int | None = None,
    rrf_k: float = DEFAULT_RRF_K,
    bm25_index_dir: Path = DEFAULT_BM25_INDEX_DIR,
) -> list[dict[str, Any]]:
    """Search BM25 and dense indexes and return their RRF-fused candidates.

    The first two parameters form the stable Role 4 contract.  Keyword-only
    parameters support controlled experiments and tests without changing the
    normal call ``search_hybrid_rrf(query, top_k)``.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    top_k = _validate_positive_int(top_k, "top_k")
    rrf_k = _validate_rrf_k(rrf_k)

    if candidate_k is None:
        candidate_k = top_k * DEFAULT_CANDIDATE_MULTIPLIER
    candidate_k = _validate_positive_int(candidate_k, "candidate_k")

    bm25_results = _run_bm25_search(Path(bm25_index_dir), query, candidate_k)
    dense_results = _run_dense_search(query, candidate_k)

    return reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        top_k,
        rrf_k=rrf_k,
    )


__all__ = [
    "DEFAULT_BM25_INDEX_DIR",
    "DEFAULT_RRF_K",
    "reciprocal_rank_fusion",
    "search_hybrid_rrf",
]
