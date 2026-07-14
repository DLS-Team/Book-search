from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# Project modules are importable because backend.py adds both the repository
# root and src/ to sys.path before importing this module.
from bm25_search import preload as preload_bm25
from bm25_search import search as search_bm25
from faiss_search import encode_query
from faiss_search import preload as preload_dense
from faiss_search import search as search_dense_raw
from hybrid_search import reciprocal_rank_fusion
from paragraph_refinement import load_chunks_db, resolve_fragments

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_HNSW_PATH = BASE_DIR / "indexes" / "faiss_ann" / "faiss_hnsw.index"
CHAPTER_IDS_PATH = BASE_DIR / "indexes" / "faiss_flat" / "chapter_ids.json"
BM25_INDEX_DIR = BASE_DIR / "outputs" / "bm25_index"

APP_STATE: dict[str, Any] = {
    "faiss_hnsw": None,
    "chapter_ids": None,
    "chunks_loaded": False,
    "is_ready": False,
    "warnings": [],
}

# RRF scores are usually small (roughly 1 / (k + rank)), so using 2.0 for
# hybrid/refined would mark practically every result as low confidence.
QUALITY_GATE_THRESHOLDS: dict[str, float] = {
    "bm25": 1.5,
    # Dense cosine scores depend heavily on the embedding model and corpus.
    # A fixed 0.4 threshold was too strict for this project and produced
    # false low-confidence warnings for otherwise valid matches.
    "dense": 0.20,
    "dense_ann": 0.20,
    "hybrid": 0.01,
    "refined": 0.01,
}


def safe_read_index(filepath: Path) -> faiss.Index:
    """Read a FAISS index through an ASCII-safe temporary path."""
    if not filepath.is_file():
        raise FileNotFoundError(f"FAISS index not found: {filepath}")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".index")
    os.close(tmp_fd)

    try:
        shutil.copyfile(filepath, tmp_path)
        return faiss.read_index(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _load_chapter_ids(path: Path) -> list[str] | dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Chapter-ID mapping not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, (list, dict)):
        raise TypeError(
            "chapter_ids.json must contain a JSON list or object, "
            f"got {type(mapping).__name__}."
        )

    return mapping


def _chapter_id_at(mapping: list[str] | dict[str, str], index: int) -> str:
    """Resolve a FAISS row number to a stored chapter ID."""
    if isinstance(mapping, list):
        if index < 0 or index >= len(mapping):
            raise IndexError(
                f"FAISS result index {index} is outside chapter-ID mapping "
                f"of size {len(mapping)}."
            )
        return str(mapping[index])

    # JSON object keys are strings even when the producer used integer keys.
    key = str(index)
    if key not in mapping:
        raise KeyError(f"FAISS result index {index} is missing from chapter_ids.json.")
    return str(mapping[key])


def initialize_server_state() -> None:
    """Load reusable online-serving resources once."""
    if APP_STATE["is_ready"]:
        return

    logger.info("Initializing Semantic Book Scene Search server state.")
    APP_STATE["warnings"] = []

    # The text database is required by refinement. Do not hide a failure here:
    # a broken/missing dataset should be visible through backend /health.
    logger.info("Loading the chunk JSONL database.")
    load_chunks_db()
    APP_STATE["chunks_loaded"] = True

    # Warm immutable retrieval resources once. This removes repeated disk
    # deserialization from the online request path.
    logger.info("Preloading BM25 resources.")
    preload_bm25(BM25_INDEX_DIR)

    logger.info("Preloading exact dense resources.")
    preload_dense()

    # ANN is optional for the other modes, so missing ANN artifacts should not
    # prevent BM25/dense/hybrid from starting.
    try:
        APP_STATE["chapter_ids"] = _load_chapter_ids(CHAPTER_IDS_PATH)
        logger.info(
            "Loaded chapter-ID mapping with %d entries.",
            len(APP_STATE["chapter_ids"]),
        )
    except Exception as exc:
        warning = f"dense_ann unavailable: {exc}"
        APP_STATE["warnings"].append(warning)
        logger.warning(warning)
        APP_STATE["chapter_ids"] = None

    try:
        APP_STATE["faiss_hnsw"] = safe_read_index(FAISS_HNSW_PATH)
        logger.info(
            "Loaded HNSW index with %d vectors.",
            APP_STATE["faiss_hnsw"].ntotal,
        )
    except Exception as exc:
        warning = f"dense_ann unavailable: {exc}"
        APP_STATE["warnings"].append(warning)
        logger.warning(warning)
        APP_STATE["faiss_hnsw"] = None

    if APP_STATE["faiss_hnsw"] is not None and APP_STATE["chapter_ids"] is not None:
        mapping_size = len(APP_STATE["chapter_ids"])
        index_size = int(APP_STATE["faiss_hnsw"].ntotal)
        if mapping_size != index_size:
            warning = (
                "dense_ann mapping/index size mismatch: "
                f"{mapping_size} IDs for {index_size} vectors."
            )
            APP_STATE["warnings"].append(warning)
            logger.warning(warning)

    APP_STATE["is_ready"] = True
    logger.info("Server state initialized.")


def _to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)

    if is_dataclass(item) and not isinstance(item, type):
        return asdict(item)  # type: ignore[arg-type]

    if hasattr(item, "model_dump"):
        # Use getattr to avoid static type checkers complaining about "model_dump"
        model_dump = getattr(item, "model_dump")
        return dict(model_dump())

    if hasattr(item, "_asdict"):
        _asdict = getattr(item, "_asdict")
        return dict(_asdict())

    raise TypeError(
        "Search result must be a dict, dataclass instance, "
        f"Pydantic model, or namedtuple; got {type(item).__name__}."
    )

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default
    return number


def _format_result(
    item: dict[str, Any],
    method_name: str,
    rank: int,
) -> dict[str, Any]:
    """Normalize outputs from all retrieval implementations."""
    chapter_id = item.get("chapter_id", item.get("chunk_id", "N/A"))
    paragraph_position = item.get(
        "paragraph_position",
        item.get("paragraph_start", item.get("paragraph_index", -1)),
    )

    score = _as_float(
        item.get(
            "score",
            item.get("similarity", item.get("rrf_score", item.get("score_or_rank", 0.0))),
        )
    )

    book = item.get("book", item.get("book_title", item.get("title", "Unknown Book")))
    chapter = item.get(
        "chapter",
        item.get("chapter_title", item.get("chapter_name", "Unknown Chapter")),
    )
    fragment = item.get(
        "best_fragment",
        item.get("fragment", item.get("text", "Full chapter returned")),
    )

    gutenberg_id = item.get("book_id", item.get("gutenberg_id", "N/A"))

    return {
        "book": str(book or "Unknown Book"),
        "author": str(item.get("author") or "Unknown Author"),
        "chapter": str(chapter or "Unknown Chapter"),
        "chapter_id": str(chapter_id),
        "best_fragment": str(fragment or ""),
        "paragraph_position": paragraph_position,
        "search_method": method_name,
        "score": score,
        "rank": int(item.get("rank", rank)),
        # Kept for compatibility with older frontend/backend contracts.
        "score_or_rank": score,
        "provenance": (
            f"Gutenberg ID: {gutenberg_id} | "
            f"Chapter ID: {chapter_id} | "
            f"Para Pos: {paragraph_position}"
        ),
    }


def apply_quality_gate(
    results: list[dict[str, Any]],
    method: str,
) -> list[dict[str, Any]]:
    if not results:
        return []

    threshold = QUALITY_GATE_THRESHOLDS.get(method)
    top_score = _as_float(results[0].get("score", results[0].get("score_or_rank", 0.0)))

    low_confidence = threshold is not None and top_score < threshold
    warning: str | None = None

    if low_confidence:
        warning = f"Top score ({top_score:.4f}) is below threshold ({threshold:.4f})."

    if method == "refined":
        fragment = str(results[0].get("best_fragment", ""))
        if "[Error:" in fragment:
            low_confidence = True
            warning = "Fragment resolution failed."

    for result in results:
        result["low_confidence"] = low_confidence
        result["warning"] = warning

    return results


def _search_dense_ann(query: str, top_k: int) -> list[dict[str, Any]]:
    index = APP_STATE.get("faiss_hnsw")
    chapter_ids = APP_STATE.get("chapter_ids")

    # Fixed typo from the previous implementation:
    # faiss_hnwn -> faiss_hnsw
    if index is None or chapter_ids is None:
        details = "; ".join(APP_STATE.get("warnings", []))
        raise RuntimeError(
            "HNSW index or chapter-ID mapping is not loaded."
            + (f" Details: {details}" if details else "")
        )

    query_vector = np.asarray(encode_query(query), dtype=np.float32).reshape(1, -1)
    if query_vector.shape[1] != index.d:
        raise ValueError(
            "Query-vector dimension does not match HNSW index: "
            f"{query_vector.shape[1]} != {index.d}."
        )

    # Cosine search uses inner product over L2-normalized vectors.
    faiss.normalize_L2(query_vector)
    scores, ids = index.search(query_vector, top_k)

    results: list[dict[str, Any]] = []
    for rank, (score, numeric_id) in enumerate(
        zip(scores[0], ids[0]),
        start=1,
    ):
        numeric_id = int(numeric_id)
        if numeric_id < 0:
            continue

        results.append(
            {
                "chapter_id": _chapter_id_at(chapter_ids, numeric_id),
                "score": float(score),
                "rank": rank,
            }
        )

    return results


def _search_uncached(
    query: str,
    method: str = "hybrid",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    total_started = time.perf_counter()
    if not APP_STATE["is_ready"]:
        raise RuntimeError(
            "Server is not initialized. Call initialize_server_state() first."
        )

    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    supported_methods = {"bm25", "dense", "dense_ann", "hybrid", "refined"}
    if method not in supported_methods:
        raise ValueError(
            f"Unknown search method: {method}. "
            f"Supported methods: {sorted(supported_methods)}"
        )

    logger.info("Search query=%r method=%s top_k=%d", query[:80], method, top_k)

    raw_results: list[Any]

    if method == "bm25":
        bm25_results = search_bm25(BM25_INDEX_DIR, query, top_k)
        raw_results = [
            {
                **_to_dict(item),
                "rank": rank,
            }
            for rank, item in enumerate(bm25_results, start=1)
        ]

    elif method == "dense":
        dense_results = search_dense_raw(query, top_k)
        raw_results = [
            {
                **_to_dict(item),
                "rank": rank,
            }
            for rank, item in enumerate(dense_results, start=1)
        ]

    elif method == "dense_ann":
        raw_results = _search_dense_ann(query, top_k)

    else:
        candidate_k = max(top_k * 5, 20)

        # Production hybrid uses BM25 + HNSW instead of exact FAISS Flat.
        # The two retrieval branches are independent and run concurrently.
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="online-hybrid",
        ) as pool:
            bm25_future = pool.submit(
                search_bm25,
                BM25_INDEX_DIR,
                query,
                candidate_k,
            )
            ann_future = pool.submit(
                _search_dense_ann,
                query,
                candidate_k,
            )
            bm25_candidates = bm25_future.result()
            ann_candidates = ann_future.result()

        raw_results = reciprocal_rank_fusion(
            bm25_candidates,
            ann_candidates,
            top_k,
        )

    normalized_raw = [_to_dict(item) for item in raw_results]

    # Every search mode must return concrete source text and metadata.
    # Retrieval implementations often return only chapter_id + score. Without
    # this step, the frontend displays Unknown Book / Unknown Chapter and has
    # no matched chunk to render.
    logger.info("Resolving chunk text and metadata for method=%s.", method)
    try:
        resolved_raw = [
            _to_dict(item)
            for item in resolve_fragments(normalized_raw)
        ]
    except Exception as exc:
        logger.exception("Fragment resolution failed for method=%s.", method)
        raise RuntimeError(
            f"Search succeeded, but chunk metadata/text resolution failed: {exc}"
        ) from exc

    # Preserve retrieval score/rank when the resolver returns only text and
    # metadata. Merge by chapter_id first; fall back to list position.
    originals_by_id: dict[str, dict[str, Any]] = {
        str(item.get("chapter_id", item.get("chunk_id"))): item
        for item in normalized_raw
        if item.get("chapter_id", item.get("chunk_id")) is not None
    }

    merged_raw: list[dict[str, Any]] = []
    for position, resolved in enumerate(resolved_raw):
        resolved_id = str(
            resolved.get("chapter_id", resolved.get("chunk_id", ""))
        )
        original = originals_by_id.get(resolved_id)

        if original is None and position < len(normalized_raw):
            original = normalized_raw[position]

        # Original retrieval fields first, resolved metadata/text override them.
        merged = dict(original or {})
        merged.update(resolved)
        merged_raw.append(merged)

    formatted = [
        _format_result(item, method, rank)
        for rank, item in enumerate(merged_raw, start=1)
    ]

    final_results = apply_quality_gate(formatted, method)

    logger.info(
        "Search completed method=%s top_k=%d total_ms=%.2f",
        method,
        top_k,
        (time.perf_counter() - total_started) * 1000,
    )
    return final_results


@lru_cache(maxsize=256)
def _cached_search(
    query: str,
    method: str,
    top_k: int,
) -> tuple[dict[str, Any], ...]:
    return tuple(_search_uncached(query, method, top_k))


def search(
    query: str,
    method: str = "hybrid",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    # Return copies because downstream presentation code may mutate results.
    return copy.deepcopy(list(_cached_search(query.strip(), method, top_k)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    try:
        initialize_server_state()
        results = search("cozy winter night", method="dense_ann", top_k=2)

        print("\n--- RESULTS ---")
        for result in results:
            print(
                f"Rank: {result['rank']} | "
                f"ID: {result['chapter_id']} | "
                f"Score: {result['score']:.4f}"
            )
            if result["low_confidence"]:
                print(f"WARNING: {result['warning']}")
            print("-" * 30)
    except Exception:
        logger.exception("Search-engine test failed.")
