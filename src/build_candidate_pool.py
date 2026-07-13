"""Build a blinded relevance-labeling pool from four retrieval methods.

The public labeling file intentionally contains no method names, retrieval
ranks, or scores.  A separate manifest preserves only the ranked chapter IDs
needed for later retrieval metrics.  Scores returned by BM25/FAISS/RRF exist
only transiently in memory and are never serialized.

Default methods and depths:
    - BM25: top 20 from 100 candidates
    - Dense FAISS Flat: top 20 from 100 candidates
    - Dense FAISS HNSW: top 20
    - Hybrid RRF: top 20 fused from the 100 BM25 + 100 Flat candidates
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import pickle
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import bm25s
import faiss
import numpy as np

from bm25_search import simple_tokenize
from embed_chapters import MODEL_NAME, NORMALIZATION, USE_MOCK_ENCODER, l2_normalize, load_encoder
from hybrid_search import reciprocal_rank_fusion


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "data" / "eval" / "query.txt"
DEFAULT_LABELING_OUTPUT = PROJECT_ROOT / "data" / "eval" / "candidate_pool_for_labeling.json"
DEFAULT_MANIFEST_OUTPUT = PROJECT_ROOT / "data" / "eval" / "retrieval_manifest.json"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "processed_chapters.jsonl"
DEFAULT_BM25_DIR = PROJECT_ROOT / "indexes" / "bm25"
DEFAULT_FLAT_INDEX_PATH = PROJECT_ROOT / "indexes" / "faiss_flat" / "flat.index"
DEFAULT_ANN_INDEX_PATH = PROJECT_ROOT / "indexes" / "faiss_ann" / "faiss_hnsw.index"
DEFAULT_CHAPTER_IDS_PATH = PROJECT_ROOT / "indexes" / "faiss_flat" / "chapter_ids.json"
DEFAULT_EMBEDDING_STATS_PATH = PROJECT_ROOT / "indexes" / "faiss_flat" / "embedding_stats.json"

DEFAULT_TOP_K = 20
DEFAULT_CANDIDATE_K = 100
DEFAULT_RRF_K = 60.0

METHODS = ("bm25", "dense_flat", "dense_ann", "hybrid_rrf")
CATEGORY_BY_PREFIX = {
    "EK": "exact_keyword",
    "SS": "semantic_scene",
    "EM": "emotion_mood",
    "AT": "atmosphere",
    "AS": "action_situation",
    "AW": "ambiguous_weak_evidence",
}
QUERY_PATTERN = re.compile(r"^(EK|SS|EM|AT|AS|AW)(\d{2})\.\s+(.+?)\s*$")
QUERY_LIKE_PATTERN = re.compile(r"^(EK|SS|EM|AT|AS|AW)\d")

LABELING_FORBIDDEN_KEYS = {
    "score",
    "rank",
    "method",
    "methods",
    "retrieval",
    "annotator",
    "notes",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    category: str
    query: str


def parse_queries(path: Path) -> list[QuerySpec]:
    """Parse Role 3's human-readable query file and validate its IDs/categories."""
    if not path.is_file():
        raise FileNotFoundError(f"Query file not found: {path}")

    queries: list[QuerySpec] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = QUERY_PATTERN.match(line)
            if match is None:
                if QUERY_LIKE_PATTERN.match(line):
                    raise ValueError(
                        f"Malformed query line {line_number} in {path}: {line!r}"
                    )
                # Title, category headings, and separator lines are intentionally ignored.
                continue

            prefix, number, query_text = match.groups()
            query_id = f"{prefix}{number}"
            if query_id in seen_ids:
                raise ValueError(f"Duplicate query ID {query_id} in {path}")
            if not query_text:
                raise ValueError(f"Query {query_id} has empty text")

            seen_ids.add(query_id)
            queries.append(
                QuerySpec(
                    query_id=query_id,
                    category=CATEGORY_BY_PREFIX[prefix],
                    query=query_text,
                )
            )

    if not queries:
        raise ValueError(f"No queries found in {path}")

    present_categories = {query.category for query in queries}
    missing_categories = set(CATEGORY_BY_PREFIX.values()) - present_categories
    if missing_categories:
        missing = ", ".join(sorted(missing_categories))
        raise ValueError(f"Query file is missing required categories: {missing}")

    return queries


def _canonical_chapter_id(value: Any) -> str:
    if value is None:
        raise ValueError("Retriever returned a candidate without chapter_id")
    chapter_id = str(value).strip()
    if not chapter_id:
        raise ValueError("Retriever returned an empty chapter_id")
    return chapter_id


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.candidate_k < args.top_k * 5:
        raise ValueError("--candidate-k must be at least five times --top-k")
    if not math.isfinite(args.rrf_k) or args.rrf_k <= 0:
        raise ValueError("--rrf-k must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    required_paths = {
        "processed corpus": args.corpus,
        "BM25 index": args.bm25_index_dir / "bm25_index",
        "BM25 metadata": args.bm25_index_dir / "metadata.pkl",
        "BM25 registry": args.bm25_index_dir / "bm25_registry.json",
        "FAISS Flat index": args.flat_index,
        "FAISS ANN index": args.ann_index,
        "FAISS chapter IDs": args.chapter_ids,
        "embedding statistics": args.embedding_stats,
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n  " + "\n  ".join(missing))

    if args.labeling_output.resolve() == args.manifest_output.resolve():
        raise ValueError("Labeling output and manifest output must be different files")

    existing_outputs = [
        path
        for path in (args.labeling_output, args.manifest_output)
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {paths}")

    if USE_MOCK_ENCODER:
        raise RuntimeError(
            "USE_MOCK_ENCODER is enabled. Real relevance pools must use the encoder "
            "that built the dense indexes."
        )


def _validate_index_metadata(
    *,
    chapter_ids: Sequence[str],
    embedding_stats: dict[str, Any],
    bm25_registry: dict[str, Any],
    flat_index: faiss.Index,
    ann_index: faiss.Index,
    metadata_count: int,
    top_k: int,
    candidate_k: int,
) -> None:
    id_count = len(chapter_ids)
    counts = {
        "chapter_ids": id_count,
        "embedding_stats": int(embedding_stats.get("num_chapters", -1)),
        "bm25_registry": int(bm25_registry.get("documents_indexed", -1)),
        "bm25_metadata": metadata_count,
        "faiss_flat": int(flat_index.ntotal),
        "faiss_ann": int(ann_index.ntotal),
    }
    if len(set(counts.values())) != 1:
        details = ", ".join(f"{name}={count}" for name, count in counts.items())
        raise ValueError(f"Corpus/index document counts do not agree: {details}")

    if id_count < max(top_k, candidate_k):
        raise ValueError(
            f"Indexes contain only {id_count} documents, fewer than requested retrieval depth"
        )
    if len(set(chapter_ids)) != id_count:
        raise ValueError("chapter_ids.json contains duplicate chapter IDs")

    expected_dimension = int(embedding_stats.get("embedding_dim", -1))
    if flat_index.d != expected_dimension or ann_index.d != expected_dimension:
        raise ValueError(
            "FAISS dimensions do not match embedding_stats.json: "
            f"stats={expected_dimension}, flat={flat_index.d}, ann={ann_index.d}"
        )
    if flat_index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS Flat index is not configured for inner product")
    if ann_index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS ANN index is not configured for inner product")

    recorded_model = embedding_stats.get("model_name")
    if recorded_model != MODEL_NAME:
        raise ValueError(
            f"Dense index model mismatch: stats={recorded_model!r}, runtime={MODEL_NAME!r}"
        )
    recorded_normalization = str(embedding_stats.get("normalization", "")).lower()
    if recorded_normalization != NORMALIZATION.lower():
        raise ValueError(
            "Dense index normalization mismatch: "
            f"stats={recorded_normalization!r}, runtime={NORMALIZATION!r}"
        )


def _retrieve_bm25_candidates(
    *,
    index_dir: Path,
    queries: Sequence[QuerySpec],
    candidate_k: int,
) -> tuple[list[list[dict[str, Any]]], int]:
    """Load BM25 once and retrieve the fusion candidate depth for every query."""
    logger.info("Loading BM25 index and metadata from %s", index_dir)
    retriever = bm25s.BM25.load(str(index_dir / "bm25_index"))
    with (index_dir / "metadata.pkl").open("rb") as source:
        metadata = pickle.load(source)

    metadata_count = len(metadata)
    raw_k = min(candidate_k * 5, metadata_count)
    tokenized_queries = [simple_tokenize(query.query) for query in queries]
    document_ids, raw_scores = retriever.retrieve(
        tokenized_queries,
        k=raw_k,
        corpus=None,
    )

    output: list[list[dict[str, Any]]] = []
    for query, row_ids, row_scores in zip(queries, document_ids, raw_scores):
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for document_id, _score in zip(row_ids, row_scores):
            meta = metadata[int(document_id)]
            chapter_id = _canonical_chapter_id(meta.get("chapter_id"))
            if chapter_id in seen:
                continue
            seen.add(chapter_id)
            candidates.append({"chapter_id": chapter_id})
            if len(candidates) == candidate_k:
                break

        if len(candidates) != candidate_k:
            raise RuntimeError(
                f"BM25 returned {len(candidates)} unique candidates for {query.query_id}; "
                f"expected {candidate_k}"
            )
        output.append(candidates)

    # Large BM25 assets are no longer needed after all queries have been retrieved.
    del retriever, metadata, document_ids, raw_scores
    gc.collect()
    return output, metadata_count


def _encode_queries(queries: Sequence[QuerySpec], batch_size: int) -> np.ndarray:
    logger.info("Loading dense encoder %s", MODEL_NAME)
    encoder = load_encoder()
    vectors = encoder.encode(
        [query.query for query in queries],
        batch_size=batch_size,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    vectors = l2_normalize(vectors).astype(np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(queries):
        raise RuntimeError(f"Encoder returned unexpected query-vector shape: {vectors.shape}")
    del encoder
    gc.collect()
    return vectors


def _faiss_rows_to_candidates(
    *,
    method_name: str,
    queries: Sequence[QuerySpec],
    chapter_ids: Sequence[str],
    indexes: np.ndarray,
    expected_k: int,
) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for query, row_indexes in zip(queries, indexes):
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index in row_indexes:
            if int(index) == -1:
                continue
            chapter_id = _canonical_chapter_id(chapter_ids[int(index)])
            if chapter_id in seen:
                continue
            seen.add(chapter_id)
            candidates.append({"chapter_id": chapter_id})
            if len(candidates) == expected_k:
                break

        if len(candidates) != expected_k:
            raise RuntimeError(
                f"{method_name} returned {len(candidates)} unique candidates for "
                f"{query.query_id}; expected {expected_k}"
            )
        output.append(candidates)
    return output


def _ranked_ids(candidates: Sequence[Any], top_k: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if isinstance(candidate, dict):
            value = candidate.get("chapter_id")
        else:
            value = getattr(candidate, "chapter_id", None)
        chapter_id = _canonical_chapter_id(value)
        if chapter_id in seen:
            continue
        seen.add(chapter_id)
        output.append({"rank": len(output) + 1, "chapter_id": chapter_id})
        if len(output) == top_k:
            break
    if len(output) != top_k:
        raise RuntimeError(f"Ranked result list has {len(output)} items; expected {top_k}")
    return output


def _build_manifest_and_pool_ids(
    *,
    queries: Sequence[QuerySpec],
    bm25_candidates: Sequence[Sequence[dict[str, Any]]],
    flat_candidates: Sequence[Sequence[dict[str, Any]]],
    ann_candidates: Sequence[Sequence[dict[str, Any]]],
    top_k: int,
    rrf_k: float,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    manifest_queries: list[dict[str, Any]] = []
    pool_ids_by_query: dict[str, set[str]] = {}

    for query, bm25_row, flat_row, ann_row in zip(
        queries,
        bm25_candidates,
        flat_candidates,
        ann_candidates,
    ):
        hybrid_row = reciprocal_rank_fusion(
            bm25_row,
            flat_row,
            top_k=top_k,
            rrf_k=rrf_k,
        )

        methods = {
            "bm25": _ranked_ids(bm25_row, top_k),
            "dense_flat": _ranked_ids(flat_row, top_k),
            "dense_ann": _ranked_ids(ann_row, top_k),
            "hybrid_rrf": _ranked_ids(hybrid_row, top_k),
        }
        pool_ids = {
            result["chapter_id"]
            for method_results in methods.values()
            for result in method_results
        }
        pool_ids_by_query[query.query_id] = pool_ids
        manifest_queries.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "query": query.query,
                "methods": methods,
            }
        )

        total_before_deduplication = top_k * len(METHODS)
        logger.info(
            "%s: %d method results -> %d unique candidates (%d overlaps)",
            query.query_id,
            total_before_deduplication,
            len(pool_ids),
            total_before_deduplication - len(pool_ids),
        )

    return manifest_queries, pool_ids_by_query


def _load_selected_corpus_rows(
    corpus_path: Path,
    required_ids: set[str],
) -> dict[str, dict[str, Any]]:
    logger.info(
        "Scanning %s once to resolve %d unique candidate IDs",
        corpus_path,
        len(required_ids),
    )
    remaining = set(required_ids)
    selected: dict[str, dict[str, Any]] = {}

    with corpus_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            chapter_id = _canonical_chapter_id(row.get("chapter_id"))
            if chapter_id in remaining:
                selected[chapter_id] = row
                remaining.remove(chapter_id)
                if not remaining:
                    break
            if line_number % 100_000 == 0:
                logger.info(
                    "Scanned %d corpus rows; %d candidate IDs remain",
                    line_number,
                    len(remaining),
                )

    if remaining:
        preview = ", ".join(sorted(remaining)[:10])
        raise KeyError(
            f"Could not resolve {len(remaining)} candidate IDs in the corpus. "
            f"First missing IDs: {preview}"
        )
    return selected


def _blind_order_key(query_id: str, chapter_id: str) -> str:
    value = f"{query_id}\0{chapter_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _build_labeling_queries(
    *,
    queries: Sequence[QuerySpec],
    pool_ids_by_query: dict[str, set[str]],
    corpus_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    required_metadata = ("book_id", "title", "author", "chapter_title", "text")

    for query in queries:
        ordered_ids = sorted(
            pool_ids_by_query[query.query_id],
            key=lambda chapter_id: _blind_order_key(query.query_id, chapter_id),
        )
        candidates: list[dict[str, Any]] = []
        for chapter_id in ordered_ids:
            row = corpus_rows[chapter_id]
            missing_fields = [field for field in required_metadata if field not in row]
            if missing_fields:
                fields = ", ".join(missing_fields)
                raise KeyError(f"Corpus row {chapter_id} is missing required fields: {fields}")
            candidates.append(
                {
                    "candidate_id": f"{query.query_id}::{chapter_id}",
                    "chapter_id": chapter_id,
                    "book_id": row["book_id"],
                    "title": row["title"],
                    "author": row["author"],
                    "chapter_title": row["chapter_title"],
                    "text": row["text"],
                    "relevance": None,
                }
            )

        output.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "query": query.query,
                "candidates": candidates,
            }
        )
    return output


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_keys(nested)


def _validate_outputs(
    *,
    labeling_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    pool_ids_by_query: dict[str, set[str]],
    top_k: int,
) -> None:
    labeling_keys = set(_walk_keys(labeling_payload))
    forbidden_found = LABELING_FORBIDDEN_KEYS & labeling_keys
    if forbidden_found:
        fields = ", ".join(sorted(forbidden_found))
        raise AssertionError(f"Blinded labeling output contains forbidden fields: {fields}")

    if "score" in set(_walk_keys(manifest_payload)):
        raise AssertionError("Retrieval manifest must not contain scores")

    labeling_queries = {
        query["query_id"]: query
        for query in labeling_payload["queries"]
    }
    for query in manifest_payload["queries"]:
        query_id = query["query_id"]
        methods = query["methods"]
        if set(methods) != set(METHODS):
            raise AssertionError(f"{query_id} does not contain all four methods")

        union: set[str] = set()
        for method_name, results in methods.items():
            if len(results) != top_k:
                raise AssertionError(
                    f"{query_id}/{method_name} has {len(results)} results; expected {top_k}"
                )
            expected_ranks = list(range(1, top_k + 1))
            actual_ranks = [result["rank"] for result in results]
            if actual_ranks != expected_ranks:
                raise AssertionError(f"{query_id}/{method_name} ranks are not sequential")
            method_ids = [result["chapter_id"] for result in results]
            if len(set(method_ids)) != top_k:
                raise AssertionError(f"{query_id}/{method_name} contains duplicate IDs")
            union.update(method_ids)

        if union != pool_ids_by_query[query_id]:
            raise AssertionError(f"{query_id} manifest union differs from the in-memory pool")
        labeling_ids = {
            candidate["chapter_id"]
            for candidate in labeling_queries[query_id]["candidates"]
        }
        if labeling_ids != union:
            raise AssertionError(f"{query_id} labeling candidates differ from manifest union")


def _write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _log_summary(
    queries: Sequence[QuerySpec],
    pool_ids_by_query: dict[str, set[str]],
    top_k: int,
) -> None:
    category_counts: dict[str, int] = defaultdict(int)
    category_candidates: dict[str, int] = defaultdict(int)
    total_candidates = 0

    for query in queries:
        candidate_count = len(pool_ids_by_query[query.query_id])
        category_counts[query.category] += 1
        category_candidates[query.category] += candidate_count
        total_candidates += candidate_count

    logger.info("Completed %d queries", len(queries))
    logger.info(
        "Pooled %d query-candidate pairs after deduplicating %d raw method results",
        total_candidates,
        len(queries) * top_k * len(METHODS),
    )
    for category in sorted(category_counts):
        count = category_counts[category]
        total = category_candidates[category]
        logger.info(
            "Category %s: %d queries, %d candidates, average %.2f/query",
            category,
            count,
            total,
            total / count,
        )


def build_candidate_pool(args: argparse.Namespace) -> None:
    _validate_arguments(args)
    queries = parse_queries(args.queries)
    if args.limit is not None:
        queries = queries[: args.limit]
    logger.info("Parsed %d queries from %s", len(queries), args.queries)

    chapter_ids_raw = _read_json(args.chapter_ids)
    if not isinstance(chapter_ids_raw, list):
        raise ValueError(f"chapter_ids.json must contain a JSON list: {args.chapter_ids}")
    chapter_ids = [_canonical_chapter_id(value) for value in chapter_ids_raw]
    embedding_stats = _read_json(args.embedding_stats)
    bm25_registry = _read_json(args.bm25_index_dir / "bm25_registry.json")

    bm25_candidates, metadata_count = _retrieve_bm25_candidates(
        index_dir=args.bm25_index_dir,
        queries=queries,
        candidate_k=args.candidate_k,
    )

    logger.info("Loading FAISS Flat index from %s", args.flat_index)
    flat_index = faiss.read_index(str(args.flat_index))
    logger.info("Loading FAISS ANN index from %s", args.ann_index)
    ann_index = faiss.read_index(str(args.ann_index))

    _validate_index_metadata(
        chapter_ids=chapter_ids,
        embedding_stats=embedding_stats,
        bm25_registry=bm25_registry,
        flat_index=flat_index,
        ann_index=ann_index,
        metadata_count=metadata_count,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    query_vectors = _encode_queries(queries, args.batch_size)
    if query_vectors.shape[1] != flat_index.d:
        raise ValueError(
            f"Runtime encoder dimension {query_vectors.shape[1]} does not match index dimension {flat_index.d}"
        )

    logger.info("Batch-searching FAISS Flat at depth %d", args.candidate_k)
    flat_scores, flat_indexes = flat_index.search(query_vectors, args.candidate_k)
    logger.info("Batch-searching FAISS ANN at depth %d", args.top_k)
    ann_scores, ann_indexes = ann_index.search(query_vectors, args.top_k)

    flat_candidates = _faiss_rows_to_candidates(
        method_name="Dense Flat",
        queries=queries,
        chapter_ids=chapter_ids,
        indexes=flat_indexes,
        expected_k=args.candidate_k,
    )
    ann_candidates = _faiss_rows_to_candidates(
        method_name="Dense ANN",
        queries=queries,
        chapter_ids=chapter_ids,
        indexes=ann_indexes,
        expected_k=args.top_k,
    )

    del flat_index, ann_index, query_vectors
    del flat_scores, flat_indexes, ann_scores, ann_indexes
    gc.collect()

    manifest_queries, pool_ids_by_query = _build_manifest_and_pool_ids(
        queries=queries,
        bm25_candidates=bm25_candidates,
        flat_candidates=flat_candidates,
        ann_candidates=ann_candidates,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )

    required_ids = set().union(*pool_ids_by_query.values())
    corpus_rows = _load_selected_corpus_rows(args.corpus, required_ids)
    labeling_queries = _build_labeling_queries(
        queries=queries,
        pool_ids_by_query=pool_ids_by_query,
        corpus_rows=corpus_rows,
    )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    labeling_payload = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at,
        "queries": labeling_queries,
    }
    manifest_payload = {
        "schema_version": "1.0",
        "generated_at_utc": generated_at,
        "configuration": {
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "rrf_k": args.rrf_k,
        },
        "queries": manifest_queries,
    }

    _validate_outputs(
        labeling_payload=labeling_payload,
        manifest_payload=manifest_payload,
        pool_ids_by_query=pool_ids_by_query,
        top_k=args.top_k,
    )
    _write_json_atomic(labeling_payload, args.labeling_output)
    _write_json_atomic(manifest_payload, args.manifest_output)

    _log_summary(queries, pool_ids_by_query, args.top_k)
    logger.info("Blinded labeling pool: %s", args.labeling_output)
    logger.info("Private retrieval manifest: %s", args.manifest_output)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pool top results from BM25, Dense Flat, Dense ANN, and Hybrid RRF."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--labeling-output", type=Path, default=DEFAULT_LABELING_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--bm25-index-dir", type=Path, default=DEFAULT_BM25_DIR)
    parser.add_argument("--flat-index", type=Path, default=DEFAULT_FLAT_INDEX_PATH)
    parser.add_argument("--ann-index", type=Path, default=DEFAULT_ANN_INDEX_PATH)
    parser.add_argument("--chapter-ids", type=Path, default=DEFAULT_CHAPTER_IDS_PATH)
    parser.add_argument("--embedding-stats", type=Path, default=DEFAULT_EMBEDDING_STATS_PATH)
    parser.add_argument("--top-k", type=_positive_int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=_positive_int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=float, default=DEFAULT_RRF_K)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--limit", type=_positive_int, help="Process only the first N queries")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = create_argument_parser()
    build_candidate_pool(parser.parse_args())


if __name__ == "__main__":
    main()
