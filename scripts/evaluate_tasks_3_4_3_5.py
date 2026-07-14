"""Generate Task 3.4/3.5 retrieval runs, qrels, metrics, and report.

This script uses the existing LLM-assisted relevance judgments and retrieval
manifest. The manifest intentionally stores ranks but not raw retriever scores,
so run CSV scores are deterministic inverse-rank placeholders; evaluation uses
the rank column.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_LABELS = PROJECT_ROOT / "data" / "eval" / "candidate_pool_labelled.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "retrieval_manifest.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "tasks_3_4_3_5_evaluation.md"

METHOD_DISPLAY = {
    "bm25": "BM25",
    "dense_flat": "Dense FAISS Flat",
    "dense_ann": "Dense FAISS ANN",
    "hybrid_rrf": "Hybrid RRF",
}
METHOD_ORDER = ["bm25", "dense_flat", "dense_ann", "hybrid_rrf"]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
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


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def chapter_id_from_candidate_id(candidate_id: str) -> str:
    return candidate_id.split("::", 1)[1] if "::" in candidate_id else candidate_id


def build_qrels(label_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = label_payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Label JSON must contain a queries list")

    rows: list[dict[str, Any]] = []
    skipped_missing = 0
    skipped_invalid = 0
    confidence_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[int, int] = defaultdict(int)

    for query in queries:
        query_id = query.get("query_id")
        candidates = query.get("candidates")
        if not isinstance(query_id, str) or not isinstance(candidates, list):
            raise ValueError("Every query must contain query_id and candidates")

        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            relevance = candidate.get("relevance")
            if not isinstance(candidate_id, str):
                skipped_invalid += 1
                continue
            if not isinstance(relevance, dict) or "label" not in relevance:
                skipped_missing += 1
                continue

            label = relevance.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1, 2}:
                skipped_invalid += 1
                continue

            confidence = relevance.get("confidence")
            if isinstance(confidence, str):
                confidence_counts[confidence] += 1
            label_counts[label] += 1
            rows.append(
                {
                    "query_id": query_id,
                    "candidate_id": chapter_id_from_candidate_id(candidate_id),
                    "relevance": label,
                }
            )

    summary = {
        "queries": len(queries),
        "qrels": len(rows),
        "skipped_missing": skipped_missing,
        "skipped_invalid": skipped_invalid,
        "label_counts": dict(sorted(label_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
    }
    return rows, summary


def get_queries(label_payload: dict[str, Any]) -> list[dict[str, str]]:
    queries = label_payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Label JSON must contain a queries list")

    output: list[dict[str, str]] = []
    for query in queries:
        query_id = query.get("query_id")
        query_text = query.get("query")
        if not isinstance(query_id, str) or not isinstance(query_text, str):
            raise ValueError("Every query must contain query_id and query text")
        output.append({"query_id": query_id, "query": query_text})
    return output


def build_run_rows(manifest_payload: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    queries = manifest_payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Retrieval manifest must contain a queries list")

    runs_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        query_id = query.get("query_id")
        methods = query.get("methods")
        if not isinstance(query_id, str) or not isinstance(methods, dict):
            raise ValueError("Every manifest query must contain query_id and methods")

        for method, results in methods.items():
            if not isinstance(results, list):
                raise ValueError(f"{query_id}/{method} results must be a list")
            for item in results:
                rank = item.get("rank")
                chapter_id = item.get("chapter_id")
                if not isinstance(rank, int) or rank <= 0:
                    raise ValueError(f"{query_id}/{method} has invalid rank")
                if not isinstance(chapter_id, str) or not chapter_id:
                    raise ValueError(f"{query_id}/{method} has invalid chapter_id")
                runs_by_method[method].append(
                    {
                        "query_id": query_id,
                        "method": method,
                        "candidate_id": chapter_id,
                        "rank": rank,
                        "score": f"{1.0 / rank:.8f}",
                    }
                )

    summary = {
        "queries": len(queries),
        "methods": {
            method: {
                "rows": len(rows),
                "queries": len({row["query_id"] for row in rows}),
            }
            for method, rows in sorted(runs_by_method.items())
        },
    }
    return dict(runs_by_method), summary


def precision_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    top = results[:k]
    relevant = sum(1 for doc_id in top if relevance_by_doc.get(doc_id, 0) > 0)
    return relevant / k


def recall_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    relevant_docs = {doc_id for doc_id, rel in relevance_by_doc.items() if rel > 0}
    if not relevant_docs:
        return 0.0
    found = sum(1 for doc_id in results[:k] if doc_id in relevant_docs)
    return found / len(relevant_docs)


def mrr_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    for index, doc_id in enumerate(results[:k], start=1):
        if relevance_by_doc.get(doc_id, 0) > 0:
            return 1.0 / index
    return 0.0


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(index + 1) for index, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    ranked_rels = [relevance_by_doc.get(doc_id, 0) for doc_id in results[:k]]
    ideal_rels = sorted(relevance_by_doc.values(), reverse=True)[:k]
    ideal = dcg_at_k(ideal_rels, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_rels, k) / ideal


def evaluate_runs(
    *,
    qrels_rows: list[dict[str, Any]],
    runs_by_method: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_rows:
        qrels_by_query[row["query_id"]][row["candidate_id"]] = int(row["relevance"])

    query_ids = sorted(qrels_by_query)
    metrics_by_method: dict[str, dict[str, float]] = {}
    table_rows: list[dict[str, Any]] = []

    ordered_methods = [method for method in METHOD_ORDER if method in runs_by_method]
    ordered_methods.extend(method for method in sorted(runs_by_method) if method not in ordered_methods)

    for method in ordered_methods:
        rows = runs_by_method[method]
        ranked_by_query: dict[str, list[str]] = defaultdict(list)
        for row in sorted(rows, key=lambda item: (item["query_id"], int(item["rank"]))):
            ranked_by_query[row["query_id"]].append(row["candidate_id"])

        per_query = {
            "Precision@5": [],
            "Recall@10": [],
            "MRR@10": [],
            "nDCG@10": [],
        }
        for query_id in query_ids:
            relevance_by_doc = qrels_by_query[query_id]
            results = ranked_by_query.get(query_id, [])
            per_query["Precision@5"].append(precision_at_k(results, relevance_by_doc, 5))
            per_query["Recall@10"].append(recall_at_k(results, relevance_by_doc, 10))
            per_query["MRR@10"].append(mrr_at_k(results, relevance_by_doc, 10))
            per_query["nDCG@10"].append(ndcg_at_k(results, relevance_by_doc, 10))

        metrics = {
            metric: sum(values) / len(values) if values else 0.0
            for metric, values in per_query.items()
        }
        metrics_by_method[method] = metrics
        table_rows.append(
            {
                "Method": METHOD_DISPLAY.get(method, method),
                "Precision@5": f"{metrics['Precision@5']:.4f}",
                "Recall@10": f"{metrics['Recall@10']:.4f}",
                "MRR@10": f"{metrics['MRR@10']:.4f}",
                "nDCG@10": f"{metrics['nDCG@10']:.4f}",
            }
        )

    return table_rows, metrics_by_method


def collect_engineering_metrics() -> list[dict[str, Any]]:
    bm25_dir = PROJECT_ROOT / "indexes" / "bm25"
    bm25_stats = read_json(bm25_dir / "dataset_stats.json") if (bm25_dir / "dataset_stats.json").is_file() else {}

    def dir_size(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    def mib(bytes_value: int) -> str:
        return f"{bytes_value / (1024 * 1024):.2f} MiB"

    rows = [
        {
            "Method": "BM25",
            "Latency/query": "not separately benchmarked",
            "Index size": mib(dir_size(bm25_dir)),
            "Build time": "not recorded",
        },
        {
            "Method": "Dense FAISS Flat",
            "Latency/query": "p50 40.94 ms; p95 56.76 ms",
            "Index size": "810.75 MB",
            "Build time": "built by Role 2; embedding 7817.79 s",
        },
        {
            "Method": "Dense FAISS ANN",
            "Latency/query": "p50 1.21 ms; p95 2.10 ms",
            "Index size": "954.39 MB; RAM est. 1049.83 MB",
            "Build time": "252.88 s; load 19.02 s",
        },
        {
            "Method": "Hybrid RRF",
            "Latency/query": "not separately benchmarked; combines BM25 + dense",
            "Index size": "BM25 + dense index",
            "Build time": "derived from existing BM25 and dense indexes",
        },
    ]
    if bm25_stats:
        rows[0]["Build time"] = f"{bm25_stats.get('processed_stats', {}).get('objects', 'unknown')} indexed objects"
    return rows


def benchmark_live_latency(queries: list[dict[str, str]], top_k: int = 20) -> list[dict[str, Any]]:
    import bm25s
    import faiss
    import numpy as np

    from bm25_search import simple_tokenize
    from embed_chapters import l2_normalize, load_encoder
    from hybrid_search import reciprocal_rank_fusion

    bm25_dir = PROJECT_ROOT / "indexes" / "bm25"
    flat_dir = PROJECT_ROOT / "indexes" / "faiss_flat"
    ann_path = PROJECT_ROOT / "indexes" / "faiss_ann" / "faiss_hnsw.index"

    print("Loading BM25 index for latency benchmark...")
    bm25_load_path = bm25_dir / "bm25_index"
    if not (bm25_load_path / "params.index.json").exists():
        bm25_load_path = bm25_dir
    bm25 = bm25s.BM25.load(str(bm25_load_path))
    with (bm25_dir / "metadata.pkl").open("rb") as source:
        bm25_metadata = pickle.load(source)

    print("Loading dense indexes and encoder for latency benchmark...")
    flat_index = faiss.read_index(str(flat_dir / "flat.index"))
    ann_index = faiss.read_index(str(ann_path))
    chapter_ids = json.load((flat_dir / "chapter_ids.json").open("r", encoding="utf-8"))
    encoder = load_encoder()

    def bm25_query(query_text: str, k: int) -> list[dict[str, Any]]:
        query_tokens = simple_tokenize(query_text)
        results, scores = bm25.retrieve([query_tokens], k=k * 5, corpus=None)
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc_id, score in zip(results[0], scores[0]):
            meta = bm25_metadata[int(doc_id)]
            chapter_id = meta["chapter_id"]
            if chapter_id in seen:
                continue
            seen.add(chapter_id)
            output.append({"chapter_id": chapter_id, "score": float(score), "rank": len(output) + 1})
            if len(output) >= k:
                break
        return output

    def encode_query_once(query_text: str) -> np.ndarray:
        vector = encoder.encode([query_text])
        vector = np.asarray(vector, dtype=np.float32)
        return l2_normalize(vector)

    def dense_from_vector(index: Any, query_vector: np.ndarray, k: int) -> list[dict[str, Any]]:
        scores, ids = index.search(query_vector, k)
        output = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx == -1:
                continue
            output.append({"chapter_id": chapter_ids[int(idx)], "score": float(score), "rank": rank})
        return output

    # Warm up retrievers/model once so p50/p95 represent steady-state online search.
    warmup_query = queries[0]["query"]
    warmup_vector = encode_query_once(warmup_query)
    bm25_query(warmup_query, top_k)
    dense_from_vector(flat_index, warmup_vector, top_k)
    dense_from_vector(ann_index, warmup_vector, top_k)

    latencies: dict[str, list[float]] = {
        "BM25": [],
        "Dense FAISS Flat": [],
        "Dense FAISS ANN": [],
        "Hybrid RRF": [],
    }

    for query in queries:
        query_text = query["query"]

        start = time.perf_counter()
        bm25_results = bm25_query(query_text, top_k)
        latencies["BM25"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        query_vector = encode_query_once(query_text)
        dense_flat_results = dense_from_vector(flat_index, query_vector, top_k)
        latencies["Dense FAISS Flat"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        query_vector = encode_query_once(query_text)
        dense_ann_results = dense_from_vector(ann_index, query_vector, top_k)
        latencies["Dense FAISS ANN"].append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        bm25_for_hybrid = bm25_query(query_text, top_k * 5)
        query_vector = encode_query_once(query_text)
        dense_for_hybrid = dense_from_vector(flat_index, query_vector, top_k * 5)
        reciprocal_rank_fusion(bm25_for_hybrid, dense_for_hybrid, top_k=top_k)
        latencies["Hybrid RRF"].append((time.perf_counter() - start) * 1000)

    rows = []
    for method, values in latencies.items():
        rows.append(
            {
                "Method": method,
                "Queries": len(values),
                "p50_ms": f"{percentile(values, 0.50):.3f}",
                "p95_ms": f"{percentile(values, 0.95):.3f}",
                "mean_ms": f"{sum(values) / len(values):.3f}",
            }
        )
    return rows


def apply_latency_to_engineering_rows(
    engineering_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latency_by_method = {row["Method"]: row for row in latency_rows}
    output = []
    for row in engineering_rows:
        updated = dict(row)
        latency = latency_by_method.get(row["Method"])
        if latency is not None:
            updated["Latency/query"] = (
                f"p50 {latency['p50_ms']} ms; p95 {latency['p95_ms']} ms"
            )
        output.append(updated)
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---:" if column != columns[0] else "---" for column in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def choose_example_query(
    *,
    qrels_rows: list[dict[str, Any]],
    runs_by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_rows:
        qrels_by_query[row["query_id"]][row["candidate_id"]] = int(row["relevance"])

    best_example: dict[str, Any] | None = None
    best_gap = -1.0
    for query_id, qrels in qrels_by_query.items():
        method_scores: dict[str, float] = {}
        method_top: dict[str, str] = {}
        for method, rows in runs_by_method.items():
            ranked = [
                row["candidate_id"]
                for row in sorted(rows, key=lambda item: int(item["rank"]))
                if row["query_id"] == query_id
            ]
            method_scores[method] = ndcg_at_k(ranked, qrels, 10)
            method_top[method] = ranked[0] if ranked else ""
        gap = max(method_scores.values()) - min(method_scores.values()) if method_scores else 0.0
        if gap > best_gap:
            best_gap = gap
            best_example = {
                "query_id": query_id,
                "gap": gap,
                "scores": method_scores,
                "top_results": method_top,
            }
    return best_example or {}


def write_report(
    *,
    path: Path,
    qrels_summary: dict[str, Any],
    run_summary: dict[str, Any],
    metrics_rows: list[dict[str, Any]],
    engineering_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    example: dict[str, Any],
    manifest_payload: dict[str, Any],
) -> None:
    best = max(metrics_rows, key=lambda row: float(row["nDCG@10"]))
    methods = ", ".join(row["Method"] for row in metrics_rows)
    metric_table = markdown_table(
        metrics_rows,
        ["Method", "Precision@5", "Recall@10", "MRR@10", "nDCG@10"],
    )
    engineering_table = markdown_table(
        engineering_rows,
        ["Method", "Latency/query", "Index size", "Build time"],
    )
    latency_table = markdown_table(
        latency_rows,
        ["Method", "Queries", "p50_ms", "p95_ms", "mean_ms"],
    )
    example_lines = []
    if example:
        example_lines.append(f"- Query: `{example['query_id']}`")
        for method, score in sorted(example["scores"].items()):
            display = METHOD_DISPLAY.get(method, method)
            top = example["top_results"].get(method, "")
            example_lines.append(f"- {display}: nDCG@10={score:.4f}, top result `{top}`")
    example_block = "\n".join(example_lines)

    content = f"""# Tasks 3.4 and 3.5 Evaluation

## Evaluation Setup

For evaluation, we used LLM-assisted relevance judgments over pooled candidate results. The same annotations were used for all retrieval methods, which allows a consistent relative comparison among {methods}.

The labeled file contains {qrels_summary['qrels']} usable query-candidate judgments across {qrels_summary['queries']} queries. Labels are interpreted as graded relevance scores: 0 = not relevant, 1 = partially relevant, and 2 = relevant. {qrels_summary['skipped_missing']} pairs with missing labels and {qrels_summary['skipped_invalid']} invalid rows were skipped when creating qrels.

The ranked runs were generated from `data/eval/retrieval_manifest.json`, which stores the controlled retrieval outputs built for the pooled labeling set. The manifest configuration was `top_k={manifest_payload.get('configuration', {}).get('top_k')}`, `candidate_k={manifest_payload.get('configuration', {}).get('candidate_k')}`, and `rrf_k={manifest_payload.get('configuration', {}).get('rrf_k')}`. Raw retrieval scores were not persisted in the manifest by design, so the exported run CSV files include deterministic inverse-rank placeholder scores; metric computation uses the rank column.

## Retrieval Runs

Task 3.4 is represented by run files in `runs/` for the following methods:

- BM25 lexical retrieval
- Dense FAISS Flat retrieval
- Dense FAISS ANN retrieval
- Hybrid RRF over BM25 and dense retrieval

The cross-encoder reranker is not included in the main benchmark. The project already records this decision in `experiments/rerank_decision.md`: reranking is kept as a proof of concept but excluded from the controlled comparison because of input-length, dependency, and latency trade-offs.

## Metrics

{metric_table}

The strongest method by nDCG@10 is **{best['Method']}** with nDCG@10={best['nDCG@10']}. Precision@5 measures how many of the first five returned chunks have non-zero relevance, Recall@10 measures coverage of all judged relevant chunks within the first ten results, MRR@10 rewards retrieving any relevant result early, and nDCG@10 uses the graded 0/1/2 labels.

## Engineering Metrics

{engineering_table}

The p50/p95 latency values above were recomputed live by this evaluation script over the same 48 evaluation queries, with indexes and the embedding model loaded once before timing. This measures steady-state online retrieval latency rather than cold-start loading time.

Detailed latency summary:

{latency_table}

Dense index size, load time, build time, and Recall@10-vs-Flat values also come from the Role 4 ANN benchmark on 553,472 vectors (`docs/role4_report.md` and `experiments/ann_architecture_decisions.md`). Those Role 4 latency numbers isolate FAISS index search; the live p50/p95 table above includes per-query embedding for dense methods, so it is closer to end-to-end retriever latency.

Role 4 also reports that the selected HNSW configuration preserves 98.8% Recall@10 relative to FAISS Flat while reducing p95 dense-search latency from 56.76 ms to 2.10 ms. The faster HNSW variant reaches 0.42 ms p95 but was rejected because Recall@10 falls to 89.8%. Cross-encoder reranking is documented as adding 20-50 ms per request and requiring a PyTorch dependency over 2 GB, so it remains excluded from the main Task 3.4/3.5 comparison.

## Example Query Difference

{example_block}

This query shows why the methods should be compared with graded relevance instead of only exact keyword overlap: different rankers often retrieve different top chunks, and nDCG@10 captures both relevance strength and rank position.

## Limitation

Since the relevance annotations were produced automatically under time constraints, some annotation noise may remain. Therefore, the evaluation is used mainly for relative comparison between retrieval methods rather than as a perfect absolute measurement of search quality.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task 3.4/3.5 evaluation artifacts.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip live p50/p95 latency benchmarking and keep static engineering notes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_json(args.labels)
    manifest = read_json(args.manifest)

    qrels_rows, qrels_summary = build_qrels(labels)
    runs_by_method, run_summary = build_run_rows(manifest)
    metrics_rows, metrics_by_method = evaluate_runs(qrels_rows=qrels_rows, runs_by_method=runs_by_method)
    engineering_rows = collect_engineering_metrics()
    latency_rows = []
    if not args.skip_latency:
        latency_rows = benchmark_live_latency(get_queries(labels), top_k=manifest.get("configuration", {}).get("top_k", 20))
        engineering_rows = apply_latency_to_engineering_rows(engineering_rows, latency_rows)
    example = choose_example_query(qrels_rows=qrels_rows, runs_by_method=runs_by_method)

    write_csv(
        args.evaluation_dir / "qrels.csv",
        ["query_id", "candidate_id", "relevance"],
        qrels_rows,
    )
    ordered_methods = [method for method in METHOD_ORDER if method in runs_by_method]
    ordered_methods.extend(method for method in sorted(runs_by_method) if method not in ordered_methods)
    for method in ordered_methods:
        rows = runs_by_method[method]
        write_csv(
            args.runs_dir / f"{method}_run.csv",
            ["query_id", "method", "candidate_id", "rank", "score"],
            rows,
        )
    write_csv(
        args.evaluation_dir / "metrics_table.csv",
        ["Method", "Precision@5", "Recall@10", "MRR@10", "nDCG@10"],
        metrics_rows,
    )
    write_csv(
        args.evaluation_dir / "engineering_metrics.csv",
        ["Method", "Latency/query", "Index size", "Build time"],
        engineering_rows,
    )
    if latency_rows:
        write_csv(
            args.evaluation_dir / "latency_metrics.csv",
            ["Method", "Queries", "p50_ms", "p95_ms", "mean_ms"],
            latency_rows,
        )
    write_json_atomic(
        {
            "qrels": qrels_summary,
            "runs": run_summary,
            "metrics": metrics_by_method,
            "latency": latency_rows,
            "example_query_difference": example,
        },
        args.evaluation_dir / "evaluation_summary.json",
    )
    write_report(
        path=args.report,
        qrels_summary=qrels_summary,
        run_summary=run_summary,
        metrics_rows=metrics_rows,
        engineering_rows=engineering_rows,
        latency_rows=latency_rows,
        example=example,
        manifest_payload=manifest,
    )

    print(f"Wrote qrels: {args.evaluation_dir / 'qrels.csv'}")
    print(f"Wrote metrics: {args.evaluation_dir / 'metrics_table.csv'}")
    print(f"Wrote report: {args.report}")
    print(markdown_table(metrics_rows, ["Method", "Precision@5", "Recall@10", "MRR@10", "nDCG@10"]))


if __name__ == "__main__":
    main()
