"""Task 3.6 error analysis for Role 3 retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = PROJECT_ROOT / "data" / "eval" / "candidate_pool_labelled.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "eval" / "retrieval_manifest.json"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "task_3_6_error_analysis.md"

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chapter_id_from_candidate_id(candidate_id: str) -> str:
    return candidate_id.split("::", 1)[1] if "::" in candidate_id else candidate_id


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum((2**rel - 1) / math.log2(index + 1) for index, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    ranked_rels = [relevance_by_doc.get(doc_id, 0) for doc_id in results[:k]]
    ideal_rels = sorted(relevance_by_doc.values(), reverse=True)[:k]
    ideal = dcg_at_k(ideal_rels, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_rels, k) / ideal


def precision_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    return sum(1 for doc_id in results[:k] if relevance_by_doc.get(doc_id, 0) > 0) / k


def recall_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    relevant_docs = {doc_id for doc_id, rel in relevance_by_doc.items() if rel > 0}
    if not relevant_docs:
        return 0.0
    found = sum(1 for doc_id in results[:k] if doc_id in relevant_docs)
    return found / len(relevant_docs)


def mrr_at_k(results: list[str], relevance_by_doc: dict[str, int], k: int) -> float:
    for rank, doc_id in enumerate(results[:k], start=1):
        if relevance_by_doc.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---:" if column != columns[0] else "---" for column in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def load_label_context(label_payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    query_info: dict[str, dict[str, Any]] = {}
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    candidates: dict[str, dict[str, Any]] = {}

    for query in label_payload["queries"]:
        query_id = query["query_id"]
        query_info[query_id] = {
            "query": query["query"],
        }
        for candidate in query["candidates"]:
            chapter_id = chapter_id_from_candidate_id(candidate["candidate_id"])
            relevance = candidate.get("relevance")
            if not isinstance(relevance, dict):
                continue
            label = relevance.get("label")
            if isinstance(label, bool) or not isinstance(label, int) or label not in {0, 1, 2}:
                continue
            qrels_by_query[query_id][chapter_id] = label
            candidates[f"{query_id}::{chapter_id}"] = {
                "text": candidate.get("text", ""),
                "evidence": relevance.get("evidence", ""),
                "label": label,
                "confidence": relevance.get("confidence", ""),
            }
    return query_info, qrels_by_query, candidates


def load_manifest_context(manifest_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for query in manifest_payload["queries"]:
        output[query["query_id"]] = {
            "query": query["query"],
            "category": query.get("category", "unknown"),
        }
    return output


def load_runs(runs_dir: Path) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[str]]] = {}
    for method in METHOD_ORDER:
        path = runs_dir / f"{method}_run.csv"
        rows = read_csv(path)
        by_query: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for row in rows:
            by_query[row["query_id"]].append((int(row["rank"]), row["candidate_id"]))
        runs[method] = {
            query_id: [candidate_id for _, candidate_id in sorted(items)]
            for query_id, items in by_query.items()
        }
    return runs


def compute_per_query_metrics(
    *,
    qrels_by_query: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[str]]],
    manifest_info: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id, relevance_by_doc in sorted(qrels_by_query.items()):
        relevant_count = sum(1 for value in relevance_by_doc.values() if value > 0)
        strong_count = sum(1 for value in relevance_by_doc.values() if value == 2)
        for method in METHOD_ORDER:
            results = runs[method].get(query_id, [])
            top_result = results[0] if results else ""
            row = {
                "query_id": query_id,
                "category": manifest_info.get(query_id, {}).get("category", "unknown"),
                "query": manifest_info.get(query_id, {}).get("query", ""),
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "Precision@5": precision_at_k(results, relevance_by_doc, 5),
                "Recall@10": recall_at_k(results, relevance_by_doc, 10),
                "MRR@10": mrr_at_k(results, relevance_by_doc, 10),
                "nDCG@10": ndcg_at_k(results, relevance_by_doc, 10),
                "relevant_judgments": relevant_count,
                "strong_judgments": strong_count,
                "top_result_id": top_result,
                "top_result_label": relevance_by_doc.get(top_result, 0),
                "top10_relevant_hits": sum(1 for doc_id in results[:10] if relevance_by_doc.get(doc_id, 0) > 0),
                "top10_strong_hits": sum(1 for doc_id in results[:10] if relevance_by_doc.get(doc_id, 0) == 2),
            }
            rows.append(row)
    return rows


def category_summary(per_query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_query_rows:
        grouped[(row["category"], row["method"])].append(row)

    output = []
    for (category, method), rows in sorted(grouped.items()):
        output.append(
            {
                "category": category,
                "method": method,
                "method_display": METHOD_DISPLAY[method],
                "queries": len(rows),
                "Precision@5": sum(row["Precision@5"] for row in rows) / len(rows),
                "Recall@10": sum(row["Recall@10"] for row in rows) / len(rows),
                "MRR@10": sum(row["MRR@10"] for row in rows) / len(rows),
                "nDCG@10": sum(row["nDCG@10"] for row in rows) / len(rows),
            }
        )
    return output


def identify_cases(
    *,
    per_query_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[str]]],
    candidate_context: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_query_rows:
        by_query[row["query_id"]][row["method"]] = row

    cases: list[dict[str, Any]] = []
    for query_id, method_rows in sorted(by_query.items()):
        hybrid = method_rows["hybrid_rrf"]
        best_method = max(METHOD_ORDER, key=lambda method: method_rows[method]["nDCG@10"])
        worst_method = min(METHOD_ORDER, key=lambda method: method_rows[method]["nDCG@10"])
        best_score = method_rows[best_method]["nDCG@10"]
        hybrid_score = hybrid["nDCG@10"]

        failure_type = ""
        focus_method = "hybrid_rrf"
        if hybrid_score < 0.55:
            failure_type = "hybrid_low_absolute_quality"
        elif best_score - hybrid_score >= 0.15:
            failure_type = f"hybrid_lags_{best_method}"
            focus_method = best_method
        elif method_rows["bm25"]["nDCG@10"] - method_rows["dense_flat"]["nDCG@10"] >= 0.15:
            failure_type = "dense_lags_lexical_baseline"
            focus_method = "dense_flat"
        elif method_rows["dense_flat"]["nDCG@10"] - method_rows["bm25"]["nDCG@10"] >= 0.15:
            failure_type = "bm25_lags_semantic_baseline"
            focus_method = "bm25"
        else:
            continue

        top_result = method_rows[focus_method]["top_result_id"]
        context = candidate_context.get(f"{query_id}::{top_result}", {})
        cases.append(
            {
                "query_id": query_id,
                "category": method_rows[focus_method]["category"],
                "query": method_rows[focus_method]["query"],
                "failure_type": failure_type,
                "focus_method": focus_method,
                "focus_method_display": METHOD_DISPLAY[focus_method],
                "best_method": best_method,
                "best_method_display": METHOD_DISPLAY[best_method],
                "worst_method": worst_method,
                "worst_method_display": METHOD_DISPLAY[worst_method],
                "bm25_nDCG@10": method_rows["bm25"]["nDCG@10"],
                "dense_flat_nDCG@10": method_rows["dense_flat"]["nDCG@10"],
                "dense_ann_nDCG@10": method_rows["dense_ann"]["nDCG@10"],
                "hybrid_rrf_nDCG@10": method_rows["hybrid_rrf"]["nDCG@10"],
                "top_result_id": top_result,
                "top_result_label": qrels_by_query[query_id].get(top_result, 0),
                "top_result_evidence": context.get("evidence", ""),
            }
        )
    return cases


def rounded_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, float):
                cleaned[key] = f"{value:.4f}"
            else:
                cleaned[key] = value
        output.append(cleaned)
    return output


def write_report(
    *,
    path: Path,
    per_query_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> None:
    hybrid_rows = [row for row in per_query_rows if row["method"] == "hybrid_rrf"]
    weakest_hybrid = sorted(hybrid_rows, key=lambda row: row["nDCG@10"])[:5]
    strongest_hybrid = sorted(hybrid_rows, key=lambda row: row["nDCG@10"], reverse=True)[:5]

    category_hybrid = [row for row in category_rows if row["method"] == "hybrid_rrf"]
    category_table = markdown_table(
        rounded_rows(category_hybrid),
        ["category", "queries", "Precision@5", "Recall@10", "MRR@10", "nDCG@10"],
    )
    weak_table = markdown_table(
        rounded_rows(
            [
                {
                    "query_id": row["query_id"],
                    "category": row["category"],
                    "nDCG@10": row["nDCG@10"],
                    "top_result_label": row["top_result_label"],
                    "query": row["query"],
                }
                for row in weakest_hybrid
            ]
        ),
        ["query_id", "category", "nDCG@10", "top_result_label", "query"],
    )
    strong_table = markdown_table(
        rounded_rows(
            [
                {
                    "query_id": row["query_id"],
                    "category": row["category"],
                    "nDCG@10": row["nDCG@10"],
                    "query": row["query"],
                }
                for row in strongest_hybrid
            ]
        ),
        ["query_id", "category", "nDCG@10", "query"],
    )
    case_table = markdown_table(
        rounded_rows(cases[:8]),
        [
            "query_id",
            "failure_type",
            "best_method_display",
            "focus_method_display",
            "bm25_nDCG@10",
            "dense_flat_nDCG@10",
            "hybrid_rrf_nDCG@10",
        ],
    )

    content = f"""# Task 3.6 Error Analysis

## Setup

This analysis uses the same LLM-assisted relevance judgments, qrels, and run files produced for Tasks 3.4 and 3.5. Relevance is graded on the 0/1/2 scale, and nDCG@10 is the primary diagnostic metric because it accounts for both rank position and label strength.

## Hybrid RRF By Query Category

{category_table}

Hybrid RRF is strongest when lexical and dense evidence complement each other, especially when one method retrieves a relevant candidate early and the other method adds semantically related alternatives. Recall@10 remains numerically low because the pooled qrels often contain many candidates with non-zero labels for a query, while each run contributes only ten results to Recall@10.

## Strongest Hybrid Queries

{strong_table}

These queries usually have a clear signal that at least one source ranks early, and RRF keeps that signal near the top.

## Weakest Hybrid Queries

{weak_table}

The weakest cases tend to be broad, ambiguous, or scene-like queries where many judged candidates are only partially relevant. In those cases, RRF can still retrieve non-zero relevance, but it may miss the few label-2 passages or rank them below several partial matches.

## Representative Failure Cases

{case_table}

The failure cases show three main patterns:

1. **Lexical-only weakness:** BM25 can over-prioritize shared words and miss paraphrased or atmospheric relevance.
2. **Dense-only weakness:** dense retrieval can retrieve semantically adjacent passages that match tone or topic but not the concrete requested situation.
3. **Fusion dilution:** Hybrid RRF is robust overall, but if one source strongly ranks weak partial matches, fusion may not fully recover the best graded candidate.

## Implications For The Next Iteration

- Keep Hybrid RRF as the default Role 3 method because it has the best aggregate nDCG@10 and MRR@10 in Task 3.5.
- Add a small calibration layer for query categories: exact-keyword queries can lean more on BM25, while ambiguous/atmospheric queries may benefit from stronger dense weighting or a second-stage rerank.
- Consider extending the dense representation beyond `title_first_middle_last` if failures show relevant scenes buried away from the sampled chapter positions.
- Keep cross-encoder reranking outside the main benchmark for now, but use it as an offline diagnostic for the worst Hybrid RRF cases if time allows.

## Limitation

The labels are automated relevance annotations, so individual examples may contain annotation noise. The analysis is therefore most useful for relative method comparison and failure-mode discovery, not as a perfect judgment of any single passage.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task 3.6 error analysis artifacts.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_json(args.labels)
    manifest = read_json(args.manifest)

    _, qrels_by_query, candidate_context = load_label_context(labels)
    manifest_info = load_manifest_context(manifest)
    runs = load_runs(args.runs_dir)

    per_query = compute_per_query_metrics(
        qrels_by_query=qrels_by_query,
        runs=runs,
        manifest_info=manifest_info,
    )
    categories = category_summary(per_query)
    cases = identify_cases(
        per_query_rows=per_query,
        qrels_by_query=qrels_by_query,
        runs=runs,
        candidate_context=candidate_context,
    )

    write_csv(
        args.evaluation_dir / "per_query_metrics.csv",
        [
            "query_id",
            "category",
            "query",
            "method",
            "method_display",
            "Precision@5",
            "Recall@10",
            "MRR@10",
            "nDCG@10",
            "relevant_judgments",
            "strong_judgments",
            "top_result_id",
            "top_result_label",
            "top10_relevant_hits",
            "top10_strong_hits",
        ],
        rounded_rows(per_query),
    )
    write_csv(
        args.evaluation_dir / "category_metrics.csv",
        [
            "category",
            "method",
            "method_display",
            "queries",
            "Precision@5",
            "Recall@10",
            "MRR@10",
            "nDCG@10",
        ],
        rounded_rows(categories),
    )
    write_csv(
        args.evaluation_dir / "failure_cases.csv",
        [
            "query_id",
            "category",
            "query",
            "failure_type",
            "focus_method",
            "focus_method_display",
            "best_method",
            "best_method_display",
            "worst_method",
            "worst_method_display",
            "bm25_nDCG@10",
            "dense_flat_nDCG@10",
            "dense_ann_nDCG@10",
            "hybrid_rrf_nDCG@10",
            "top_result_id",
            "top_result_label",
            "top_result_evidence",
        ],
        rounded_rows(cases),
    )
    write_report(
        path=args.report,
        per_query_rows=per_query,
        category_rows=categories,
        cases=cases,
    )

    print(f"Wrote per-query metrics: {args.evaluation_dir / 'per_query_metrics.csv'}")
    print(f"Wrote category metrics: {args.evaluation_dir / 'category_metrics.csv'}")
    print(f"Wrote failure cases: {args.evaluation_dir / 'failure_cases.csv'}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
