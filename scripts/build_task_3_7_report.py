"""Build Task 3.7 final ranking recommendation artifacts.

The script reads the measured outputs from Tasks 3.4-3.6 and writes:
- docs/task_3_7_final_recommendation.md
- slides/role3_slide.md

It intentionally has no third-party dependencies so the report can be
regenerated after rerunning the benchmarks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METHOD_ORDER = ["BM25", "Dense FAISS Flat", "Dense FAISS ANN", "Hybrid RRF"]
PRIMARY_METRIC = "nDCG@10"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def format_metric(value: str | float) -> str:
    return f"{float(value):.4f}"


def method_sort_key(row: dict[str, str]) -> int:
    method = row.get("Method") or row.get("method_display") or ""
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_experiment_rows(
    metrics: list[dict[str, str]],
    latency: list[dict[str, str]],
    engineering: list[dict[str, str]],
) -> list[list[str]]:
    latency_by_method = {row["Method"]: row for row in latency}
    engineering_by_method = {row["Method"]: row for row in engineering}
    rows = []
    for row in sorted(metrics, key=method_sort_key):
        method = row["Method"]
        latency_row = latency_by_method.get(method, {})
        engineering_row = engineering_by_method.get(method, {})
        rows.append(
            [
                method,
                format_metric(row["Precision@5"]),
                format_metric(row["Recall@10"]),
                format_metric(row["MRR@10"]),
                format_metric(row["nDCG@10"]),
                latency_row.get("p50_ms", ""),
                latency_row.get("p95_ms", ""),
                engineering_row.get("Index size", ""),
                engineering_row.get("Build time", ""),
            ]
        )
    return rows


def hybrid_category_rows(category_metrics: list[dict[str, str]]) -> list[list[str]]:
    rows = [
        row
        for row in category_metrics
        if row.get("method_display") == "Hybrid RRF"
    ]
    rows.sort(key=lambda row: row["category"])
    return [
        [
            row["category"],
            row["queries"],
            format_metric(row["Precision@5"]),
            format_metric(row["Recall@10"]),
            format_metric(row["MRR@10"]),
            format_metric(row["nDCG@10"]),
        ]
        for row in rows
    ]


def representative_failure_rows(failures: list[dict[str, str]], limit: int = 6) -> list[list[str]]:
    selected = sorted(failures, key=lambda row: float(row["hybrid_rrf_nDCG@10"]))[:limit]
    return [
        [
            row["query_id"],
            row["category"],
            row["failure_type"],
            row["best_method_display"],
            row["hybrid_rrf_nDCG@10"],
            row["top_result_label"],
            row["query"].replace("|", "\\|"),
        ]
        for row in selected
    ]


def build_report(repo: Path) -> tuple[str, str]:
    metrics = read_csv(repo / "evaluation" / "metrics_table.csv")
    latency = read_csv(repo / "evaluation" / "latency_metrics.csv")
    engineering = read_csv(repo / "evaluation" / "engineering_metrics.csv")
    category_metrics = read_csv(repo / "evaluation" / "category_metrics.csv")
    failures = read_csv(repo / "evaluation" / "failure_cases.csv")
    summary = read_json(repo / "evaluation" / "evaluation_summary.json")

    metrics_by_method = {row["Method"]: row for row in metrics}
    latency_by_method = {row["Method"]: row for row in latency}

    best_quality = max(metrics, key=lambda row: as_float(row, PRIMARY_METRIC))
    fastest = min(latency, key=lambda row: as_float(row, "p50_ms"))
    best_quality_method = best_quality["Method"]
    fastest_method = fastest["Method"]

    hybrid = metrics_by_method["Hybrid RRF"]
    hybrid_latency = latency_by_method["Hybrid RRF"]
    ann = metrics_by_method["Dense FAISS ANN"]
    ann_latency = latency_by_method["Dense FAISS ANN"]
    bm25 = metrics_by_method["BM25"]
    bm25_latency = latency_by_method["BM25"]

    experiment_table = markdown_table(
        [
            "Method",
            "P@5",
            "R@10",
            "MRR@10",
            "nDCG@10",
            "p50 ms",
            "p95 ms",
            "Index size",
            "Build/load note",
        ],
        build_experiment_rows(metrics, latency, engineering),
    )

    category_table = markdown_table(
        ["Category", "Queries", "P@5", "R@10", "MRR@10", "nDCG@10"],
        hybrid_category_rows(category_metrics),
    )

    failure_table = markdown_table(
        [
            "Query",
            "Category",
            "Failure type",
            "Best method",
            "Hybrid nDCG@10",
            "Top label",
            "Query text",
        ],
        representative_failure_rows(failures),
    )

    qrels = summary["qrels"]
    runs = summary["runs"]
    label_counts = qrels["label_counts"]
    confidence_counts = qrels["confidence_counts"]

    report = f"""# Task 3.7 Final Experiment Recommendation

## Data Used

- Evaluation set: {qrels["queries"]} queries, {qrels["qrels"]} usable relevance labels, {qrels["skipped_missing"]} missing candidate ids skipped.
- Compared methods: {", ".join(METHOD_ORDER)}.
- Run depth: {runs["methods"]["hybrid_rrf"]["rows"] // runs["queries"]} candidates per method per query in the saved run files; metrics reported at P@5, R@10, MRR@10, and nDCG@10.
- Label distribution: 0={label_counts["0"]}, 1={label_counts["1"]}, 2={label_counts["2"]}; confidence distribution: low={confidence_counts["low"]}, medium={confidence_counts["medium"]}, high={confidence_counts["high"]}.

## Final Experiment Table

{experiment_table}

## Quality-Efficiency Recommendation

**Recommendation:** use Hybrid RRF as the default ranking method for the final demo and offline evaluation, and keep Dense FAISS ANN as the low-latency serving fallback.

Hybrid RRF has the best aggregate quality: nDCG@10={format_metric(hybrid["nDCG@10"])} and MRR@10={format_metric(hybrid["MRR@10"])}. It also has the best Recall@10 among the tested methods ({format_metric(hybrid["Recall@10"])}), while matching the best Precision@5 ({format_metric(hybrid["Precision@5"])}). The cost is latency: p50={hybrid_latency["p50_ms"]} ms and p95={hybrid_latency["p95_ms"]} ms because it runs both lexical and dense retrieval before fusion.

Dense FAISS ANN is the best speed-quality trade-off when the online latency budget is strict: p50={ann_latency["p50_ms"]} ms and p95={ann_latency["p95_ms"]} ms, with nDCG@10={format_metric(ann["nDCG@10"])}. It is much faster than Dense FAISS Flat and close to BM25 latency, but loses about {as_float(hybrid, "nDCG@10") - as_float(ann, "nDCG@10"):.4f} nDCG@10 versus Hybrid RRF.

BM25 remains a strong lexical baseline: p50={bm25_latency["p50_ms"]} ms, p95={bm25_latency["p95_ms"]} ms, and nDCG@10={format_metric(bm25["nDCG@10"])}. It is especially useful for exact terms and as a fallback, but it does not beat Hybrid RRF overall. Dense FAISS Flat should not be the serving default because it is slower than ANN with no measured quality advantage.

## Error-Analysis Summary

Hybrid RRF by query category:

{category_table}

Representative low-quality or diagnostic cases from Task 3.6:

{failure_table}

The main failure modes are:

- **Ambiguous or weak-evidence queries:** broad prompts such as "something feels wrong" often retrieve partially relevant passages but not a clearly defensible label-2 result.
- **Semantic scene drift:** dense retrieval can match tone or topic while missing the concrete event requested by the query.
- **Literal trap:** BM25 can over-rank exact word overlap, for example terms such as "chandelier" without the full scene.
- **Fusion dilution:** Hybrid RRF is robust overall, but if one source ranks weak partial matches very highly, fusion can still leave the best evidence below the top positions.

## Slide-Ready Result

**One result:** Hybrid RRF is the best measured ranking approach: nDCG@10={format_metric(hybrid["nDCG@10"])}, MRR@10={format_metric(hybrid["MRR@10"])}, p95={hybrid_latency["p95_ms"]} ms.

**One decision explanation:** Hybrid RRF wins because lexical retrieval catches exact named/keyword evidence while dense retrieval adds paraphrase and atmosphere matches; Reciprocal Rank Fusion keeps candidates that are supported by either signal near the top without training a new model.

**One limitation:** the best-quality method is also the slowest tested method, and weak ambiguous queries still need a quality gate or refinement before the result can be presented as confident evidence.

## Acceptance Check

- Final experiment table: included above with quality, latency, index size, and build/load notes for all four methods.
- Error-analysis summary: included from Task 3.6 category metrics and failure cases.
- Quality-efficiency recommendation: explicitly based on Task 3.5 metrics and Task 3.6 failures.
- Defense answer: Hybrid RRF works best overall; it fails on ambiguous/weak-evidence and scene-drift cases; its trade-off is higher latency for better ranking quality.
"""

    slide = f"""# Role 3 Slide: Ranking Experiments

## Final Result

Hybrid RRF is the best measured ranking method: nDCG@10={format_metric(hybrid["nDCG@10"])}, MRR@10={format_metric(hybrid["MRR@10"])}, P@5={format_metric(hybrid["Precision@5"])}, p95={hybrid_latency["p95_ms"]} ms.

| Method | nDCG@10 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| BM25 | {format_metric(bm25["nDCG@10"])} | {format_metric(bm25["MRR@10"])} | {bm25_latency["p50_ms"]} | {bm25_latency["p95_ms"]} |
| Dense ANN | {format_metric(ann["nDCG@10"])} | {format_metric(ann["MRR@10"])} | {ann_latency["p50_ms"]} | {ann_latency["p95_ms"]} |
| Hybrid RRF | {format_metric(hybrid["nDCG@10"])} | {format_metric(hybrid["MRR@10"])} | {hybrid_latency["p50_ms"]} | {hybrid_latency["p95_ms"]} |

## Decision

Use Hybrid RRF for the final demo and offline ranking story. It combines exact lexical matches with dense semantic matches and gives the best measured quality across the controlled query set.

Use Dense FAISS ANN when serving latency is the priority: it is close to BM25 speed and much faster than Hybrid RRF, with a moderate quality drop.

## Limitation

Hybrid RRF is slower because it runs both retrieval paths, and ambiguous weak-evidence queries still need quality gating/refinement before we should present a result as confident.
"""

    return report, slide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/task_3_7_final_recommendation.md"),
    )
    parser.add_argument("--slide", type=Path, default=Path("slides/role3_slide.md"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    report, slide = build_report(repo)

    report_path = repo / args.report
    slide_path = repo / args.slide
    report_path.parent.mkdir(parents=True, exist_ok=True)
    slide_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    slide_path.write_text(slide, encoding="utf-8")

    print(f"Wrote {report_path}")
    print(f"Wrote {slide_path}")


if __name__ == "__main__":
    main()
