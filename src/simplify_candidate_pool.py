"""Create a compact candidate-pool JSON file for manual labeling.

The compact file keeps only:
    - query_id
    - query
    - candidate_id
    - text
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval" / "candidate_pool_for_labeling.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "eval" / "candidate_pool_for_labeling_short.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


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


def simplify_candidate_pool(payload: dict[str, Any]) -> dict[str, Any]:
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise ValueError("Input JSON must contain a 'queries' list")

    compact_queries: list[dict[str, Any]] = []
    for query_index, query_row in enumerate(queries, start=1):
        if not isinstance(query_row, dict):
            raise ValueError(f"Query #{query_index} must be an object")

        query_id = query_row.get("query_id")
        query_text = query_row.get("query")
        candidates = query_row.get("candidates")
        if not query_id:
            raise ValueError(f"Query #{query_index} is missing query_id")
        if not isinstance(query_text, str) or not query_text:
            raise ValueError(f"Query {query_id} is missing query text")
        if not isinstance(candidates, list):
            raise ValueError(f"Query {query_id} must contain a candidates list")

        compact_candidates: list[dict[str, str]] = []
        for candidate_index, candidate_row in enumerate(candidates, start=1):
            if not isinstance(candidate_row, dict):
                raise ValueError(
                    f"Candidate #{candidate_index} for query {query_id} must be an object"
                )

            candidate_id = candidate_row.get("candidate_id")
            text = candidate_row.get("text")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(
                    f"Candidate #{candidate_index} for query {query_id} is missing candidate_id"
                )
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"Candidate {candidate_id} for query {query_id} is missing text"
                )

            compact_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "text": text,
                }
            )

        compact_queries.append(
            {
                "query_id": str(query_id),
                "query": query_text,
                "candidates": compact_candidates,
            }
        )

    return {"queries": compact_queries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only query/candidate IDs, query text, and candidate text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {args.output}")

    payload = _read_json(args.input)
    compact_payload = simplify_candidate_pool(payload)
    _write_json_atomic(compact_payload, args.output)

    query_count = len(compact_payload["queries"])
    candidate_count = sum(len(query["candidates"]) for query in compact_payload["queries"])
    print(f"Wrote {query_count} queries and {candidate_count} candidates to {args.output}")


if __name__ == "__main__":
    main()
