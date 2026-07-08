from __future__ import annotations

import json
from pathlib import Path

from bm25_search import search


EXACT_KEYWORD_QUERIES = [
    "fireplace winter night",
    "murder knife blood",
    "ship storm sea",
    "lonely dark city",
    "letter from mother",
    "old castle ghost",
    "battle sword king",
]

BM25_FAILURE_CASES = [
    {
        "query": "a lonely person walking through a dark city",
        "why_bm25_may_fail": (
            "BM25 depends on exact words. A relevant passage may describe the same scene "
            "using words like street, lamps, fog, alone, or midnight without saying lonely or dark city."
        ),
    },
    {
        "query": "hero feels guilty after betrayal",
        "why_bm25_may_fail": (
            "The emotional meaning may be expressed indirectly. BM25 can miss passages where guilt "
            "and betrayal are implied through actions or dialogue rather than literal keywords."
        ),
    },
    {
        "query": "a child is afraid but trying to be brave",
        "why_bm25_may_fail": (
            "BM25 may retrieve passages containing child, afraid, and brave literally, but miss "
            "semantically relevant scenes where fear and courage are described with different wording."
        ),
    },
]


def main() -> None:
    output_path = Path("outputs/bm25_examples.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []

    for query in EXACT_KEYWORD_QUERIES:
        results = search(
            index_dir=Path("outputs/bm25_index"),
            query=query,
            top_k=5,
        )

        examples.append(
            {
                "query_type": "exact_keyword",
                "query": query,
                "top_results": results,
            }
        )

    report = {
        "bm25_role": (
            "BM25 is the lexical sparse baseline. It is useful for exact words, names, "
            "rare terms, and phrases, and later can be combined with dense retrieval."
        ),
        "exact_keyword_examples": examples,
        "failure_cases": BM25_FAILURE_CASES,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved BM25 examples to {output_path}")


if __name__ == "__main__":
    main()