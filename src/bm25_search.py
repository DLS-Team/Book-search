from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bm25s
from tqdm import tqdm


INDEX_VERSION = "role1-bm25-v1"


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def load_processed_jsonl(path: Path, max_docs: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if max_docs is not None and index >= max_docs:
                break

            row = json.loads(line)

            yield {
                "internal_id": index,
                "book_id": row["book_id"],
                "title": row["title"],
                "author": row["author"],
                "chapter_id": row["chapter_id"],
                "chapter_title": row["chapter_title"],
                "text": row["text"],
                "paragraphs": row["paragraphs"],
            }


def build_index(input_jsonl: Path, index_dir: Path, max_docs: int | None) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    docs: list[str] = []
    metadata: list[dict[str, Any]] = []

    print(f"Loading processed corpus from {input_jsonl}...")

    for row in tqdm(load_processed_jsonl(input_jsonl, max_docs=max_docs)):
        docs.append(row["text"])
        metadata.append(
            {
                "internal_id": row["internal_id"],
                "book_id": row["book_id"],
                "title": row["title"],
                "author": row["author"],
                "chapter_id": row["chapter_id"],
                "chapter_title": row["chapter_title"],
                "paragraphs": row["paragraphs"],
                "preview": row["text"][:500],
            }
        )

    print(f"Loaded documents: {len(docs)}")
    print("Tokenizing...")
    tokenized_docs = [simple_tokenize(doc) for doc in tqdm(docs)]

    print("Building BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(tokenized_docs)

    print("Saving index and metadata...")
    retriever.save(str(index_dir / "bm25_index"), corpus=None)

    with (index_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    registry = {
        "version": INDEX_VERSION,
        "build_date_utc": datetime.now(timezone.utc).isoformat(),
        "index_name": "bm25_sparse_baseline",
        "index_type": "BM25",
        "library": "bm25s",
        "documents_indexed": len(metadata),
        "source_file": str(input_jsonl),
        "index_directory": str(index_dir),
        "tokenization": {
            "lowercase": True,
            "pattern": r"[a-z0-9]+(?:'[a-z]+)?",
            "punctuation": "removed except apostrophes inside words",
            "stemming": False,
        },
        "bm25_parameters": {
            "implementation_defaults": True,
            "note": "bm25s.BM25() default parameters are used.",
        },
        "note": (
            "BM25 lexical baseline over stable pseudo-chapter / scene chunks. "
            "Used as exact-keyword baseline and future hybrid retrieval input."
        ),
    }

    (index_dir / "bm25_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(registry, ensure_ascii=False, indent=2))


def search(index_dir: Path, query: str, top_k: int) -> list[dict[str, Any]]:
    retriever = bm25s.BM25.load(str(index_dir / "bm25_index"))

    with (index_dir / "metadata.pkl").open("rb") as f:
        metadata = pickle.load(f)

    query_tokens = simple_tokenize(query)

    results, scores = retriever.retrieve([query_tokens], k=top_k * 5, corpus=None)

    output = []
    seen_chapter_ids: set[str] = set()

    for doc_id, score in zip(results[0], scores[0]):
        meta = metadata[int(doc_id)]
        chapter_id = meta["chapter_id"]

        if chapter_id in seen_chapter_ids:
            continue

        seen_chapter_ids.add(chapter_id)
        output.append(
            {
                "score": float(score),
                "book_id": meta["book_id"],
                "title": meta["title"],
                "author": meta["author"],
                "chapter_id": meta["chapter_id"],
                "chapter_title": meta["chapter_title"],
                "preview": meta["preview"],
                "paragraphs": meta["paragraphs"],
            }
        )

        if len(output) >= top_k:
            break

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query BM25 index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/processed_chapters.jsonl"))
    build_parser.add_argument("--index-dir", type=Path, default=Path("outputs/bm25_index"))
    build_parser.add_argument(
        "--max-docs",
        type=int,
        default=600000,
        help="Use 600k by default to satisfy scale requirement without overloading laptop RAM.",
    )

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--index-dir", type=Path, default=Path("outputs/bm25_index"))
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "build":
        build_index(input_jsonl=args.input_jsonl, index_dir=args.index_dir, max_docs=args.max_docs)
    elif args.command == "search":
        results = search(index_dir=args.index_dir, query=args.query, top_k=args.top_k)
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
