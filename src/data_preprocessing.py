from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass
class ProcessedChunk:
    book_id: str
    title: str
    author: str
    chapter_id: str
    chapter_title: str
    text: str
    paragraphs: list[dict[str, Any]]
    char_length: int
    token_length: int


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_gutenberg_boilerplate(text: str) -> str:
    start_match = re.search(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if start_match:
        text = text[start_match.end():]

    end_match = re.search(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if end_match:
        text = text[:end_match.start()]

    return normalize_text(text)


def extract_metadata(text: str, book_id: str) -> tuple[str, str]:
    title_match = re.search(r"^Title:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    author_match = re.search(r"^Author:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)

    title = title_match.group(1).strip() if title_match else f"Book {book_id}"
    author = author_match.group(1).strip() if author_match else "Unknown author"

    return title, author


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if p]


def paragraph_to_words(paragraph: str) -> list[str]:
    return re.findall(r"\S+", paragraph)


def build_chunks_from_paragraphs(
    paragraphs: list[str],
    target_words: int,
    max_words: int,
) -> list[tuple[str, int, int]]:
    """
    Build stable scene-like chunks from paragraphs.

    Returns:
        (chunk_text, start_paragraph_position, end_paragraph_position)
    """
    chunks: list[tuple[str, int, int]] = []

    current_parts: list[str] = []
    current_words = 0
    start_pos: int | None = None
    end_pos: int | None = None

    for paragraph_position, paragraph in enumerate(paragraphs):
        words = paragraph_to_words(paragraph)
        word_count = len(words)

        if word_count == 0:
            continue

        if start_pos is None:
            start_pos = paragraph_position

        current_parts.append(paragraph)
        current_words += word_count
        end_pos = paragraph_position

        if current_words >= target_words or current_words >= max_words:
            chunk_text = "\n\n".join(current_parts).strip()
            if chunk_text and start_pos is not None and end_pos is not None:
                chunks.append((chunk_text, start_pos, end_pos))

            current_parts = []
            current_words = 0
            start_pos = None
            end_pos = None

    if current_parts and start_pos is not None and end_pos is not None:
        chunk_text = "\n\n".join(current_parts).strip()
        chunks.append((chunk_text, start_pos, end_pos))

    return chunks


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, {"_invalid_json": line}


def make_processed_chunk(
    *,
    book_id: str,
    title: str,
    author: str,
    chunk_index: int,
    chunk_text: str,
    start_paragraph: int,
    end_paragraph: int,
) -> ProcessedChunk:
    tokens = simple_tokenize(chunk_text)

    return ProcessedChunk(
        book_id=book_id,
        title=title,
        author=author,
        chapter_id=f"{book_id}_chunk_{chunk_index:06d}",
        chapter_title=f"Chunk {chunk_index}",
        text=chunk_text,
        paragraphs=[
            {
                "paragraph_position": pos,
                "source": "book_paragraph",
            }
            for pos in range(start_paragraph, end_paragraph + 1)
        ],
        char_length=len(chunk_text),
        token_length=len(tokens),
    )


def is_valid_chunk(
    chunk: ProcessedChunk,
    min_chars: int,
    min_tokens: int,
) -> tuple[bool, str]:
    if chunk.char_length < min_chars:
        return False, "too_few_characters"
    if chunk.token_length < min_tokens:
        return False, "too_few_tokens"
    return True, "accepted"


def build_stats(records: list[ProcessedChunk]) -> dict[str, Any]:
    if not records:
        return {
            "objects": 0,
            "books": 0,
            "authors": 0,
            "avg_chars": 0,
            "median_chars": 0,
            "avg_tokens": 0,
            "median_tokens": 0,
            "avg_paragraph_pointers": 0,
            "median_paragraph_pointers": 0,
        }

    char_lengths = [r.char_length for r in records]
    token_lengths = [r.token_length for r in records]
    paragraph_counts = [len(r.paragraphs) for r in records]

    return {
        "objects": len(records),
        "books": len({r.book_id for r in records}),
        "authors": len({r.author for r in records}),
        "avg_chars": round(mean(char_lengths), 2),
        "median_chars": round(median(char_lengths), 2),
        "avg_tokens": round(mean(token_lengths), 2),
        "median_tokens": round(median(token_lengths), 2),
        "avg_paragraph_pointers": round(mean(paragraph_counts), 2),
        "median_paragraph_pointers": round(median(paragraph_counts), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data/raw/project_gutenberg_books.jsonl"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/processed/processed_chapters.jsonl"),
    )
    parser.add_argument(
        "--stats-json",
        type=Path,
        default=Path("outputs/dataset_stats.json"),
    )
    parser.add_argument(
        "--failures-jsonl",
        type=Path,
        default=Path("outputs/preprocessing_failures.jsonl"),
    )
    parser.add_argument("--target-words", type=int, default=300)
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--min-tokens", type=int, default=50)

    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.failures_jsonl.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[ProcessedChunk] = []
    failures: list[dict[str, Any]] = []

    raw_books = 0

    for line_number, row in load_jsonl(args.input_jsonl):
        raw_books += 1

        if "_invalid_json" in row:
            failures.append(
                {
                    "line_number": line_number,
                    "reason": "invalid_json",
                }
            )
            continue

        book_id = str(row.get("id", f"book_{line_number}"))
        raw_text = str(row.get("text", ""))

        if not raw_text.strip():
            failures.append(
                {
                    "line_number": line_number,
                    "book_id": book_id,
                    "reason": "empty_book_text",
                }
            )
            continue

        title, author = extract_metadata(raw_text, book_id)
        clean_text = clean_gutenberg_boilerplate(raw_text)
        paragraphs = split_paragraphs(clean_text)

        if not paragraphs:
            failures.append(
                {
                    "line_number": line_number,
                    "book_id": book_id,
                    "reason": "no_paragraphs_after_cleaning",
                }
            )
            continue

        chunks = build_chunks_from_paragraphs(
            paragraphs,
            target_words=args.target_words,
            max_words=args.max_words,
        )

        for chunk_index, (chunk_text, start_paragraph, end_paragraph) in enumerate(chunks):
            chunk = make_processed_chunk(
                book_id=book_id,
                title=title,
                author=author,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                start_paragraph=start_paragraph,
                end_paragraph=end_paragraph,
            )

            valid, reason = is_valid_chunk(
                chunk,
                min_chars=args.min_chars,
                min_tokens=args.min_tokens,
            )

            if not valid:
                failures.append(
                    {
                        "line_number": line_number,
                        "book_id": book_id,
                        "chapter_id": chunk.chapter_id,
                        "reason": reason,
                        "char_length": chunk.char_length,
                        "token_length": chunk.token_length,
                    }
                )
                continue

            accepted.append(chunk)

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for chunk in accepted:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    with args.failures_jsonl.open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    stats = {
        "object_definition": "stable pseudo-chapter / scene chunk searchable object",
        "reason_for_fallback": (
            "The accessible chapter-level Hugging Face subset contained too few objects. "
            "We therefore build stable text chunks from Gutenberg books while preserving "
            "book_id, chunk/chapter_id, and paragraph-position provenance."
        ),
        "raw_books": raw_books,
        "accepted_rows": len(accepted),
        "rejected_items": len(failures),
        "removed_percent_relative_to_objects": round(
            (len(failures) / (len(accepted) + len(failures))) * 100,
            4,
        )
        if accepted or failures
        else 0,
        "chunking": {
            "target_words": args.target_words,
            "max_words": args.max_words,
            "min_chars": args.min_chars,
            "min_tokens": args.min_tokens,
        },
        "tokenization_decision": {
            "lowercase": True,
            "punctuation": "removed except apostrophes inside words",
            "stemming": False,
            "reason": "First iteration keeps BM25 simple and preserves names, places, and literary phrases.",
        },
        "processed_stats": build_stats(accepted),
    }

    args.stats_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()