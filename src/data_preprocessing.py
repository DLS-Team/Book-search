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


class RunningStats:
    def __init__(self) -> None:
        self.char_lengths: list[int] = []
        self.token_lengths: list[int] = []
        self.paragraph_counts: list[int] = []
        self.book_ids: set[str] = set()
        self.authors: set[str] = set()

    def add(self, chunk: ProcessedChunk) -> None:
        self.char_lengths.append(chunk.char_length)
        self.token_lengths.append(chunk.token_length)
        self.paragraph_counts.append(len(chunk.paragraphs))
        self.book_ids.add(chunk.book_id)
        self.authors.add(chunk.author)

    def to_dict(self) -> dict[str, Any]:
        if not self.char_lengths:
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

        return {
            "objects": len(self.char_lengths),
            "books": len(self.book_ids),
            "authors": len(self.authors),
            "avg_chars": round(mean(self.char_lengths), 2),
            "median_chars": round(median(self.char_lengths), 2),
            "avg_tokens": round(mean(self.token_lengths), 2),
            "median_tokens": round(median(self.token_lengths), 2),
            "avg_paragraph_pointers": round(mean(self.paragraph_counts), 2),
            "median_paragraph_pointers": round(median(self.paragraph_counts), 2),
        }


def normalize_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_gutenberg_noise(text: str) -> str:
    text = str(text)

    text = re.sub(r"\[Illustration:.*?\]", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[Transcriber's Note:.*?\]", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[Transcriber’s Note:.*?\]", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[Footnote.*?\]", " ", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove long visual separator lines.
    text = re.sub(r"^[\s*_\-=~•·]{5,}$", " ", text, flags=re.MULTILINE)

    noisy_line_patterns = [
        r"Produced by .*",
        r"Distributed Proofreading Team.*",
        r"Project Gutenberg.*",
        r"www\.gutenberg\.org.*",
        r"Internet Archive.*",
        r"Release Date:.*",
        r"Language:.*",
        r"Character set encoding:.*",
        r"\[eBook #.*?\]",
        r"Transcriber's note:.*",
        r"Transcriber’s note:.*",
    ]

    for pattern in noisy_line_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return text


def clean_paragraph_text(text: str) -> str:
    text = str(text)

    # Join hyphenated words split by line wrapping: "educa-\ntion" -> "education".
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Convert remaining line breaks to spaces so stored chunk text is readable.
    text = text.replace("\n", " ")

    # Remove simple Gutenberg formatting markers.
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)

    # Normalize common typography.
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", " -- ").replace("–", " - ")

    # Remove repeated decoration symbols and page-number-like fragments.
    text = re.sub(r"[*_=~]{2,}", " ", text)
    text = re.sub(r"\bPage\s+\d+\b", " ", text, flags=re.IGNORECASE)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_gutenberg_boilerplate(text: str) -> str:
    text = normalize_text(text)

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

    text = remove_gutenberg_noise(text)
    return normalize_text(text)


def extract_metadata(text: str, book_id: str) -> tuple[str, str]:
    title_match = re.search(r"^Title:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    author_match = re.search(r"^Author:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)

    title = title_match.group(1).strip() if title_match else f"Book {book_id}"
    author = author_match.group(1).strip() if author_match else "Unknown author"

    return clean_paragraph_text(title), clean_paragraph_text(author)


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)


def split_paragraphs(text: str) -> list[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    cleaned_paragraphs: list[str] = []

    for paragraph in raw_paragraphs:
        paragraph = clean_paragraph_text(paragraph)

        if not paragraph:
            continue
        if len(paragraph) < 40:
            continue
        if re.fullmatch(r"[A-Z0-9 .,'\-:;]+", paragraph) and len(paragraph.split()) < 12:
            continue

        cleaned_paragraphs.append(paragraph)

    return cleaned_paragraphs


def paragraph_to_words(paragraph: str) -> list[str]:
    return re.findall(r"\S+", paragraph)


def build_chunks_from_paragraphs(
    paragraphs: list[str],
    target_words: int,
    max_words: int,
) -> list[tuple[str, int, int]]:
    """
    Build scene-like chunks from complete paragraphs.

    The chunk boundary is always a paragraph boundary. This avoids cutting
    sentences in the middle unless the original source paragraph itself is broken.
    Returns: (chunk_text, start_paragraph_position, end_paragraph_position)
    """
    chunks: list[tuple[str, int, int]] = []
    current_parts: list[str] = []
    current_words = 0
    start_pos: int | None = None
    end_pos: int | None = None

    for paragraph_position, paragraph in enumerate(paragraphs):
        word_count = len(paragraph_to_words(paragraph))
        if word_count == 0:
            continue

        if start_pos is None:
            start_pos = paragraph_position

        current_parts.append(paragraph)
        current_words += word_count
        end_pos = paragraph_position

        # Stop only after a full paragraph, so stored text is not cut mid-sentence.
        if current_words >= target_words or current_words >= max_words:
            chunk_text = clean_paragraph_text(" ".join(current_parts))
            if chunk_text and start_pos is not None and end_pos is not None:
                chunks.append((chunk_text, start_pos, end_pos))

            current_parts = []
            current_words = 0
            start_pos = None
            end_pos = None

    if current_parts and start_pos is not None and end_pos is not None:
        chunk_text = clean_paragraph_text(" ".join(current_parts))
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
    chunk_text = clean_paragraph_text(chunk_text)
    tokens = simple_tokenize(chunk_text)

    return ProcessedChunk(
        book_id=book_id,
        title=title,
        author=author,
        chapter_id=f"{book_id}_chunk_{chunk_index:06d}",
        chapter_title=f"Chunk {chunk_index}",
        text=chunk_text,
        paragraphs=[
            {"paragraph_position": pos, "source": "book_paragraph"}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/raw/project_gutenberg_books.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/processed/processed_chapters.jsonl"))
    parser.add_argument("--stats-json", type=Path, default=Path("outputs/dataset_stats.json"))
    parser.add_argument("--failures-jsonl", type=Path, default=Path("outputs/preprocessing_failures.jsonl"))
    parser.add_argument("--target-words", type=int, default=300)
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--min-tokens", type=int, default=50)
    args = parser.parse_args()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.failures_jsonl.parent.mkdir(parents=True, exist_ok=True)

    stats_tracker = RunningStats()
    accepted_rows = 0
    failures_count = 0
    raw_books = 0
    seen_book_ids: set[str] = set()

    with args.output_jsonl.open("w", encoding="utf-8") as output_f, args.failures_jsonl.open(
        "w", encoding="utf-8"
    ) as failures_f:
        for line_number, row in load_jsonl(args.input_jsonl):
            raw_books += 1

            if "_invalid_json" in row:
                failures_count += 1
                failures_f.write(json.dumps({"line_number": line_number, "reason": "invalid_json"}) + "\n")
                continue

            book_id = str(row.get("id", f"book_{line_number}"))

            if book_id in seen_book_ids:
                failures_count += 1
                failures_f.write(
                    json.dumps(
                        {"line_number": line_number, "book_id": book_id, "reason": "duplicate_book_id"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            seen_book_ids.add(book_id)

            raw_text = str(row.get("text", ""))
            if not raw_text.strip():
                failures_count += 1
                failures_f.write(
                    json.dumps(
                        {"line_number": line_number, "book_id": book_id, "reason": "empty_book_text"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue

            title, author = extract_metadata(raw_text, book_id)
            clean_text = clean_gutenberg_boilerplate(raw_text)
            paragraphs = split_paragraphs(clean_text)

            if not paragraphs:
                failures_count += 1
                failures_f.write(
                    json.dumps(
                        {"line_number": line_number, "book_id": book_id, "reason": "no_paragraphs_after_cleaning"},
                        ensure_ascii=False,
                    )
                    + "\n"
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

                valid, reason = is_valid_chunk(chunk, min_chars=args.min_chars, min_tokens=args.min_tokens)
                if not valid:
                    failures_count += 1
                    failures_f.write(
                        json.dumps(
                            {
                                "line_number": line_number,
                                "book_id": book_id,
                                "chapter_id": chunk.chapter_id,
                                "reason": reason,
                                "char_length": chunk.char_length,
                                "token_length": chunk.token_length,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                output_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
                stats_tracker.add(chunk)
                accepted_rows += 1

    total_items = accepted_rows + failures_count
    stats = {
        "object_definition": "stable pseudo-chapter / scene chunk searchable object",
        "reason_for_fallback": (
            "The accessible chapter-level Hugging Face subset contained too few objects. "
            "We therefore build stable text chunks from Gutenberg books while preserving "
            "book_id, chunk/chapter_id, and paragraph-position provenance."
        ),
        "raw_books": raw_books,
        "accepted_rows": accepted_rows,
        "rejected_items": failures_count,
        "removed_percent_relative_to_objects": round((failures_count / total_items) * 100, 4)
        if total_items
        else 0,
        "chunking": {
            "target_words": args.target_words,
            "max_words": args.max_words,
            "min_chars": args.min_chars,
            "min_tokens": args.min_tokens,
            "boundary_policy": "chunks end only at paragraph boundaries",
        },
        "cleaning": {
            "gutenberg_boilerplate_removed": True,
            "line_breaks_in_stored_text": "converted to spaces",
            "hyphenated_line_breaks": "joined",
            "separator_lines_removed": True,
            "transcriber_notes_removed": True,
            "illustration_notes_removed": True,
            "repeated_formatting_symbols_removed": True,
        },
        "tokenization_decision": {
            "lowercase": True,
            "punctuation": "removed except apostrophes inside words",
            "stemming": False,
            "reason": "First iteration keeps BM25 simple and preserves names, places, and literary phrases.",
        },
        "processed_stats": stats_tracker.to_dict(),
    }

    args.stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
