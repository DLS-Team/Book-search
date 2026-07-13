"""
Task 2.1 — Long chapter representation strategy.

Decision (see docs/role2_report.md, section "2.1"):
    We do NOT embed the raw full chapter text directly. Embedding models have
    input-length limits and long chapters dilute scene-level signal (section 4.3
    of the architecture plan). Instead we build a *representation proxy* that is
    optimized for matching, while the returned/refined text (full chapter +
    paragraphs, owned by Role 1's metadata schema) stays optimized for reading.

    We implement TWO alternative proxy strategies (minimum required by 4.3):

    1. `first_n_tokens`          — baseline: title + the first N tokens of the
                                    chapter. Cheap, but systematically misses
                                    scenes that occur later in the chapter.
    2. `title_first_middle_last` — title + a window from the beginning, a window
                                    from the middle, and a window from the end of
                                    the chapter, concatenated. Meant to reduce the
                                    "scene buried in the middle" failure mode.

    Both strategies keep pointers back to:
      - the full chapter text (for display / cross-encoder rerank later),
      - the paragraph list (for Role 4's paragraph-level refinement).

    These pointers reuse the exact field names from Role 1's metadata schema
    (task 1.2) so no format is duplicated: `book_id`, `chapter_id`,
    `paragraph_ids` (ordered list, index == paragraph position).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal

Strategy = Literal["first_n_tokens", "title_first_middle_last"]


def split_paragraphs(chapter_text: str) -> List[str]:
    """Split a chapter into paragraphs on blank lines. Used only to build the
    pointer list here; the authoritative paragraph store belongs to Role 1's
    schema (task 1.2) / Role 4's refinement module (task 4.2)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", chapter_text) if p.strip()]
    return paras if paras else [chapter_text.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\S+", text)


@dataclass
class ChapterRecord:
    book_id: str
    chapter_id: str
    title: str
    author: str
    chapter_title: str
    full_text: str
    paragraphs: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.paragraphs:
            self.paragraphs = split_paragraphs(self.full_text)


@dataclass
class RepresentationResult:
    book_id: str
    chapter_id: str
    proxy_text: str
    strategy: Strategy
    # pointers, matching Role 1's schema field names (task 1.2)
    full_text_pointer: str          # here: chapter_id (lookup key into processed store)
    paragraph_ids: List[int]        # positions 0..len(paragraphs)-1


def represent_first_n_tokens(chapter: ChapterRecord, n_tokens: int = 220) -> RepresentationResult:
    """Strategy 1: title + first N tokens of the chapter body."""
    tokens = _tokenize(chapter.full_text)
    proxy = f"{chapter.title} — {chapter.chapter_title}. " + " ".join(tokens[:n_tokens])
    return RepresentationResult(
        book_id=chapter.book_id,
        chapter_id=chapter.chapter_id,
        proxy_text=proxy.strip(),
        strategy="first_n_tokens",
        full_text_pointer=chapter.chapter_id,
        paragraph_ids=list(range(len(chapter.paragraphs))),
    )


def represent_title_first_middle_last(
    chapter: ChapterRecord, window_tokens: int = 90
) -> RepresentationResult:
    """Strategy 2: title + beginning + middle + ending windows.

    Reduces the risk that a scene mentioned only in the middle or end of a
    chapter is invisible to the embedding, at the cost of a slightly longer
    proxy text than strategy 1.
    """
    tokens = _tokenize(chapter.full_text)
    n = len(tokens)
    beginning = tokens[:window_tokens]
    mid_start = max(0, n // 2 - window_tokens // 2)
    middle = tokens[mid_start: mid_start + window_tokens]
    ending = tokens[max(0, n - window_tokens):]

    proxy = (
        f"{chapter.title} — {chapter.chapter_title}. "
        + " ".join(beginning)
        + " ... " + " ".join(middle)
        + " ... " + " ".join(ending)
    )
    return RepresentationResult(
        book_id=chapter.book_id,
        chapter_id=chapter.chapter_id,
        proxy_text=proxy.strip(),
        strategy="title_first_middle_last",
        full_text_pointer=chapter.chapter_id,
        paragraph_ids=list(range(len(chapter.paragraphs))),
    )


STRATEGIES = {
    "first_n_tokens": represent_first_n_tokens,
    "title_first_middle_last": represent_title_first_middle_last,
}


def build_proxy(chapter: ChapterRecord, strategy: Strategy) -> RepresentationResult:
    return STRATEGIES[strategy](chapter)


# -- Chosen default for the production dense pipeline (documented in 2.6) --
# title_first_middle_last is the default: qualitative comparison (docs/role2_report.md)
# showed first_n_tokens systematically misses "climax" scenes placed at the end of a
# chapter (a common pattern in the sampled Gutenberg fiction chapters).
DEFAULT_STRATEGY: Strategy = "title_first_middle_last"
