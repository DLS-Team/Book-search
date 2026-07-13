"""
Task 2.7 — Backend /search endpoint for the web interface (architecture
section 3.5, "Web interface flow").

Design constraint from the plan (must hold): "intentionally thin: it does not
contain ranking logic". This module never computes scores itself — it only
routes (query, mode) to the already-existing search modules owned by each
role, and shapes the response into the schema from section 2.3
(book/author/chapter/fragment/method/score/provenance).

Modes wired here:
    - "dense"  -> src/faiss_search.py::search()          (Role 2, ready)
    - "bm25"   -> src/bm25_search.py::search()            (Role 1 interface, stub)
    - "hybrid" -> src/hybrid_search.py::search()           (Role 3 interface, stub)
    - "refined"-> src/paragraph_refinement.py::refine()    (Role 4 interface, stub)

The stubs raise a clear NotImplementedError with a pointer to the owning
task, so integration failures are obvious rather than silent. Once Role 1 /
Role 3 / Role 4 land their modules with the agreed function signatures, only
the `import` lines below need to change — the endpoint itself does not.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from faiss_search import search as dense_search  # Role 2 — ready

app = FastAPI(title="Semantic Book Scene Search — API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SearchMode = Literal["dense", "bm25", "hybrid", "refined"]

# Minimal in-memory metadata lookup so the demo can render full result cards
# (book title / author / chapter / fragment) without waiting on Role 1's
# metadata store. Keyed by chapter_id, matching the schema from task 1.2.
try:
    from data_loader import load_sample_chapters

    _DEMO_METADATA = {
        c.chapter_id: {
            "book_title": c.title,
            "author": c.author,
            "chapter_title": c.chapter_title,
            "fragment": c.paragraphs[0] if c.paragraphs else c.full_text[:200],
            "book_id": c.book_id,
            "chapter_id": c.chapter_id,
        }
        for c in load_sample_chapters()
    }
except Exception:
    _DEMO_METADATA = {}


class ResultCard(BaseModel):
    book_title: str
    author: str
    chapter: str
    fragment: str
    method: str
    score: float
    rank: int
    provenance: str


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: List[ResultCard]
    latency_ms: float
    low_confidence: bool
    error: Optional[str] = None


LOW_CONFIDENCE_THRESHOLD = 0.15  # cosine score; see docs/role2_report.md limitations


def _bm25_search(query: str, top_k: int):
    raise NotImplementedError(
        "bm25_search.search() is owned by Role 1 (task 1.4). "
        "Wire it in here once the interface (query, top_k) -> ranked candidates is delivered."
    )


def _hybrid_search(query: str, top_k: int):
    raise NotImplementedError(
        "hybrid_search.search() is owned by Role 3 (task 3.3). "
        "Wire it in here once RRF over bm25_search + faiss_search is delivered."
    )


def _refined_search(query: str, top_k: int):
    raise NotImplementedError(
        "paragraph_refinement.refine() is owned by Role 4 (task 4.2). "
        "Wire it in here once paragraph-level refinement over top chapters is delivered."
    )


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(..., min_length=1, description="Free-form scene/mood/situation query"),
    mode: SearchMode = Query("dense", description="Search mode"),
    top_k: int = Query(5, ge=1, le=50),
):
    t0 = time.time()
    try:
        if mode == "dense":
            raw_results = dense_search(q, top_k=top_k)
        elif mode == "bm25":
            raw_results = _bm25_search(q, top_k)
        elif mode == "hybrid":
            raw_results = _hybrid_search(q, top_k)
        else:
            raw_results = _refined_search(q, top_k)
    except NotImplementedError as e:
        return SearchResponse(
            query=q, mode=mode, results=[], latency_ms=(time.time() - t0) * 1000,
            low_confidence=True, error=str(e),
        )

    cards: List[ResultCard] = []
    for r in raw_results:
        meta = _DEMO_METADATA.get(r.chapter_id, {})
        cards.append(
            ResultCard(
                book_title=meta.get("book_title", "unknown"),
                author=meta.get("author", "unknown"),
                chapter=meta.get("chapter_title", r.chapter_id),
                fragment=meta.get("fragment", ""),
                method=mode,
                score=round(r.score, 4),
                rank=r.rank,
                provenance=f"book_id={meta.get('book_id', '?')} chapter_id={r.chapter_id} paragraph_position=0",
            )
        )

    low_confidence = (not cards) or (cards[0].score < LOW_CONFIDENCE_THRESHOLD)

    return SearchResponse(
        query=q,
        mode=mode,
        results=cards,
        latency_ms=round((time.time() - t0) * 1000, 2),
        low_confidence=low_confidence,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


# Run with: uvicorn backend:app --reload --port 8000
