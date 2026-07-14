"""
FastAPI serving layer for Semantic Book Scene Search.

Works whether this file is stored as:
    Book-search/app/backend.py
or:
    Book-search/backend.py

Recommended command from the repository root:
    python -m uvicorn app.backend:app --reload --port 8000
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("book_search.backend")

SearchMode = Literal["bm25", "dense", "dense_ann", "hybrid", "refined"]


def _find_project_root() -> Path:
    """Find the directory that contains src/search_engine.py."""
    current_file = Path(__file__).resolve()

    candidates = [
        current_file.parent,
        current_file.parent.parent,
        Path.cwd(),
        *current_file.parents,
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if (candidate / "src" / "search_engine.py").is_file():
            return candidate

    checked = "\n".join(
        f"  - {candidate / 'src' / 'search_engine.py'}"
        for candidate in seen
    )
    raise RuntimeError(
        "Could not locate src/search_engine.py.\n"
        "Run Uvicorn from the Book-search repository root.\n"
        f"Checked:\n{checked}"
    )


PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"

# Needed for both package imports (`src.search_engine`) and the project's
# existing top-level imports (`from bm25_search import ...`).
for path in (PROJECT_ROOT, SRC_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _load_search_engine():
    try:
        module = importlib.import_module("src.search_engine")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import src.search_engine. "
            f"Project root: {PROJECT_ROOT}. Original error: {exc}"
        ) from exc

    initialize = getattr(module, "initialize_server_state", None)
    search_function = getattr(module, "search", None)

    if not callable(initialize):
        raise RuntimeError(
            "src.search_engine must expose initialize_server_state()."
        )
    if not callable(search_function):
        raise RuntimeError("src.search_engine must expose search().")

    return initialize, search_function


initialize_server_state, search_engine_search = _load_search_engine()


class ResultCard(BaseModel):
    book_title: str
    author: str
    chapter: str
    fragment: str
    method: str
    score: float
    rank: int
    provenance: str
    low_confidence: bool = False
    warning: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[ResultCard] = Field(default_factory=list)
    latency_ms: float
    low_confidence: bool
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    project_root: str
    error: str | None = None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.initialized = False
    app.state.initialization_error = None

    try:
        initialize_server_state()
        app.state.initialized = True
        logger.info("Search engine initialized successfully.")
    except Exception as exc:
        app.state.initialization_error = str(exc)
        logger.exception("Search-engine initialization failed.")

    yield


app = FastAPI(
    title="Semantic Book Scene Search — API",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    initialized = bool(getattr(app.state, "initialized", False))
    error = getattr(app.state, "initialization_error", None)

    return HealthResponse(
        status="ok" if initialized else "degraded",
        initialized=initialized,
        project_root=str(PROJECT_ROOT),
        error=error,
    )


@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    q: str = Query(
        ...,
        min_length=1,
        description="Free-form scene, mood, or situation query",
    ),
    mode: SearchMode = Query("hybrid", description="Search mode"),
    top_k: int = Query(5, ge=1, le=50),
) -> SearchResponse:
    started_at = time.perf_counter()

    if not bool(getattr(app.state, "initialized", False)):
        error = getattr(
            app.state,
            "initialization_error",
            "Search engine has not been initialized.",
        )
        return SearchResponse(
            query=q,
            mode=mode,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            low_confidence=True,
            error=error,
        )

    try:
        raw_results = search_engine_search(
            query=q.strip(),
            method=mode,
            top_k=top_k,
        )
    except Exception as exc:
        logger.exception("Search request failed.")
        return SearchResponse(
            query=q,
            mode=mode,
            latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            low_confidence=True,
            error=str(exc),
        )

    if raw_results is None:
        raw_results = []

    cards: list[ResultCard] = []
    overall_low_confidence = False

    for position, result in enumerate(raw_results, start=1):
        if not isinstance(result, dict):
            logger.warning("Ignoring non-dictionary result: %r", result)
            continue

        low_confidence = bool(result.get("low_confidence", False))
        overall_low_confidence = overall_low_confidence or low_confidence

        rank = _as_int(result.get("rank"), position)
        score = _as_float(
            result.get("score", result.get("score_or_rank", 0.0))
        )

        fragment = str(
            result.get(
                "best_fragment",
                result.get("fragment", result.get("text", "")),
            )
            or ""
        )

        cards.append(
            ResultCard(
                book_title=str(
                    result.get("book_title", result.get("book", "Unknown Book"))
                ),
                author=str(result.get("author", "Unknown Author")),
                chapter=str(
                    result.get(
                        "chapter",
                        result.get(
                            "chapter_title",
                            result.get("chapter_id", "Unknown Chapter"),
                        ),
                    )
                ),
                fragment=fragment[:5000],
                method=str(
                    result.get("search_method", result.get("method", mode))
                ),
                score=score,
                rank=rank,
                provenance=str(result.get("provenance", "")),
                low_confidence=low_confidence,
                warning=(
                    str(result["warning"])
                    if result.get("warning") is not None
                    else None
                ),
            )
        )

    return SearchResponse(
        query=q,
        mode=mode,
        results=cards,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        low_confidence=overall_low_confidence,
    )
