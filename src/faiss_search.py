"""
Task 2.4 — FAISS Flat baseline index (exact dense reference search).
Task 2.5 — Registration of dense-pipeline metadata in the shared index registry.

Decision (docs/role2_report.md):
    IndexFlatIP over L2-normalized vectors == exact cosine similarity search.
    Flat gives reference-quality dense results (section 4.7). Role 4 later
    compares an ANN index (HNSW/IVF) against this Flat baseline for the
    speed/recall trade-off (task 4.1) — Flat is not replaced, it stays as the
    quality reference throughout the project.

Search interface (kept intentionally simple and stable, per section 6.1
"expose simple shared interfaces early"):

    search(query: str, top_k: int) -> List[DenseResult]

Role 3 (hybrid RRF, task 3.3) and Role 4 (search_engine.py, task 4.4) both
depend on exactly this function signature.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import faiss
import numpy as np

from embed_chapters import encode_query, MODEL_NAME, USE_MOCK_ENCODER, NORMALIZATION
from representation import DEFAULT_STRATEGY

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = _PROJECT_ROOT / "indexes" / "faiss_flat"
REGISTRY_PATH = _PROJECT_ROOT / "indexes" / "index_registry.json"


@dataclass
class DenseResult:
    chapter_id: str
    score: float          # cosine similarity, higher is better
    rank: int              # 1-based rank within this result list


def build_flat_index(index_dir: str = str(INDEX_DIR)) -> faiss.Index:
    """Build (or rebuild) the FAISS Flat inner-product index from the
    embeddings produced by embed_chapters.embed_chapters()."""
    index_dir = Path(index_dir)
    embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_dir / "flat.index"))
    return index


def load_flat_index(index_dir: str = str(INDEX_DIR)) -> faiss.Index:
    index_dir = Path(index_dir)
    return faiss.read_index(str(index_dir / "flat.index"))


def load_chapter_ids(index_dir: str = str(INDEX_DIR)) -> List[str]:
    index_dir = Path(index_dir)
    with open(index_dir / "chapter_ids.json", "r", encoding="utf-8") as f:
        return json.load(f)


def search(query: str, top_k: int = 10, index_dir: str = str(INDEX_DIR)) -> List[DenseResult]:
    """Dense semantic search. This is the exact interface Role 3 (hybrid RRF)
    and Role 4 (search_engine.py) call into."""
    index = load_flat_index(index_dir)
    chapter_ids = load_chapter_ids(index_dir)

    q_vec = encode_query(query).astype(np.float32).reshape(1, -1)
    scores, idxs = index.search(q_vec, top_k)

    results: List[DenseResult] = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx == -1:
            continue
        results.append(DenseResult(chapter_id=chapter_ids[idx], score=float(score), rank=rank))
    return results


def register_dense_pipeline(
    index_dir: str = str(INDEX_DIR),
    representation_strategy: str = DEFAULT_STRATEGY,
    registry_path: str = str(REGISTRY_PATH),
) -> dict:
    """Task 2.5 — write dense-pipeline metadata into the shared
    metadata + index version registry (architecture section 3.2, model lifecycle).
    Reproducibility contract: encoder + representation strategy + index type +
    metric + build date must be enough to rebuild the index from scratch.
    """
    index_dir = Path(index_dir)
    with open(index_dir / "embedding_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    entry = {
        "component": "dense_retrieval",
        "encoder_name": MODEL_NAME if not USE_MOCK_ENCODER else "MOCK_ENCODER (pipeline validation only)",
        "embedding_dim": stats["embedding_dim"],
        "pooling": "mean",
        "max_tokens": 256,
        "representation_strategy": representation_strategy,
        "index_type": "FAISS IndexFlatIP",
        "metric": "cosine (via L2-normalized vectors + inner product)",
        "normalization": NORMALIZATION,
        "num_chapters_indexed": stats["num_chapters"],
        "embedding_seconds_total": stats["embedding_seconds_total"],
        "vector_size_bytes_total": stats["vector_size_bytes_total"],
        "build_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    registry_path = Path(registry_path)
    registry: dict = {}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

    registry["dense_retrieval"] = entry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    return entry


if __name__ == "__main__":
    build_flat_index()
    entry = register_dense_pipeline()
    print(json.dumps(entry, ensure_ascii=False, indent=2))

    demo_queries = [
        "cozy winter night near a fireplace",
        "hero feels guilty after betrayal",
        "a lonely person walking through a dark city",
        "tense conversation before a murder",
        "a child is afraid but trying to be brave",
    ]
    for q in demo_queries:
        print(f"\nQuery: {q}")
        t0 = time.time()
        res = search(q, top_k=3)
        dt = (time.time() - t0) * 1000
        for r in res:
            print(f"  rank={r.rank} chapter_id={r.chapter_id} score={r.score:.4f}")
        print(f"  latency_ms={dt:.2f}")
