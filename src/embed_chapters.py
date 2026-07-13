"""
Task 2.2 — Embedding model choice + offline embedding generation.
Task 2.3 — Vector normalization (L2) for cosine-style retrieval.

Decision (docs/role2_report.md, sections "2.2" and "2.3"):
    - Encoder: pretrained open-source sentence-transformers bi-encoder,
      default `sentence-transformers/all-MiniLM-L6-v2`
      (384-dim, mean pooling, max 256 tokens, reproducible, fast enough for
      >500k chapters on CPU in the given 1-week window).
    - We do NOT train a custom contrastive retriever as a primary component:
      we do not have a large trusted set of query-chapter positive/negative
      pairs, and training on weak/self-labeled pairs risks optimizing noise
      instead of scene-level relevance (section 4.4, "Not doing, and why").
    - All document and query embeddings are L2-normalized so that FAISS inner
      product search behaves as cosine similarity (section 4.6). Direction
      captures semantic similarity for text embeddings better than magnitude.

This module is written against the real `sentence-transformers` API. Because
weight downloads require network access to huggingface.co (not available in
this execution sandbox), a lightweight deterministic `MockEncoder` is also
provided so the rest of the pipeline (representation -> embedding -> FAISS ->
search) can be built, tested, and demoed end-to-end without internet access.
Swap `USE_MOCK_ENCODER = False` and install `sentence-transformers` + `torch`
to run with the real model.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import numpy as np

from representation import ChapterRecord, RepresentationResult, build_proxy, DEFAULT_STRATEGY

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = str(_PROJECT_ROOT / "indexes" / "faiss_flat")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
POOLING = "mean"
MAX_TOKENS = 256
NORMALIZATION = "l2"

USE_MOCK_ENCODER = True  # flip to False once sentence-transformers + weights are available


class MockEncoder:
    """Deterministic hash-based pseudo-embedding, used ONLY to validate the
    pipeline (representation -> embedding -> normalization -> FAISS -> search)
    without network access to download real model weights. It is NOT a
    semantic encoder and must never be used for the real evaluation in task 3.x.
    """

    def __init__(self, dim: int = EMBEDDING_DIM, seed: int = 13):
        self.dim = dim
        self.seed = seed

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = t.lower().split()
            v = np.zeros(self.dim, dtype=np.float32)
            for w in words:
                h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                rng = np.random.default_rng(h % (2**32))
                v += rng.normal(size=self.dim).astype(np.float32)
            if words:
                v /= len(words)
            vecs[i] = v
        return vecs


def load_encoder():
    if USE_MOCK_ENCODER:
        return MockEncoder(dim=EMBEDDING_DIM)
    from sentence_transformers import SentenceTransformer  # noqa: local import, heavy dep

    return SentenceTransformer(MODEL_NAME)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def embed_chapters(
    chapters: Iterable[ChapterRecord],
    strategy: str = DEFAULT_STRATEGY,
    out_dir: str = DEFAULT_OUT_DIR,
    batch_size: int = 64,
) -> dict:
    """Offline embedding generation for the whole corpus.

    Writes:
      - {out_dir}/embeddings.npy      float32 [N, dim], L2-normalized
      - {out_dir}/chapter_ids.json    list[str], row i -> chapter_id of embeddings[i]
      - {out_dir}/proxies.jsonl       one RepresentationResult per line (audit trail)

    Returns a dict of measured stats (embedding time, vector size) that Role 3
    needs for the metrics/benchmark tables (task 3.5).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    encoder = load_encoder()

    chapter_ids: List[str] = []
    proxies: List[RepresentationResult] = []
    for ch in chapters:
        rep = build_proxy(ch, strategy)
        chapter_ids.append(rep.chapter_id)
        proxies.append(rep)

    texts = [p.proxy_text for p in proxies]

    t0 = time.time()
    raw_vecs = encoder.encode(texts, batch_size=batch_size, show_progress_bar=False)
    raw_vecs = np.asarray(raw_vecs, dtype=np.float32)
    embed_seconds = time.time() - t0

    norm_vecs = l2_normalize(raw_vecs)

    np.save(out_path / "embeddings.npy", norm_vecs)
    with open(out_path / "chapter_ids.json", "w", encoding="utf-8") as f:
        json.dump(chapter_ids, f, ensure_ascii=False, indent=2)
    with open(out_path / "proxies.jsonl", "w", encoding="utf-8") as f:
        for p in proxies:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")

    stats = {
        "num_chapters": len(chapter_ids),
        "embedding_dim": int(norm_vecs.shape[1]) if len(norm_vecs) else EMBEDDING_DIM,
        "embedding_seconds_total": round(embed_seconds, 4),
        "embedding_seconds_per_chapter": round(embed_seconds / max(1, len(chapter_ids)), 6),
        "vector_size_bytes_total": int(norm_vecs.nbytes),
        "vector_size_bytes_per_chapter": int(norm_vecs.nbytes / max(1, len(chapter_ids))),
        "model_name": MODEL_NAME if not USE_MOCK_ENCODER else "MOCK_ENCODER (pipeline validation only)",
        "representation_strategy": strategy,
        "normalization": NORMALIZATION,
    }
    with open(out_path / "embedding_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def encode_query(query: str) -> np.ndarray:
    """Fast online query embedding (same encoder + same L2 normalization)."""
    encoder = load_encoder()
    vec = encoder.encode([query])
    vec = np.asarray(vec, dtype=np.float32)
    return l2_normalize(vec)[0]


if __name__ == "__main__":
    from data_loader import load_sample_chapters

    chapters = load_sample_chapters()
    stats = embed_chapters(chapters)
    print(json.dumps(stats, indent=2))
