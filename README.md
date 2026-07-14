# Semantic Book Scene Search

Semantic Book Scene Search finds scenes, moods, situations, and events in public-domain fiction. The system combines lexical retrieval, dense semantic retrieval, approximate nearest-neighbor search, hybrid rank fusion, and chunk-level text resolution behind one web interface.

## Documentation Map

The repository already contains detailed reports and decision records. This README is the entry point; the linked documents contain the full rationale, measurements, limitations, and implementation notes.

### Role reports

- [Role 1 — Data, Tokenization, and BM25 Baseline](docs/role1_notes.md)  
  Corpus construction, preprocessing schema, dataset statistics, tokenization decisions, and the BM25 baseline.

- [Role 2 — Semantic Representation and Dense Retrieval](docs/role2_report.md)  
  Chapter representation, embedding model selection, normalization, similarity metric, and dense-retrieval limitations.

- [Role 3 — Hybrid BM25 + Dense RRF](docs/role3_hybrid_rrf.md)  
  Reciprocal Rank Fusion, candidate merging, deduplication, score interpretation, and the Role 4 handoff contract.

- [Role 4 — ANN, Refinement, and Serving Integration](docs/role4_report.md)  
  HNSW benchmarking, metadata resolution, cross-encoder decision, serving architecture, and quality gates.

### Architecture and experiment decisions

- [ANN Architecture Decision](experiments/ann_architecture_decisions.md)  
  Why HNSW was selected over IVF and compressed alternatives, including latency, memory, and recall measurements.

- [Paragraph/Chunk Resolution Decision](experiments/paragraph_refinement_decision.md)  
  Why the original local TF-IDF refinement plan was replaced with direct chunk metadata resolution.

- [Serving Integration Decision](experiments/serving_integration_decision.md)  
  Search facade, startup lifecycle, caching, unified result schema, graceful degradation, and concurrent hybrid retrieval.

- [Quality Gate Decision](experiments/quality_gate_decision.md)  
  Method-specific confidence thresholds, abstention behavior, provenance, and frontend integration.

- [Cross-Encoder Rerank Decision](experiments/rerank_decision.md)  
  Proof-of-concept results and the reason cross-encoder reranking was excluded from the main online serving path.

- [Role 2 Slide Notes](slides/role2_slide.md)  
  Slide-ready summary of the dense-retrieval contribution.

## System Overview

The online pipeline is:

```text
User query
   |
   v
FastAPI backend
   |
   +--> BM25 lexical retrieval
   |
   +--> Exact dense FAISS Flat retrieval
   |
   +--> Dense ANN retrieval with FAISS HNSW
   |
   +--> Hybrid BM25 + HNSW via RRF
   |
   v
Chunk text and metadata resolution
   |
   v
Quality gate and provenance
   |
   v
Streamlit frontend
```

The project uses chunk-level searchable objects rather than entire books or very long chapters. Role 1 produced 3,662,966 accepted searchable objects from 20,000 raw Gutenberg books, while the initial BM25 scalable experiment indexed 600,000 objects. See [Role 1 notes](docs/role1_notes.md) for the complete dataset statistics and preprocessing rationale.

## Search Modes

| Mode | Description | Recommended use |
|---|---|---|
| `bm25` | Lexical BM25 retrieval | Exact words, names, phrases, and rare terms |
| `dense` | Exact FAISS Flat semantic retrieval | Dense quality reference and controlled comparison |
| `dense_ann` | FAISS HNSW approximate retrieval | Fast semantic serving |
| `hybrid` | BM25 + dense ANN combined with RRF | Best general-purpose retrieval mode |
| `refined` | Hybrid retrieval with resolved chunk text and metadata | User-facing readable results |

### BM25

BM25 is the lexical baseline. Text is lowercased and tokenized with a regex that preserves apostrophes inside words. Stemming is intentionally disabled in the first version to preserve literary names and phrases.

Details: [Role 1 notes](docs/role1_notes.md).

### Dense retrieval

The dense pipeline uses `sentence-transformers/all-MiniLM-L6-v2`, 384-dimensional L2-normalized embeddings, and inner-product search. The production representation strategy combines the title with beginning, middle, and ending windows.

Details: [Role 2 report](docs/role2_report.md).

### ANN retrieval

The serving candidate is FAISS HNSW with the balanced configuration:

```text
M = 32
efConstruction = 200
efSearch = 128
```

On the documented benchmark, HNSW reached 98.8% Recall@10 relative to Flat while reducing p95 search latency from about 56.8 ms to about 2.1 ms.

Details: [ANN architecture decision](experiments/ann_architecture_decisions.md).

### Hybrid retrieval

Hybrid retrieval uses Reciprocal Rank Fusion:

```text
RRF(document) = sum(1 / (60 + source_rank))
```

Raw BM25 and cosine scores are not added because they use unrelated scales. The online hybrid path runs BM25 and HNSW concurrently and fuses their ranked candidate lists.

Details: [Role 3 report](docs/role3_hybrid_rrf.md) and [serving integration decision](experiments/serving_integration_decision.md).

### Chunk resolution

The initial plan proposed finding a chapter and then locating a paragraph within it. The real dataset already consists of small scene-level chunks, so the refinement module now resolves retrieved IDs directly to text and metadata using an in-memory dictionary.

Details: [paragraph/chunk resolution decision](experiments/paragraph_refinement_decision.md).

## Repository Structure

```text
Book-search/
├── app/
│   ├── backend.py
│   └── frontend.py
├── data/
│   ├── processed/
│   │   └── processed_chapters.jsonl
│   └── raw/
├── docs/
│   ├── instructions.md
│   ├── role1_notes.md
│   ├── role2_report.md
│   ├── role3_hybrid_rrf.md
│   └── role4_report.md
├── experiments/
│   ├── ann_architecture_decisions.md
│   ├── paragraph_refinement_decision.md
│   ├── quality_gate_decision.md
│   ├── rerank_decision.md
│   └── serving_integration_decision.md
├── indexes/
│   ├── bm25/
│   ├── faiss_ann/
│   │   └── faiss_hnsw.index
│   └── faiss_flat/
│       ├── chapter_ids.json
│       ├── embeddings.npy
│       ├── flat.index
│       └── proxies.jsonl
├── outputs/
│   ├── dataset_stats.json
│   └── bm25_index/
├── slides/
│   └── role2_slide.md
├── src/
│   ├── ann_benchmark.py
│   ├── bm25_search.py
│   ├── data_preprocessing.py
│   ├── embed_chapters.py
│   ├── faiss_search.py
│   ├── hybrid_search.py
│   ├── paragraph_refinement.py
│   ├── representation.py
│   ├── rerank.py
│   ├── sanity_checks.py
│   └── search_engine.py
├── tests/
│   └── test_hybrid_search.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.11 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Application

Run commands from the repository root.

### Start the backend

```powershell
python -m uvicorn app.backend:app --port 8000
```

During development:

```powershell
python -m uvicorn app.backend:app --reload --port 8000
```

Useful endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/search
```

### Start the frontend

Open another terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/frontend.py
```

Then open:

```text
http://localhost:8501
```

The backend must remain active on port `8000`.

## API Example

```text
GET /search?q=cozy+winter+night&mode=hybrid&top_k=5
```

PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/search?q=cozy%20winter%20night&mode=hybrid&top_k=5"
```

The unified result includes:

```json
{
  "book_title": "Example Book",
  "author": "Example Author",
  "chapter": "Example Chapter",
  "fragment": "Matched chunk text...",
  "method": "hybrid",
  "score": 0.0325,
  "rank": 1,
  "provenance": "Gutenberg ID: ...",
  "low_confidence": false,
  "warning": null
}
```

The output contract and normalization logic are documented in [serving integration decision](experiments/serving_integration_decision.md).

## Example Queries

### Lexical

```text
fireplace winter night
storm at sea
murder in the library
```

### Semantic

```text
a lonely person walking through a dark city
someone feels guilty after betraying a friend
a child is frightened but trying to be brave
```

### Atmosphere and events

```text
an eerie abandoned house at night
a tense silence before something terrible happens
two people saying goodbye before a dangerous journey
```

## Building the BM25 Index

```powershell
python src\bm25_search.py build `
  --input-jsonl data\processed\processed_chapters.jsonl `
  --index-dir outputs\bm25_index `
  --max-docs 600000
```

Direct BM25 test:

```powershell
python src\bm25_search.py search `
  --index-dir outputs\bm25_index `
  --query "fireplace winter night" `
  --top-k 5
```

The BM25 design and corpus scale are explained in [Role 1 notes](docs/role1_notes.md).

## Dense and ANN Artifacts

Exact dense retrieval expects:

```text
indexes/faiss_flat/flat.index
indexes/faiss_flat/chapter_ids.json
```

ANN retrieval expects:

```text
indexes/faiss_ann/faiss_hnsw.index
indexes/faiss_flat/chapter_ids.json
```

The ID list must be built from the same embedding run and in the same order as the FAISS vectors. The ANN trade-offs and Windows-safe FAISS I/O workaround are documented in [ANN architecture decision](experiments/ann_architecture_decisions.md).

## Performance Design

The serving layer follows a bootstrap/online split:

### Bootstrap phase

- Load processed chunks into memory
- Load BM25 index and metadata
- Load FAISS Flat resources
- Load HNSW index
- Load chapter ID mapping

### Online phase

- Encode the query
- Run the selected retriever
- Run BM25 and HNSW concurrently for hybrid mode
- Resolve only returned chunk IDs
- Apply the quality gate
- Return the normalized response

Repeated identical searches can be served from an in-process LRU cache. Full details are in [serving integration decision](experiments/serving_integration_decision.md).

For realistic latency measurements, run Uvicorn without `--reload`:

```powershell
python -m uvicorn app.backend:app --port 8000
```

## Quality Gate

The system does not silently hide weak matches. It returns the closest results with:

```json
{
  "low_confidence": true,
  "warning": "Top score is below the method threshold."
}
```

The thresholds differ by retrieval method because BM25, cosine similarity, and RRF scores have different scales. See [quality gate decision](experiments/quality_gate_decision.md).

## Cross-Encoder Status

`src/rerank.py` contains a working cross-encoder proof of concept, but it is not part of the normal online path. The team excluded it because of dependency size, CPU latency, and limited incremental value for already-small chunks.

See [cross-encoder rerank decision](experiments/rerank_decision.md).

## Testing

Run the available tests from the repository root:

```powershell
pytest -v
```

Hybrid-specific tests:

```powershell
pytest tests\test_hybrid_search.py -v
```

Role 2 sanity checks are described in [Role 2 report](docs/role2_report.md).

## Troubleshooting

### Backend cannot locate `search_engine.py`

Run the backend from the repository root:

```powershell
python -m uvicorn app.backend:app --port 8000
```

### Health status is degraded

Open:

```text
http://127.0.0.1:8000/health
```

Common causes are missing indexes, a missing processed dataset, incompatible vector dimensions, or a mismatched ID mapping.

### Results show `Unknown Book`

The retrieved `chapter_id` may not exist in `data/processed/processed_chapters.jsonl`, or the index and processed corpus may have been generated from different versions.

### Hybrid or ANN mode fails

Verify:

```text
indexes/faiss_ann/faiss_hnsw.index
indexes/faiss_flat/chapter_ids.json
```

The serving layer intentionally allows BM25 and exact dense search to continue when ANN resources are unavailable. See [serving integration decision](experiments/serving_integration_decision.md).

## Known Limitations

- BM25 requires lexical overlap.
- The beginning/middle/end dense representation can miss scenes outside those windows.
- HNSW trades a small amount of recall for much lower latency.
- Chunk boundaries may still occasionally produce imperfect reading fragments.
- Quality thresholds are heuristic and should be calibrated on a labeled evaluation set.
- The cross-encoder is implemented only as a proof of concept.

Each limitation is discussed in the corresponding role or decision document linked above.

## License

See [LICENSE](LICENSE).
