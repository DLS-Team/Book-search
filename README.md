# Semantic Book Scene Search

A search system for finding scenes, moods, situations, and events in public-domain fiction.

Instead of relying only on exact keywords, the project combines lexical retrieval, dense semantic retrieval, approximate nearest-neighbor search, hybrid rank fusion, and chunk-level text resolution.

## Features

- Free-form scene and mood search
- BM25 lexical retrieval
- Exact dense retrieval with FAISS Flat
- Fast dense retrieval with FAISS HNSW
- Hybrid BM25 + dense retrieval using Reciprocal Rank Fusion
- Chunk text and metadata resolution for every search mode
- Low-confidence warnings
- FastAPI backend
- Streamlit frontend
- Cached indexes and repeated-query results
- Public-domain Gutenberg-based corpus

## Project Structure

```text
Book-search/
├── app/
│   ├── backend.py
│   └── frontend.py
├── data/
│   ├── raw/
│   └── processed/
│       └── processed_chapters.jsonl
├── indexes/
│   ├── faiss_ann/
│   │   └── faiss_hnsw.index
│   └── faiss_flat/
│       ├── chapter_ids.json
│       ├── embeddings.npy
│       └── flat.index
├── outputs/
│   └── bm25_index/
│       ├── bm25_index/
│       ├── metadata.pkl
│       └── bm25_registry.json
├── src/
│   ├── bm25_search.py
│   ├── data_loader.py
│   ├── faiss_search.py
│   ├── hybrid_search.py
│   ├── paragraph_refinement.py
│   ├── representation.py
│   ├── rerank.py
│   └── search_engine.py
├── requirements.txt
└── README.md
```

## Search Modes

### `bm25`

Keyword-oriented lexical search.

Best for:

```text
fireplace winter night
storm at sea
murder in the library
```

### `dense`

Exact semantic search using FAISS `IndexFlatIP`.

Best for evaluating dense retrieval quality, but slower than ANN on large collections.

Example:

```text
a lonely person walking through a dark city
```

### `dense_ann`

Fast semantic retrieval using an HNSW approximate nearest-neighbor index.

This is the recommended production dense mode.

### `hybrid`

Combines BM25 and dense ANN results using Reciprocal Rank Fusion.

Hybrid search is useful when a query contains both important keywords and broader semantic meaning.

### `refined`

Runs hybrid retrieval and returns resolved chunk text and metadata.

In the current chunk-based dataset, the retrieved chunk is already the display fragment, so refinement mainly resolves IDs into readable text and metadata.

## Requirements

- Python 3.11 or newer
- Windows, Linux, or macOS
- Enough RAM to load the processed chunk database and indexes
- Prebuilt BM25 and FAISS indexes

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Project

Run all commands from the repository root.

### 1. Start the backend

```powershell
python -m uvicorn app.backend:app --port 8000
```

For development with automatic reload:

```powershell
python -m uvicorn app.backend:app --reload --port 8000
```

Backend endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/search
```

Check the health endpoint before starting the frontend:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "initialized": true,
  "project_root": "C:\\path\\to\\Book-search",
  "error": null
}
```

### 2. Start the frontend

Open a second terminal:

```powershell
cd "C:\path\to\Book-search"
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/frontend.py
```

Open:

```text
http://localhost:8501
```

The backend must remain active on port `8000`.

## Example Queries

Lexical:

```text
fireplace winter night
storm at sea
lost child in the forest
```

Semantic:

```text
a child is frightened but trying to be brave
someone feels guilty after betraying a friend
two people saying goodbye before a dangerous journey
```

Atmosphere:

```text
an eerie abandoned house at night
a peaceful morning in the countryside
a tense silence before something terrible happens
```

Action:

```text
someone escaping from pursuers through the woods
a secret meeting between two enemies
a character discovering a hidden room
```

## API Usage

Example request:

```text
GET /search?q=cozy+winter+night&mode=hybrid&top_k=5
```

PowerShell example:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/search?q=cozy%20winter%20night&mode=hybrid&top_k=5"
```

Example response shape:

```json
{
  "query": "cozy winter night",
  "mode": "hybrid",
  "results": [
    {
      "book_title": "Example Book",
      "author": "Example Author",
      "chapter": "Chapter 3",
      "fragment": "Matched chunk text...",
      "method": "hybrid",
      "score": 0.0325,
      "rank": 1,
      "provenance": "Gutenberg ID: ...",
      "low_confidence": false,
      "warning": null
    }
  ],
  "latency_ms": 215.4,
  "low_confidence": false,
  "error": null
}
```

## Building the BM25 Index

```powershell
python src\bm25_search.py build `
  --input-jsonl data\processed\processed_chapters.jsonl `
  --index-dir outputs\bm25_index `
  --max-docs 600000
```

Test BM25 directly:

```powershell
python src\bm25_search.py search `
  --index-dir outputs\bm25_index `
  --query "fireplace winter night" `
  --top-k 5
```

## Dense Index Files

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

The number and order of entries in `chapter_ids.json` must match the vectors stored in the corresponding FAISS index.

## Performance

The server preloads major resources during startup:

- processed chunk database
- BM25 index and metadata
- FAISS Flat index
- FAISS HNSW index
- chapter ID mapping

The first startup may take some time, but query latency should be significantly lower afterward.

Recommended serving modes:

```text
bm25      -> lexical baseline
dense     -> exact dense reference
dense_ann -> fast semantic retrieval
hybrid    -> BM25 + HNSW + RRF
refined   -> hybrid results with resolved readable text
```

For accurate latency measurements, run Uvicorn without `--reload`:

```powershell
python -m uvicorn app.backend:app --port 8000
```

Repeated identical queries may be served from the in-process query cache.

## Troubleshooting

### `Could not import module "backend"`

Run Uvicorn from the repository root:

```powershell
python -m uvicorn app.backend:app --port 8000
```

Do not use:

```powershell
uvicorn backend:app
```

unless `backend.py` is located in the repository root.

### Backend health is degraded

Open:

```text
http://127.0.0.1:8000/health
```

Check the `error` field and the backend terminal logs.

Common causes:

- missing BM25 index
- missing FAISS index
- missing `chapter_ids.json`
- missing processed JSONL file
- incompatible embedding dimensions
- incorrect working directory

### Frontend cannot reach backend

Make sure the backend is running at:

```text
http://127.0.0.1:8000
```

Then restart Streamlit:

```powershell
python -m streamlit run app/frontend.py
```

### Results show `Unknown Book`

This usually means the retrieved `chapter_id` does not exist in:

```text
data/processed/processed_chapters.jsonl
```

It can also indicate that the FAISS index and `chapter_ids.json` were built from a different version of the dataset.

### Dense ANN mapping mismatch

The number of HNSW vectors must match the number of IDs in:

```text
indexes/faiss_flat/chapter_ids.json
```

Rebuild both from the same embedding run if they differ.

### High latency

Check that:

- BM25 is not reloaded per query
- FAISS indexes are loaded once at startup
- hybrid uses HNSW rather than exact Flat search
- fragment resolution uses the in-memory chunk dictionary
- Uvicorn is running without `--reload`
- the embedding model is cached and not recreated for every query

## Git Workflow

Update a role branch with the latest `main`:

```powershell
git switch role1
git fetch origin
git merge origin/main
git push origin role1
```

Restore stashed changes:

```powershell
git stash list
git stash pop
```

Create a safety branch before major changes:

```powershell
git switch -c backup-before-optimization
git push -u origin backup-before-optimization
```

## Notes

- The corpus contains public-domain fiction.
- Retrieval works over preprocessed scene/chapter chunks.
- `dense` is retained as an exact reference baseline.
- `dense_ann` is intended for low-latency serving.
- Raw BM25 and dense scores are not directly added together; hybrid search uses rank-based RRF.
