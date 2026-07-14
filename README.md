# Semantic Book Scene Search

Semantic Book Scene Search is a retrieval system for finding scenes, moods, situations, and events in public-domain fiction.

The project combines:

- BM25 lexical retrieval
- exact dense retrieval with FAISS Flat
- low-latency dense retrieval with FAISS HNSW
- hybrid BM25 + dense retrieval using Reciprocal Rank Fusion
- chunk-level metadata and text resolution
- confidence warnings and provenance
- a FastAPI backend
- a Streamlit frontend

The system searches over stable pseudo-chapter / scene chunks created from Project Gutenberg books.

---

## Documentation Map

This README is the project entry point. Detailed design decisions, experiments, limitations, and role-specific work are documented in the existing Markdown files.

### Role reports

- [Role 1 — Data, Tokenization, and BM25 Baseline](docs/role1_notes.md)  
  Corpus construction, updated dataset statistics, chunking, cleaning, tokenization, and BM25.

- [Role 2 — Semantic Representation and Dense Retrieval](docs/role2_report.md)  
  Representation strategy, embedding model, normalization, and exact dense search.

- [Role 3 — Hybrid BM25 + Dense RRF](docs/role3_hybrid_rrf.md)  
  Reciprocal Rank Fusion, candidate merging, deduplication, and the hybrid interface.

- [Role 4 — ANN Indexing, Refinement, and Serving Integration](docs/role4_report.md)  
  HNSW benchmarking, metadata resolution, serving architecture, and quality gates.

### Role 3 evaluation and recommendation

- [Tasks 3.4 and 3.5 Evaluation](docs/tasks_3_4_3_5_evaluation.md)  
  Evaluation setup, relevance judgments, quality metrics, latency, and engineering metrics.

- [Task 3.6 Error Analysis](docs/task_3_6_error_analysis.md)  
  Category-level results, strongest and weakest queries, and representative failure cases.

- [Task 3.7 Final Recommendation](docs/task_3_7_final_recommendation.md)  
  Final comparison table and recommendation for the default and fallback retrieval modes.

- [Evaluation Query Set](docs/query.txt)  
  The 48-query benchmark grouped into exact keyword, semantic scene, mood, atmosphere, action, and ambiguous-query categories.

### Architecture decisions

- [ANN Architecture Decision](experiments/ann_architecture_decisions.md)
- [Paragraph/Chunk Resolution Decision](experiments/paragraph_refinement_decision.md)
- [Serving Integration Decision](experiments/serving_integration_decision.md)
- [Quality Gate Decision](experiments/quality_gate_decision.md)
- [Cross-Encoder Rerank Decision](experiments/rerank_decision.md)

---

## Updated Dataset

The latest preprocessing run used:

| Statistic | Value |
|---|---:|
| Raw books | 5,000 |
| Accepted searchable objects | 553,472 |
| Rejected items | 2,457 |
| Removed percentage | 0.442% |
| Books after processing | 3,150 |
| Authors after processing | 2,254 |
| Average characters per object | 2,165.85 |
| Median characters per object | 1,961 |
| Average tokens per object | 385.74 |
| Median tokens per object | 348 |
| Average paragraph pointers per object | 5.75 |
| Median paragraph pointers per object | 5 |

The searchable object is a stable pseudo-chapter / scene chunk with:

- `book_id`
- `title`
- `author`
- `chapter_id`
- `chapter_title`
- `text`
- paragraph-position pointers
- character and token lengths

The latest chunking configuration is:

| Parameter | Value |
|---|---:|
| Target words | 300 |
| Maximum words | 450 |
| Minimum characters | 300 |
| Minimum tokens | 50 |
| Boundary policy | paragraph boundaries only |

The preprocessing pipeline also removes Gutenberg boilerplate, transcriber notes, illustration notes, separator lines, repeated formatting symbols, broken line wrapping, and hyphenated line-break artifacts.

See [Role 1 notes](docs/role1_notes.md) for full details.

---

## System Architecture

```text
User query
   |
   v
Streamlit frontend
   |
   v
FastAPI backend
   |
   +--> BM25
   |
   +--> Dense FAISS Flat
   |
   +--> Dense FAISS HNSW
   |
   +--> Hybrid BM25 + HNSW + RRF
   |
   v
Chunk text and metadata resolution
   |
   v
Quality gate + provenance
   |
   v
Search results
```

The serving layer is implemented as a facade in `src/search_engine.py`. It loads reusable resources during startup, routes each query to the selected retriever, resolves returned chunk IDs into readable text, normalizes output fields, and applies a method-specific confidence gate.

---

## Search Modes

| Mode | Description | Recommended use |
|---|---|---|
| `bm25` | Lexical BM25 retrieval | names, rare terms, exact phrases |
| `dense` | Exact FAISS Flat semantic retrieval | reference-quality dense baseline |
| `dense_ann` | FAISS HNSW semantic retrieval | low-latency semantic serving |
| `hybrid` | BM25 + HNSW combined with RRF | best general-purpose mode |
| `refined` | Hybrid retrieval with resolved chunk text | user-facing readable results |

### BM25

BM25 is the lexical baseline. Text is lowercased and tokenized with a regex tokenizer. Punctuation is removed except apostrophes inside words, and stemming is disabled.

This preserves names, places, and literary phrases while providing a stable classical baseline.

### Dense FAISS Flat

The dense baseline uses:

```text
sentence-transformers/all-MiniLM-L6-v2
embedding dimension = 384
L2 normalization = enabled
similarity = inner product
```

Because document and query vectors are normalized, FAISS inner product is equivalent to cosine similarity.

### Dense FAISS ANN

The selected HNSW serving configuration is:

```text
M = 32
efConstruction = 200
efSearch = 128
```

The recorded ANN benchmark showed:

| Metric | FAISS Flat | HNSW Balanced |
|---|---:|---:|
| p95 FAISS search latency | 56.76 ms | 2.10 ms |
| Recall@10 vs Flat | 100% | 98.8% |
| Index size | 810.75 MB | 954.39 MB |

### Hybrid RRF

Hybrid retrieval combines BM25 and dense candidates using:

```text
RRF(document) = sum(1 / (60 + source_rank))
```

Raw BM25 and dense scores are not added because their scales are unrelated.

The online hybrid path runs BM25 and HNSW concurrently, then fuses the ranked candidate lists.

---

## Evaluation

The controlled evaluation used:

- 48 queries
- 2,304 usable relevance judgments
- graded labels: `0`, `1`, and `2`
- 20 saved candidates per method per query
- metrics: Precision@5, Recall@10, MRR@10, and nDCG@10

The query set covers six categories:

1. exact keyword
2. semantic scene
3. emotion and mood
4. atmosphere
5. action and situation
6. ambiguous or weak evidence

The full benchmark queries are available in [docs/query.txt](docs/query.txt).

### Final quality and latency table

| Method | P@5 | R@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.9250 | 0.2100 | 0.9375 | 0.6591 | 15.907 | 34.051 |
| Dense FAISS Flat | 0.9250 | 0.2100 | 0.9583 | 0.6423 | 82.451 | 89.346 |
| Dense FAISS ANN | 0.9208 | 0.2141 | 0.9583 | 0.6365 | 17.326 | 20.697 |
| Hybrid RRF | 0.9250 | 0.2147 | 0.9792 | 0.6753 | 89.742 | 110.120 |

### Final recommendation

Use:

```text
Hybrid RRF as the default quality-oriented mode
Dense FAISS ANN as the low-latency fallback
```

Hybrid RRF achieved the best aggregate ranking quality:

```text
nDCG@10 = 0.6753
MRR@10 = 0.9792
Recall@10 = 0.2147
```

Dense ANN offers the best strict-latency trade-off:

```text
p50 = 17.326 ms
p95 = 20.697 ms
nDCG@10 = 0.6365
```

See:

- [Tasks 3.4 and 3.5 Evaluation](docs/tasks_3_4_3_5_evaluation.md)
- [Task 3.6 Error Analysis](docs/task_3_6_error_analysis.md)
- [Task 3.7 Final Recommendation](docs/task_3_7_final_recommendation.md)

---

## Error Analysis

The strongest Hybrid RRF category was atmosphere:

```text
nDCG@10 = 0.7653
```

Other strong categories included:

```text
exact keyword = 0.6983
emotion/mood = 0.6960
action/situation = 0.6576
semantic scene = 0.6194
ambiguous/weak evidence = 0.6152
```

The main failure modes were:

- lexical overmatching;
- semantic scene drift;
- broad or ambiguous queries;
- fusion dilution when one retriever ranks weak partial matches too highly.

Examples of difficult queries include:

```text
something feels wrong
rivals become allies
a masked ball with a broken chandelier
```

The full analysis is in [Task 3.6 Error Analysis](docs/task_3_6_error_analysis.md).

---

## Repository Structure

```text
Book-search/
├── app/
│   ├── backend.py
│   └── frontend.py
├── data/
│   ├── processed/
│   │   └── processed_chapters.jsonl
│   ├── raw/
│   └── eval/
├── docs/
│   ├── role1_notes.md
│   ├── role2_report.md
│   ├── role3_hybrid_rrf.md
│   ├── role4_report.md
│   ├── tasks_3_4_3_5_evaluation.md
│   ├── task_3_6_error_analysis.md
│   ├── task_3_7_final_recommendation.md
│   └── query.txt
├── experiments/
│   ├── ann_architecture_decisions.md
│   ├── paragraph_refinement_decision.md
│   ├── quality_gate_decision.md
│   ├── rerank_decision.md
│   └── serving_integration_decision.md
├── indexes/
│   ├── faiss_ann/
│   │   └── faiss_hnsw.index
│   └── faiss_flat/
│       ├── chapter_ids.json
│       ├── embeddings.npy
│       └── flat.index
├── outputs/
│   ├── bm25_index/
│   └── dataset_stats.json
├── runs/
├── src/
│   ├── bm25_search.py
│   ├── data_preprocessing.py
│   ├── embed_chapters.py
│   ├── faiss_search.py
│   ├── hybrid_search.py
│   ├── paragraph_refinement.py
│   ├── representation.py
│   ├── rerank.py
│   └── search_engine.py
├── tests/
├── requirements.txt
└── README.md
```

---

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

---

## Running the Application

Run all commands from the repository root.

### Start the backend

```powershell
python -m uvicorn app.backend:app --port 8000
```

For development:

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

Open a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/frontend.py
```

Then open:

```text
http://localhost:8501
```

The backend must remain active on port `8000`.

---

## API Example

```text
GET /search?q=cozy+winter+night&mode=hybrid&top_k=5
```

PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/search?q=cozy%20winter%20night&mode=hybrid&top_k=5"
```

Example result:

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

---

## Example Queries

### Exact keyword

```text
red wax letter
silver key in snow
a masked ball with a broken chandelier
```

### Semantic scene

```text
rivals become allies
homecoming changes everything
an investigator starts doubting a witness they used to trust
```

### Emotion and mood

```text
guilt after betrayal
lonely in a crowd
someone hides their fear and tries to look brave
```

### Atmosphere

```text
haunted castle at night
a warm kitchen on a winter morning that feels safe
a quiet frontier town where everyone expects trouble
```

### Action and situation

```text
prisoner escapes captivity
someone follows a suspicious person without being noticed
a person interrupts a ceremony to reveal the truth
```

### Ambiguous or weak evidence

```text
something feels wrong
the rules stop working
a place feels familiar but also completely wrong
```

The full query pool is in [docs/query.txt](docs/query.txt).

---

## Building the BM25 Index

```powershell
python src\bm25_search.py build `
  --input-jsonl data\processed\processed_chapters.jsonl `
  --index-dir outputs\bm25_index `
  --max-docs 553472
```

Test BM25 directly:

```powershell
python src\bm25_search.py search `
  --index-dir outputs\bm25_index `
  --query "fireplace winter night" `
  --top-k 5
```

---

## Performance Design

The backend separates startup work from online query work.

### Startup phase

- load the chunk database;
- load BM25 index and metadata;
- load FAISS Flat resources;
- load HNSW index;
- load chapter ID mapping;
- initialize reusable models and caches.

### Online phase

- encode the query;
- run the selected retriever;
- run BM25 and HNSW concurrently for hybrid mode;
- resolve returned chunk IDs;
- apply the quality gate;
- return normalized results.

For realistic latency measurements, run without `--reload`:

```powershell
python -m uvicorn app.backend:app --port 8000
```

---

## Quality Gate

Weak matches are not hidden. They are returned with:

```json
{
  "low_confidence": true,
  "warning": "Top score is below the method threshold."
}
```

The thresholds differ by method because BM25, cosine similarity, and RRF scores use different numerical scales.

See [Quality Gate Decision](experiments/quality_gate_decision.md).

---

## Cross-Encoder Status

`src/rerank.py` contains a working cross-encoder proof of concept, but it is excluded from the normal serving path.

Reasons:

- large PyTorch dependency;
- additional CPU latency;
- limited value after chunk-level retrieval;
- unnecessary complexity for the final demo.

See [Cross-Encoder Rerank Decision](experiments/rerank_decision.md).

---

## Testing

Run all tests:

```powershell
pytest -v
```

Run hybrid-specific tests:

```powershell
pytest tests\test_hybrid_search.py -v
```

---

## Known Limitations

- BM25 depends on lexical overlap.
- Dense retrieval may match tone while missing the requested event.
- The title-first-middle-last representation can miss scenes outside the sampled windows.
- HNSW trades a small amount of recall for speed.
- Ambiguous queries often produce partially relevant rather than clearly relevant results.
- Quality thresholds are heuristic.
- Automated relevance labels may contain annotation noise.
- Chunk boundaries can omit neighboring context.

---

## License

See [LICENSE](LICENSE).
