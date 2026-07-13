# Slide: Dense Retrieval & Representation (Role 2)

**Problem:** long chapters exceed embedding input limits and dilute scene-level signal.

**Representation:** `title + beginning + middle + ending` proxy text for embedding;
full chapter/paragraphs kept separately for reading (pointers match Role 1's schema).

**Encoder:** pretrained `sentence-transformers/all-MiniLM-L6-v2` (384-dim, mean pooling,
256 max tokens). No custom training — no trusted query-chapter pair dataset exists yet.

**Metric:** L2-normalized vectors + FAISS inner product = cosine similarity.
Avoids norm-dominated rankings; sanity-checked against raw inner product.

**Index:** FAISS `IndexFlatIP` — exact reference index, later compared against
Role 4's HNSW/IVF ANN index for the speed/recall trade-off.

**Limitation:** beginning/middle/end windows can miss scenes far from those
three anchor points in unusually long chapters — flagged for iteration 3 if
error analysis shows it's a frequent failure mode.
