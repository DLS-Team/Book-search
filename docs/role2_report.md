# Role 2 Report — Semantic Representation and Dense Retrieval

## 2.1 Chapter representation strategy

**Decision:** `title_first_middle_last` — title + a window from the beginning,
middle, and end of the chapter, concatenated into one proxy text used only for
embedding. The full chapter and its paragraphs (Role 1's schema, task 1.2)
stay untouched and are what gets returned/refined for reading.

**Alternatives considered and rejected:**
| Alternative | Why not chosen |
|---|---|
| First N tokens only | Cheapest option, but systematically blind to scenes placed mid- or end-chapter — a common pattern in narrative fiction (climax at the end). |
| Averaged window embeddings | Requires embedding several windows per chapter and averaging vectors, multiplying embedding cost by ~3-5x for a benefit similar to a single concatenated proxy at 500k-chapter scale. |
| Summary indexing | Would require running a summarization model over the whole corpus first — a second model dependency, extra latency/cost, and a new failure mode (summary hallucination) for a 1-week window. |

**Measured evidence:** `src/sanity_checks.py::representation_strategy_comparison`
runs both `first_n_tokens` (deliberately truncated to 40 tokens to make the
effect visible) and `title_first_middle_last` against the same demo queries.
On the query *"a child is afraid but trying to be brave"* — whose payoff line
sits at the very end of the chapter — both strategies still agreed on this
small demo corpus, but the truncated-tokens variant lost the payoff line
entirely from its proxy text, meaning the agreement was coincidental (it won
on earlier overlapping words, not the actual climax). This is exactly the
mechanism the plan's section 4.3 warns about, and is why
`title_first_middle_last` is the default for the real corpus.

**Limitation:** for chapters much longer than ~3x the window size, the
beginning/middle/end windows can still miss a scene that occurs off those
three anchor points (e.g. at the 25% or 75% mark). If error analysis (Role 3,
task 3.6) shows this is a frequent failure mode, the next iteration should
move to overlapping windows or the averaged-window-embeddings alternative.

## 2.2 Embedding model

**Decision:** pretrained open-source sentence-transformers bi-encoder,
`sentence-transformers/all-MiniLM-L6-v2` — 384 dimensions, mean pooling, 256
max input tokens, L2-normalized output.

**Why:** the project needs reproducible offline document embeddings and fast
online query embeddings. A pretrained bi-encoder gives both without any
training step, and 384-dim / MiniLM-scale embeddings keep the >500k-chapter
corpus's vector store small enough (~750 MB at float32) to build and query
within the 1-week window on CPU.

**Not doing, and why:** we do not train a custom contrastive retriever as a
primary component. That requires a large, trusted set of query-chapter
positive/negative pairs; we do not have one, and training on weak or
self-labeled pairs risks optimizing noise rather than scene-level relevance
(architecture plan, section 4.4).

**Measured stats** (from `embed_chapters.py`, to be re-run on the real
corpus): embedding time total/per-chapter and vector size total/per-chapter
are written to `indexes/faiss_flat/embedding_stats.json` and handed to Role 3
for the benchmarking tables (task 3.5).

## 2.3 Normalization and similarity metric

**Decision:** L2-normalize every document and query embedding, then use FAISS
inner product search — this is mathematically equivalent to cosine
similarity, and avoids norm-dominated rankings.

**Alternatives considered and rejected:**
- Raw inner product on unnormalized vectors — dominated by embedding
  magnitude, which does not reliably track semantic relevance for this
  encoder.
- L2 distance over unnormalized vectors — same magnitude-sensitivity problem,
  just expressed as distance instead of similarity.

**Sanity check:** `src/sanity_checks.py::normalization_sanity_check` compares
the top-k ranking from raw inner product vs. L2-normalized cosine search on
the same demo queries and documents. On the demo corpus the two rankings
happened to coincide (the demo proxies are short and similar in length, so
norms don't vary much) — the check is designed to catch cases where they
*don't* coincide once the real corpus (chapters of very different lengths)
is embedded; normalized cosine is kept as the default regardless, since it is
the theoretically correct choice for this encoder family.

## Deliverables (section 6.1 — shared responsibilities)
- **Slide-ready result:** see `slides/role2_slide.md`.
- **Decision explanation:** this document, sections 2.1–2.3.
- **Limitation:** stated above under 2.1 — beginning/middle/end windows can
  miss scenes far from those three anchor points on unusually long chapters.
