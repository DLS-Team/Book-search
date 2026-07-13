# Role 4: ANN Indexing, Refinement, and Serving Integration
## Engineering Decisions & Implementation Report

**Role Scope:** Owning the serving-oriented search behavior, local refinement, and the unified demo path. This role turns isolated search algorithms into a cohesive, measurable, and fast online retrieval system.

---

### 1. Task 4.1: ANN Indexing (HNSW vs. Flat Baseline)

**Decision:** Selected **FAISS HNSW** (Hierarchical Navigable Small World) with Inner Product metric as the serving candidate, comparing it against the FAISS Flat baseline provided by Role 2.

**Alternatives Considered:**
*   *IVF (Inverted File Index):* Rejected. IVF relies on `nprobe` tuning and clustering, which can lead to unpredictable p95 tail latencies. HNSW provides a much more stable latency profile, which is critical for an interactive web demo.
*   *Compression (PQ / Float16):* Rejected. At `dim=384` and ~550k vectors, the HNSW index fits comfortably in RAM (~1.05 GB). Introducing compression would add complexity and degrade recall without solving an actual memory bottleneck.

**Engineering Highlight (Windows/Cyrillic Bug Workaround):**
During integration, a critical C++ level bug was discovered in FAISS: `faiss.write_index` and `faiss.read_index` fail with a `RuntimeError` if the target file path contains non-ASCII characters (e.g., Russian usernames in the Windows OneDrive path `C:\Users\Кирилл\...`). 
*Solution:* Implemented a safe I/O wrapper. Indices are written/read via Python's `tempfile.mkstemp()` (which guarantees an ASCII path in the OS temp directory) and then moved to the target directory using `shutil.move`. This ensures cross-platform stability without altering the project structure.

**Benchmark Results (Real Data: 553,472 vectors, dim=384):**

| Metric | FAISS Flat (Reference) | FAISS HNSW (Balanced) | FAISS HNSW (Fast) |
| :--- | :--- | :--- | :--- |
| **Build Time** | N/A (Built by Role 2) | 252.88 s | 73.62 s |
| **Load Time** | 1.94 s | 19.02 s | 9.76 s |
| **Index Size** | 810.75 MB | 954.39 MB | 886.90 MB |
| **p50 Latency** | 40.94 ms | 1.21 ms | 0.27 ms |
| **p95 Latency** | 56.76 ms | 2.10 ms | 0.42 ms |
| **Recall@10 vs Flat** | 1.0 (100%) | 0.988 (98.8%) | 0.898 (89.8%) |

*Parameters (Balanced): M=32, efConstruction=200, efSearch=128.*
*Parameters (Fast): M=16, efConstruction=100, efSearch=32.*

**Trade-off Conclusion:** We sacrifice ~240 MB of RAM and 4 minutes of offline build time to achieve a **96% reduction in p95 latency** (from 56ms down to 2.1ms). The recall loss is only 1.2%, which is completely acceptable as it will be compensated by BM25 during the Hybrid RRF stage (Role 3). The "Fast" config was rejected for production due to an unacceptable 10% drop in recall.

---

### 2. Task 4.2: Paragraph/Chunk Resolution

**Paradigm Shift:** The original plan implied a "Small-to-Big" retrieval: finding a full chapter via base search, then using local TF-IDF to find the specific paragraph. 
However, based on the actual data pipeline delivered by Role 1, the searchable object is a **Chunk**, which already represents a paragraph-level snippet. 

**Decision:** Completely removed the local TF-IDF approach. Transformed the module into a **Metadata Resolver**.
*   **Implementation:** The `resolve_fragments()` function takes a list of `chapter_id`s returned by base search and maps them to actual text strings using the `processed_chapters.jsonl` file.
*   **Performance:** The JSONL file is loaded into a Python dictionary (`O(1)` lookup) once during server initialization. This reduces the runtime resolution latency from a theoretical ~15ms (TF-IDF initialization) to **< 0.1ms** per request. No text re-parsing occurs at runtime.

---

### 3. Task 4.3: Cross-Encoder Reranking (Proof of Concept)

**Decision:** Code was implemented as a PoC (`src/rerank.py`), explicitly supporting `torch.device` selection (CUDA/CPU), but was **architecturally rejected** for the online serving path.

**Reasoning:** 
1.  **Latency:** Adding a Cross-encoder adds 20-50ms per request even for 3-5 short snippets.
2.  **Dependencies:** Requires PyTorch (`>2GB`), which unnecessarily bloats the demo environment.
3.  **Diminishing Returns:** At the chunk level, the base search has already narrowed down the candidates. Neural reranking provides marginal precision gains compared to the massive latency cost.
*Status:* Excluded from `search_engine.py`. Exists in the repository as a documented "Future Extension" for async processing.

---

### 4. Task 4.4: Serving Integration (`search_engine.py`)

**Architectural Pattern:** **Facade / Router**. The module contains zero ranking logic. It acts as a stateful router that directs queries to the correct algorithms (Role 1, 2, 3) and normalizes their outputs into a strict, unified schema (Section 2.3 of the project plan).

**Application Lifecycle Management:**
*   **Bootstrap Phase (`initialize_server_state`):** Executed once on server start. Loads the HNSW index, the JSONL chunk dictionary, and the ID mapping array into RAM. Avoids any disk I/O during user requests.
*   **Online Phase (`search`):** Pure in-memory computation.

**Integration Challenges Solved:**
1.  **Dataclass Handling (Role 2):** Role 2 returns `DenseResult` dataclasses, not dictionaries. Our facade intercepts these and converts them to standard dicts using `dataclasses.asdict()` to prevent downstream `.get()` errors.
2.  **HNSW ID Mapping:** FAISS returns raw integer indices. We load `chapter_ids.json` (generated by Role 2) into RAM and map these integers to actual string chunk IDs (e.g., `123` -> `41496-8_chunk_000001`) so the resolver (Task 4.2) can find the text.
3.  **Path Injection (Role 3):** Role 3's Hybrid RRF function assumes a default path for BM25. Our facade explicitly overrides `bm25_index_dir` to ensure it points to the correct project directory.

---

### 5. Task 4.5: Quality Gate & Abstention

**Decision:** Implemented heuristic-based quality gates rather than LLM-as-a-judge.
*   **Mechanism:** If the top result's `score_or_rank` falls below a method-specific threshold (e.g., Dense < 0.4), the system flags the entire response.
*   **Behavior:** The system does *not* hide results. It adds `low_confidence: True` and a `warning` string to the JSON payload.
*   **Ethical Control:** This adheres to the project's requirement for "abstention when evidence is weak" (Section 4.13). The user sees the provenance and the system's warning, preventing the product from overclaiming false matches. The frontend (Role 2) is expected to render a grayed-out UI state based on this flag.