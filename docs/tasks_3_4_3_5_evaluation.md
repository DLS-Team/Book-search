# Tasks 3.4 and 3.5 Evaluation

## Evaluation Setup

For evaluation, we used LLM-assisted relevance judgments over pooled candidate results. The same annotations were used for all retrieval methods, which allows a consistent relative comparison among BM25, Dense FAISS Flat, Dense FAISS ANN, Hybrid RRF.

The labeled file contains 2304 usable query-candidate judgments across 48 queries. Labels are interpreted as graded relevance scores: 0 = not relevant, 1 = partially relevant, and 2 = relevant. 7 pairs with missing labels and 0 invalid rows were skipped when creating qrels.

The ranked runs were generated from `data/eval/retrieval_manifest.json`, which stores the controlled retrieval outputs built for the pooled labeling set. The manifest configuration was `top_k=20`, `candidate_k=100`, and `rrf_k=60.0`. Raw retrieval scores were not persisted in the manifest by design, so the exported run CSV files include deterministic inverse-rank placeholder scores; metric computation uses the rank column.

## Retrieval Runs

Task 3.4 is represented by run files in `runs/` for the following methods:

- BM25 lexical retrieval
- Dense FAISS Flat retrieval
- Dense FAISS ANN retrieval
- Hybrid RRF over BM25 and dense retrieval

The cross-encoder reranker is not included in the main benchmark. The project already records this decision in `experiments/rerank_decision.md`: reranking is kept as a proof of concept but excluded from the controlled comparison because of input-length, dependency, and latency trade-offs.

## Metrics

| Method | Precision@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.9250 | 0.2100 | 0.9375 | 0.6591 |
| Dense FAISS Flat | 0.9250 | 0.2100 | 0.9583 | 0.6423 |
| Dense FAISS ANN | 0.9208 | 0.2141 | 0.9583 | 0.6365 |
| Hybrid RRF | 0.9250 | 0.2147 | 0.9792 | 0.6753 |

The strongest method by nDCG@10 is **Hybrid RRF** with nDCG@10=0.6753. Precision@5 measures how many of the first five returned chunks have non-zero relevance, Recall@10 measures coverage of all judged relevant chunks within the first ten results, MRR@10 rewards retrieving any relevant result early, and nDCG@10 uses the graded 0/1/2 labels.

## Engineering Metrics

| Method | Latency/query | Index size | Build time |
| --- | ---: | ---: | ---: |
| BM25 | p50 15.907 ms; p95 34.051 ms | 1346.60 MiB | 553472 indexed objects |
| Dense FAISS Flat | p50 82.451 ms; p95 89.346 ms | 810.75 MB | built by Role 2; embedding 7817.79 s |
| Dense FAISS ANN | p50 17.326 ms; p95 20.697 ms | 954.39 MB; RAM est. 1049.83 MB | 252.88 s; load 19.02 s |
| Hybrid RRF | p50 89.742 ms; p95 110.120 ms | BM25 + dense index | derived from existing BM25 and dense indexes |

The p50/p95 latency values above were recomputed live by this evaluation script over the same 48 evaluation queries, with indexes and the embedding model loaded once before timing. This measures steady-state online retrieval latency rather than cold-start loading time.

Detailed latency summary:

| Method | Queries | p50_ms | p95_ms | mean_ms |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 48 | 15.907 | 34.051 | 18.207 |
| Dense FAISS Flat | 48 | 82.451 | 89.346 | 81.285 |
| Dense FAISS ANN | 48 | 17.326 | 20.697 | 17.801 |
| Hybrid RRF | 48 | 89.742 | 110.120 | 92.120 |

Dense index size, load time, build time, and Recall@10-vs-Flat values also come from the Role 4 ANN benchmark on 553,472 vectors (`docs/role4_report.md` and `experiments/ann_architecture_decisions.md`). Those Role 4 latency numbers isolate FAISS index search; the live p50/p95 table above includes per-query embedding for dense methods, so it is closer to end-to-end retriever latency.

Role 4 also reports that the selected HNSW configuration preserves 98.8% Recall@10 relative to FAISS Flat while reducing p95 dense-search latency from 56.76 ms to 2.10 ms. The faster HNSW variant reaches 0.42 ms p95 but was rejected because Recall@10 falls to 89.8%. Cross-encoder reranking is documented as adding 20-50 ms per request and requiring a PyTorch dependency over 2 GB, so it remains excluded from the main Task 3.4/3.5 comparison.

## Example Query Difference

- Query: `AW04`
- BM25: nDCG@10=0.4540, top result `58534-8_chunk_000254`
- Dense FAISS ANN: nDCG@10=0.9665, top result `37179-8_chunk_000036`
- Dense FAISS Flat: nDCG@10=0.9375, top result `37179-8_chunk_000036`
- Hybrid RRF: nDCG@10=0.7538, top result `37179-8_chunk_000010`

This query shows why the methods should be compared with graded relevance instead of only exact keyword overlap: different rankers often retrieve different top chunks, and nDCG@10 captures both relevance strength and rank position.

## Limitation

Since the relevance annotations were produced automatically under time constraints, some annotation noise may remain. Therefore, the evaluation is used mainly for relative comparison between retrieval methods rather than as a perfect absolute measurement of search quality.
