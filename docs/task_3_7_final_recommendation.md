# Task 3.7 Final Experiment Recommendation

## Data Used

- Evaluation set: 48 queries, 2304 usable relevance labels, 7 missing candidate ids skipped.
- Compared methods: BM25, Dense FAISS Flat, Dense FAISS ANN, Hybrid RRF.
- Run depth: 20 candidates per method per query in the saved run files; metrics reported at P@5, R@10, MRR@10, and nDCG@10.
- Label distribution: 0=184, 1=1966, 2=154; confidence distribution: low=960, medium=957, high=387.

## Final Experiment Table

| Method | P@5 | R@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Index size | Build/load note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.9250 | 0.2100 | 0.9375 | 0.6591 | 15.907 | 34.051 | 1346.60 MiB | 553472 indexed objects |
| Dense FAISS Flat | 0.9250 | 0.2100 | 0.9583 | 0.6423 | 82.451 | 89.346 | 810.75 MB | built by Role 2; embedding 7817.79 s |
| Dense FAISS ANN | 0.9208 | 0.2141 | 0.9583 | 0.6365 | 17.326 | 20.697 | 954.39 MB; RAM est. 1049.83 MB | 252.88 s; load 19.02 s |
| Hybrid RRF | 0.9250 | 0.2147 | 0.9792 | 0.6753 | 89.742 | 110.120 | BM25 + dense index | derived from existing BM25 and dense indexes |

## Quality-Efficiency Recommendation

**Recommendation:** use Hybrid RRF as the default ranking method for the final demo and offline evaluation, and keep Dense FAISS ANN as the low-latency serving fallback.

Hybrid RRF has the best aggregate quality: nDCG@10=0.6753 and MRR@10=0.9792. It also has the best Recall@10 among the tested methods (0.2147), while matching the best Precision@5 (0.9250). The cost is latency: p50=89.742 ms and p95=110.120 ms because it runs both lexical and dense retrieval before fusion.

Dense FAISS ANN is the best speed-quality trade-off when the online latency budget is strict: p50=17.326 ms and p95=20.697 ms, with nDCG@10=0.6365. It is much faster than Dense FAISS Flat and close to BM25 latency, but loses about 0.0388 nDCG@10 versus Hybrid RRF.

BM25 remains a strong lexical baseline: p50=15.907 ms, p95=34.051 ms, and nDCG@10=0.6591. It is especially useful for exact terms and as a fallback, but it does not beat Hybrid RRF overall. Dense FAISS Flat should not be the serving default because it is slower than ANN with no measured quality advantage.

## Error-Analysis Summary

Hybrid RRF by query category:

| Category | Queries | P@5 | R@10 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- | --- |
| action_situation | 8 | 0.9250 | 0.2071 | 1.0000 | 0.6576 |
| ambiguous_weak_evidence | 8 | 0.9250 | 0.2201 | 1.0000 | 0.6152 |
| atmosphere | 8 | 0.9000 | 0.2176 | 1.0000 | 0.7653 |
| emotion_mood | 8 | 1.0000 | 0.2147 | 1.0000 | 0.6960 |
| exact_keyword | 8 | 0.8750 | 0.1985 | 0.8750 | 0.6983 |
| semantic_scene | 8 | 0.9250 | 0.2301 | 1.0000 | 0.6194 |

Representative low-quality or diagnostic cases from Task 3.6:

| Query | Category | Failure type | Best method | Hybrid nDCG@10 | Top label | Query text |
| --- | --- | --- | --- | --- | --- | --- |
| SS05 | semantic_scene | hybrid_low_absolute_quality | Dense FAISS ANN | 0.3902 | 1 | a young person leaves home and learns to make independent decisions |
| SS01 | semantic_scene | hybrid_low_absolute_quality | BM25 | 0.4074 | 1 | rivals become allies |
| AW08 | ambiguous_weak_evidence | hybrid_low_absolute_quality | Hybrid RRF | 0.4259 | 1 | a character experiences a scene that could be a memory, a dream, a lie, or several of these at the same time |
| AW01 | ambiguous_weak_evidence | hybrid_low_absolute_quality | Dense FAISS ANN | 0.4300 | 1 | something feels wrong |
| EK03 | exact_keyword | hybrid_low_absolute_quality | BM25 | 0.4402 | 1 | a masked ball with a broken chandelier |
| AT01 | atmosphere | hybrid_low_absolute_quality | Dense FAISS Flat | 0.4663 | 1 | haunted castle at night |

The main failure modes are:

- **Ambiguous or weak-evidence queries:** broad prompts such as "something feels wrong" often retrieve partially relevant passages but not a clearly defensible label-2 result.
- **Semantic scene drift:** dense retrieval can match tone or topic while missing the concrete event requested by the query.
- **Literal trap:** BM25 can over-rank exact word overlap, for example terms such as "chandelier" without the full scene.
- **Fusion dilution:** Hybrid RRF is robust overall, but if one source ranks weak partial matches very highly, fusion can still leave the best evidence below the top positions.

## Slide-Ready Result

**One result:** Hybrid RRF is the best measured ranking approach: nDCG@10=0.6753, MRR@10=0.9792, p95=110.120 ms.

**One decision explanation:** Hybrid RRF wins because lexical retrieval catches exact named/keyword evidence while dense retrieval adds paraphrase and atmosphere matches; Reciprocal Rank Fusion keeps candidates that are supported by either signal near the top without training a new model.

**One limitation:** the best-quality method is also the slowest tested method, and weak ambiguous queries still need a quality gate or refinement before the result can be presented as confident evidence.

## Acceptance Check

- Final experiment table: included above with quality, latency, index size, and build/load notes for all four methods.
- Error-analysis summary: included from Task 3.6 category metrics and failure cases.
- Quality-efficiency recommendation: explicitly based on Task 3.5 metrics and Task 3.6 failures.
- Defense answer: Hybrid RRF works best overall; it fails on ambiguous/weak-evidence and scene-drift cases; its trade-off is higher latency for better ranking quality.
