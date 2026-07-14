# Role 3 Slide: Ranking Experiments

## Final Result

Hybrid RRF is the best measured ranking method: nDCG@10=0.6753, MRR@10=0.9792, P@5=0.9250, p95=110.120 ms.

| Method | nDCG@10 | MRR@10 | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.6591 | 0.9375 | 15.907 | 34.051 |
| Dense ANN | 0.6365 | 0.9583 | 17.326 | 20.697 |
| Hybrid RRF | 0.6753 | 0.9792 | 89.742 | 110.120 |

## Decision

Use Hybrid RRF for the final demo and offline ranking story. It combines exact lexical matches with dense semantic matches and gives the best measured quality across the controlled query set.

Use Dense FAISS ANN when serving latency is the priority: it is close to BM25 speed and much faster than Hybrid RRF, with a moderate quality drop.

## Limitation

Hybrid RRF is slower because it runs both retrieval paths, and ambiguous weak-evidence queries still need quality gating/refinement before we should present a result as confident.
