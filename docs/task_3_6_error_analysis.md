# Task 3.6 Error Analysis

## Setup

This analysis uses the same LLM-assisted relevance judgments, qrels, and run files produced for Tasks 3.4 and 3.5. Relevance is graded on the 0/1/2 scale, and nDCG@10 is the primary diagnostic metric because it accounts for both rank position and label strength.

## Hybrid RRF By Query Category

| category | queries | Precision@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| action_situation | 8 | 0.9250 | 0.2071 | 1.0000 | 0.6576 |
| ambiguous_weak_evidence | 8 | 0.9250 | 0.2201 | 1.0000 | 0.6152 |
| atmosphere | 8 | 0.9000 | 0.2176 | 1.0000 | 0.7653 |
| emotion_mood | 8 | 1.0000 | 0.2147 | 1.0000 | 0.6960 |
| exact_keyword | 8 | 0.8750 | 0.1985 | 0.8750 | 0.6983 |
| semantic_scene | 8 | 0.9250 | 0.2301 | 1.0000 | 0.6194 |

Hybrid RRF is strongest when lexical and dense evidence complement each other, especially when one method retrieves a relevant candidate early and the other method adds semantically related alternatives. Recall@10 remains numerically low because the pooled qrels often contain many candidates with non-zero labels for a query, while each run contributes only ten results to Recall@10.

## Strongest Hybrid Queries

| query_id | category | nDCG@10 | query |
| --- | ---: | ---: | ---: |
| AS08 | action_situation | 1.0000 | two rival political groups meet in secret and decide to work together against a threat that neither side can defeat alone |
| AT05 | atmosphere | 1.0000 | a warm kitchen on a winter morning that feels safe |
| AT06 | atmosphere | 1.0000 | a quiet frontier town where everyone expects trouble |
| EM08 | emotion_mood | 1.0000 | after being ignored and underestimated for years, a character develops a quiet resentment that they hide behind patient and respectful behavior |
| SS04 | semantic_scene | 1.0000 | an investigator starts doubting a witness they used to trust |

These queries usually have a clear signal that at least one source ranks early, and RRF keeps that signal near the top.

## Weakest Hybrid Queries

| query_id | category | nDCG@10 | top_result_label | query |
| --- | ---: | ---: | ---: | ---: |
| SS05 | semantic_scene | 0.3902 | 1 | a young person leaves home and learns to make independent decisions |
| SS01 | semantic_scene | 0.4074 | 1 | rivals become allies |
| AW08 | ambiguous_weak_evidence | 0.4259 | 1 | a character experiences a scene that could be a memory, a dream, a lie, or several of these at the same time |
| AW01 | ambiguous_weak_evidence | 0.4300 | 1 | something feels wrong |
| EK03 | exact_keyword | 0.4402 | 1 | a masked ball with a broken chandelier |

The weakest cases tend to be broad, ambiguous, or scene-like queries where many judged candidates are only partially relevant. In those cases, RRF can still retrieve non-zero relevance, but it may miss the few label-2 passages or rank them below several partial matches.

## Representative Failure Cases

| query_id | failure_type | best_method_display | focus_method_display | bm25_nDCG@10 | dense_flat_nDCG@10 | hybrid_rrf_nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AS01 | hybrid_low_absolute_quality | BM25 | Hybrid RRF | 0.6956 | 0.4271 | 0.5144 |
| AS03 | hybrid_low_absolute_quality | Hybrid RRF | Hybrid RRF | 0.5091 | 0.4700 | 0.5323 |
| AS05 | hybrid_low_absolute_quality | BM25 | Hybrid RRF | 0.6769 | 0.3602 | 0.4978 |
| AS06 | hybrid_low_absolute_quality | Hybrid RRF | Hybrid RRF | 0.4609 | 0.4540 | 0.5013 |
| AS07 | dense_lags_lexical_baseline | BM25 | Dense FAISS Flat | 0.8126 | 0.6484 | 0.6944 |
| AT01 | hybrid_low_absolute_quality | Dense FAISS Flat | Hybrid RRF | 0.4843 | 0.6779 | 0.4663 |
| AT02 | hybrid_low_absolute_quality | Dense FAISS ANN | Hybrid RRF | 0.4556 | 0.6592 | 0.5325 |
| AT07 | dense_lags_lexical_baseline | Hybrid RRF | Dense FAISS Flat | 0.7129 | 0.4700 | 0.7392 |

The failure cases show three main patterns:

1. **Lexical-only weakness:** BM25 can over-prioritize shared words and miss paraphrased or atmospheric relevance.
2. **Dense-only weakness:** dense retrieval can retrieve semantically adjacent passages that match tone or topic but not the concrete requested situation.
3. **Fusion dilution:** Hybrid RRF is robust overall, but if one source strongly ranks weak partial matches, fusion may not fully recover the best graded candidate.

## Implications For The Next Iteration

- Keep Hybrid RRF as the default Role 3 method because it has the best aggregate nDCG@10 and MRR@10 in Task 3.5.
- Add a small calibration layer for query categories: exact-keyword queries can lean more on BM25, while ambiguous/atmospheric queries may benefit from stronger dense weighting or a second-stage rerank.
- Consider extending the dense representation beyond `title_first_middle_last` if failures show relevant scenes buried away from the sampled chapter positions.
- Keep cross-encoder reranking outside the main benchmark for now, but use it as an offline diagnostic for the worst Hybrid RRF cases if time allows.

## Limitation

The labels are automated relevance annotations, so individual examples may contain annotation noise. The analysis is therefore most useful for relative method comparison and failure-mode discovery, not as a perfect judgment of any single passage.
