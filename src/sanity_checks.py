"""
Task 2.3 — sanity check: normalized (cosine-style) vs unnormalized (raw inner
product) dense search, on a small set of queries (section 4.6, "How we
evaluate it").

Task 2.1 — qualitative comparison of the two representation strategies
(first_n_tokens vs title_first_middle_last), used as evidence in the 2.6
report and to harvest representation-related failure cases for the shared
pool (section 6.1).

Both checks run on the demo corpus (src/data_loader.py) so they can be
executed without the real >=500k chapter corpus from Role 1.
"""

from __future__ import annotations

import json

import numpy as np

from data_loader import load_sample_chapters
from embed_chapters import load_encoder, l2_normalize
from representation import represent_first_n_tokens, represent_title_first_middle_last


def normalization_sanity_check(queries):
    chapters = load_sample_chapters()
    encoder = load_encoder()

    proxies = [represent_title_first_middle_last(c) for c in chapters]
    texts = [p.proxy_text for p in proxies]
    chapter_ids = [p.chapter_id for p in proxies]

    raw_doc_vecs = np.asarray(encoder.encode(texts), dtype=np.float32)
    norm_doc_vecs = l2_normalize(raw_doc_vecs)

    report = []
    for q in queries:
        raw_q = np.asarray(encoder.encode([q]), dtype=np.float32)[0]
        norm_q = l2_normalize(raw_q.reshape(1, -1))[0]

        raw_scores = raw_doc_vecs @ raw_q
        norm_scores = norm_doc_vecs @ norm_q

        raw_order = [chapter_ids[i] for i in np.argsort(-raw_scores)]
        norm_order = [chapter_ids[i] for i in np.argsort(-norm_scores)]

        report.append(
            {
                "query": q,
                "raw_inner_product_ranking": raw_order,
                "normalized_cosine_ranking": norm_order,
                "rankings_match": raw_order == norm_order,
            }
        )
    return report


def representation_strategy_comparison(queries):
    chapters = load_sample_chapters()
    encoder = load_encoder()

    strat_a = [represent_first_n_tokens(c, n_tokens=40) for c in chapters]  # deliberately short
    strat_b = [represent_title_first_middle_last(c) for c in chapters]

    def rank(proxies, query):
        texts = [p.proxy_text for p in proxies]
        ids = [p.chapter_id for p in proxies]
        doc_vecs = l2_normalize(np.asarray(encoder.encode(texts), dtype=np.float32))
        q_vec = l2_normalize(np.asarray(encoder.encode([query]), dtype=np.float32))[0]
        scores = doc_vecs @ q_vec
        order = [ids[i] for i in np.argsort(-scores)]
        return order

    report = []
    for q in queries:
        order_a = rank(strat_a, q)
        order_b = rank(strat_b, q)
        report.append(
            {
                "query": q,
                "first_n_tokens_top1": order_a[0],
                "title_first_middle_last_top1": order_b[0],
                "strategies_agree_top1": order_a[0] == order_b[0],
            }
        )
    return report


if __name__ == "__main__":
    demo_queries = [
        "cozy winter night near a fireplace",
        "a child is afraid but trying to be brave",
        "tense conversation before a murder",
    ]

    print("=== 2.3 Normalization sanity check ===")
    norm_report = normalization_sanity_check(demo_queries)
    print(json.dumps(norm_report, indent=2))

    print("\n=== 2.1 Representation strategy comparison ===")
    rep_report = representation_strategy_comparison(demo_queries)
    print(json.dumps(rep_report, indent=2))

    with open("experiments_role2_sanity.json", "w", encoding="utf-8") as f:
        json.dump({"normalization_check": norm_report, "representation_comparison": rep_report}, f, indent=2)
