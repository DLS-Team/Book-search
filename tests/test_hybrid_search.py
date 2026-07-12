from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import hybrid_search


@dataclass
class DenseStub:
    chapter_id: str | int
    score: float
    rank: int = 999  # Input rank is intentionally ignored; list order wins.


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_formula_overlap_metadata_and_diagnostics(self) -> None:
        bm25 = [
            {"chapter_id": "shared", "score": 1000.0, "title": "Book A", "author": "Author A"},
            {"chapter_id": "bm-only", "score": 500.0, "title": "Book B"},
        ]
        dense = [DenseStub("dense-only", 0.99), DenseStub("shared", 0.01)]

        results = hybrid_search.reciprocal_rank_fusion(bm25, dense, top_k=3, rrf_k=60)
        by_id = {result["chapter_id"]: result for result in results}

        self.assertEqual(results[0]["chapter_id"], "shared")
        self.assertAlmostEqual(by_id["shared"]["score"], 1 / 61 + 1 / 62)
        self.assertEqual(by_id["shared"]["bm25_rank"], 1)
        self.assertEqual(by_id["shared"]["dense_rank"], 2)
        self.assertEqual(by_id["shared"]["bm25_score"], 1000.0)
        self.assertEqual(by_id["shared"]["dense_score"], 0.01)
        self.assertEqual(by_id["shared"]["retrieval_sources"], ["bm25", "dense"])
        self.assertEqual(by_id["shared"]["title"], "Book A")
        self.assertEqual(by_id["shared"]["author"], "Author A")
        self.assertEqual([result["rank"] for result in results], [1, 2, 3])

    def test_raw_scores_do_not_change_ranking(self) -> None:
        first = hybrid_search.reciprocal_rank_fusion(
            [{"chapter_id": "a", "score": 1e9}, {"chapter_id": "b", "score": -1e9}],
            [],
            top_k=2,
        )
        second = hybrid_search.reciprocal_rank_fusion(
            [{"chapter_id": "a", "score": -1e9}, {"chapter_id": "b", "score": 1e9}],
            [],
            top_k=2,
        )

        self.assertEqual([item["chapter_id"] for item in first], ["a", "b"])
        self.assertEqual(
            [item["chapter_id"] for item in first],
            [item["chapter_id"] for item in second],
        )

    def test_deduplicates_within_source_and_canonicalizes_ids(self) -> None:
        results = hybrid_search.reciprocal_rank_fusion(
            [
                {"chapter_id": 7, "score": 4.0},
                {"chapter_id": "7", "score": 3.0},
                {"chapter_id": None, "score": 2.0},
                {"chapter_id": "", "score": 1.0},
            ],
            [DenseStub("7", 0.8)],
            top_k=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chapter_id"], "7")
        self.assertAlmostEqual(results[0]["score"], 2 / 61)
        self.assertEqual(results[0]["bm25_rank"], 1)

    def test_supports_single_source_results_and_deterministic_ties(self) -> None:
        results = hybrid_search.reciprocal_rank_fusion(
            [{"chapter_id": "z", "score": 5.0}],
            [DenseStub("a", 0.5)],
            top_k=1,
        )

        # Same RRF score, source count, and best rank: chapter_id breaks the tie.
        self.assertEqual(results[0]["chapter_id"], "a")
        self.assertIsNone(results[0]["bm25_rank"])
        self.assertEqual(results[0]["dense_rank"], 1)

    def test_rejects_invalid_fusion_arguments(self) -> None:
        for invalid_top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=invalid_top_k):
                with self.assertRaises(ValueError):
                    hybrid_search.reciprocal_rank_fusion([], [], invalid_top_k)

        for invalid_rrf_k in (0, -1, True, "60", float("nan"), float("inf")):
            with self.subTest(rrf_k=invalid_rrf_k):
                with self.assertRaises(ValueError):
                    hybrid_search.reciprocal_rank_fusion([], [], 1, rrf_k=invalid_rrf_k)


class HybridSearchWrapperTests(unittest.TestCase):
    def test_calls_both_retrievers_at_configured_depth(self) -> None:
        bm25_output = [{"chapter_id": "a", "score": 8.0, "title": "A"}]
        dense_output = [DenseStub("a", 0.8)]

        with (
            patch.object(hybrid_search, "_run_bm25_search", return_value=bm25_output) as bm25_search,
            patch.object(hybrid_search, "_run_dense_search", return_value=dense_output) as dense_search,
        ):
            results = hybrid_search.search_hybrid_rrf("winter fire", top_k=2, candidate_k=12)

        bm25_search.assert_called_once_with(
            hybrid_search.DEFAULT_BM25_INDEX_DIR,
            "winter fire",
            12,
        )
        dense_search.assert_called_once_with("winter fire", 12)
        self.assertEqual(results[0]["chapter_id"], "a")
        self.assertIn("score", results[0])

    def test_default_candidate_depth_is_five_times_top_k(self) -> None:
        with (
            patch.object(hybrid_search, "_run_bm25_search", return_value=[]) as bm25_search,
            patch.object(hybrid_search, "_run_dense_search", return_value=[]) as dense_search,
        ):
            self.assertEqual(hybrid_search.search_hybrid_rrf("query", top_k=3), [])

        bm25_search.assert_called_once_with(hybrid_search.DEFAULT_BM25_INDEX_DIR, "query", 15)
        dense_search.assert_called_once_with("query", 15)

    def test_rejects_invalid_wrapper_arguments_before_search(self) -> None:
        invalid_calls = [
            {"query": "", "top_k": 5},
            {"query": "   ", "top_k": 5},
            {"query": None, "top_k": 5},
            {"query": "valid", "top_k": 0},
            {"query": "valid", "top_k": 5, "candidate_k": 0},
            {"query": "valid", "top_k": 5, "rrf_k": 0},
        ]
        for kwargs in invalid_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    hybrid_search.search_hybrid_rrf(**kwargs)

    def test_retriever_errors_are_not_swallowed(self) -> None:
        with patch.object(
            hybrid_search,
            "_run_bm25_search",
            side_effect=FileNotFoundError("missing index"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "missing index"):
                hybrid_search.search_hybrid_rrf("query")


if __name__ == "__main__":
    unittest.main()
