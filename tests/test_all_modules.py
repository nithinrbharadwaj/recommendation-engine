"""
tests/test_all_modules.py
--------------------------
Unit tests for all four recommendation engine modules.
Run with:  python -m pytest tests/test_all_modules.py -v
Or:        python tests/test_all_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import math

from modules.similarity_calculator import SimilarityCalculator
from modules.candidate_generator   import CandidateGenerator
from modules.scorer                import Scorer
from modules.evaluator             import Evaluator
from data.sample_data              import USER_RATINGS, ITEM_METADATA, GROUND_TRUTH, ALL_ITEMS


# ── Similarity Calculator Tests ───────────────────────────────────────────────

class TestSimilarityCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = SimilarityCalculator()

    def test_cosine_identical_vectors(self):
        v = {"a": 1, "b": 2, "c": 3}
        self.assertAlmostEqual(self.calc.cosine_similarity(v, v), 1.0, places=4)

    def test_cosine_orthogonal_vectors(self):
        v1 = {"a": 1, "b": 0}
        v2 = {"a": 0, "b": 1}
        self.assertAlmostEqual(self.calc.cosine_similarity(v1, v2), 0.0, places=4)

    def test_cosine_empty_vectors(self):
        self.assertEqual(self.calc.cosine_similarity({}, {"a": 1}), 0.0)
        self.assertEqual(self.calc.cosine_similarity({}, {}), 0.0)

    def test_cosine_no_overlap(self):
        v1 = {"a": 1}
        v2 = {"b": 1}
        self.assertEqual(self.calc.cosine_similarity(v1, v2), 0.0)

    def test_cosine_partial_overlap(self):
        v1 = {"a": 3, "b": 4}
        v2 = {"a": 3, "c": 4}
        # dot = 9, |v1|=5, |v2|=5  → 9/25 = 0.36
        expected = 9 / 25
        self.assertAlmostEqual(self.calc.cosine_similarity(v1, v2), expected, places=4)

    def test_jaccard_identical_sets(self):
        s = {"a", "b", "c"}
        self.assertEqual(self.calc.jaccard_similarity(s, s), 1.0)

    def test_jaccard_disjoint_sets(self):
        self.assertEqual(self.calc.jaccard_similarity({"a"}, {"b"}), 0.0)

    def test_jaccard_both_empty(self):
        self.assertEqual(self.calc.jaccard_similarity(set(), set()), 1.0)

    def test_jaccard_one_empty(self):
        self.assertEqual(self.calc.jaccard_similarity(set(), {"a"}), 0.0)

    def test_jaccard_partial_overlap(self):
        s1 = {"a", "b", "c"}
        s2 = {"b", "c", "d"}
        # intersection=2, union=4 → 0.5
        self.assertAlmostEqual(self.calc.jaccard_similarity(s1, s2), 0.5, places=4)

    def test_most_similar_users_returns_list(self):
        result = self.calc.most_similar_users("alice", USER_RATINGS)
        self.assertIsInstance(result, list)

    def test_most_similar_users_excludes_target(self):
        result = self.calc.most_similar_users("alice", USER_RATINGS)
        users  = [u for u, _ in result]
        self.assertNotIn("alice", users)

    def test_most_similar_users_unknown_user(self):
        result = self.calc.most_similar_users("unknown", USER_RATINGS)
        self.assertEqual(result, [])

    def test_most_similar_users_sorted_descending(self):
        result = self.calc.most_similar_users("alice", USER_RATINGS, top_n=10)
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_most_similar_users_invalid_method(self):
        with self.assertRaises(ValueError):
            self.calc.most_similar_users("alice", USER_RATINGS, method="invalid")


# ── Candidate Generator Tests ─────────────────────────────────────────────────

class TestCandidateGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = CandidateGenerator(USER_RATINGS, ITEM_METADATA)

    def test_collaborative_returns_set(self):
        result = self.gen.collaborative_candidates("alice")
        self.assertIsInstance(result, set)

    def test_collaborative_excludes_already_rated(self):
        already_rated = set(USER_RATINGS["alice"].keys())
        result = self.gen.collaborative_candidates("alice")
        self.assertTrue(result.isdisjoint(already_rated))

    def test_collaborative_unknown_user(self):
        self.assertEqual(self.gen.collaborative_candidates("unknown"), set())

    def test_content_returns_set(self):
        result = self.gen.content_based_candidates("alice")
        self.assertIsInstance(result, set)

    def test_content_excludes_already_rated(self):
        already_rated = set(USER_RATINGS["alice"].keys())
        result = self.gen.content_based_candidates("alice")
        self.assertTrue(result.isdisjoint(already_rated))

    def test_content_unknown_user(self):
        self.assertEqual(self.gen.content_based_candidates("unknown"), set())

    def test_popular_returns_set(self):
        result = self.gen.popular_candidates("alice")
        self.assertIsInstance(result, set)

    def test_popular_excludes_already_rated(self):
        already_rated = set(USER_RATINGS["alice"].keys())
        result = self.gen.popular_candidates("alice")
        self.assertTrue(result.isdisjoint(already_rated))

    def test_generate_all_is_union(self):
        collab  = self.gen.collaborative_candidates("alice")
        content = self.gen.content_based_candidates("alice")
        popular = self.gen.popular_candidates("alice")
        combined = self.gen.generate("alice", strategy="all")
        self.assertEqual(combined, collab | content | popular)

    def test_generate_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.gen.generate("alice", strategy="magic")

    def test_generate_no_metadata(self):
        gen = CandidateGenerator(USER_RATINGS)   # No metadata
        result = gen.generate("alice", strategy="content")
        self.assertIsInstance(result, set)


# ── Scorer Tests ──────────────────────────────────────────────────────────────

class TestScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = Scorer(USER_RATINGS, ITEM_METADATA)

    def test_score_item_returns_float(self):
        score = self.scorer.score_item("alice", "monitor")
        self.assertIsInstance(score, float)

    def test_score_item_in_range(self):
        score = self.scorer.score_item("alice", "monitor")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_unknown_item(self):
        score = self.scorer.score_item("alice", "nonexistent_item")
        self.assertIsInstance(score, float)

    def test_rank_candidates_returns_list(self):
        candidates = {"monitor", "webcam", "speaker"}
        result = self.scorer.rank_candidates("alice", candidates, top_n=3)
        self.assertIsInstance(result, list)

    def test_rank_candidates_sorted_descending(self):
        candidates = {"monitor", "webcam", "speaker", "usb_hub"}
        result = self.scorer.rank_candidates("alice", candidates, top_n=4)
        scores = [s for _, s in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_rank_candidates_respects_top_n(self):
        candidates = {"monitor", "webcam", "speaker", "usb_hub", "led_strip"}
        result = self.scorer.rank_candidates("alice", candidates, top_n=3)
        self.assertLessEqual(len(result), 3)

    def test_rank_candidates_empty(self):
        result = self.scorer.rank_candidates("alice", set(), top_n=5)
        self.assertEqual(result, [])

    def test_explain_score_keys(self):
        breakdown = self.scorer.explain_score("alice", "monitor")
        for key in ("collaborative", "content", "popularity", "final"):
            self.assertIn(key, breakdown)

    def test_explain_score_components_sum_to_final(self):
        bd = self.scorer.explain_score("alice", "monitor")
        w  = self.scorer.weights
        expected = round(
            w["collaborative"] * bd["collaborative"]
            + w["content"]      * bd["content"]
            + w["popularity"]   * bd["popularity"],
            4,
        )
        self.assertAlmostEqual(bd["final"], expected, places=3)


# ── Evaluator Tests ───────────────────────────────────────────────────────────

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()
        self.recs      = ["item1", "item2", "item3", "item4", "item5"]
        self.relevant  = {"item1", "item3", "item6"}

    def test_precision_at_k_perfect(self):
        recs = ["item1", "item3"]
        rel  = {"item1", "item3"}
        self.assertAlmostEqual(self.evaluator.precision_at_k(recs, rel, k=2), 1.0)

    def test_precision_at_k_zero(self):
        recs = ["item99", "item98"]
        rel  = {"item1", "item2"}
        self.assertAlmostEqual(self.evaluator.precision_at_k(recs, rel, k=2), 0.0)

    def test_precision_at_k_partial(self):
        p = self.evaluator.precision_at_k(self.recs, self.relevant, k=5)
        self.assertAlmostEqual(p, 2 / 5)

    def test_precision_empty_recommended(self):
        self.assertEqual(self.evaluator.precision_at_k([], self.relevant, k=5), 0.0)

    def test_precision_k_zero(self):
        self.assertEqual(self.evaluator.precision_at_k(self.recs, self.relevant, k=0), 0.0)

    def test_recall_at_k_perfect(self):
        recs = ["item1", "item3", "item6"]
        rel  = {"item1", "item3", "item6"}
        self.assertAlmostEqual(self.evaluator.recall_at_k(recs, rel, k=3), 1.0)

    def test_recall_at_k_zero(self):
        self.assertAlmostEqual(
            self.evaluator.recall_at_k(["item99"], {"item1"}, k=1), 0.0
        )

    def test_recall_empty_relevant(self):
        self.assertEqual(self.evaluator.recall_at_k(self.recs, set(), k=5), 0.0)

    def test_f1_at_k(self):
        p = self.evaluator.precision_at_k(self.recs, self.relevant, k=5)
        r = self.evaluator.recall_at_k(self.recs, self.relevant, k=5)
        f = self.evaluator.f1_at_k(self.recs, self.relevant, k=5)
        expected = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        self.assertAlmostEqual(f, expected, places=4)

    def test_f1_zero_when_no_hits(self):
        self.assertEqual(
            self.evaluator.f1_at_k(["item99"], {"item1"}, k=1), 0.0
        )

    def test_hit_rate_hit(self):
        self.assertEqual(
            self.evaluator.hit_rate_at_k(["item1", "item99"], {"item1"}, k=2), 1.0
        )

    def test_hit_rate_miss(self):
        self.assertEqual(
            self.evaluator.hit_rate_at_k(["item99", "item98"], {"item1"}, k=2), 0.0
        )

    def test_coverage_full(self):
        all_items = {"a", "b", "c"}
        all_recs  = [["a", "b"], ["b", "c"]]
        self.assertAlmostEqual(self.evaluator.coverage(all_recs, all_items), 1.0)

    def test_coverage_partial(self):
        all_items = {"a", "b", "c", "d"}
        all_recs  = [["a", "b"]]
        self.assertAlmostEqual(self.evaluator.coverage(all_recs, all_items), 0.5)

    def test_coverage_empty_catalogue(self):
        self.assertEqual(self.evaluator.coverage([["a"]], set()), 0.0)

    def test_evaluate_all_returns_dict(self):
        user_results = [
            ("u1", ["item1", "item2"], {"item1"}),
            ("u2", ["item3", "item4"], {"item3", "item4"}),
        ]
        result = self.evaluator.evaluate_all(user_results, {"item1","item2","item3","item4"}, k=2)
        self.assertIsInstance(result, dict)
        for key in ("precision@k", "recall@k", "f1@k", "hit_rate@k", "coverage"):
            self.assertIn(key, result)

    def test_evaluate_all_empty(self):
        result = self.evaluator.evaluate_all([], ALL_ITEMS, k=5)
        self.assertEqual(result["precision@k"], 0.0)
        self.assertEqual(result["num_users"], 0)


# ── Integration Test ──────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    def test_full_pipeline(self):
        """Run the complete pipeline end-to-end."""
        from recommendation_engine import RecommendationEngine

        engine = RecommendationEngine(USER_RATINGS, ITEM_METADATA)
        recs   = engine.recommend("alice", top_n=5)

        self.assertIsInstance(recs, list)
        self.assertLessEqual(len(recs), 5)

        if recs:
            item, score = recs[0]
            self.assertIsInstance(item, str)
            self.assertIsInstance(score, float)

        # Scores should be descending
        scores = [s for _, s in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_evaluation_pipeline(self):
        from recommendation_engine import RecommendationEngine

        engine  = RecommendationEngine(USER_RATINGS, ITEM_METADATA)
        metrics = engine.evaluate(GROUND_TRUTH, top_n=5, k=5)

        self.assertIn("precision@k", metrics)
        self.assertGreaterEqual(metrics["precision@k"], 0.0)
        self.assertLessEqual(metrics["precision@k"], 1.0)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
