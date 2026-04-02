"""
test.py — Comprehensive tests for recommendation engine modules
================================================================
Tests for all four components: similarity, candidate_gen, scorer, evaluator

Run with: python test.py
"""

import sys
from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

USER_RATINGS = {
    "alice": {"item1": 5, "item2": 3, "item3": 4, "item4": 2},
    "bob":   {"item1": 4, "item2": 3, "item3": 5, "item5": 4},
    "carol": {"item1": 1, "item2": 5, "item3": 2, "item6": 4},
    "dave":  {"item2": 4, "item3": 4, "item4": 3, "item5": 5},
}

ITEM_METADATA = {
    "item1": {"genre": 0.8, "action": 0.9, "drama": 0.2},
    "item2": {"genre": 0.5, "action": 0.3, "drama": 0.8},
    "item3": {"genre": 0.7, "action": 0.8, "drama": 0.5},
    "item4": {"genre": 0.6, "action": 0.7, "drama": 0.4},
    "item5": {"genre": 0.9, "action": 0.95, "drama": 0.1},
    "item6": {"genre": 0.4, "action": 0.2, "drama": 0.9},
}

GROUND_TRUTH = {
    "alice": {"item3", "item5"},
    "bob":   {"item1", "item4"},
    "carol": {"item2", "item5"},
    "dave":  {"item1", "item6"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: SIMILARITY CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def test_similarity_calculator():
    """Test Module 1: Similarity Calculator"""
    print("\n" + "=" * 70)
    print("TEST 1: SIMILARITY CALCULATOR")
    print("=" * 70)

    calc = SimilarityCalculator()
    tests_passed = 0
    tests_total = 0

    # Test 1.1: Cosine similarity
    print("\n[1.1] Testing cosine_similarity()")
    v1 = {"a": 1, "b": 2}
    v2 = {"a": 1, "b": 2}
    sim = calc.cosine_similarity(v1, v2)
    tests_total += 1
    if abs(sim - 1.0) < 0.001:  # Should be identical (1.0)
        print("  ✅ Identical vectors: PASS")
        tests_passed += 1
    else:
        print(f"  ❌ Identical vectors: FAIL (got {sim}, expected 1.0)")

    # Test 1.2: Orthogonal vectors
    print("\n[1.2] Testing orthogonal vectors")
    v1 = {"a": 1, "b": 0}
    v2 = {"a": 0, "b": 1}
    sim = calc.cosine_similarity(v1, v2)
    tests_total += 1
    if abs(sim - 0.0) < 0.001:
        print("  ✅ Orthogonal vectors: PASS")
        tests_passed += 1
    else:
        print(f"  ❌ Orthogonal vectors: FAIL (got {sim}, expected 0.0)")

    # Test 1.3: Jaccard similarity
    print("\n[1.3] Testing jaccard_similarity()")
    s1 = {"a", "b", "c"}
    s2 = {"b", "c", "d"}
    sim = calc.jaccard_similarity(s1, s2)
    expected = 2 / 4  # intersection=2, union=4
    tests_total += 1
    if abs(sim - expected) < 0.001:
        print(f"  ✅ Jaccard overlap: PASS (got {sim})")
        tests_passed += 1
    else:
        print(f"  ❌ Jaccard overlap: FAIL (got {sim}, expected {expected})")

    # Test 1.4: Pearson correlation
    print("\n[1.4] Testing pearson_correlation()")
    r1 = [1, 2, 3, 4, 5]
    r2 = [2, 4, 6, 8, 10]  # Perfect positive correlation
    corr = calc.pearson_correlation(r1, r2)
    tests_total += 1
    if corr > 0.99:  # Very close to 1.0
        print(f"  ✅ Perfect correlation: PASS (got {corr})")
        tests_passed += 1
    else:
        print(f"  ❌ Perfect correlation: FAIL (got {corr})")

    # Test 1.5: Most similar users
    print("\n[1.5] Testing most_similar_users()")
    similar = calc.most_similar_users("alice", USER_RATINGS, top_n=2)
    tests_total += 1
    if isinstance(similar, list) and len(similar) <= 2:
        print(f"  ✅ Similar users: PASS (found {len(similar)} neighbors)")
        tests_passed += 1
    else:
        print(f"  ❌ Similar users: FAIL (got {similar})")

    print(f"\n→ Similarity Calculator: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: CANDIDATE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def test_candidate_generator():
    """Test Module 2: Candidate Generator"""
    print("\n" + "=" * 70)
    print("TEST 2: CANDIDATE GENERATOR")
    print("=" * 70)

    gen = CandidateGenerator(USER_RATINGS, ITEM_METADATA)
    tests_passed = 0
    tests_total = 0

    # Test 2.1: Collaborative candidates
    print("\n[2.1] Testing collaborative_candidates()")
    collab = gen.collaborative_candidates("alice")
    tests_total += 1
    if isinstance(collab, set):
        print(f"  ✅ Returns set: PASS (found {len(collab)} candidates)")
        tests_passed += 1
    else:
        print(f"  ❌ Returns set: FAIL (got {type(collab)})")

    # Test 2.2: Excludes already rated
    print("\n[2.2] Checking already-rated items excluded")
    already_rated = set(USER_RATINGS["alice"].keys())
    tests_total += 1
    if collab.isdisjoint(already_rated):
        print("  ✅ Excludes already-rated: PASS")
        tests_passed += 1
    else:
        print("  ❌ Excludes already-rated: FAIL")

    # Test 2.3: Content-based candidates
    print("\n[2.3] Testing content_based_candidates()")
    content = gen.content_based_candidates("alice")
    tests_total += 1
    if isinstance(content, set):
        print(f"  ✅ Returns set: PASS (found {len(content)} candidates)")
        tests_passed += 1
    else:
        print(f"  ❌ Returns set: FAIL")

    # Test 2.4: Popular candidates
    print("\n[2.4] Testing popularity_candidates()")
    popular = gen.popularity_candidates("alice", top_n=5)
    tests_total += 1
    if isinstance(popular, set) and len(popular) <= 5:
        print(f"  ✅ Popular items: PASS (found {len(popular)} items)")
        tests_passed += 1
    else:
        print(f"  ❌ Popular items: FAIL")

    # Test 2.5: Hybrid candidates
    print("\n[2.5] Testing hybrid_candidates()")
    hybrid = gen.hybrid_candidates("alice")
    tests_total += 1
    if isinstance(hybrid, set):
        expected_union = collab | content | popular
        if hybrid == expected_union:
            print(f"  ✅ Hybrid union: PASS (found {len(hybrid)} total)")
            tests_passed += 1
        else:
            print(f"  ❌ Hybrid union: FAIL (expected {expected_union}, got {hybrid})")
    else:
        print(f"  ❌ Hybrid hybrid: FAIL")

    print(f"\n→ Candidate Generator: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: SCORER & RANKER
# ═══════════════════════════════════════════════════════════════════════════════

def test_scorer():
    """Test Module 3: Scorer & Ranker"""
    print("\n" + "=" * 70)
    print("TEST 3: SCORER & RANKER")
    print("=" * 70)

    scorer = RecommendationScorer(USER_RATINGS, ITEM_METADATA)
    tests_passed = 0
    tests_total = 0

    # Test 3.1: Score item
    print("\n[3.1] Testing calculate_score()")
    score = scorer.calculate_score("alice", "item5")
    tests_total += 1
    if 0.0 <= score <= 1.0:
        print(f"  ✅ Score in range [0,1]: PASS (score={score})")
        tests_passed += 1
    else:
        print(f"  ❌ Score in range [0,1]: FAIL (got {score})")

    # Test 3.2: Rank candidates
    print("\n[3.2] Testing rank_candidates()")
    candidates = {"item2", "item5", "item6"}
    ranked = scorer.rank_candidates("alice", candidates, limit=3)
    tests_total += 1
    if isinstance(ranked, list) and len(ranked) <= 3:
        print(f"  ✅ Ranked list: PASS (returned {len(ranked)} items)")
        tests_passed += 1
    else:
        print(f"  ❌ Ranked list: FAIL")

    # Test 3.3: Descending order
    print("\n[3.3] Checking scores are descending")
    scores = [s for _, s in ranked]
    tests_total += 1
    if scores == sorted(scores, reverse=True):
        print("  ✅ Descending order: PASS")
        tests_passed += 1
    else:
        print(f"  ❌ Descending order: FAIL")

    # Test 3.4: Explain score
    print("\n[3.4] Testing explain_score()")
    breakdown = scorer.explain_score("alice", "item5")
    tests_total += 1
    required_keys = {"collaborative", "content", "popularity", "final"}
    if all(k in breakdown for k in required_keys):
        print(f"  ✅ Score breakdown: PASS")
        print(f"     Collaborative: {breakdown['collaborative']}")
        print(f"     Content:       {breakdown['content']}")
        print(f"     Popularity:    {breakdown['popularity']}")
        print(f"     Final:         {breakdown['final']}")
        tests_passed += 1
    else:
        print(f"  ❌ Score breakdown: FAIL")

    # Test 3.5: Add custom scorer
    print("\n[3.5] Testing add_scorer()")
    def custom_scorer(user_id, item_id):
        return 0.5
    
    scorer.add_scorer("custom", custom_scorer, 0.1)
    tests_total += 1
    if "custom" in scorer.scorers:
        print("  ✅ Custom scorer registered: PASS")
        tests_passed += 1
    else:
        print("  ❌ Custom scorer registered: FAIL")

    print(f"\n→ Scorer & Ranker: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def test_evaluator():
    """Test Module 4: Evaluator"""
    print("\n" + "=" * 70)
    print("TEST 4: EVALUATOR")
    print("=" * 70)

    evaluator = RecommendationEvaluator()
    tests_passed = 0
    tests_total = 0

    # Test 4.1: Precision@K
    print("\n[4.1] Testing precision_at_k()")
    recs = ["item1", "item3", "item5"]
    relevant = {"item1", "item5"}
    precision = evaluator.precision_at_k(recs, relevant, k=3)
    expected = 2 / 3
    tests_total += 1
    if abs(precision - expected) < 0.001:
        print(f"  ✅ Precision@3: PASS (got {precision:.4f})")
        tests_passed += 1
    else:
        print(f"  ❌ Precision@3: FAIL (got {precision}, expected {expected})")

    # Test 4.2: Recall@K
    print("\n[4.2] Testing recall_at_k()")
    recall = evaluator.recall_at_k(recs, relevant, k=3)
    expected = 2 / 2  # 1.0 (found both relevant items)
    tests_total += 1
    if abs(recall - expected) < 0.001:
        print(f"  ✅ Recall@3: PASS (got {recall:.4f})")
        tests_passed += 1
    else:
        print(f"  ❌ Recall@3: FAIL (got {recall}, expected {expected})")

    # Test 4.3: NDCG@K
    print("\n[4.3] Testing ndcg_at_k()")
    ndcg = evaluator.ndcg_at_k(recs, relevant, k=3)
    tests_total += 1
    if 0.0 <= ndcg <= 1.0:
        print(f"  ✅ NDCG@3: PASS (got {ndcg:.4f})")
        tests_passed += 1
    else:
        print(f"  ❌ NDCG@3: FAIL (got {ndcg})")

    # Test 4.4: Evaluate all
    print("\n[4.4] Testing evaluate_all()")
    recommendations = {
        "alice": ["item3", "item5", "item1"],
        "bob":   ["item1", "item4", "item2"],
        "carol": ["item2", "item5", "item1"],
    }
    results = evaluator.evaluate_all(recommendations, GROUND_TRUTH, k=3)
    tests_total += 1
    required_metrics = {"precision@k", "recall@k", "ndcg@k", "num_users"}
    if all(m in results for m in required_metrics):
        print(f"  ✅ Evaluation metrics: PASS")
        print(f"     Precision@3: {results['precision@k']}")
        print(f"     Recall@3:    {results['recall@k']}")
        print(f"     NDCG@3:      {results['ndcg@k']}")
        print(f"     Users:       {results['num_users']}")
        tests_passed += 1
    else:
        print(f"  ❌ Evaluation metrics: FAIL")

    # Test 4.5: Empty inputs
    print("\n[4.5] Testing edge cases (empty inputs)")
    tests_total += 1
    precision_empty = evaluator.precision_at_k([], set(), k=5)
    recall_empty = evaluator.recall_at_k([], set(), k=5)
    if precision_empty == 0.0 and recall_empty == 0.0:
        print("  ✅ Handles empty inputs: PASS")
        tests_passed += 1
    else:
        print("  ❌ Handles empty inputs: FAIL")

    print(f"\n→ Evaluator: {tests_passed}/{tests_total} tests passed")
    return tests_passed, tests_total


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  DAY 29: RECOMMENDATION ENGINE — COMPREHENSIVE TEST SUITE  ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    total_passed = 0
    total_tests = 0

    # Run all tests
    passed, tests = test_similarity_calculator()
    total_passed += passed
    total_tests += tests

    passed, tests = test_candidate_generator()
    total_passed += passed
    total_tests += tests

    passed, tests = test_scorer()
    total_passed += passed
    total_tests += tests

    passed, tests = test_evaluator()
    total_passed += passed
    total_tests += tests

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed:      {total_passed}")
    print(f"Failed:      {total_tests - total_passed}")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%")

    if total_passed == total_tests:
        print("\n" + "🎉 " * 10)
        print("ALL TESTS PASSED! ✅")
        print("🎉 " * 10)
        return 0
    else:
        print(f"\n❌ {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
