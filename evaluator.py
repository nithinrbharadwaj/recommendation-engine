"""
Module 4: Evaluator
--------------------
Checks how good the recommendations are by comparing predicted
recommendations against known ground-truth items.

Metrics implemented:
  - Precision@K   : fraction of top-K recs that are relevant
  - Recall@K      : fraction of relevant items found in top-K
  - NDCG@K        : Accounts for ranking position
  - Average scores over all test users
"""


class RecommendationEvaluator:
    """Evaluates recommendation quality against ground-truth data."""

    # ── Per-user Metrics ─────────────────────────────────────────────────────

    def precision_at_k(self, recommendations: list, relevant_items: set, k: int) -> float:
        """
        Precision@K — of the top-K items recommended, how many are relevant?

        Args:
            recommendations: Ordered list of recommended item IDs (best first)
            relevant_items:  Set of ground-truth relevant item IDs
            k:               Cut-off position

        Returns:
            Float in [0.0, 1.0]
        """
        if not recommendations or k <= 0:
            return 0.0

        top_k = recommendations[:k]
        hits  = sum(1 for item in top_k if item in relevant_items)
        return hits / k

    def recall_at_k(self, recommendations: list, relevant_items: set, k: int) -> float:
        """
        Recall@K — of all relevant items, how many appear in the top-K?

        Args:
            recommendations: Ordered list of recommended item IDs
            relevant_items:  Set of ground-truth relevant item IDs
            k:               Cut-off position

        Returns:
            Float in [0.0, 1.0]; returns 0.0 if relevant set is empty.
        """
        if not relevant_items or not recommendations or k <= 0:
            return 0.0

        top_k = set(recommendations[:k])
        hits  = len(top_k & relevant_items)
        return hits / len(relevant_items)

    def ndcg_at_k(self, recommendations: list, relevant_items: set, k: int) -> float:
        """
        NDCG@K — Normalized Discounted Cumulative Gain.
        Accounts for ranking position (relevant items earlier = better score).

        Args:
            recommendations: Ordered list of recommended item IDs
            relevant_items:  Set of ground-truth relevant item IDs
            k:               Cut-off position

        Returns:
            Float in [0.0, 1.0]
        """
        if not recommendations or not relevant_items or k <= 0:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommendations[:k], start=1):
            if item in relevant_items:
                dcg += 1.0 / (i ** 0.5)  # log2(i+1) approximation

        # Calculate IDCG (ideal DCG - all relevant items ranked first)
        idcg = 0.0
        for i in range(1, min(k, len(relevant_items)) + 1):
            idcg += 1.0 / (i ** 0.5)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_all(
        self,
        recommendations_dict: dict,
        ground_truth_dict: dict,
        k: int = 5,
    ) -> dict:
        """
        Run a full evaluation suite for the recommendation system.

        Args:
            recommendations_dict: {user_id: [recommended_item_ids]}
            ground_truth_dict:    {user_id: {relevant_item_ids}}
            k:                    Cut-off position for ranking metrics

        Returns:
            Dict with metric names as keys and float scores as values.
        """
        if not recommendations_dict or not ground_truth_dict:
            return {
                "precision@k": 0.0,
                "recall@k":    0.0,
                "ndcg@k":      0.0,
                "k":           k,
                "num_users":   0,
            }

        precisions = []
        recalls    = []
        ndcgs      = []

        for user_id, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user_id, set())
            if not isinstance(relevant, set):
                relevant = set(relevant)

            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))

        n = len(recommendations_dict)
        if n == 0:
            return {
                "precision@k": 0.0,
                "recall@k":    0.0,
                "ndcg@k":      0.0,
                "k":           k,
                "num_users":   0,
            }

        return {
            "precision@k": round(sum(precisions) / n, 4),
            "recall@k":    round(sum(recalls) / n, 4),
            "ndcg@k":      round(sum(ndcgs) / n, 4),
            "k":           k,
            "num_users":   n,
        }


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = RecommendationEvaluator()

    # Simulate results for 3 test users
    recommendations = {
        "alice": ["item5", "item6", "item7", "item8", "item9"],
        "bob":   ["item3", "item5", "item8", "item9", "item10"],
        "carol": ["item1", "item4", "item6", "item7", "item8"],
    }

    ground_truth = {
        "alice": {"item5", "item7"},
        "bob":   {"item3", "item10"},
        "carol": {"item6", "item4", "item8"},
    }

    print("=== Evaluation Results (K=5) ===")
    results = evaluator.evaluate_all(recommendations, ground_truth, k=5)
    for metric, value in results.items():
        print(f"  {metric:<15}: {value}")

    print("\n=== Per-User Breakdown ===")
    for user, recs in recommendations.items():
        relevant = ground_truth.get(user, set())
        p = evaluator.precision_at_k(recs, relevant, 5)
        r = evaluator.recall_at_k(recs, relevant, 5)
        n = evaluator.ndcg_at_k(recs, relevant, 5)
        print(
            f"  {user:<8}  P={p:.4f}  R={r:.4f}  NDCG={n:.4f}"
        )
