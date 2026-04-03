"""
Module 4: Evaluator
--------------------
Checks how good the recommendations are by comparing predicted
recommendations against known ground-truth items.

Metrics implemented:
  - Precision@K   : fraction of top-K recs that are relevant
  - Recall@K      : fraction of relevant items found in top-K
  - F1@K          : harmonic mean of Precision@K and Recall@K
  - NDCG@K        : accounts for ranking position
  - Hit Rate@K    : 1 if at least one relevant item is in the top-K
  - Coverage      : fraction of the full catalogue ever recommended
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

    def f1_at_k(self, recommendations: list, relevant_items: set, k: int) -> float:
        """
        F1@K — harmonic mean of Precision@K and Recall@K.

        Balances the trade-off between precision and recall.  Returns 0.0
        when both are zero (avoids division by zero).

        Args:
            recommendations: Ordered list of recommended item IDs
            relevant_items:  Set of ground-truth relevant item IDs
            k:               Cut-off position

        Returns:
            Float in [0.0, 1.0]
        """
        precision = self.precision_at_k(recommendations, relevant_items, k)
        recall    = self.recall_at_k(recommendations, relevant_items, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

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

    def hit_rate_at_k(self, recommendations: list, relevant_items: set, k: int) -> float:
        """
        Hit Rate@K — 1.0 if at least one relevant item appears in the top-K,
        0.0 otherwise.

        A coarse but intuitive metric: did we surface *anything* useful?

        Args:
            recommendations: Ordered list of recommended item IDs
            relevant_items:  Set of ground-truth relevant item IDs
            k:               Cut-off position

        Returns:
            1.0 or 0.0
        """
        if not recommendations or not relevant_items or k <= 0:
            return 0.0

        top_k = set(recommendations[:k])
        return 1.0 if top_k & relevant_items else 0.0

    def coverage(self, recommendations_dict: dict, all_items: set) -> float:
        """
        Coverage — fraction of the full catalogue that appears in at least
        one user's recommendation list.

        A low coverage score means the engine keeps recommending the same
        popular items while ignoring most of the catalogue (popularity bias).

        Args:
            recommendations_dict: {user_id: [recommended_item_ids]}
            all_items:            Complete set of item IDs in the catalogue

        Returns:
            Float in [0.0, 1.0]; returns 0.0 if the catalogue is empty.
        """
        if not all_items:
            return 0.0

        recommended = set()
        for recs in recommendations_dict.values():
            recommended.update(recs)

        return len(recommended & all_items) / len(all_items)

    def evaluate_all(
        self,
        recommendations_dict: dict,
        ground_truth_dict: dict,
        k: int = 5,
        all_items: set | None = None,
    ) -> dict:
        """
        Run a full evaluation suite for the recommendation system.

        Args:
            recommendations_dict: {user_id: [recommended_item_ids]}
            ground_truth_dict:    {user_id: {relevant_item_ids}}
            k:                    Cut-off position for ranking metrics
            all_items:            Complete catalogue item set (used for
                                  coverage). If omitted, coverage is derived
                                  from all items seen in ground_truth_dict.

        Returns:
            Dict with metric names as keys and float scores as values.
        """
        empty_result = {
            "precision@k": 0.0,
            "recall@k":    0.0,
            "f1@k":        0.0,
            "ndcg@k":      0.0,
            "hit_rate@k":  0.0,
            "coverage":    0.0,
            "k":           k,
            "num_users":   0,
        }

        if not recommendations_dict or not ground_truth_dict:
            return empty_result

        precisions = []
        recalls    = []
        f1s        = []
        ndcgs      = []
        hit_rates  = []

        for user_id, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user_id, set())
            if not isinstance(relevant, set):
                relevant = set(relevant)

            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            f1s.append(self.f1_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))
            hit_rates.append(self.hit_rate_at_k(recs, relevant, k))

        n = len(recommendations_dict)
        if n == 0:
            return empty_result

        # Derive catalogue from ground truth if not supplied explicitly
        catalogue = all_items
        if catalogue is None:
            catalogue = set()
            for items in ground_truth_dict.values():
                catalogue.update(items)

        return {
            "precision@k": round(sum(precisions) / n, 4),
            "recall@k":    round(sum(recalls) / n, 4),
            "f1@k":        round(sum(f1s) / n, 4),
            "ndcg@k":      round(sum(ndcgs) / n, 4),
            "hit_rate@k":  round(sum(hit_rates) / n, 4),
            "coverage":    round(self.coverage(recommendations_dict, catalogue), 4),
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

    all_items = {"item1", "item2", "item3", "item4", "item5",
                 "item6", "item7", "item8", "item9", "item10"}

    print("=== Evaluation Results (K=5) ===")
    results = evaluator.evaluate_all(recommendations, ground_truth, k=5, all_items=all_items)
    for metric, value in results.items():
        print(f"  {metric:<15}: {value}")

    print("\n=== Per-User Breakdown ===")
    for user, recs in recommendations.items():
        relevant = ground_truth.get(user, set())
        p  = evaluator.precision_at_k(recs, relevant, 5)
        r  = evaluator.recall_at_k(recs, relevant, 5)
        f1 = evaluator.f1_at_k(recs, relevant, 5)
        n  = evaluator.ndcg_at_k(recs, relevant, 5)
        h  = evaluator.hit_rate_at_k(recs, relevant, 5)
        print(
            f"  {user:<8}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}  "
            f"NDCG={n:.4f}  Hit={h:.0f}"
        )