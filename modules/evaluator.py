"""
Module 4: Evaluator
--------------------
Checks how good the recommendations are by comparing predicted
recommendations against known ground-truth items.

Metrics implemented:
  - Precision@K   : fraction of top-K recs that are relevant
  - Recall@K      : fraction of relevant items found in top-K
  - F1@K          : harmonic mean of Precision and Recall
  - Hit Rate@K    : 1 if at least one relevant item is in top-K
  - Coverage      : fraction of all items that ever appear in recs
  - Average scores over all test users
"""


class Evaluator:
    """Evaluates recommendation quality against ground-truth data."""

    # ── Per-user Metrics ─────────────────────────────────────────────────────

    def precision_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """
        Precision@K — of the top-K items recommended, how many are relevant?

        Args:
            recommended: Ordered list of recommended item IDs (best first)
            relevant:    Set of ground-truth relevant item IDs
            k:           Cut-off position

        Returns:
            Float in [0.0, 1.0]
        """
        if not recommended or k <= 0:
            return 0.0

        top_k = recommended[:k]
        hits  = sum(1 for item in top_k if item in relevant)
        return hits / k

    def recall_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """
        Recall@K — of all relevant items, how many appear in the top-K?

        Args:
            recommended: Ordered list of recommended item IDs
            relevant:    Set of ground-truth relevant item IDs
            k:           Cut-off position

        Returns:
            Float in [0.0, 1.0]; returns 0.0 if relevant set is empty.
        """
        if not relevant or not recommended or k <= 0:
            return 0.0

        top_k = set(recommended[:k])
        hits  = len(top_k & relevant)
        return hits / len(relevant)

    def f1_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """
        F1@K — harmonic mean of Precision@K and Recall@K.

        Returns:
            Float in [0.0, 1.0]
        """
        p = self.precision_at_k(recommended, relevant, k)
        r = self.recall_at_k(recommended, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def hit_rate_at_k(self, recommended: list, relevant: set, k: int) -> float:
        """
        Hit Rate@K — 1 if at least one relevant item appears in top-K, else 0.

        Returns:
            1.0 or 0.0
        """
        top_k = set(recommended[:k])
        return 1.0 if top_k & relevant else 0.0

    # ── System-Wide Metrics ──────────────────────────────────────────────────

    def coverage(self, all_recommendations: list, all_items: set) -> float:
        """
        Item Coverage — fraction of the full catalogue that was recommended
        to at least one user.

        Args:
            all_recommendations: List of recommendation lists (one per user)
            all_items:           Set of all possible item IDs in the system

        Returns:
            Float in [0.0, 1.0]
        """
        if not all_items:
            return 0.0

        recommended_items = set()
        for rec_list in all_recommendations:
            recommended_items.update(rec_list)

        return len(recommended_items & all_items) / len(all_items)

    def average_precision_at_k(
        self, user_results: list, k: int
    ) -> float:
        """
        Mean Precision@K across all users.

        Args:
            user_results: List of (recommended_list, relevant_set) tuples
            k:            Cut-off position

        Returns:
            Float in [0.0, 1.0]
        """
        if not user_results:
            return 0.0

        scores = [
            self.precision_at_k(recs, rel, k) for recs, rel in user_results
        ]
        return sum(scores) / len(scores)

    def evaluate_all(
        self,
        user_results: list,
        all_items: set,
        k: int = 5,
    ) -> dict:
        """
        Run a full evaluation suite for the recommendation system.

        Args:
            user_results: List of (user_id, recommended_list, relevant_set) tuples
            all_items:    Complete item catalogue
            k:            Cut-off position for ranking metrics

        Returns:
            Dict with metric names as keys and float scores as values.
        """
        if not user_results:
            return {
                "precision@k": 0.0,
                "recall@k":    0.0,
                "f1@k":        0.0,
                "hit_rate@k":  0.0,
                "coverage":    0.0,
                "k":           k,
                "num_users":   0,
            }

        precisions  = []
        recalls     = []
        f1s         = []
        hit_rates   = []
        all_recs    = []

        for _user, recs, relevant in user_results:
            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            f1s.append(self.f1_at_k(recs, relevant, k))
            hit_rates.append(self.hit_rate_at_k(recs, relevant, k))
            all_recs.append(recs)

        n = len(user_results)

        return {
            "precision@k": round(sum(precisions) / n, 4),
            "recall@k":    round(sum(recalls)    / n, 4),
            "f1@k":        round(sum(f1s)        / n, 4),
            "hit_rate@k":  round(sum(hit_rates)  / n, 4),
            "coverage":    round(self.coverage(all_recs, all_items), 4),
            "k":           k,
            "num_users":   n,
        }


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = Evaluator()

    # Simulate results for 3 test users
    user_results = [
        # (user_id, recommended_items, ground_truth_relevant_items)
        ("alice", ["item5", "item6", "item7", "item8", "item9"], {"item5", "item7"}),
        ("bob",   ["item3", "item5", "item8", "item9", "item10"], {"item3", "item10"}),
        ("carol", ["item1", "item4", "item6", "item7", "item8"], {"item6", "item4", "item8"}),
    ]

    all_items = {f"item{i}" for i in range(1, 15)}

    print("=== Evaluation Results (K=5) ===")
    results = evaluator.evaluate_all(user_results, all_items, k=5)
    for metric, value in results.items():
        print(f"  {metric:<15}: {value}")

    print("\n=== Per-User Breakdown ===")
    for user, recs, relevant in user_results:
        p = evaluator.precision_at_k(recs, relevant, 5)
        r = evaluator.recall_at_k(recs, relevant, 5)
        f = evaluator.f1_at_k(recs, relevant, 5)
        h = evaluator.hit_rate_at_k(recs, relevant, 5)
        print(
            f"  {user:<8}  P={p:.2f}  R={r:.2f}  F1={f:.2f}  Hit={int(h)}"
        )
