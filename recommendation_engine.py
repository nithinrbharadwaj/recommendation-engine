"""
recommendation_engine.py
-------------------------
Orchestrates all four modules into a single, easy-to-use RecommendationEngine
class, plus a CLI demo.
"""

from modules.similarity_calculator import SimilarityCalculator
from modules.candidate_generator   import CandidateGenerator
from modules.scorer                import Scorer
from modules.evaluator             import Evaluator


class RecommendationEngine:
    """
    End-to-end recommendation engine.

    Pipeline:
        User → CandidateGenerator → Scorer → top-N recommendations
    Then optionally:
        Recommendations + ground truth → Evaluator → quality metrics
    """

    def __init__(
        self,
        user_ratings: dict,
        item_metadata: dict | None = None,
        scorer_weights: dict | None = None,
    ):
        """
        Args:
            user_ratings:   {user_id: {item_id: rating}}
            item_metadata:  {item_id: {attribute: value}}
            scorer_weights: {'collaborative': w1, 'content': w2, 'popularity': w3}
        """
        self.user_ratings  = user_ratings
        self.item_metadata = item_metadata or {}

        self.sim_calc  = SimilarityCalculator()
        self.generator = CandidateGenerator(user_ratings, item_metadata)
        self.scorer    = Scorer(user_ratings, item_metadata, scorer_weights)
        self.evaluator = Evaluator()

    # ── Core Recommendation ──────────────────────────────────────────────────

    def recommend(
        self,
        user_id: str,
        top_n: int = 5,
        strategy: str = "all",
        verbose: bool = False,
    ) -> list:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id:  Target user
            top_n:    Number of items to return
            strategy: Candidate generation strategy ('collaborative' | 'content' |
                      'popular' | 'all')
            verbose:  Print score breakdown if True

        Returns:
            List of (item_id, score) tuples sorted best-first.
        """
        if user_id not in self.user_ratings:
            print(f"[Warning] Unknown user '{user_id}'. Returning popular items.")
            strategy = "popular"

        # Step 1: Generate candidates
        candidates = self.generator.generate(user_id, strategy=strategy)

        if not candidates:
            print(f"[Warning] No candidates found for '{user_id}'.")
            return []

        # Step 2: Score & rank
        ranked = self.scorer.rank_candidates(user_id, candidates, top_n=top_n)

        # Step 3 (optional): Print explanation
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Recommendations for '{user_id}'  (strategy={strategy})")
            print(f"{'='*55}")
            print(f"  Candidates generated : {len(candidates)}")
            print(f"  {'Rank':<5} {'Item':<15} {'Score':>6}  Breakdown")
            print(f"  {'-'*50}")
            for rank, (item, score) in enumerate(ranked, 1):
                bd = self.scorer.explain_score(user_id, item)
                print(
                    f"  {rank:<5} {item:<15} {score:>6.4f}  "
                    f"collab={bd['collaborative']}  "
                    f"content={bd['content']}  "
                    f"pop={bd['popularity']}"
                )
            print()

        return ranked

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        ground_truth: dict,
        top_n: int = 5,
        k: int = 5,
        strategy: str = "all",
    ) -> dict:
        """
        Evaluate the engine against held-out ground-truth data.

        Args:
            ground_truth: {user_id: set_of_relevant_items}
            top_n:        Items to recommend per user
            k:            Cut-off for ranking metrics
            strategy:     Candidate generation strategy

        Returns:
            Dict of evaluation metrics.
        """
        all_items    = set(self.item_metadata.keys()) or set(
            item for ratings in self.user_ratings.values() for item in ratings
        )
        user_results = []

        for user_id, relevant_items in ground_truth.items():
            recs = self.recommend(user_id, top_n=top_n, strategy=strategy)
            recommended_list = [item for item, _score in recs]
            user_results.append((user_id, recommended_list, relevant_items))

        return self.evaluator.evaluate_all(user_results, all_items, k=k)

    # ── Utility ──────────────────────────────────────────────────────────────

    def find_similar_users(self, user_id: str, top_n: int = 3) -> list:
        """Return the top-N most similar users to a given user."""
        return self.sim_calc.most_similar_users(
            user_id, self.user_ratings, top_n=top_n
        )

    def add_user(self, user_id: str, ratings: dict) -> None:
        """Add or update a user's ratings at runtime."""
        self.user_ratings[user_id] = ratings
        # Re-initialise modules that cache data
        self.generator = CandidateGenerator(self.user_ratings, self.item_metadata)
        self.scorer    = Scorer(self.user_ratings, self.item_metadata)

    def add_item(self, item_id: str, metadata: dict) -> None:
        """Add or update an item's metadata at runtime."""
        self.item_metadata[item_id] = metadata
        self.generator = CandidateGenerator(self.user_ratings, self.item_metadata)
        self.scorer    = Scorer(self.user_ratings, self.item_metadata)


# ── CLI Demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.sample_data import USER_RATINGS, ITEM_METADATA, GROUND_TRUTH, ALL_ITEMS

    print("\n" + "█" * 55)
    print("   RECOMMENDATION ENGINE  — Full Demo")
    print("█" * 55)

    engine = RecommendationEngine(USER_RATINGS, ITEM_METADATA)

    # 1. Recommend for known users
    for user in ["alice", "bob", "carol"]:
        engine.recommend(user, top_n=5, verbose=True)

    # 2. Similar users
    print("=== Similar Users (alice) ===")
    for user, score in engine.find_similar_users("alice"):
        print(f"  {user}: {score}")

    # 3. Cold-start: new user with no ratings
    print("\n=== Cold-Start Test (new user) ===")
    engine.recommend("zara", top_n=5, verbose=True)

    # 4. Dynamic update
    print("=== Adding New User 'zara' with ratings ===")
    engine.add_user("zara", {"laptop": 5, "headphones": 4})
    engine.recommend("zara", top_n=5, verbose=True)

    # 5. Evaluation
    print("=== Evaluation Against Ground Truth (K=5) ===")
    metrics = engine.evaluate(GROUND_TRUTH, top_n=5, k=5)
    for metric, value in metrics.items():
        print(f"  {metric:<15}: {value}")
