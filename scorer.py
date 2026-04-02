"""
Module 3: Scorer & Ranker
--------------------------
Ranks candidate items for a specific user and returns the top-N picks.

Scoring strategy (weighted sum):
  score = w_collab  * collaborative_score
        + w_content * content_score
        + w_popular * popularity_score

All component scores are normalised to [0, 1] before weighting.
The final score is clamped to [0.0, 1.0] and rounded to 4 decimal places.
"""

from similarity import SimilarityCalculator


class RecommendationScorer:
    """Scores and ranks recommendation candidates for a user."""

    # Default weights — must sum to 1.0 for a normalised final score.
    # Collaborative gets the highest weight because user-to-user similarity
    # is generally the strongest signal in a ratings-based system.
    DEFAULT_WEIGHTS = {
        "collaborative": 0.5,
        "content":       0.3,
        "popularity":    0.2,
    }

    def __init__(
        self,
        user_ratings: dict,
        item_metadata: dict | None = None,
        weights: dict | None = None,
    ):
        """
        Args:
            user_ratings:  {user_id: {item_id: rating}}
            item_metadata: {item_id: {attribute: value}}
            weights:       {'collaborative': float, 'content': float, 'popularity': float}
                           Values should sum to 1.0.
        """
        self.user_ratings  = user_ratings
        self.item_metadata = item_metadata or {}
        self.weights       = weights or dict(self.DEFAULT_WEIGHTS)

        # Registry for optional custom scoring functions added at runtime
        self.scorers: dict = {}

        # Shared similarity calculator used by all internal scorers
        self.sim_calc = SimilarityCalculator()

        # Popularity scores are computed once on first use and cached here
        # to avoid re-iterating over all ratings on every score call.
        self._popularity_cache: dict = {}

    # ── Scorer Registration ──────────────────────────────────────────────────

    def add_scorer(self, name: str, function, weight: float) -> None:
        """
        Register a custom scoring function to extend the composite score.

        The function is called with (user_id, item_id) and must return a
        float in [0, 1]. Its contribution is added on top of the base
        weighted score, so total weights may exceed 1.0 if custom scorers
        are added without adjusting the base weights.

        Args:
            name:     Unique name for the scorer (e.g., 'recency')
            function: Callable(user_id: str, item_id: str) -> float
            weight:   Contribution weight for this scorer
        """
        self.scorers[name] = {"function": function, "weight": weight}

    # ── Internal Scorers ─────────────────────────────────────────────────────

    def _collaborative_score(self, target_user: str, item_id: str) -> float:
        """
        Predicts how much the target user would enjoy an item based on
        what similar users rated it.

        Algorithm:
          1. Compute cosine similarity between target user and every other
             user who has rated the item.
          2. Take a similarity-weighted average of their ratings.
          3. Normalise by dividing by the maximum possible rating (5).

        Returns a value in [0, 1]. Returns 0.0 if no other user has rated
        the item or if the target user has no ratings to compare against.
        """
        target_vec = self.user_ratings.get(target_user, {})
        total_sim = total_weighted = 0.0

        for user, ratings in self.user_ratings.items():
            # Skip the target user themselves and users who haven't rated this item
            if user == target_user or item_id not in ratings:
                continue

            sim = self.sim_calc.cosine_similarity(target_vec, ratings)
            total_weighted += sim * ratings[item_id]   # weight rating by similarity
            total_sim      += abs(sim)                 # accumulate denominator

        if total_sim == 0:
            return 0.0  # No comparable users found

        predicted_rating = total_weighted / total_sim
        return max(0.0, min(1.0, predicted_rating / 5.0))  # Normalise to [0, 1]

    def _content_score(self, target_user: str, item_id: str) -> float:
        """
        Measures how well a candidate item matches the user's taste profile
        by comparing its metadata to the user's top-3 highest-rated items.

        Algorithm:
          1. Find the user's top-3 rated items (by rating value).
          2. Compute cosine similarity between the candidate item's metadata
             vector and each of those top-3 items' metadata vectors.
          3. Return the average similarity across all comparisons.

        Returns a value in [0, 1]. Returns 0.0 if the item has no metadata
        or if the user has no ratings to build a taste profile from.
        """
        # Cannot score without metadata for the candidate item
        if not self.item_metadata or item_id not in self.item_metadata:
            return 0.0

        user_ratings = self.user_ratings.get(target_user, {})
        if not user_ratings:
            return 0.0  # No taste profile to compare against

        # Build taste profile from the user's top-3 liked items
        top_liked = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:3]

        candidate_vec = self.item_metadata[item_id]
        sims = []
        for liked_item, _rating in top_liked:
            if liked_item in self.item_metadata:
                # Compare candidate against each liked item in feature space
                sims.append(
                    self.sim_calc.cosine_similarity(
                        self.item_metadata[liked_item], candidate_vec
                    )
                )

        # Average similarity across all taste-profile comparisons
        return sum(sims) / len(sims) if sims else 0.0

    def _popularity_score(self, item_id: str) -> float:
        """
        Estimates item popularity as the fraction of all users who have
        rated it (rating-count / total-users).

        The cache is built lazily on the first call and reused for all
        subsequent calls, so the O(users × items) scan happens only once.

        Returns a value in [0, 1].
        """
        # Build and cache popularity scores on first call
        if not self._popularity_cache:
            total_users = len(self.user_ratings)
            if total_users == 0:
                return 0.0

            # Count how many users have rated each item
            raw_counts: dict = {}
            for ratings in self.user_ratings.values():
                for item in ratings:
                    raw_counts[item] = raw_counts.get(item, 0) + 1

            # Normalise counts to [0, 1] by dividing by total users
            self._popularity_cache = {
                item: count / total_users
                for item, count in raw_counts.items()
            }

        # Return 0.0 for items not seen in any user's ratings
        return self._popularity_cache.get(item_id, 0.0)

    # ── Public API ───────────────────────────────────────────────────────────

    def calculate_score(self, user_id: str, item_id: str, context: dict | None = None) -> float:
        """
        Calculate the composite recommendation score for a single item.

        Combines collaborative, content-based, and popularity scores using
        the configured weights. Any registered custom scorers are added on
        top. The result is clamped to [0.0, 1.0].

        Args:
            user_id:  Target user ID
            item_id:  Item to score
            context:  Reserved for future context-aware scoring (unused)

        Returns:
            Float in [0.0, 1.0] rounded to 4 decimal places.
        """
        # Compute each component score independently
        collab  = self._collaborative_score(user_id, item_id)
        content = self._content_score(user_id, item_id)
        popular = self._popularity_score(item_id)

        # Weighted sum of the three base signals
        score = (
            self.weights["collaborative"] * collab
            + self.weights["content"]     * content
            + self.weights["popularity"]  * popular
        )

        # Add contributions from any custom scorers registered via add_scorer()
        for name, scorer_info in self.scorers.items():
            custom_score = scorer_info["function"](user_id, item_id)
            score += scorer_info["weight"] * custom_score

        # Clamp to valid range and round for consistent output
        return round(min(1.0, max(0.0, score)), 4)

    def rank_candidates(
        self,
        target_user: str,
        candidates: set,
        limit: int = 5,
    ) -> list:
        """
        Score every candidate item and return the top-N ranked by score.

        Args:
            target_user: User to generate recommendations for
            candidates:  Set of candidate item IDs to evaluate
            limit:       Maximum number of results to return (default: 5)

        Returns:
            List of (item_id, score) tuples, sorted descending by score.
            Returns an empty list if candidates is empty.
        """
        if not candidates:
            return []  # Nothing to rank

        # Score every candidate, then sort highest-first
        scored = [(item, self.calculate_score(target_user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:limit]  # Return only the top-N

    def explain_score(self, target_user: str, item_id: str) -> dict:
        """
        Return a human-readable breakdown of how a score was computed.

        Useful for debugging, logging, and showing users *why* an item
        was recommended ("because users similar to you rated it highly").

        Returns:
            Dict with keys: 'collaborative', 'content', 'popularity', 'final'
        """
        # Re-compute each component so the breakdown matches calculate_score()
        collab  = self._collaborative_score(target_user, item_id)
        content = self._content_score(target_user, item_id)
        popular = self._popularity_score(item_id)
        final   = self.calculate_score(target_user, item_id)

        return {
            "collaborative": round(collab,  4),
            "content":       round(content, 4),
            "popularity":    round(popular, 4),
            "final":         final,            # Already rounded by calculate_score
        }


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal sample data to verify the scorer runs end-to-end
    USER_RATINGS = {
        "alice": {"item1": 5, "item2": 3, "item3": 4},
        "bob":   {"item1": 4, "item2": 3, "item3": 5},
        "carol": {"item1": 1, "item2": 5, "item3": 2},
    }

    # Metadata vectors — each key is a feature, value is its strength [0, 1]
    ITEM_METADATA = {
        "item1": {"genre": 0.8, "action": 0.9},
        "item2": {"genre": 0.5, "action": 0.3},
        "item3": {"genre": 0.7, "action": 0.8},
    }

    scorer = RecommendationScorer(USER_RATINGS, ITEM_METADATA)

    candidates = {"item2", "item3"}
    print("=== Top Recommendations for alice ===")
    ranked = scorer.rank_candidates("alice", candidates, limit=5)
    for rank, (item, score) in enumerate(ranked, 1):
        breakdown = scorer.explain_score("alice", item)
        print(f"  {rank}. {item}: {score}")
        print(f"     Breakdown: {breakdown}")