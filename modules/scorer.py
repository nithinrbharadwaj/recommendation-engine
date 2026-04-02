"""
Module 3: Scorer
-----------------
Ranks candidate items for a specific user and returns the top-N picks.

Scoring strategy (weighted sum):
  score = w_collab  * collaborative_score
        + w_content * content_score
        + w_popular * popularity_score
"""

from modules.similarity_calculator import SimilarityCalculator


class Scorer:
    """Scores and ranks recommendation candidates for a user."""

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
        self.user_ratings = user_ratings
        self.item_metadata = item_metadata or {}
        self.weights = weights or {
            "collaborative": 0.5,
            "content":       0.3,
            "popularity":    0.2,
        }
        self.sim_calc = SimilarityCalculator()
        self._popularity_cache: dict = {}   # Pre-computed once

    # ── Internal Scorers ─────────────────────────────────────────────────────

    def _collaborative_score(self, target_user: str, item_id: str) -> float:
        """
        Weighted-average predicted rating from similar users.
        Returns a value in [0, 1] (normalized by max rating 5).
        """
        target_vec = self.user_ratings.get(target_user, {})
        total_sim = total_weighted = 0.0

        for user, ratings in self.user_ratings.items():
            if user == target_user or item_id not in ratings:
                continue
            sim = self.sim_calc.cosine_similarity(target_vec, ratings)
            total_weighted += sim * ratings[item_id]
            total_sim += abs(sim)

        if total_sim == 0:
            return 0.0

        predicted_rating = total_weighted / total_sim
        return max(0.0, min(1.0, predicted_rating / 5.0))   # Normalize to [0,1]

    def _content_score(self, target_user: str, item_id: str) -> float:
        """
        Average cosine similarity between candidate item and the user's
        top-3 liked items based on metadata.
        Returns a value in [0, 1].
        """
        if not self.item_metadata or item_id not in self.item_metadata:
            return 0.0

        user_ratings = self.user_ratings.get(target_user, {})
        if not user_ratings:
            return 0.0

        # Top-3 liked items by this user
        top_liked = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:3]

        candidate_vec = self.item_metadata[item_id]
        sims = []
        for liked_item, _rating in top_liked:
            if liked_item in self.item_metadata:
                sims.append(
                    self.sim_calc.cosine_similarity(
                        self.item_metadata[liked_item], candidate_vec
                    )
                )

        return sum(sims) / len(sims) if sims else 0.0

    def _popularity_score(self, item_id: str) -> float:
        """
        Fraction of users who have rated this item (proxy for popularity).
        Returns a value in [0, 1].
        """
        if not self._popularity_cache:
            total_users = len(self.user_ratings)
            if total_users == 0:
                return 0.0
            for ratings in self.user_ratings.values():
                for item in ratings:
                    self._popularity_cache[item] = (
                        self._popularity_cache.get(item, 0) + 1
                    )
            self._popularity_cache = {
                k: v / total_users
                for k, v in self._popularity_cache.items()
            }

        return self._popularity_cache.get(item_id, 0.0)

    # ── Public API ───────────────────────────────────────────────────────────

    def score_item(self, target_user: str, item_id: str) -> float:
        """
        Compute the weighted composite score for a single candidate item.

        Returns:
            Float in [0.0, 1.0]
        """
        collab  = self._collaborative_score(target_user, item_id)
        content = self._content_score(target_user, item_id)
        popular = self._popularity_score(item_id)

        score = (
            self.weights["collaborative"] * collab
            + self.weights["content"]      * content
            + self.weights["popularity"]   * popular
        )
        return round(score, 4)

    def rank_candidates(
        self,
        target_user: str,
        candidates: set,
        top_n: int = 5,
    ) -> list:
        """
        Score all candidates and return the top-N.

        Args:
            target_user: User to rank items for
            candidates:  Set of candidate item IDs
            top_n:       How many recommendations to return

        Returns:
            List of (item_id, score) tuples, sorted descending by score.
        """
        if not candidates:
            return []

        scored = [(item, self.score_item(target_user, item)) for item in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def explain_score(self, target_user: str, item_id: str) -> dict:
        """
        Return a breakdown of the score components for transparency.

        Returns:
            Dict with 'collaborative', 'content', 'popularity', and 'final' keys.
        """
        collab  = self._collaborative_score(target_user, item_id)
        content = self._content_score(target_user, item_id)
        popular = self._popularity_score(item_id)
        final   = self.score_item(target_user, item_id)

        return {
            "collaborative": round(collab,  4),
            "content":       round(content, 4),
            "popularity":    round(popular, 4),
            "final":         final,
        }


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.sample_data import USER_RATINGS, ITEM_METADATA
    from modules.candidate_generator import CandidateGenerator

    gen    = CandidateGenerator(USER_RATINGS, ITEM_METADATA)
    scorer = Scorer(USER_RATINGS, ITEM_METADATA)

    target     = "alice"
    candidates = gen.generate(target)

    print(f"=== Top Recommendations for '{target}' ===")
    ranked = scorer.rank_candidates(target, candidates, top_n=5)
    for rank, (item, score) in enumerate(ranked, 1):
        breakdown = scorer.explain_score(target, item)
        print(
            f"  {rank}. {item}  |  final={score}  "
            f"| collab={breakdown['collaborative']}  "
            f"content={breakdown['content']}  "
            f"pop={breakdown['popularity']}"
        )
