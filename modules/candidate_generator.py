"""
Module 2: Candidate Generator
-------------------------------
Generates candidate items for a user to consider, based on:
  - Items rated highly by similar users (collaborative filtering)
  - Items sharing attributes with things the user already likes (content-based)
  - Popular items the user hasn't seen yet (popularity-based fallback)
"""

from modules.similarity_calculator import SimilarityCalculator


class CandidateGenerator:
    """Generates recommendation candidates for a given user."""

    def __init__(self, user_ratings: dict, item_metadata: dict | None = None):
        """
        Args:
            user_ratings:  {user_id: {item_id: rating}}
            item_metadata: {item_id: {attribute: value}} — optional, used for
                           content-based generation.
        """
        self.user_ratings = user_ratings
        self.item_metadata = item_metadata or {}
        self.sim_calc = SimilarityCalculator()

    # ── Collaborative Filtering ──────────────────────────────────────────────

    def collaborative_candidates(
        self,
        target_user: str,
        top_similar_users: int = 3,
        min_rating: float = 3.5,
    ) -> set:
        """
        Return items that similar users rated highly but the target user
        hasn't rated yet.

        Args:
            target_user:        User to generate candidates for
            top_similar_users:  How many neighbours to consider
            min_rating:         Minimum rating to qualify as 'highly rated'

        Returns:
            Set of candidate item IDs
        """
        if target_user not in self.user_ratings:
            return set()

        already_rated = set(self.user_ratings[target_user].keys())

        similar_users = self.sim_calc.most_similar_users(
            target_user, self.user_ratings, method="cosine", top_n=top_similar_users
        )

        candidates = set()
        for neighbour, _score in similar_users:
            for item, rating in self.user_ratings[neighbour].items():
                if item not in already_rated and rating >= min_rating:
                    candidates.add(item)

        return candidates

    # ── Content-Based Filtering ──────────────────────────────────────────────

    def content_based_candidates(
        self,
        target_user: str,
        top_n_liked: int = 3,
    ) -> set:
        """
        Return items whose metadata is similar to items the user liked most.

        Args:
            target_user:  User to generate candidates for
            top_n_liked:  Consider this many of the user's top-rated items

        Returns:
            Set of candidate item IDs
        """
        if target_user not in self.user_ratings or not self.item_metadata:
            return set()

        user_ratings = self.user_ratings[target_user]
        already_rated = set(user_ratings.keys())

        # Find the user's top-rated items
        liked_items = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[
            :top_n_liked
        ]

        candidates = set()
        for liked_item, _rating in liked_items:
            if liked_item not in self.item_metadata:
                continue
            liked_vec = self.item_metadata[liked_item]

            # Compare against every unrated item
            for item_id, item_vec in self.item_metadata.items():
                if item_id in already_rated:
                    continue
                score = self.sim_calc.cosine_similarity(liked_vec, item_vec)
                if score > 0.5:          # Threshold — tune as needed
                    candidates.add(item_id)

        return candidates

    # ── Popularity-Based Fallback ────────────────────────────────────────────

    def popular_candidates(
        self,
        target_user: str,
        top_n: int = 10,
    ) -> set:
        """
        Return globally popular items the user hasn't rated yet.

        Args:
            target_user: User to generate candidates for
            top_n:       Number of popular items to include

        Returns:
            Set of candidate item IDs
        """
        already_rated = set(
            self.user_ratings.get(target_user, {}).keys()
        )

        # Count how many users rated each item
        item_counts: dict = {}
        for ratings in self.user_ratings.values():
            for item in ratings:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Sort by popularity, exclude already-rated
        popular = sorted(item_counts, key=item_counts.get, reverse=True)
        return {item for item in popular[:top_n] if item not in already_rated}

    # ── Combined ─────────────────────────────────────────────────────────────

    def generate(self, target_user: str, strategy: str = "all") -> set:
        """
        Generate candidates using one or all strategies.

        Args:
            target_user: Target user ID
            strategy:    'collaborative' | 'content' | 'popular' | 'all'

        Returns:
            Combined set of candidate item IDs
        """
        strategies = {
            "collaborative": self.collaborative_candidates,
            "content":       self.content_based_candidates,
            "popular":       self.popular_candidates,
        }

        if strategy == "all":
            candidates = set()
            for fn in strategies.values():
                candidates |= fn(target_user)
            return candidates

        if strategy not in strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(strategies.keys())} or 'all'."
            )

        return strategies[strategy](target_user)


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.sample_data import USER_RATINGS, ITEM_METADATA

    gen = CandidateGenerator(USER_RATINGS, ITEM_METADATA)

    target = "alice"
    print(f"=== Candidates for '{target}' ===")
    print(f"Collaborative : {gen.collaborative_candidates(target)}")
    print(f"Content-Based : {gen.content_based_candidates(target)}")
    print(f"Popular       : {gen.popular_candidates(target)}")
    print(f"All combined  : {gen.generate(target)}")
