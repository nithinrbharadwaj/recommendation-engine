"""
Module 1: Similarity Calculator
--------------------------------
Provides cosine similarity and Jaccard similarity functions to measure
how alike two users or items are based on their attributes/ratings.
"""

import math


class SimilarityCalculator:
    """Calculates similarity scores between users or items."""

    def cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """
        Compute cosine similarity between two sparse vectors (dicts).

        Args:
            vec_a: {feature: value} mapping for entity A
            vec_b: {feature: value} mapping for entity B

        Returns:
            Float in [0.0, 1.0]; 0.0 means orthogonal, 1.0 means identical direction.
        """
        if not vec_a or not vec_b:
            return 0.0

        # Only iterate over shared keys for the dot product
        common_keys = set(vec_a.keys()) & set(vec_b.keys())
        dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)

        mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """
        Compute Jaccard similarity between two sets.

        Args:
            set_a: First set (e.g., items purchased by user A)
            set_b: Second set (e.g., items purchased by user B)

        Returns:
            Float in [0.0, 1.0]; 1.0 means identical sets.
        """
        if not set_a and not set_b:
            return 1.0  # Both empty — treat as identical
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union

    def most_similar_users(
        self,
        target_user: str,
        user_ratings: dict,
        method: str = "cosine",
        top_n: int = 3,
    ) -> list:
        """
        Find the top-N users most similar to the target user.

        Args:
            target_user:  ID of the user we want to find neighbours for
            user_ratings: {user_id: {item_id: rating}} nested dict
            method:       'cosine' or 'jaccard'
            top_n:        How many similar users to return

        Returns:
            List of (user_id, similarity_score) tuples, sorted descending.
        """
        if target_user not in user_ratings:
            return []

        target_vec = user_ratings[target_user]
        scores = []

        for user, ratings in user_ratings.items():
            if user == target_user:
                continue

            if method == "cosine":
                score = self.cosine_similarity(target_vec, ratings)
            elif method == "jaccard":
                score = self.jaccard_similarity(
                    set(target_vec.keys()), set(ratings.keys())
                )
            else:
                raise ValueError(f"Unknown method '{method}'. Use 'cosine' or 'jaccard'.")

            scores.append((user, round(score, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


# ── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    calc = SimilarityCalculator()

    # Cosine similarity demo
    alice = {"action": 5, "comedy": 3, "drama": 0}
    bob   = {"action": 4, "comedy": 2, "drama": 1}
    carol = {"action": 0, "comedy": 0, "drama": 5}

    print("=== Cosine Similarity ===")
    print(f"Alice vs Bob  : {calc.cosine_similarity(alice, bob):.4f}")   # High
    print(f"Alice vs Carol: {calc.cosine_similarity(alice, carol):.4f}") # Low

    # Jaccard similarity demo
    alice_items = {"sword", "shield", "potion"}
    bob_items   = {"sword", "bow", "potion"}
    carol_items = {"staff", "robe"}

    print("\n=== Jaccard Similarity ===")
    print(f"Alice vs Bob  : {calc.jaccard_similarity(alice_items, bob_items):.4f}")
    print(f"Alice vs Carol: {calc.jaccard_similarity(alice_items, carol_items):.4f}")

    # Most similar users
    user_ratings = {
        "alice": {"item1": 5, "item2": 3, "item3": 4},
        "bob":   {"item1": 4, "item2": 3, "item3": 5},
        "carol": {"item1": 1, "item2": 5, "item3": 2},
        "dave":  {"item2": 4, "item3": 4, "item4": 3},
    }
    print("\n=== Most Similar to Alice ===")
    for user, score in calc.most_similar_users("alice", user_ratings):
        print(f"  {user}: {score}")
