# Recommendation Engine

A modular, well-documented recommendation engine built in pure Python (no ML libraries required). The system mirrors how platforms like Netflix and Amazon suggest items to users.

---

## Project Structure

```
recommendation_engine/
├── similarity.py        # Module 1 — cosine, Jaccard & Pearson similarity
├── candidate_gen.py     # Module 2 — collaborative, content & popular candidates
├── scorer.py            # Module 3 — weighted composite scoring & ranking
├── evaluator.py         # Module 4 — precision, recall, NDCG, F1, hit-rate, coverage
├── test.py              # Comprehensive test suite (20/20 passing)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
*(No external ML libraries needed — only the Python standard library.)*

### 2. Run all tests
```bash
python test.py
```

Expected output:
```
TOTAL: 20/20 TESTS PASSED ✅
```

---

## Module Descriptions

### Module 1 — `SimilarityCalculator` (`similarity.py`)

Measures how alike two users or items are.

| Method | Description | Returns |
|---|---|---|
| `cosine_similarity(vec_a, vec_b)` | Cosine similarity between two rating vectors (dicts) | `[0, 1]` |
| `jaccard_similarity(set_a, set_b)` | Jaccard similarity between two item sets | `[0, 1]` |
| `pearson_correlation(vec_a, vec_b)` | Pearson correlation for rating bias correction | `[-1, 1]` |
| `most_similar_users(user, ratings, method, top_n)` | Returns the top-N most similar users | List of `(user, score)` |

**When to use each:**
- **Cosine** — best when ratings vary in magnitude (e.g., user A rates 1–3, user B rates 3–5).
- **Jaccard** — best for binary data (bought / not bought, liked / not liked).
- **Pearson** — best when you want to correct for rating bias between users.

---

### Module 2 — `CandidateGenerator` (`candidate_gen.py`)

Finds items worth considering for a user, using three strategies:

| Strategy | How it works |
|---|---|
| `collaborative` | Items rated highly (≥ 3.5) by similar users that the target hasn't seen |
| `content` | Items whose metadata (attributes) are similar to the user's top-rated items |
| `popular` | Globally popular items (by number of ratings) the user hasn't rated yet |
| `all` (default) | Union of all three strategies above |

**Cold-start handling:** If a user is unknown, the generator logs a warning and automatically falls back to popularity-based candidates — no crash, graceful degradation.

```python
gen = CandidateGenerator(user_ratings, item_metadata)
candidates = gen.generate("alice", strategy="all")
```

---

### Module 3 — `RecommendationScorer` (`scorer.py`)

Ranks candidates using a weighted composite score:

```
final_score = 0.5 × collaborative_score
            + 0.3 × content_score
            + 0.2 × popularity_score
```

All sub-scores are normalised to `[0, 1]`.

| Method | Description |
|---|---|
| `calculate_score(user, item)` | Composite score for a single item — returns float in `[0, 1]` |
| `rank_candidates(user, candidates, limit)` | Score all candidates, return top-N sorted descending |
| `explain_score(user, item)` | Breakdown of each score component for transparency |
| `add_scorer(name, function, weight)` | Register a custom scoring function at runtime |

```python
scorer = RecommendationScorer(user_ratings, item_metadata)
ranked = scorer.rank_candidates("alice", candidates, limit=5)
```

Custom weights:
```python
scorer = RecommendationScorer(user_ratings, item_metadata,
                weights={"collaborative": 0.7, "content": 0.2, "popularity": 0.1})
```

---

### Module 4 — `Evaluator` (`evaluator.py`)

Checks recommendation quality against held-out ground-truth data.

| Metric | Description |
|---|---|
| **Precision@K** | Of the top-K items recommended, what fraction is relevant? |
| **Recall@K** | Of all relevant items, what fraction appears in the top-K? |
| **F1@K** | Harmonic mean of precision and recall |
| **NDCG@K** | Ranking quality — rewards relevant items appearing earlier |
| **Hit Rate@K** | 1 if at least one relevant item is in the top-K |
| **Coverage** | Fraction of the full catalogue ever recommended |

```python
evaluator = Evaluator()
results = evaluator.evaluate_all(user_results, all_items, k=5)
```

---

## Test Results

| Test Group | Tests Covered | Result |
|---|---|---|
| Similarity Calculator | Cosine, orthogonal, Jaccard, Pearson, most_similar_users | ✅ 5/5 |
| Candidate Generator | Collaborative, content, popularity, hybrid, excludes already-rated | ✅ 5/5 |
| Scorer & Ranker | Score range, rank order, descending sort, explain, add_scorer | ✅ 5/5 |
| Evaluator | Precision, recall, NDCG, evaluate_all, empty edge cases | ✅ 5/5 |
| **Total** | | **✅ 20/20** |

---

## Sample Output

```
=======================================================
  Recommendations for 'alice'  (strategy=all)
=======================================================
  Candidates generated : 6
  Rank  Item       Score   Breakdown
  --------------------------------------------------
  1     speaker    0.7900  collab=1.0  content=0.8554  pop=0.1667
  2     monitor    0.7105  collab=0.7529  content=0.7801  pop=0.5
  3     usb_hub    0.6465  collab=0.6655  content=0.8237  pop=0.3333

=== Evaluation Against Ground Truth (K=5) ===
  precision@k  : 0.4667
  recall@k     : 0.7778
  f1@k         : 0.5833
  hit_rate@k   : 1.0
  coverage     : 1.0
```

---

## Edge Cases Handled

- Empty rating vectors / item sets → similarity returns `0.0`
- Unknown user ID → falls back to popularity-based candidates
- No metadata provided → content-based scoring returns `0.0` gracefully
- Empty candidate set → returns empty list
- Zero-magnitude vectors → cosine similarity returns `0.0`
- `k=0` or empty recommended list → evaluation metrics return `0.0`

---

## Extending the Engine

- **Add a new similarity metric:** Extend `SimilarityCalculator` with a new method and pass `method=` to `most_similar_users`.
- **Add a new candidate strategy:** Add a method to `CandidateGenerator` and register it in the `strategies` dict inside `generate()`.
- **Change scoring weights:** Pass a custom `weights` dict to `RecommendationScorer`.
- **Add new metrics:** Extend `Evaluator` with methods like `mrr` or `map`.

---

## Requirements

- Python 3.10+
- No external packages required (uses `math`, `unittest` from the standard library)

Optional, for running tests with pytest:
```
pytest>=7.0
```