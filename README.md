# Recommendation Engine

A modular, well-documented recommendation engine built in pure Python (no ML libraries required). The system mirrors how platforms like Netflix and Amazon suggest items to users.

---

## Project Structure

```
recommendation_engine/
├── modules/
│   ├── similarity_calculator.py   # Module 1 — cosine & Jaccard similarity
│   ├── candidate_generator.py     # Module 2 — collaborative, content & popular candidates
│   ├── scorer.py                  # Module 3 — weighted composite scoring & ranking
│   └── evaluator.py               # Module 4 — precision, recall, F1, hit-rate, coverage
├── data/
│   └── sample_data.py             # Sample user ratings, item metadata & ground truth
├── tests/
│   └── test_all_modules.py        # Full unit + integration test suite
├── recommendation_engine.py       # Orchestrator — ties all modules together
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

### 2. Run the full demo
```bash
python recommendation_engine.py
```

### 3. Run all tests
```bash
python -m pytest tests/test_all_modules.py -v
# or
python tests/test_all_modules.py
```

---

## Module Descriptions

### Module 1 — `SimilarityCalculator`

**File:** `modules/similarity_calculator.py`

Measures how alike two users or items are.

| Method | Description |
|---|---|
| `cosine_similarity(vec_a, vec_b)` | Cosine similarity between two rating vectors (dicts). Returns `[0, 1]`. |
| `jaccard_similarity(set_a, set_b)` | Jaccard similarity between two item sets. Returns `[0, 1]`. |
| `most_similar_users(user, ratings, method, top_n)` | Returns the top-N most similar users using either method. |

**When to use each:**
- **Cosine** — best when ratings vary in magnitude (e.g., user A rates 1–3, user B rates 3–5).
- **Jaccard** — best for binary data (bought / not bought, liked / not liked).

---

### Module 2 — `CandidateGenerator`

**File:** `modules/candidate_generator.py`

Finds items worth considering for a user, using three strategies:

| Strategy | How it works |
|---|---|
| `collaborative` | Items rated highly (≥ 3.5) by similar users that the target hasn't seen. |
| `content` | Items whose metadata (attributes) are similar to the user's top-rated items. |
| `popular` | Globally popular items (by number of ratings) the user hasn't rated yet. |
| `all` | Union of all three strategies (default). |

```python
gen = CandidateGenerator(user_ratings, item_metadata)
candidates = gen.generate("alice", strategy="all")
```

---

### Module 3 — `Scorer`

**File:** `modules/scorer.py`

Ranks candidates using a weighted composite score:

```
final_score = 0.5 × collaborative_score
            + 0.3 × content_score
            + 0.2 × popularity_score
```

All sub-scores are normalised to `[0, 1]`.

| Method | Description |
|---|---|
| `score_item(user, item)` | Composite score for a single item. |
| `rank_candidates(user, candidates, top_n)` | Rank all candidates, return top-N. |
| `explain_score(user, item)` | Breakdown of each score component. |

```python
scorer = Scorer(user_ratings, item_metadata)
ranked = scorer.rank_candidates("alice", candidates, top_n=5)
```

Custom weights:
```python
scorer = Scorer(user_ratings, item_metadata,
                weights={"collaborative": 0.7, "content": 0.2, "popularity": 0.1})
```

---

### Module 4 — `Evaluator`

**File:** `modules/evaluator.py`

Checks recommendation quality against held-out ground-truth data.

| Metric | Description |
|---|---|
| **Precision@K** | Of the top-K items recommended, what fraction is relevant? |
| **Recall@K** | Of all relevant items, what fraction appears in the top-K? |
| **F1@K** | Harmonic mean of precision and recall. |
| **Hit Rate@K** | 1 if at least one relevant item is in the top-K. |
| **Coverage** | Fraction of the full catalogue ever recommended. |

```python
evaluator = Evaluator()
results = evaluator.evaluate_all(user_results, all_items, k=5)
```

---

## Using the Orchestrator

```python
from recommendation_engine import RecommendationEngine
from data.sample_data import USER_RATINGS, ITEM_METADATA, GROUND_TRUTH

engine = RecommendationEngine(USER_RATINGS, ITEM_METADATA)

# Get recommendations
recs = engine.recommend("alice", top_n=5, verbose=True)

# Find similar users
similar = engine.find_similar_users("alice", top_n=3)

# Add a new user at runtime (dynamic update)
engine.add_user("zara", {"laptop": 5, "headphones": 4})
recs = engine.recommend("zara", top_n=5)

# Evaluate the system
metrics = engine.evaluate(GROUND_TRUTH, top_n=5, k=5)
```

---

## Sample Output

```
=====================================================
  Recommendations for 'alice'  (strategy=all)
=====================================================
  Candidates generated : 6
  Rank  Item            Score  Breakdown
  --------------------------------------------------
  1     monitor         0.4821  collab=0.72  content=0.35  pop=0.83
  2     speaker         0.3614  collab=0.45  content=0.52  pop=0.67
  3     webcam          0.3102  collab=0.41  content=0.28  pop=0.67
  ...

=== Evaluation Results (K=5) ===
  precision@k    : 0.3667
  recall@k       : 0.4444
  f1@k           : 0.3966
  hit_rate@k     : 0.8333
  coverage       : 0.7
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
- **Change scoring weights:** Pass a custom `weights` dict to `Scorer` or `RecommendationEngine`.
- **Add new metrics:** Extend `Evaluator` with methods like `ndcg_at_k` or `mrr`.

---

## Requirements

- Python 3.10+
- No external packages required (uses `math`, `unittest` from the standard library)

Optional, for running tests with pytest:
```
pytest>=7.0
```
