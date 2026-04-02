"""
Sample data for the Recommendation Engine
------------------------------------------
  USER_RATINGS  : {user_id: {item_id: rating (1-5)}}
  ITEM_METADATA : {item_id: {attribute: value}}  — used by content-based module
  GROUND_TRUTH  : {user_id: set_of_items_they_would_like}  — used by Evaluator
"""

# ── User–Item Rating Matrix ──────────────────────────────────────────────────
USER_RATINGS: dict = {
    "alice": {
        "laptop":       5,
        "headphones":   4,
        "keyboard":     3,
        "mouse":        4,
    },
    "bob": {
        "laptop":       4,
        "headphones":   5,
        "monitor":      4,
        "webcam":       3,
    },
    "carol": {
        "keyboard":     5,
        "mouse":        5,
        "mousepad":     4,
        "usb_hub":      3,
    },
    "dave": {
        "laptop":       3,
        "monitor":      5,
        "webcam":       4,
        "led_strip":    2,
    },
    "eve": {
        "headphones":   4,
        "speaker":      5,
        "usb_hub":      4,
        "led_strip":    3,
    },
    "frank": {
        "laptop":       5,
        "keyboard":     4,
        "monitor":      3,
        "mousepad":     3,
    },
}

# ── Item Metadata (for Content-Based Filtering) ──────────────────────────────
# Attributes are one-hot encoded or scored numerically.
ITEM_METADATA: dict = {
    "laptop":     {"electronics": 1, "portable": 1, "input": 0, "audio": 0, "display": 0, "price_tier": 3},
    "headphones": {"electronics": 1, "portable": 1, "input": 0, "audio": 1, "display": 0, "price_tier": 2},
    "keyboard":   {"electronics": 1, "portable": 0, "input": 1, "audio": 0, "display": 0, "price_tier": 1},
    "mouse":      {"electronics": 1, "portable": 0, "input": 1, "audio": 0, "display": 0, "price_tier": 1},
    "monitor":    {"electronics": 1, "portable": 0, "input": 0, "audio": 0, "display": 1, "price_tier": 2},
    "webcam":     {"electronics": 1, "portable": 0, "input": 1, "audio": 0, "display": 0, "price_tier": 1},
    "speaker":    {"electronics": 1, "portable": 1, "input": 0, "audio": 1, "display": 0, "price_tier": 2},
    "mousepad":   {"electronics": 0, "portable": 0, "input": 1, "audio": 0, "display": 0, "price_tier": 0},
    "usb_hub":    {"electronics": 1, "portable": 0, "input": 0, "audio": 0, "display": 0, "price_tier": 1},
    "led_strip":  {"electronics": 1, "portable": 0, "input": 0, "audio": 0, "display": 1, "price_tier": 0},
}

# ── Ground Truth for Evaluation ───────────────────────────────────────────────
# Items each user would like (withheld from training data, used for testing).
GROUND_TRUTH: dict = {
    "alice": {"monitor", "webcam", "speaker"},
    "bob":   {"keyboard", "mousepad", "speaker"},
    "carol": {"headphones", "speaker", "led_strip"},
    "dave":  {"laptop", "keyboard", "usb_hub"},
    "eve":   {"laptop", "monitor", "mousepad"},
    "frank": {"headphones", "speaker", "webcam"},
}

# All items in the catalogue
ALL_ITEMS: set = set(ITEM_METADATA.keys())
