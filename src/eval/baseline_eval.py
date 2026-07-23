"""Evaluation helpers for non-sequential baselines (Popularity, BPR-MF).

These models score by (user, item) pairs directly rather than from an input
sequence, so they don't fit the `score_fn(input_batch, ...)` interface used by
`sampled.py` / `full_ranking.py` (built for SASRec-style sequence encoders).
Same metrics, same fixed negatives file -- just a simpler per-user callable.
"""

from typing import Callable

import numpy as np

from src.eval.metrics import compute_rank, summarize


def evaluate_baseline_sampled(
    score_for_user: Callable[[int], np.ndarray],
    targets: dict[int, int],
    negatives: dict[int, list[int]],
    k: int = 10,
) -> dict[str, float]:
    """score_for_user(user) -> full score array of length n_items + 1 (index 0 unused)."""
    ranks = []
    for user, target in targets.items():
        scores_full = score_for_user(user)
        candidates = [target] + negatives[user]
        ranks.append(compute_rank(scores_full[candidates]))
    return summarize(np.array(ranks), k=k)


def evaluate_baseline_full_ranking(
    score_for_user: Callable[[int], np.ndarray],
    train: dict[int, list[int]],
    targets: dict[int, int],
    exclude_extra: dict[int, list[int]] | None = None,
    k: int = 10,
) -> dict[str, float]:
    ranks = []
    for user, target in targets.items():
        scores = score_for_user(user).copy()
        scores[0] = -np.inf
        exclude = set(train.get(user, []))
        if exclude_extra:
            exclude |= set(exclude_extra.get(user, []))
        exclude.discard(target)
        for item in exclude:
            scores[item] = -np.inf
        target_score = scores[target]
        scores[target] = -np.inf
        ranks.append(int((scores > target_score).sum()))
    return summarize(np.array(ranks), k=k)
