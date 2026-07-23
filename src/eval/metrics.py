"""Rank-based metrics shared by the sampled and full-ranking evaluators.

Convention: for each user we get a 1-D score array where index 0 is always the
score of the ground-truth item. `rank` = number of *other* items scored higher
(ties broken in favor of the ground truth, i.e. optimistic tie-breaking is not
used -- ties count against the truth, which matches common SASRec-derived
implementations).
"""

import numpy as np


def compute_rank(scores: np.ndarray) -> int:
    """scores: 1-D array, scores[0] is the ground-truth item's score."""
    true_score = scores[0]
    return int((scores[1:] > true_score).sum())


def compute_ranks_batch(scores: np.ndarray) -> np.ndarray:
    """scores: [B, C] array, scores[:, 0] is the ground-truth item's score."""
    true_scores = scores[:, 0:1]
    return (scores[:, 1:] > true_scores).sum(axis=1)


def hit_rate_at_k(ranks: np.ndarray, k: int = 10) -> float:
    return float((ranks < k).mean())


def ndcg_at_k(ranks: np.ndarray, k: int = 10) -> float:
    hits = ranks < k
    dcg = np.where(hits, 1.0 / np.log2(ranks + 2), 0.0)
    return float(dcg.mean())


def summarize(ranks: np.ndarray, k: int = 10) -> dict[str, float]:
    return {
        f"HR@{k}": hit_rate_at_k(ranks, k),
        f"NDCG@{k}": ndcg_at_k(ranks, k),
    }
