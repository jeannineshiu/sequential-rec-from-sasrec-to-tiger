"""Hand-verified correctness tests for the sampled and full-ranking evaluators.

A 3-user toy dataset with a deterministic scoring function (score == item id)
lets us hand-compute the expected HR@k / NDCG@k and check the evaluator
matches exactly.
"""

import numpy as np
import pytest

from src.eval.full_ranking import evaluate_full_ranking
from src.eval.metrics import compute_rank, hit_rate_at_k, ndcg_at_k
from src.eval.sampled import evaluate_sampled


def test_compute_rank_top_and_bottom():
    assert compute_rank(np.array([5, 1, 2, 3, 4])) == 0
    assert compute_rank(np.array([1, 5, 4, 3, 2])) == 4


def test_hit_rate_and_ndcg_known_values():
    ranks = np.array([0, 3, 1])
    assert hit_rate_at_k(ranks, k=2) == pytest.approx(2 / 3)
    expected_ndcg = (1.0 + 0.0 + 1 / np.log2(3)) / 3
    assert ndcg_at_k(ranks, k=2) == pytest.approx(expected_ndcg)


def _score_by_item_id(input_batch, candidate_batch):
    # deterministic "model": score of an item == its id
    return candidate_batch.astype(float)


def test_evaluate_sampled_matches_hand_computation():
    train = {1: [10, 11, 12], 2: [20, 21], 3: [30]}
    targets = {1: 5, 2: 2, 3: 6}
    negatives = {1: [1, 2, 3], 2: [9, 8, 7], 3: [10, 1, 2]}

    metrics = evaluate_sampled(_score_by_item_id, train, targets, negatives, maxlen=5, k=2)

    assert metrics["HR@2"] == pytest.approx(2 / 3)
    expected_ndcg = (1.0 + 0.0 + 1 / np.log2(3)) / 3
    assert metrics["NDCG@2"] == pytest.approx(expected_ndcg)


def _score_full_catalog_by_item_id(input_batch, n_items=5):
    batch_size = input_batch.shape[0]
    row = np.arange(n_items + 1, dtype=float)
    return np.tile(row, (batch_size, 1))


def test_evaluate_full_ranking_excludes_train_items():
    train = {1: [2, 3]}
    targets = {1: 4}
    n_items = 5

    metrics = evaluate_full_ranking(
        lambda batch: _score_full_catalog_by_item_id(batch, n_items),
        train,
        targets,
        n_items=n_items,
        maxlen=5,
        k=2,
    )
    # competitors after excluding {2,3}: item 1 (score 1) and item 5 (score 5).
    # target score = 4, only item 5 beats it -> rank = 1 -> hit at k=2.
    assert metrics["HR@2"] == pytest.approx(1.0)
    assert metrics["NDCG@2"] == pytest.approx(1 / np.log2(1 + 2))


def test_evaluate_full_ranking_exclude_extra_removes_more_competitors():
    train = {1: [2, 3]}
    targets = {1: 4}
    n_items = 5

    metrics = evaluate_full_ranking(
        lambda batch: _score_full_catalog_by_item_id(batch, n_items),
        train,
        targets,
        n_items=n_items,
        maxlen=5,
        exclude_extra={1: [5]},
        k=1,
    )
    # now item 5 is also excluded -> only item 1 (score 1) competes, rank = 0 -> hit.
    assert metrics["HR@1"] == pytest.approx(1.0)
