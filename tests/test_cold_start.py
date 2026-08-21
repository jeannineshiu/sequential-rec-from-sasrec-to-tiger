"""Cold-start bucketing: frequency counting, bucket edges, and per-bucket slicing."""

import numpy as np
import pytest

from src.eval.cold_start import (
    DEFAULT_BUCKETS,
    bucket_of,
    bucket_users,
    bucketed_metrics,
    item_train_frequency,
)


def test_frequency_counts_repeats_and_ignores_padding():
    train = {1: [2, 3, 3], 2: [3], 3: []}
    freq = item_train_frequency(train, n_items=4)

    assert freq.tolist() == [0, 0, 1, 3, 0]


def test_bucket_edges_are_contiguous_and_exhaustive():
    for frequency in range(0, 100):
        bucket_of(frequency)  # must not raise anywhere in the range

    assert bucket_of(0) == "unseen"
    assert bucket_of(1) == "tail"
    assert bucket_of(4) == "tail"
    assert bucket_of(5) == "torso"
    assert bucket_of(19) == "torso"
    assert bucket_of(20) == "head"


def test_five_core_does_not_prevent_unseen_items():
    """An item with 5 interactions can still be absent from the train split.

    All 5 of this item's interactions land in other users' valid/test slots, so
    it is genuinely unseen during training -- which is why the bucket exists.
    """
    train = {1: [9], 2: [9]}
    freq = item_train_frequency(train, n_items=10)
    assert bucket_of(int(freq[7])) == "unseen"


def test_bucket_users_partitions_every_user_exactly_once():
    users = [10, 11, 12, 13]
    targets = {10: 1, 11: 2, 12: 3, 13: 4}
    frequency = np.array([0, 0, 3, 8, 50])

    indices = bucket_users(users, targets, frequency)

    assert indices["unseen"].tolist() == [0]
    assert indices["tail"].tolist() == [1]
    assert indices["torso"].tolist() == [2]
    assert indices["head"].tolist() == [3]
    assert sum(len(v) for v in indices.values()) == len(users)


def test_bucketed_metrics_slice_the_right_users():
    users = [10, 11, 12, 13]
    targets = {10: 1, 11: 2, 12: 3, 13: 4}
    frequency = np.array([0, 0, 3, 8, 50])
    # ranks: the head-item user is the only hit.
    ranks = {"m": np.array([50, 50, 50, 0])}

    rows = {row["bucket"]: row for row in bucketed_metrics(users, ranks, targets, frequency, k=10)}

    assert rows["head"]["m"]["HR@10"] == 1.0
    assert rows["tail"]["m"]["HR@10"] == 0.0
    assert rows["overall"]["m"]["HR@10"] == 0.25
    assert rows["overall"]["n_users"] == 4


def test_empty_bucket_reports_nan_rather_than_crashing():
    users = [10]
    targets = {10: 1}
    frequency = np.array([0, 100])
    rows = {
        row["bucket"]: row
        for row in bucketed_metrics(users, {"m": np.array([0])}, targets, frequency)
    }

    assert rows["head"]["m"]["HR@10"] == 1.0
    assert np.isnan(rows["unseen"]["m"]["HR@10"])
    assert rows["unseen"]["n_users"] == 0


@pytest.mark.parametrize("label,low,high", DEFAULT_BUCKETS)
def test_every_bucket_boundary_maps_to_its_own_label(label, low, high):
    assert bucket_of(low) == label
    if high is not None:
        assert bucket_of(high) == label
