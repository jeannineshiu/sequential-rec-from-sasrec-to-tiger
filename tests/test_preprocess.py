import pandas as pd

from src.data.preprocess import build_sequences, k_core_filter, leave_one_out_split, reindex_ids


def test_k_core_filter_drops_sparse_users_and_items():
    # a dense clique of 5 users x 5 items satisfies the 5-core everywhere,
    # plus a sparse user (2 interactions) that should be dropped entirely,
    # which then drops item 99 (only interacted with by that user) below
    # the item threshold too.
    rows = []
    for u in range(1, 6):
        for i in range(1, 6):
            rows.append({"user": u, "item": i, "timestamp": i})
    rows.append({"user": 99, "item": 99, "timestamp": 0})
    rows.append({"user": 99, "item": 98, "timestamp": 1})
    df = pd.DataFrame(rows)

    filtered = k_core_filter(df, k=5)

    assert set(filtered["user"].unique()) == {1, 2, 3, 4, 5}
    assert 99 not in filtered["item"].values
    assert 98 not in filtered["item"].values


def test_reindex_ids_start_at_one_and_are_contiguous():
    df = pd.DataFrame({"user": [10, 10, 20], "item": [200, 201, 200], "timestamp": [1, 2, 3]})
    reindexed, user_map, item_map = reindex_ids(df)

    assert sorted(reindexed["user"].unique()) == [1, 2]
    assert sorted(reindexed["item"].unique()) == [1, 2]
    assert user_map[10] == 1 and user_map[20] == 2
    assert min(item_map.values()) == 1  # 0 reserved for padding


def test_leave_one_out_split_no_leakage():
    sequences = {1: [1, 2, 3, 4, 5], 2: [10, 11, 12]}
    train, valid, test = leave_one_out_split(sequences)

    assert train[1] == [1, 2, 3]
    assert valid[1] == 4
    assert test[1] == 5

    for user in test:
        assert test[user] not in train[user]
        assert valid[user] not in train[user]


def test_leave_one_out_split_drops_too_short_sequences():
    # a sequence of length < 3 can't produce non-overlapping train/valid/test
    sequences = {1: [1, 2], 2: [1, 2, 3]}
    train, valid, test = leave_one_out_split(sequences)

    assert 1 not in test
    assert 2 in test


def test_build_sequences_sorted_by_timestamp():
    df = pd.DataFrame(
        {
            "user": [1, 1, 1],
            "item": [30, 10, 20],
            "timestamp": [3, 1, 2],
        }
    )
    sequences = build_sequences(df)
    assert sequences[1] == [10, 20, 30]
