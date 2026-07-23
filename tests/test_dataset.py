from src.data.dataset import SASRecTrainDataset, build_eval_input, pad_left


def test_pad_left_pads_short_sequence():
    assert pad_left([1, 2, 3], 5) == [0, 0, 1, 2, 3]


def test_pad_left_truncates_to_most_recent():
    assert pad_left([1, 2, 3, 4, 5], 3) == [3, 4, 5]


def test_build_eval_input_appends_extra_before_truncation():
    result = build_eval_input([1, 2, 3], maxlen=5, extra=[4])
    assert result.tolist() == [0, 1, 2, 3, 4]


def test_train_dataset_shapes_and_shift():
    train_sequences = {1: [10, 11, 12, 13], 2: [20, 21]}
    ds = SASRecTrainDataset(train_sequences, n_items=100, maxlen=4, seed=0)
    assert len(ds) == 2

    input_seq, target_seq, neg_seq = ds[0]
    assert input_seq.shape == (4,)
    assert target_seq.shape == (4,)
    assert neg_seq.shape == (4,)
    # target is input shifted by one position
    nonzero_input = input_seq[input_seq != 0].tolist()
    nonzero_target = target_seq[target_seq != 0].tolist()
    assert nonzero_input == [10, 11, 12]
    assert nonzero_target == [11, 12, 13]


def test_train_dataset_negatives_never_in_users_history():
    train_sequences = {1: list(range(1, 10))}
    ds = SASRecTrainDataset(train_sequences, n_items=10, maxlen=8, seed=1)
    _, _, neg_seq = ds[0]
    user_items = set(train_sequences[1])
    for n in neg_seq.tolist():
        if n != 0:
            assert n not in user_items


def test_train_dataset_excludes_users_with_short_sequences():
    train_sequences = {1: [10], 2: [20, 21, 22]}
    ds = SASRecTrainDataset(train_sequences, n_items=100, maxlen=4, seed=0)
    assert ds.users == [2]


def test_popularity_negative_sampling_never_in_users_history():
    train_sequences = {1: [1, 2, 3, 4, 5], 2: [6, 7, 8]}
    ds = SASRecTrainDataset(
        train_sequences, n_items=10, maxlen=8, seed=1, neg_sampling="popularity"
    )
    for idx in range(len(ds)):
        _, _, neg_seq = ds[idx]
        user_items = set(train_sequences[ds.users[idx]])
        for n in neg_seq.tolist():
            if n != 0:
                assert n not in user_items


def test_popularity_negative_sampling_favors_frequent_items():
    # item 1 appears 4 times (frequent), item 8 appears once (rare). Neither is
    # in user 3's own history, so both are valid negatives -- popularity
    # sampling should pick item 1 far more often than item 8.
    train_sequences = {
        1: [1, 2],
        2: [1, 3],
        3: [4, 5],  # user 3's history excludes items 1 and 8
        4: [1, 6],
        5: [1, 7],
        6: [8, 2],
    }
    ds = SASRecTrainDataset(train_sequences, n_items=9, maxlen=4, seed=0, neg_sampling="popularity")
    user_items = set(train_sequences[3])
    samples = [ds._sample_negative(user_items) for _ in range(500)]
    assert samples.count(1) > samples.count(8) * 3


def test_invalid_neg_sampling_raises():
    import pytest

    with pytest.raises(ValueError):
        SASRecTrainDataset({1: [1, 2]}, n_items=10, neg_sampling="bogus")
