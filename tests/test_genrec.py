"""Semantic-ID vocabulary, the generative model, and constrained decoding."""

import numpy as np
import pytest
import torch

from src.beam_search import beam_search, greedy_decode
from src.data.genrec_dataset import GenRecTrainDataset, build_eval_input_tokens
from src.models.genrec import GenRec
from src.semantic_ids.vocab import BOS, N_SPECIAL, PAD, SemanticIdVocab


@pytest.fixture
def vocab():
    """6 items over 2 levels of 3 codes, with one collision forced onto level 3."""
    codes = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 1],  # collides with item 3 on the first two levels
            [2, 2, 0],
            [2, 0, 0],
        ]
    )
    return SemanticIdVocab(item_ids=np.arange(1, 7), codes=codes)


@pytest.fixture
def model(vocab):
    torch.manual_seed(0)
    return GenRec(
        vocab_size=vocab.vocab_size,
        n_levels=vocab.n_levels,
        level_offsets=vocab.level_offsets,
        level_sizes=vocab.level_sizes,
        maxlen_items=4,
        hidden_dim=16,
        num_blocks=2,
        num_heads=2,
        dropout=0.0,
    )


# -- vocabulary ---------------------------------------------------------


def test_levels_occupy_disjoint_token_ranges(vocab):
    seen = set()
    for level in range(vocab.n_levels):
        start, end = vocab.level_slice(level)
        assert start >= N_SPECIAL, "level tokens must not collide with PAD/BOS"
        assert not (seen & set(range(start, end)))
        seen |= set(range(start, end))


def test_round_trip_item_to_tokens(vocab):
    for item in range(1, 7):
        assert vocab.decode_tokens(vocab.item_tokens[item]) == item


def test_padding_item_encodes_to_padding_tokens(vocab):
    assert (vocab.item_tokens[0] == PAD).all()
    assert BOS not in set(vocab.item_tokens.reshape(-1).tolist())


def test_encode_items_keeps_item_order_and_grid(vocab):
    tokens = vocab.encode_items(np.array([[0, 2, 5]]))
    assert tokens.shape == (1, 3 * vocab.n_levels)
    assert (tokens[0, : vocab.n_levels] == PAD).all()
    assert tokens[0, vocab.n_levels : 2 * vocab.n_levels].tolist() == (
        vocab.item_tokens[2].tolist()
    )


def test_non_unique_codes_are_rejected():
    codes = np.array([[0, 0], [0, 0]])
    with pytest.raises(ValueError, match="not unique"):
        SemanticIdVocab(item_ids=np.array([1, 2]), codes=codes)


def test_trie_allows_exactly_the_real_continuations(vocab):
    # items 3 and 4 share the level-1/2 prefix and differ only on level 3.
    prefix = tuple(int(t) for t in vocab.item_tokens[3][:2])
    allowed = vocab.allowed_next(prefix)
    expected = {int(vocab.item_tokens[3][2]), int(vocab.item_tokens[4][2])}
    assert set(int(a) for a in allowed) == expected


def test_trie_returns_nothing_for_an_impossible_prefix(vocab):
    bogus = (int(vocab.item_tokens[1][0]), int(vocab.item_tokens[5][1]))
    assert len(vocab.allowed_next(bogus)) == 0


# -- model --------------------------------------------------------------


def test_forward_masks_out_other_levels(model, vocab):
    tokens = torch.from_numpy(build_eval_input_tokens([1, 2], vocab, 4)).unsqueeze(0)
    logits = model(tokens)

    assert logits.shape == (1, model.maxlen_tokens, vocab.vocab_size)
    levels = model.target_levels_for(model.maxlen_tokens, tokens.device)
    for position in range(model.maxlen_tokens):
        start, end = vocab.level_slice(int(levels[position]))
        finite = torch.isfinite(logits[0, position])
        assert finite[start:end].all(), "the target level must be scorable"
        assert finite.sum() == end - start, "no other level may be scorable"


def test_target_levels_follow_the_token_grid(model):
    levels = model.target_levels_for(model.maxlen_tokens, torch.device("cpu"))
    # Position p predicts grid position p+1.
    assert levels[0].item() == 1 % model.n_levels
    assert levels[model.n_levels - 1].item() == 0


def test_score_item_tokens_is_a_proper_log_probability(model, vocab):
    """Scores must exp-sum to 1 over the whole code space, not over the items.

    The per-level softmax spreads mass across every code combination, and only
    6 of this fixture's 3x3x2 = 18 combinations are real items. So the items'
    probabilities sum to *less* than 1 -- the missing mass is what constrained
    decoding redistributes and post-hoc filtering throws away.
    """
    import itertools

    history = torch.from_numpy(build_eval_input_tokens([1, 2], vocab, 4)).unsqueeze(0)
    every_combination = torch.tensor(
        [
            [vocab.level_offsets[level] + code for level, code in enumerate(combo)]
            for combo in itertools.product(*[range(s) for s in vocab.level_sizes])
        ]
    ).unsqueeze(0)

    all_scores = model.score_item_tokens(history, every_combination)
    assert torch.exp(all_scores).sum().item() == pytest.approx(1.0, abs=1e-4)

    item_scores = model.score_item_tokens(history, torch.from_numpy(vocab.item_tokens[1:]).unsqueeze(0))
    assert item_scores.shape == (1, 6)
    assert torch.exp(item_scores).sum().item() < 1.0


def test_scoring_survives_a_completely_full_window(model, vocab):
    """A full window plus candidate tokens overflows the trained positions."""
    history = torch.from_numpy(build_eval_input_tokens([1, 2, 3, 4, 5, 6], vocab, 4)).unsqueeze(0)
    assert history.shape[1] == model.maxlen_tokens

    scores = model.score_item_tokens(history, torch.from_numpy(vocab.item_tokens[1:]).unsqueeze(0))
    assert torch.isfinite(scores).all()


def test_empty_history_does_not_produce_nan(model, vocab):
    """All-padding context: attention has no keys to attend to."""
    history = torch.from_numpy(build_eval_input_tokens([], vocab, 4)).unsqueeze(0)
    scores = model.score_item_tokens(history, torch.from_numpy(vocab.item_tokens[1:]).unsqueeze(0))
    assert torch.isfinite(scores).all()


# -- decoding -----------------------------------------------------------


def test_beam_search_only_ever_returns_real_items(model, vocab):
    """The plan's acceptance criterion: any decode result must be a legal item."""
    histories = np.stack([build_eval_input_tokens([i], vocab, 4) for i in range(1, 7)])
    items, scores = beam_search(model, vocab, torch.from_numpy(histories), beam_size=4)

    assert items.shape == (6, 4)
    for row in items:
        for item in row:
            assert 1 <= item <= 6, "beam search emitted something that is not an item"
    assert np.isfinite(scores).all()


def test_beam_search_returns_distinct_items_in_descending_score(model, vocab):
    histories = np.stack([build_eval_input_tokens([1, 2], vocab, 4)])
    items, scores = beam_search(model, vocab, torch.from_numpy(histories), beam_size=6)

    assert len(set(items[0].tolist())) == len(items[0]), "an item was returned twice"
    assert (np.diff(scores[0]) <= 1e-6).all(), "scores are not descending"


def test_beam_scores_agree_with_direct_scoring(model, vocab):
    """Decoded scores and candidate scores must be the same quantity."""
    history = torch.from_numpy(build_eval_input_tokens([1, 2], vocab, 4)).unsqueeze(0)
    items, scores = beam_search(model, vocab, history, beam_size=6)

    direct = model.score_item_tokens(
        history, torch.from_numpy(vocab.item_tokens[items[0]]).unsqueeze(0)
    )
    np.testing.assert_allclose(scores[0], direct[0].numpy(), atol=1e-4)


def test_beam_search_finds_the_true_top_item(model, vocab):
    """With a beam as wide as the catalogue, beam search must be exact."""
    history = torch.from_numpy(build_eval_input_tokens([3], vocab, 4)).unsqueeze(0)
    all_scores = model.score_item_tokens(
        history, torch.from_numpy(vocab.item_tokens[1:]).unsqueeze(0)
    )[0]
    best_item = int(all_scores.argmax()) + 1

    items, _ = beam_search(model, vocab, history, beam_size=6)
    assert items[0, 0] == best_item


def test_greedy_decode_returns_none_for_illegal_sequences(model, vocab):
    """Unconstrained decoding is allowed to walk off the catalogue."""
    history = torch.from_numpy(build_eval_input_tokens([1], vocab, 4)).unsqueeze(0)
    items, logprobs = greedy_decode(model, vocab, history)

    assert len(items) == 1
    assert items[0] is None or 1 <= items[0] <= 6
    assert np.isfinite(logprobs).all()


# -- training tensors ---------------------------------------------------


def test_train_sample_is_the_token_sequence_shifted_by_one(vocab):
    dataset = GenRecTrainDataset({7: [1, 2, 3]}, vocab, maxlen_items=4)
    inputs, targets = dataset[0]

    assert inputs.shape == targets.shape == (4 * vocab.n_levels,)
    # Window is [1, 2] with [3] as the next item; input is left-padded.
    assert inputs[-2 * vocab.n_levels :].tolist() == (
        vocab.item_tokens[1].tolist() + vocab.item_tokens[2].tolist()
    )
    assert targets[:-1].tolist() == inputs[1:].tolist()
    assert targets[-1].item() == vocab.item_tokens[3][0], "last position predicts the next item"


def test_train_targets_stay_on_their_level(vocab):
    dataset = GenRecTrainDataset({7: [1, 2, 3, 4, 5]}, vocab, maxlen_items=4)
    inputs, targets = dataset[0]

    for position, target in enumerate(targets.tolist()):
        if target == PAD:
            continue
        start, end = vocab.level_slice((position + 1) % vocab.n_levels)
        assert start <= target < end, f"target at {position} is on the wrong level"


def test_users_with_one_interaction_are_dropped(vocab):
    dataset = GenRecTrainDataset({7: [1], 8: [1, 2]}, vocab, maxlen_items=4)
    assert len(dataset) == 1


# -- cached scoring -----------------------------------------------------


@pytest.mark.parametrize("num_heads", [1, 2])
def test_cached_scoring_equals_uncached(vocab, num_heads):
    """The cache is an optimization, so it must change nothing at all."""
    torch.manual_seed(3)
    model = GenRec(
        vocab_size=vocab.vocab_size,
        n_levels=vocab.n_levels,
        level_offsets=vocab.level_offsets,
        level_sizes=vocab.level_sizes,
        maxlen_items=4,
        hidden_dim=16,
        num_blocks=2,
        num_heads=num_heads,
        dropout=0.0,
    ).eval()

    histories = torch.from_numpy(
        np.stack(
            [
                build_eval_input_tokens([1, 2], vocab, 4),
                build_eval_input_tokens([], vocab, 4),  # empty history
                build_eval_input_tokens([1, 2, 3, 4, 5, 6], vocab, 4),  # full window
            ]
        )
    )
    candidates = torch.from_numpy(np.stack([vocab.item_tokens[1:]] * 3)).long()

    uncached = model.score_item_tokens(histories, candidates)
    cached = model.score_item_tokens_cached(histories, candidates)

    torch.testing.assert_close(cached, uncached, atol=1e-5, rtol=1e-4)


def test_cached_scoring_handles_a_single_level():
    """L=1 skips the decode step entirely; it must still be a distribution."""
    codes = np.array([[0], [1], [2]])
    vocab = SemanticIdVocab(item_ids=np.arange(1, 4), codes=codes)
    torch.manual_seed(4)
    model = GenRec(
        vocab_size=vocab.vocab_size,
        n_levels=1,
        level_offsets=vocab.level_offsets,
        level_sizes=vocab.level_sizes,
        maxlen_items=4,
        hidden_dim=16,
        num_blocks=1,
        num_heads=1,
        dropout=0.0,
    ).eval()

    history = torch.from_numpy(build_eval_input_tokens([1, 2], vocab, 4)).unsqueeze(0)
    all_items = torch.from_numpy(vocab.item_tokens[1:]).unsqueeze(0)

    cached = model.score_item_tokens_cached(history, all_items)
    torch.testing.assert_close(cached, model.score_item_tokens(history, all_items), atol=1e-5, rtol=1e-4)
    assert torch.exp(cached).sum().item() == pytest.approx(1.0, abs=1e-4)
