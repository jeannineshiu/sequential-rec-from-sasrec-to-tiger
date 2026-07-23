"""Correctness checks for the known SASRec footguns listed in EXECUTION_PLAN.md:
causal mask direction, padding exclusion, and positional-embedding index range.
"""

import torch

from src.models.sasrec import SASRec, build_masks, sinusoidal_position_table


def test_causal_mask_upper_triangle_true():
    causal_mask, _ = build_masks(torch.tensor([[1, 2, 3, 4]]))
    expected = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.equal(causal_mask, expected)
    # position i must be allowed to attend to itself and the past, not the future
    assert causal_mask[0].tolist() == [False, True, True, True]
    assert causal_mask[3].tolist() == [False, False, False, False]


def test_key_padding_mask_matches_zeros():
    input_seqs = torch.tensor([[0, 0, 1, 2], [0, 5, 6, 7]])
    _, key_padding_mask = build_masks(input_seqs)
    expected = input_seqs == 0
    assert torch.equal(key_padding_mask, expected)


def test_no_future_leakage_in_encoder_output():
    torch.manual_seed(0)
    model = SASRec(n_items=20, maxlen=6, hidden_dim=8, num_blocks=2, num_heads=2, dropout=0.0)
    model.eval()

    seq_a = torch.tensor([[0, 0, 1, 2, 3, 4]])
    seq_b = torch.tensor([[0, 0, 1, 2, 3, 5]])  # only the last position differs

    with torch.no_grad():
        out_a = model.encode(seq_a)
        out_b = model.encode(seq_b)

    # every position except the last (which saw the changed future token) must
    # be identical -- if it isn't, the model is "peeking" at the future.
    assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out_a[:, -1, :], out_b[:, -1, :], atol=1e-6)


def test_positional_embedding_handles_full_length_sequence():
    maxlen = 10
    model = SASRec(n_items=20, maxlen=maxlen, hidden_dim=8, num_blocks=1, num_heads=1)
    model.eval()
    full_seq = torch.arange(1, maxlen + 1).unsqueeze(0)  # no padding, length == maxlen

    with torch.no_grad():
        out = model.encode(full_seq)  # must not raise an index-out-of-range error

    assert out.shape == (1, maxlen, 8)


def test_score_and_score_full_catalog_agree():
    torch.manual_seed(0)
    model = SASRec(n_items=15, maxlen=5, hidden_dim=8, num_blocks=1, num_heads=1, dropout=0.0)
    model.eval()
    input_seqs = torch.tensor([[0, 1, 2, 3, 4]])
    candidates = torch.tensor([[5, 10, 15]])

    with torch.no_grad():
        direct_scores = model.score(input_seqs, candidates)
        full_scores = model.score_full_catalog(input_seqs)

    for i, item in enumerate([5, 10, 15]):
        assert torch.allclose(direct_scores[0, i], full_scores[0, item], atol=1e-5)


def test_pos_emb_type_none_ignores_position():
    """With pos_emb_type='none', shuffling non-padding positions in a way that
    keeps the causal structure trivial (single real item) must not change that
    item's contribution -- there is no positional signal to distinguish slots."""
    torch.manual_seed(0)
    model = SASRec(
        n_items=20, maxlen=6, hidden_dim=8, num_blocks=1, num_heads=1, pos_emb_type="none"
    )
    model.eval()
    assert not hasattr(model, "pos_emb")
    assert not hasattr(model, "pos_emb_table")

    # same single real item at two different absolute slots (rest padding)
    seq_early = torch.tensor([[7, 0, 0, 0, 0, 0]])
    seq_late = torch.tensor([[0, 0, 0, 0, 0, 7]])
    with torch.no_grad():
        out_early = model.encode(seq_early)[:, 0, :]
        out_late = model.encode(seq_late)[:, -1, :]

    assert torch.allclose(out_early, out_late, atol=1e-6)


def test_pos_emb_type_sinusoidal_is_fixed_not_learnable():
    model = SASRec(
        n_items=20, maxlen=6, hidden_dim=8, num_blocks=1, num_heads=1, pos_emb_type="sinusoidal"
    )
    assert "pos_emb_table" in dict(model.named_buffers())
    assert "pos_emb_table" not in dict(model.named_parameters())
    # padding slot (index 0) must be all-zero so padded positions get no signal
    assert torch.allclose(model.pos_emb_table[0], torch.zeros(8))


def test_sinusoidal_position_table_shape_and_padding_row():
    table = sinusoidal_position_table(maxlen=10, hidden_dim=16)
    assert table.shape == (11, 16)
    assert torch.equal(table[0], torch.zeros(16))
    assert not torch.allclose(table[1], torch.zeros(16))
    # distinct positions must get distinct encodings
    assert not torch.allclose(table[1], table[2])
