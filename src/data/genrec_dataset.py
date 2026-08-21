"""Training/eval tensors for the semantic-ID generative model.

Same windowing as `SASRecTrainDataset` -- one sample per user, left-padded to
`maxlen_items` -- but every item expands to its L semantic tokens, so a window
of 50 items is a 200-token sequence and one user still produces ~maxlen*L
supervised next-token predictions per epoch. That parity matters: it keeps the
per-epoch training signal comparable to SASRec's rather than quietly making the
generative model a lower-budget run.

No negative sampling here. The generative objective is a softmax over the level's
codebook (256 entries), so it is closer to a full-catalog cross-entropy than to
SASRec's BCE-against-one-negative -- which is exactly the difference flagged as
the largest unexplained effect in this repo's cross-framework SASRec comparison,
and here shows up as a deliberate part of the design rather than an accident.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import pad_left
from src.semantic_ids.vocab import SemanticIdVocab


class GenRecTrainDataset(Dataset):
    """One sample per user: (input_tokens [T], target_tokens [T]).

    target[p] is the token at grid position p+1, so the last input position
    predicts the first code of the item following the window.
    """

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        vocab: SemanticIdVocab,
        maxlen_items: int = 50,
    ):
        self.users = [u for u, seq in train_sequences.items() if len(seq) >= 2]
        self.sequences = train_sequences
        self.vocab = vocab
        self.maxlen_items = maxlen_items
        self.n_levels = vocab.n_levels

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        seq = self.sequences[self.users[idx]]

        window = seq[-(self.maxlen_items + 1) :]
        input_items = pad_left(window[:-1], self.maxlen_items)
        next_item = window[-1]

        input_tokens = self.vocab.item_tokens[np.asarray(input_items)].reshape(-1)
        # Shift by one token; the position past the window predicts the first
        # code of the next item, which is the only supervision that item gives.
        target_tokens = np.concatenate([input_tokens[1:], self.vocab.item_tokens[next_item][:1]])

        return (
            torch.from_numpy(input_tokens.astype(np.int64)),
            torch.from_numpy(target_tokens.astype(np.int64)),
        )


def build_eval_input_tokens(
    history: list[int],
    vocab: SemanticIdVocab,
    maxlen_items: int,
    extra: list[int] | None = None,
) -> np.ndarray:
    """Left-padded token window for inference -> [maxlen_items * L]."""
    full = list(history) + (extra or [])
    items = pad_left(full, maxlen_items)
    return vocab.item_tokens[np.asarray(items)].reshape(-1).astype(np.int64)


def build_eval_batch(
    users: list[int],
    train: dict[int, list[int]],
    vocab: SemanticIdVocab,
    maxlen_items: int,
    extra_history: dict[int, list[int]] | None = None,
) -> np.ndarray:
    return np.stack(
        [
            build_eval_input_tokens(
                train.get(u, []),
                vocab,
                maxlen_items,
                extra=(extra_history or {}).get(u, []),
            )
            for u in users
        ]
    )
