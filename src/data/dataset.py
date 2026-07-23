"""PyTorch Dataset for SASRec-style next-item training with left padding."""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


def pad_left(seq: list[int], maxlen: int) -> list[int]:
    """Left-pad with 0, or keep the most recent `maxlen` items if longer."""
    if len(seq) >= maxlen:
        return seq[-maxlen:]
    return [0] * (maxlen - len(seq)) + seq


class SASRecTrainDataset(Dataset):
    """One sample per user: (input, positive target, negative target), all
    padded/truncated to maxlen. Positions with input==0 are ignored in the loss.
    """

    def __init__(
        self,
        train_sequences: dict[int, list[int]],
        n_items: int,
        maxlen: int = 200,
        seed: int = 42,
    ):
        # SASRec needs at least 2 items per user to form one (input, target) pair.
        self.users = [u for u, seq in train_sequences.items() if len(seq) >= 2]
        self.sequences = train_sequences
        self.n_items = n_items
        self.maxlen = maxlen
        self.item_sets = {u: set(seq) for u, seq in train_sequences.items()}
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.users)

    def _sample_negative(self, exclude: set[int]) -> int:
        item = self._rng.randint(1, self.n_items)
        while item in exclude:
            item = self._rng.randint(1, self.n_items)
        return item

    def __getitem__(self, idx: int):
        user = self.users[idx]
        seq = self.sequences[user]

        # window: use up to maxlen+1 most recent items so we can shift by one
        window = seq[-(self.maxlen + 1) :]
        input_seq = window[:-1]
        target_seq = window[1:]

        neg_seq = [self._sample_negative(self.item_sets[user]) if t != 0 else 0 for t in target_seq]

        input_seq = pad_left(input_seq, self.maxlen)
        target_seq = pad_left(target_seq, self.maxlen)
        neg_seq = pad_left(neg_seq, self.maxlen)

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long),
        )


def build_eval_input(history: list[int], maxlen: int, extra: list[int] | None = None) -> np.ndarray:
    """Build a single left-padded input sequence for inference.

    `extra` items (e.g. the valid item, when evaluating on test) are appended
    to `history` before truncation, matching SASRec's original evaluation code.
    """
    full = list(history) + (extra or [])
    return np.array(pad_left(full, maxlen), dtype=np.int64)
