"""Sampled-metrics evaluator: 1 positive + 100 uniform random negatives.

Aligns with the original SASRec paper's protocol. Negatives are generated once
with a fixed seed and cached to disk so every model (SASRec, BERT4Rec via
RecBole, TIGER-style) is scored against the *identical* negative set --
otherwise cross-model comparisons are meaningless.
"""

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np

from src.data.dataset import build_eval_input
from src.eval.metrics import compute_ranks_batch, summarize


def generate_negatives(
    train: dict[int, list[int]],
    valid: dict[int, int],
    test: dict[int, int],
    n_items: int,
    n_negs: int = 100,
    seed: int = 42,
) -> dict[int, list[int]]:
    """Per user, sample n_negs items never interacted with (train+valid+test)."""
    rng = random.Random(seed)
    negatives: dict[int, list[int]] = {}
    for user in test:
        exclude = set(train.get(user, [])) | {valid[user], test[user]}
        negs = []
        while len(negs) < n_negs:
            item = rng.randint(1, n_items)
            if item not in exclude and item not in negs:
                negs.append(item)
        negatives[user] = negs
    return negatives


def save_negatives(negatives: dict[int, list[int]], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(negatives, f)


def load_negatives(path: Path) -> dict[int, list[int]]:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def evaluate_sampled(
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    train: dict[int, list[int]],
    targets: dict[int, int],
    negatives: dict[int, list[int]],
    maxlen: int,
    extra_history: dict[int, list[int]] | None = None,
    k: int = 10,
    batch_size: int = 256,
) -> dict[str, float]:
    """
    score_fn(input_batch [B, maxlen], candidates_batch [B, 1+n_negs]) -> scores [B, 1+n_negs]
    extra_history: e.g. {user: [valid_item]} appended before the target when
                   evaluating on test (so the model has seen valid at inference time).
    """
    users = list(targets.keys())
    all_ranks = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        input_batch = []
        candidate_batch = []
        for user in batch_users:
            history = train.get(user, [])
            extra = extra_history.get(user, []) if extra_history else []
            input_seq = build_eval_input(history, maxlen, extra=extra)
            input_batch.append(input_seq)
            candidate_batch.append([targets[user]] + negatives[user])

        input_batch = np.stack(input_batch)
        candidate_batch = np.array(candidate_batch)
        scores = score_fn(input_batch, candidate_batch)
        all_ranks.append(compute_ranks_batch(scores))

    ranks = np.concatenate(all_ranks)
    return summarize(ranks, k=k)
