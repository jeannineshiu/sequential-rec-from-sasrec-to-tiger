"""Full-ranking evaluator: rank the target item against the entire catalog,
excluding items the user has already interacted with. This is the modern
standard complement to `sampled.py`'s paper-aligned protocol -- run both and
compare (see ablation A3 in EXECUTION_PLAN.md).
"""

from typing import Callable

import numpy as np

from src.data.dataset import build_eval_input
from src.eval.metrics import summarize


def evaluate_full_ranking(
    score_fn: Callable[[np.ndarray], np.ndarray],
    train: dict[int, list[int]],
    targets: dict[int, int],
    n_items: int,
    maxlen: int,
    extra_history: dict[int, list[int]] | None = None,
    exclude_extra: dict[int, list[int]] | None = None,
    k: int = 10,
    batch_size: int = 128,
) -> dict[str, float]:
    """
    score_fn(input_batch [B, maxlen]) -> scores [B, n_items + 1]
        (column index == item id; column 0 is the padding slot and ignored)
    extra_history: items to append to the input sequence before truncation
                   (e.g. valid item, when evaluating on test)
    exclude_extra: additional items to mask out of the ranking besides `train`
                   (e.g. the valid item itself, so it can't be "recommended"
                   again when scoring test)
    """
    users = list(targets.keys())
    all_ranks = []

    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        input_batch = []
        for user in batch_users:
            history = train.get(user, [])
            extra = extra_history.get(user, []) if extra_history else []
            input_batch.append(build_eval_input(history, maxlen, extra=extra))
        input_batch = np.stack(input_batch)

        scores = score_fn(input_batch).copy()  # [B, n_items + 1]
        scores[:, 0] = -np.inf

        for row, user in enumerate(batch_users):
            exclude = set(train.get(user, []))
            if exclude_extra:
                exclude |= set(exclude_extra.get(user, []))
            exclude.discard(targets[user])
            for item in exclude:
                scores[row, item] = -np.inf

        target_scores = scores[np.arange(len(batch_users)), [targets[u] for u in batch_users]]
        scores[np.arange(len(batch_users)), [targets[u] for u in batch_users]] = -np.inf
        ranks = (scores > target_scores[:, None]).sum(axis=1)
        all_ranks.append(ranks)

    ranks = np.concatenate(all_ranks)
    return summarize(ranks, k=k)
