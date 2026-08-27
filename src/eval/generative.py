"""Evaluators for a model that *generates* its recommendations.

The sampled protocol needs no new evaluator: `GenRec.score_item_tokens_cached`
gives a log-probability for any candidate item, so the existing
`evaluate_sampled` runs unchanged and the numbers land on the same scale as
every SASRec row in the repo. That is the whole reason scoring was built
alongside decoding.

Full ranking is different. Scoring all 12,101 items per user with a decoder is
far more expensive than one matrix product, so the generative model produces its
top-K by constrained beam search instead -- which is TIGER's own protocol, and
carries an approximation the dot-product models do not have: an item the beam
never reaches counts as a miss even if exhaustive scoring would have ranked it
top-10. Widening `beam_size` bounds how much that costs.
"""

from typing import Callable

import numpy as np
import torch

import time

from src.beam_search import batched_beam_search, greedy_decode
from src.data.genrec_dataset import build_eval_batch
from src.eval.metrics import summarize
from src.models.genrec import GenRec
from src.semantic_ids.vocab import SemanticIdVocab


def make_sampled_score_fn(
    model: GenRec, vocab: SemanticIdVocab, device: torch.device, chunk_size: int = 12800
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """-> score_fn(item_input_batch [B, maxlen_items], candidates [B, C]) -> [B, C].

    Takes *item* ids, as `evaluate_sampled` produces them, and expands to tokens
    here so the shared evaluator needs no knowledge of semantic IDs.
    """
    model.eval()

    def score_fn(input_batch: np.ndarray, candidate_batch: np.ndarray) -> np.ndarray:
        history = torch.from_numpy(
            vocab.item_tokens[input_batch].reshape(len(input_batch), -1)
        ).long()
        cand_tokens = torch.from_numpy(vocab.item_tokens[candidate_batch]).long()

        scores = []
        # Cached scoring keeps the [B, C, L, T] attention tensor in memory,
        # so rows are still chunked -- just far fewer, far cheaper passes.
        rows_per_chunk = max(1, chunk_size // candidate_batch.shape[1])
        for start in range(0, len(history), rows_per_chunk):
            h = history[start : start + rows_per_chunk].to(device)
            c = cand_tokens[start : start + rows_per_chunk].to(device)
            scores.append(model.score_item_tokens_cached(h, c).cpu().numpy())
        return np.concatenate(scores)

    return score_fn


def generative_full_ranking_ranks(
    model: GenRec,
    vocab: SemanticIdVocab,
    train: dict[int, list[int]],
    targets: dict[int, int],
    maxlen_items: int,
    device: torch.device,
    extra_history: dict[int, list[int]] | None = None,
    exclude_extra: dict[int, list[int]] | None = None,
    k: int = 10,
    beam_size: int = 20,
    batch_size: int = 128,
) -> tuple[list[int], np.ndarray]:
    """Beam-search top-K against the whole catalogue, returning per-user ranks.

    Already-seen items are dropped from the returned list *after* decoding
    (the model is free to generate them), which matches how the dot-product
    full-ranking evaluator masks history before ranking.
    """
    users = list(targets.keys())
    history_tokens = build_eval_batch(users, train, vocab, maxlen_items, extra_history)

    # Beam wider than k, since filtering seen items shortens the list.
    items, _ = batched_beam_search(
        model,
        vocab,
        history_tokens,
        device,
        beam_size=beam_size,
        n_return=beam_size,
        batch_size=batch_size,
    )

    ranks = np.full(len(users), k, dtype=np.int64)  # k == "not retrieved" == a miss
    for row, user in enumerate(users):
        seen = set(train.get(user, []))
        if exclude_extra:
            seen |= set(exclude_extra.get(user, []))
        seen.discard(targets[user])

        rank = 0
        for item in items[row]:
            if item == 0 or item in seen:
                continue
            if item == targets[user]:
                ranks[row] = rank
                break
            rank += 1
            if rank >= k:
                break

    return users, ranks


def evaluate_generative_full_ranking(
    model: GenRec,
    vocab: SemanticIdVocab,
    train: dict[int, list[int]],
    targets: dict[int, int],
    maxlen_items: int,
    device: torch.device,
    extra_history: dict[int, list[int]] | None = None,
    exclude_extra: dict[int, list[int]] | None = None,
    k: int = 10,
    beam_size: int = 20,
    batch_size: int = 128,
) -> dict[str, float]:
    _, ranks = generative_full_ranking_ranks(
        model,
        vocab,
        train,
        targets,
        maxlen_items=maxlen_items,
        device=device,
        extra_history=extra_history,
        exclude_extra=exclude_extra,
        k=k,
        beam_size=beam_size,
        batch_size=batch_size,
    )
    metrics = summarize(ranks, k=k)
    metrics["beam_size"] = float(beam_size)
    return metrics


def decode_legality(
    model: GenRec,
    vocab: SemanticIdVocab,
    train: dict[int, list[int]],
    targets: dict[int, int],
    maxlen_items: int,
    device: torch.device,
    extra_history: dict[int, list[int]] | None = None,
    batch_size: int = 256,
) -> dict[str, float]:
    """How often does *unconstrained* greedy decoding produce a real item?

    This is the number that says whether the Trie is load-bearing or decorative.
    Step 1 of the plan's de-risking sequence: if greedy decoding is already
    ~100% legal, constrained decoding is a correctness guarantee rather than a
    quality intervention -- worth knowing which of the two it is.
    """
    users = list(targets.keys())
    history_tokens = build_eval_batch(users, train, vocab, maxlen_items, extra_history)

    n_legal = 0
    n_hit = 0
    for start in range(0, len(history_tokens), batch_size):
        chunk = torch.from_numpy(history_tokens[start : start + batch_size]).long().to(device)
        items, _ = greedy_decode(model, vocab, chunk)
        for offset, item in enumerate(items):
            if item is None:
                continue
            n_legal += 1
            if item == targets[users[start + offset]]:
                n_hit += 1

    return {
        "greedy_legal_rate": n_legal / len(users),
        "greedy_HR@1": n_hit / len(users),
    }


def log_prior(frequency: np.ndarray) -> np.ndarray:
    """Add-one smoothed log P(item) over the training split; index 0 unused."""
    counts = frequency.astype(np.float64) + 1.0
    counts[0] = 0.0
    return np.log(counts / counts.sum(), out=np.full_like(counts, -np.inf), where=counts > 0)


class NaNScoreError(RuntimeError):
    """The model produced NaN scores for at least one user.

    Its own type because the caller has to be able to tell this apart from an
    ordinary crash. A NaN score is not a failure that announces itself: rank is
    computed as `(others > target).sum()`, and every comparison against NaN is
    False, so a NaN *target* score used to come back as rank 0 -- a top-1 hit
    for exactly the users the model failed on. That is the one error direction
    nothing downstream can catch, because it makes the metrics better rather
    than worse.
    """


@torch.no_grad()
def exhaustive_ranks(
    model: GenRec,
    vocab: SemanticIdVocab,
    train: dict[int, list[int]],
    targets: dict[int, int],
    maxlen_items: int,
    device: torch.device,
    alphas: list[float],
    prior: np.ndarray,
    extra_history: dict[int, list[int]] | None = None,
    user_batch: int = 64,
    cand_chunk: int = 2048,
    topk: int = 10,
    on_nan: str = "raise",
    attn_budget: int = 100_000_000,
) -> tuple[list[int], dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Score every catalogue item for every user.

    Returns per-alpha rank arrays and the per-alpha top-k item matrices. The
    top-k matrices are what any recommendation-diversity question has to be
    answered from: a beam-ranked top-k only contains items the beam kept, so
    it understates coverage by construction.

    `on_nan` decides what happens when the model returns NaN for a user:
    "raise" (the default) stops with a `NaNScoreError` naming the users, and
    "miss" scores them as a definite miss and reports the count on the way out.
    There is deliberately no option to keep the old behaviour, which was to let
    the comparison arithmetic silently read a NaN as a rank-0 hit. Whatever
    produces the NaN is a bug somewhere else; turning it into an inflated
    metric is a bug here.

    `attn_budget` caps the scorer's transient attention tensor, which is
    [users, candidates, heads, L-1, T+L-1] elements -- 313M of them at the
    nominal 64 x 2048 on ML-1M's 796-token histories. Past roughly 2e8 the MPS
    backend returns NaN for part of the batch rather than failing, and does it
    only when an earlier model has already fragmented its allocator. `cand_chunk`
    stays the ceiling; the budget lowers it when the histories are long. Beauty's
    196-token histories leave the chunk at 2048, so its tables do not move.
    """
    if on_nan not in ("raise", "miss"):
        raise ValueError(f"on_nan must be 'raise' or 'miss', got {on_nan!r}")
    users = list(targets.keys())
    n_items = len(vocab.item_ids)
    all_tokens = torch.from_numpy(vocab.item_tokens).long().to(device)  # [n_items+1, L]
    prior_t = torch.from_numpy(prior).float().to(device)
    n_heads = model.attn_layers[0].num_heads
    # The blocks a previous model left behind are what fragment the allocator
    # into the state where the NaN appears -- scoring GenRec straight after
    # SASRec reproduces it, scoring GenRec alone does not.
    if device.type == "mps":
        torch.mps.empty_cache()

    ranks = {alpha: np.empty(len(users), dtype=np.int64) for alpha in alphas}
    # The top-k items themselves, not just how many were distinct: the
    # popularity profile of what gets recommended is read off these.
    recommended = {alpha: np.empty((len(users), topk), dtype=np.int64) for alpha in alphas}
    nan_users: list[int] = []
    start_time = time.time()

    for start in range(0, len(users), user_batch):
        chunk_users = users[start : start + user_batch]
        history = (
            torch.from_numpy(
                build_eval_batch(chunk_users, train, vocab, maxlen_items, extra_history)
            )
            .long()
            .to(device)
        )
        cache = model.build_cache(history)

        # Candidates are scored independently, so narrowing the chunk changes
        # only how the same work is dispatched -- never the numbers, except for
        # the float noise any change of kernel shape carries.
        span = cache["length"] + vocab.n_levels - 1
        per_candidate = len(chunk_users) * n_heads * (vocab.n_levels - 1) * span
        chunk = max(1, min(cand_chunk, attn_budget // max(1, per_candidate)))

        scores = torch.empty(len(chunk_users), n_items + 1, device=device)
        scores[:, 0] = -float("inf")
        for c0 in range(1, n_items + 1, chunk):
            c1 = min(c0 + chunk, n_items + 1)
            candidates = all_tokens[c0:c1].unsqueeze(0).expand(len(chunk_users), -1, -1)
            scores[:, c0:c1] = model.score_with_cache(cache, candidates)

        # Mask history exactly as the dot-product full-ranking evaluator does.
        for row, user in enumerate(chunk_users):
            seen = set(train.get(user, []))
            if extra_history:
                seen |= set(extra_history.get(user, []))
            seen.discard(targets[user])
            if seen:
                scores[row, torch.tensor(sorted(seen), device=device)] = -float("inf")

        # Trap NaN before any of it reaches the ranking arithmetic. A whole row
        # is condemned by a single NaN, not just a NaN in the target column: a
        # NaN *candidate* never counts as beating the target either, so it
        # understates the rank of a user whose own score was fine.
        nan_rows = torch.isnan(scores).any(dim=1)
        if nan_rows.any():
            offsets = nan_rows.nonzero(as_tuple=True)[0].cpu().numpy()
            bad = [chunk_users[int(o)] for o in offsets]
            nan_users.extend(bad)
            if on_nan == "raise":
                raise NaNScoreError(
                    f"{len(bad)} of {len(chunk_users)} users in the batch starting at "
                    f"{start} scored NaN (first: user {bad[0]}, history "
                    f"{len(train.get(bad[0], []))} items). Ranking these would report "
                    "them as rank-0 hits. Fix the scorer, or pass on_nan='miss' to "
                    "score them as misses on purpose."
                )

        target_idx = torch.tensor([targets[u] for u in chunk_users], device=device)
        for alpha in alphas:
            if alpha:
                adjusted = scores - alpha * prior_t.unsqueeze(0)
                # Index 0 is padding, not an item: its score is -inf and so is
                # its prior, and -inf - (-inf) is NaN. `beaten` shrugs that off,
                # but topk sorts NaN above every real score, so left alone the
                # padding id leads every debiased user's top-k. Restore the
                # sentinel rather than widen the NaN guard to accept it.
                adjusted[:, 0] = -float("inf")
            else:
                adjusted = scores
            target_scores = adjusted[torch.arange(len(chunk_users), device=device), target_idx]
            beaten = (adjusted > target_scores.unsqueeze(1)).sum(dim=1)
            ranks[alpha][start : start + len(chunk_users)] = beaten.cpu().numpy()
            top = torch.topk(adjusted, k=topk, dim=1).indices.cpu().numpy()
            recommended[alpha][start : start + len(chunk_users)] = top
            if nan_rows.any():  # on_nan == "miss"; rank n_items is past every k
                ranks[alpha][start + offsets] = n_items
                recommended[alpha][start + offsets] = 0

        done = start + len(chunk_users)
        if done % (user_batch * 20) == 0 or done == len(users):
            elapsed = time.time() - start_time
            print(f"  {done}/{len(users)} users, {elapsed:.0f}s elapsed", flush=True)

    if nan_users:  # only reachable under on_nan="miss"
        print(
            f"  WARNING: {len(nan_users)}/{len(users)} users scored NaN and were "
            f"counted as misses (first: {nan_users[:5]})",
            flush=True,
        )

    return users, ranks, recommended
