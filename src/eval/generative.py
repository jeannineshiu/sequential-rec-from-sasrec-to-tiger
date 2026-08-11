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
) -> tuple[list[int], dict[float, np.ndarray], dict[float, int]]:
    """Score every catalogue item for every user; return per-alpha rank arrays."""
    users = list(targets.keys())
    n_items = len(vocab.item_ids)
    all_tokens = torch.from_numpy(vocab.item_tokens).long().to(device)  # [n_items+1, L]
    prior_t = torch.from_numpy(prior).float().to(device)

    ranks = {alpha: np.empty(len(users), dtype=np.int64) for alpha in alphas}
    # Distinct items appearing in any top-10: the direct measure of whether
    # debiasing actually reverses the recommendation-diversity collapse.
    recommended = {alpha: set() for alpha in alphas}
    start_time = time.time()

    for start in range(0, len(users), user_batch):
        chunk_users = users[start : start + user_batch]
        history = torch.from_numpy(
            build_eval_batch(chunk_users, train, vocab, maxlen_items, extra_history)
        ).long().to(device)
        cache = model.build_cache(history)

        scores = torch.empty(len(chunk_users), n_items + 1, device=device)
        scores[:, 0] = -float("inf")
        for c0 in range(1, n_items + 1, cand_chunk):
            c1 = min(c0 + cand_chunk, n_items + 1)
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

        target_idx = torch.tensor([targets[u] for u in chunk_users], device=device)
        for alpha in alphas:
            adjusted = scores - alpha * prior_t.unsqueeze(0) if alpha else scores
            target_scores = adjusted[torch.arange(len(chunk_users), device=device), target_idx]
            beaten = (adjusted > target_scores.unsqueeze(1)).sum(dim=1)
            ranks[alpha][start : start + len(chunk_users)] = beaten.cpu().numpy()
            top = torch.topk(adjusted, k=10, dim=1).indices.cpu().numpy()
            recommended[alpha].update(int(i) for i in np.unique(top))

        done = start + len(chunk_users)
        if done % (user_batch * 20) == 0 or done == len(users):
            elapsed = time.time() - start_time
            print(f"  {done}/{len(users)} users, {elapsed:.0f}s elapsed", flush=True)

    return users, ranks, {a: len(v) for a, v in recommended.items()}
