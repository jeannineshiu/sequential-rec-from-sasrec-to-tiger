"""Constrained beam search over semantic ID tokens (Week 5 Day 5-7, step 2).

A generative recommender can emit any token sequence, and most sequences are not
items. Two ways to deal with that, and the plan runs both so the difference is
measurable rather than assumed:

  greedy_decode  -- generate freely, filter illegal results afterwards. Simple,
                    and it quantifies how often the model goes off-catalog.
  beam_search    -- restrict each step to tokens the Trie says continue toward a
                    real item, so every beam is legal by construction and the
                    beam is never spent on sequences that cannot be redeemed.

Both return items ranked by total log-probability, the same quantity
`GenRec.score_item_tokens` computes, so decoded lists and candidate scores are
on one scale -- asserted in the tests rather than assumed.

Both also run off `GenRec.build_cache`, so the history is encoded once per row
instead of once per beam per level.
"""

import numpy as np
import torch

from src.models.genrec import GenRec
from src.semantic_ids.vocab import SemanticIdVocab


def _level_logprobs(model: GenRec, cache: dict, prefixes: torch.Tensor, level: int) -> torch.Tensor:
    """Log-probs of the level-`level` token for each beam -> [B, K, V].

    prefixes: [B, K, level] tokens generated so far (empty at level 0).
    """
    head = model.token_emb.weight.T
    if level == 0:
        logits = cache["last_hidden"] @ head  # [B, V]
        logits = logits.unsqueeze(1).expand(-1, prefixes.shape[1], -1)
    else:
        hidden = model._decode_with_cache(cache, prefixes)[:, :, -1, :]  # [B, K, D]
        logits = hidden @ head
    logits = logits.masked_fill(~model.level_legal[level].view(1, 1, -1), float("-inf"))
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def greedy_decode(
    model: GenRec,
    vocab: SemanticIdVocab,
    history_tokens: torch.Tensor,
) -> tuple[list[int | None], np.ndarray]:
    """Unconstrained argmax decoding of one item per row.

    -> (item ids, one per row, None where the emitted code tuple is not an item;
        total log-probability per row)
    """
    model.eval()
    cache = model.build_cache(history_tokens)
    B = history_tokens.shape[0]

    prefixes = torch.zeros(B, 1, 0, dtype=torch.long, device=history_tokens.device)
    logprob_total = torch.zeros(B, device=history_tokens.device)

    for level in range(vocab.n_levels):
        logprobs = _level_logprobs(model, cache, prefixes, level)[:, 0, :]  # [B, V]
        best_logprob, best_token = logprobs.max(dim=-1)
        logprob_total += best_logprob
        prefixes = torch.cat([prefixes, best_token.view(B, 1, 1)], dim=2)

    codes = prefixes[:, 0, :].cpu().numpy()
    items = [vocab.decode_tokens(row) for row in codes]
    return items, logprob_total.cpu().numpy()


@torch.no_grad()
def beam_search(
    model: GenRec,
    vocab: SemanticIdVocab,
    history_tokens: torch.Tensor,
    beam_size: int = 10,
    n_return: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Trie-constrained beam search.

    history_tokens: [B, T] left-padded, grid-aligned
    -> (items [B, n_return] internal ids, scores [B, n_return] log-probabilities)

    Every returned id is a real item: at each step the legal continuations come
    from the Trie, so an illegal sequence is never scored, let alone emitted.
    Rows with fewer than n_return legal completions are padded with item 0 and
    score -inf.
    """
    model.eval()
    device = history_tokens.device
    B = history_tokens.shape[0]
    L = vocab.n_levels
    V = vocab.vocab_size
    n_return = n_return or beam_size

    prefix_index, prefix_masks = vocab.build_mask_table()
    cache = model.build_cache(history_tokens)

    prefixes = torch.zeros(B, 1, 0, dtype=torch.long, device=device)
    scores = torch.zeros(B, 1, device=device)

    for level in range(L):
        logprobs = _level_logprobs(model, cache, prefixes, level)  # [B, K, V]

        # Trie constraint as a gather: one row of the mask table per beam.
        n_beams = prefixes.shape[0] * prefixes.shape[1]
        if level == 0:
            keys = [()] * n_beams  # every beam starts at the Trie root
        else:
            keys = [
                tuple(int(t) for t in prefix)
                for prefix in prefixes.cpu().numpy().reshape(n_beams, level)
            ]
        rows = np.array([prefix_index.get(key, -1) for key in keys])
        mask = np.zeros((len(rows), V), dtype=bool)
        known = rows >= 0
        mask[known] = prefix_masks[rows[known]]
        logprobs = logprobs.masked_fill(
            ~torch.from_numpy(mask).to(device).view(*logprobs.shape), float("-inf")
        )

        candidates = (scores.unsqueeze(-1) + logprobs).reshape(B, -1)  # [B, K*V]
        k = min(beam_size, candidates.shape[1])
        scores, flat = torch.topk(candidates, k=k, dim=-1)
        parent, token = flat // V, flat % V

        prefixes = torch.cat(
            [torch.gather(prefixes, 1, parent.unsqueeze(-1).expand(-1, -1, level)),
             token.unsqueeze(-1)],
            dim=2,
        )

    items = np.zeros((B, n_return), dtype=np.int64)
    out_scores = np.full((B, n_return), -np.inf, dtype=np.float32)
    codes = prefixes.cpu().numpy()
    finite = torch.isfinite(scores).cpu().numpy()
    score_values = scores.cpu().numpy()
    for b in range(B):
        rank = 0
        for beam in range(codes.shape[1]):
            if rank >= n_return or not finite[b, beam]:
                break
            item = vocab.decode_tokens(codes[b, beam])
            if item is None:  # unreachable while the Trie is enforced
                continue
            items[b, rank] = item
            out_scores[b, rank] = score_values[b, beam]
            rank += 1
    return items, out_scores


def batched_beam_search(
    model: GenRec,
    vocab: SemanticIdVocab,
    history_tokens: np.ndarray,
    device: torch.device,
    beam_size: int = 10,
    n_return: int | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """beam_search over a numpy input, in chunks that fit in memory."""
    all_items, all_scores = [], []
    for start in range(0, len(history_tokens), batch_size):
        chunk = torch.from_numpy(history_tokens[start : start + batch_size]).long().to(device)
        items, scores = beam_search(model, vocab, chunk, beam_size=beam_size, n_return=n_return)
        all_items.append(items)
        all_scores.append(scores)
    return np.concatenate(all_items), np.concatenate(all_scores)
