"""What is the first code actually costing the generative model?

Per-level code accuracy (teacher-forced) is 9.8% / 17.9% / 22.4% / 86.3% on
Beauty, which says the binding constraint is level 1. That is an observation
about the model's *logits*. It does not say what fixing level 1 would be worth
in retrieval terms, and "the first code is the bottleneck" is worth nothing as a
finding until it has a number attached.

This attaches one. For each oracle depth d, the target's true first d codes are
handed to the model for free -- scoring is restricted to the items sharing that
prefix, and the model's own scores rank what remains:

    d = 0   the whole catalogue; reproduces the reported GenRec full-ranking row
    d = 1   only items sharing the target's level-1 code
    d = 2   ... level-1 and level-2
    d = 3   the collision group: items the content signal could not tell apart

Read as a decomposition, not a leaderboard. An oracle that shrinks 12,101
candidates to a few dozen is *supposed* to help, so "HR@10 goes up" is not the
finding; what the oracle fails to fix is. The candidate-set column is printed
alongside so the size of the hint is never hidden, and the share-of-total-gain
statistic is deliberately absent: the deepest depth leaves one candidate and
scores 1.0 by construction, which makes it a meaningless denominator.

Two further questions separate "cannot find the region" from "finds it and
cannot rank inside it", and the answer turns out to be both. The rank of the
true level-1 code among all 256 says how well the model localizes; the unaided
HR@10 *conditioned* on the model having placed level 1 correctly says what it
does with a region it found on its own.

    uv run python -m scripts.first_code_ceiling

Writes results/tables/first_code_ceiling.md.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.genrec_dataset import build_eval_batch
from src.eval.metrics import summarize
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import pick_device
from src.utils import load_processed
from scripts.compare_atomic_vs_semantic import load_genrec

import time


def prefix_group_ids(item_tokens: np.ndarray, depth: int) -> np.ndarray:
    """[n_items+1] -> group id, where two items share an id iff their first
    `depth` codes match. depth=0 puts everything in one group."""
    if depth == 0:
        return np.zeros(len(item_tokens), dtype=np.int64)
    _, inverse = np.unique(item_tokens[:, :depth], axis=0, return_inverse=True)
    return inverse.astype(np.int64)


@torch.no_grad()
def oracle_ladder(
    model,
    vocab: SemanticIdVocab,
    train: dict[int, list[int]],
    targets: dict[int, int],
    maxlen_items: int,
    device: torch.device,
    extra_history: dict[int, list[int]] | None = None,
    k: int = 10,
    user_batch: int = 64,
    cand_chunk: int = 2048,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    """Ranks and candidate-set sizes per oracle depth, plus level-1 rank of the
    true first code.

    One scoring pass serves every depth: the score vector does not depend on the
    oracle, only the mask over it does, so the depths are free once the scores
    exist.
    """
    users = list(targets.keys())
    n_items = len(vocab.item_ids)
    depths = list(range(vocab.n_levels))  # 0..L-1; depth L is the item itself

    groups = {
        d: torch.from_numpy(prefix_group_ids(vocab.item_tokens, d)).to(device) for d in depths
    }
    all_tokens = torch.from_numpy(vocab.item_tokens).long().to(device)

    ranks = {d: np.empty(len(users), dtype=np.int64) for d in depths}
    cand_sizes = {d: np.empty(len(users), dtype=np.int64) for d in depths}
    level1_rank = np.empty(len(users), dtype=np.int64)
    start_time = time.time()

    for start in range(0, len(users), user_batch):
        chunk = users[start : start + user_batch]
        history = (
            torch.from_numpy(build_eval_batch(chunk, train, vocab, maxlen_items, extra_history))
            .long()
            .to(device)
        )
        cache = model.build_cache(history)

        scores = torch.empty(len(chunk), n_items + 1, device=device)
        scores[:, 0] = -float("inf")
        for c0 in range(1, n_items + 1, cand_chunk):
            c1 = min(c0 + cand_chunk, n_items + 1)
            candidates = all_tokens[c0:c1].unsqueeze(0).expand(len(chunk), -1, -1)
            scores[:, c0:c1] = model.score_with_cache(cache, candidates)

        # Same history masking as every other full-ranking evaluator here.
        for row, user in enumerate(chunk):
            seen = set(train.get(user, []))
            if extra_history:
                seen |= set(extra_history.get(user, []))
            seen.discard(targets[user])
            if seen:
                scores[row, torch.tensor(sorted(seen), device=device)] = -float("inf")

        target_idx = torch.tensor([targets[u] for u in chunk], device=device)
        rows = torch.arange(len(chunk), device=device)
        target_scores = scores[rows, target_idx]

        for d in depths:
            in_group = groups[d].unsqueeze(0) == groups[d][target_idx].unsqueeze(1)
            in_group[:, 0] = False
            # A history item excluded above is already -inf, so it can never
            # outrank the target; count it out of the candidate size too, or the
            # oracle would look weaker than it is.
            live = in_group & torch.isfinite(scores)
            beaten = ((scores > target_scores.unsqueeze(1)) & live).sum(dim=1)
            ranks[d][start : start + len(chunk)] = beaten.cpu().numpy()
            cand_sizes[d][start : start + len(chunk)] = live.sum(dim=1).cpu().numpy()

        # Where does the true level-1 code sit among the model's level-1 logits?
        head = model.token_emb.weight.T
        logits0 = cache["last_hidden"] @ head
        logits0 = logits0.masked_fill(~model.level_legal[0].unsqueeze(0), float("-inf"))
        truth0 = all_tokens[target_idx][:, 0]
        true_logit = logits0[rows, truth0]
        level1_rank[start : start + len(chunk)] = (
            (logits0 > true_logit.unsqueeze(1)).sum(dim=1).cpu().numpy()
        )

        done = start + len(chunk)
        if done % (user_batch * 40) == 0 or done == len(users):
            print(f"  {done}/{len(users)} users, {time.time() - start_time:.0f}s", flush=True)

    return ranks, cand_sizes, level1_rank


def main(genrec_config: str, k: int, limit_users: int | None = None) -> None:
    device = pick_device()
    with open(genrec_config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    vocab = SemanticIdVocab.from_data_dir(data_dir)
    if limit_users:
        # Sanity-check path only: d=0 must reproduce the reported GenRec row
        # before spending the full pass. Never report a subsampled table.
        test = {u: test[u] for u in list(test)[:limit_users]}
        print(f"[subsample] {len(test)} users -- sanity check, not a reportable number")
    extra = {u: [valid[u]] for u in test}
    model = load_genrec(
        cfg, vocab, device, Path("results/checkpoints") / f"{cfg['mlflow']['run_name']}.pt"
    )

    print(f"scoring {len(test)} users exhaustively (one pass serves every oracle depth) ...")
    ranks, cand_sizes, level1_rank = oracle_ladder(
        model,
        vocab,
        train,
        test,
        maxlen_items=cfg["model"]["maxlen"],
        device=device,
        extra_history=extra,
        k=k,
    )

    lines = [
        f"# What the first code costs — {cfg['dataset']}, full ranking, test set, k={k}",
        "",
        "Oracle depth *d* hands the model the target's true first *d* codes and ranks only the",
        "items sharing that prefix. `d=0` is the model as it actually runs. The candidate column",
        "is the median number of items still competing, so the size of the hint stays visible.",
        "**Not comparable to SASRec** — SASRec gets no oracle.",
        "",
        f"| oracle depth | median candidates | HR@{k} | NDCG@{k} | vs d=0 |",
        "|---|---|---|---|---|",
    ]
    base = summarize(ranks[0], k=k)[f"HR@{k}"]
    summaries = {}
    for d in sorted(ranks):
        s = summarize(ranks[d], k=k)
        summaries[d] = s
        med = int(np.median(cand_sizes[d]))
        rel = "—" if d == 0 else f"{(s[f'HR@{k}'] - base) / base * 100:+.0f}%"
        label = "0 (as trained)" if d == 0 else str(d)
        lines.append(f"| {label} | {med:,} | {s[f'HR@{k}']:.4f} | {s[f'NDCG@{k}']:.4f} | {rel} |")

    # Deliberately not reported as a share of the full oracle: d = L-1 leaves one
    # candidate, so its HR@k is 1.0 by construction and makes a meaningless
    # denominator. The informative quantities are the d=1 multiplier and what is
    # still being missed *after* the region is handed over.
    d1 = summaries[1]
    hit_at_1 = d1[f"HR@{k}"]
    lines += [
        "",
        f"Handing over the level-1 code alone multiplies HR@{k} by "
        f"**{hit_at_1 / base:.1f}x** ({base:.4f} -> {hit_at_1:.4f}).",
        "",
        f"But it does not rescue the model: with the right region and a median of "
        f"{int(np.median(cand_sizes[1]))} candidates left, **{(1 - hit_at_1) * 100:.0f}% of "
        f"targets still miss the top {k}**. Level 2 is where retrieval actually becomes "
        f"reliable ({summaries[2][f'HR@{k}']:.4f} from a median of "
        f"{int(np.median(cand_sizes[2]))} candidates). So the first code is not a lone "
        "bottleneck -- selecting the region and ranking within it are both weak.",
        "",
        "## Is the model near the right first code, or nowhere near it?",
        "",
        "Rank of the true level-1 code among all 256, from the model's own logits:",
        "",
        "| true level-1 code in top-n | share of users |",
        "|---|---|",
    ]
    for n in (1, 5, 10, 25, 64, 128):
        lines.append(f"| {n} | {(level1_rank < n).mean() * 100:.1f}% |")

    # The conditional is what separates "cannot find the region" from "finds the
    # region and cannot rank inside it". Both are measured on the *unaided* d=0
    # ranks, so this is the model as it actually runs.
    buckets = [
        ("top-1 correct", level1_rank == 0),
        ("in top-10", (level1_rank > 0) & (level1_rank < 10)),
        ("in top-64", (level1_rank >= 10) & (level1_rank < 64)),
        ("outside top-64", level1_rank >= 64),
    ]
    lines += [
        "",
        f"Median rank of the true level-1 code: **{int(np.median(level1_rank))}** of 256 "
        f"(random would be 128), so the model localizes the region far better than chance "
        "while rarely nailing it.",
        "",
        f"### Unaided HR@{k}, split by how well the model placed the level-1 code",
        "",
        f"| level-1 outcome | users | HR@{k} (no oracle) |",
        "|---|---|---|",
    ]
    for label, mask in buckets:
        n_users = int(mask.sum())
        hr = float((ranks[0][mask] < k).mean()) if n_users else float("nan")
        lines.append(f"| {label} | {n_users:,} | {hr:.4f} |")
    top1_hr = float((ranks[0][level1_rank == 0] < k).mean())
    lines += [
        "",
        f"Even when the model's own first choice of level-1 code is correct, unaided "
        f"HR@{k} is only **{top1_hr:.4f}** -- the residual loss is inside the region, not in "
        "reaching it.",
        "",
        "Reproduce: `uv run python -m scripts.first_code_ceiling`",
        "",
    ]

    print("\n".join(lines))
    if limit_users:
        print("\n[subsample] table NOT written")
        return
    out = Path("results/tables/first_code_ceiling.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genrec-config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--limit-users", type=int, default=None, help="sanity-check subsample; writes no table"
    )
    args = parser.parse_args()
    main(args.genrec_config, args.k, args.limit_users)
