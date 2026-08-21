"""Why does the generative model lose, and lose most on rare items?

The cold-start table says the gap widens monotonically toward the tail, which is
the opposite of what semantic IDs are supposed to buy. Two candidate mechanisms,
both measurable here rather than argued:

  popularity profile -- what the model actually recommends. If generation
      collapses onto head items, the tail result is a symptom of that collapse
      and not of anything about semantic IDs specifically.

  per-level accuracy -- an item is retrieved only if all four codes are right.
      If accuracy is fine at level 1 and falls off a cliff by level 4, the loss
      is error compounding along the code sequence, and the disambiguation token
      (which carries no content signal) is the prime suspect.

    uv run python -m scripts.diagnose_genrec
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.genrec_dataset import build_eval_batch
from src.eval.cold_start import DEFAULT_BUCKETS, bucket_of, item_train_frequency
from src.eval.generative import exhaustive_ranks, log_prior
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import make_score_fn, pick_device
from src.utils import load_processed
from scripts.compare_atomic_vs_semantic import load_genrec, load_sasrec

import src.data.dataset as ds_mod


def sasrec_topk(model, train, test, extra, n_items, maxlen, k, device, batch_size=128):
    score_fn = make_score_fn(model, device, mode="full")
    users = list(test.keys())
    tops = []
    for start in range(0, len(users), batch_size):
        chunk = users[start : start + batch_size]
        inputs = np.stack(
            [ds_mod.build_eval_input(train.get(u, []), maxlen, extra=extra.get(u, [])) for u in chunk]
        )
        scores = score_fn(inputs).copy()
        scores[:, 0] = -np.inf
        for row, user in enumerate(chunk):
            seen = set(train.get(user, [])) | set(extra.get(user, []))
            seen.discard(test[user])
            for item in seen:
                scores[row, item] = -np.inf
        tops.append(np.argpartition(-scores, k, axis=1)[:, :k])
    return users, np.concatenate(tops)


def popularity_profile(name: str, topk: np.ndarray, frequency: np.ndarray) -> dict:
    flat = topk.reshape(-1)
    freqs = frequency[flat]
    labels = np.array([bucket_of(int(f)) for f in freqs])
    profile = {
        "model": name,
        "distinct_items_recommended": int(len(np.unique(flat))),
        "median_train_freq": float(np.median(freqs)),
        "mean_train_freq": float(freqs.mean()),
    }
    for label, _, _ in DEFAULT_BUCKETS:
        profile[f"share_{label}"] = float((labels == label).mean())
    return profile


@torch.no_grad()
def per_level_accuracy(model, vocab, train, test, extra, maxlen_items, device, batch_size=256):
    """Fraction of users whose true item's level-l code is the argmax, given the
    true prefix. Teacher-forced, so each level is scored independently of the
    previous level's mistakes -- which is what isolates compounding."""
    users = list(test.keys())
    correct = np.zeros(vocab.n_levels)
    total = 0

    for start in range(0, len(users), batch_size):
        chunk = users[start : start + batch_size]
        history = torch.from_numpy(
            build_eval_batch(chunk, train, vocab, maxlen_items, extra)
        ).long().to(device)
        truth = torch.from_numpy(vocab.item_tokens[[test[u] for u in chunk]]).long().to(device)

        cache = model.build_cache(history)
        head = model.token_emb.weight.T

        logits0 = cache["last_hidden"] @ head
        logits0 = logits0.masked_fill(~model.level_legal[0].unsqueeze(0), float("-inf"))
        correct[0] += (logits0.argmax(-1) == truth[:, 0]).sum().item()

        if vocab.n_levels > 1:
            hidden = model._decode_with_cache(cache, truth[:, : vocab.n_levels - 1].unsqueeze(1))
            logits = (hidden @ head)[:, 0]  # [B, L-1, V]
            levels = torch.arange(1, vocab.n_levels, device=device)
            logits = logits.masked_fill(~model.level_legal[levels].unsqueeze(0), float("-inf"))
            preds = logits.argmax(-1)  # [B, L-1]
            for level in range(1, vocab.n_levels):
                correct[level] += (preds[:, level - 1] == truth[:, level]).sum().item()
        total += len(chunk)

    return correct / total


def main(sasrec_config: str, genrec_config: str, k: int, alphas: list[float]) -> None:
    device = pick_device()
    with open(sasrec_config) as f:
        sasrec_cfg = yaml.safe_load(f)
    with open(genrec_config) as f:
        genrec_cfg = yaml.safe_load(f)

    data_dir = Path(sasrec_cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    n_items = meta["n_items"]
    vocab = SemanticIdVocab.from_data_dir(data_dir)
    extra = {u: [valid[u]] for u in test}
    frequency = item_train_frequency(train, n_items)

    sasrec = load_sasrec(
        sasrec_cfg, n_items, device, Path("results/checkpoints") / f"{sasrec_cfg['mlflow']['run_name']}.pt"
    )
    genrec = load_genrec(
        genrec_cfg, vocab, device, Path("results/checkpoints") / f"{genrec_cfg['mlflow']['run_name']}.pt"
    )

    print("SASRec top-10 ...")
    users, top_a = sasrec_topk(
        sasrec, train, test, extra, n_items, sasrec_cfg["model"]["maxlen"], k, device
    )
    # Exhaustive rather than beam-ranked. A beam-20 top-10 can only contain items
    # reachable through 20 of the 256 first codes, so reading coverage off it
    # measures the beam's width as much as the model's diversity -- which is how
    # an earlier version of this table reported 839 distinct items.
    print("GenRec top-10 (exhaustive) ...")
    users_b, _, topk = exhaustive_ranks(
        genrec, vocab, train, test,
        maxlen_items=genrec_cfg["model"]["maxlen"], device=device,
        alphas=alphas, prior=log_prior(frequency), extra_history=extra, topk=k,
    )
    assert users == users_b, "the two models were scored on different user orderings"

    profiles = [popularity_profile("SASRec (atomic)", top_a, frequency)]
    for alpha in alphas:
        label = "GenRec (semantic)" + (f", debiased a={alpha:g}" if alpha else "")
        profiles.append(popularity_profile(label, topk[alpha], frequency))

    lines = [
        "# What the two models actually recommend — Amazon Beauty, test top-10",
        "",
        "Both models ranked exhaustively over the whole catalogue, so no beam pruning is",
        "constraining what can appear in a top-10 on either side.",
        "",
        "| model | distinct items in all top-10s | median train freq | mean | % head | % torso | % tail | % unseen |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in profiles:
        lines.append(
            f"| {p['model']} | {p['distinct_items_recommended']:,} | {p['median_train_freq']:.0f} | "
            f"{p['mean_train_freq']:.1f} | {p['share_head']:.1%} | {p['share_torso']:.1%} | "
            f"{p['share_tail']:.1%} | {p['share_unseen']:.1%} |"
        )
    lines += ["", f"Catalogue size: {n_items:,} items.", ""]

    print("per-level accuracy ...")
    accuracy = per_level_accuracy(
        genrec, vocab, train, test, extra, genrec_cfg["model"]["maxlen"], device
    )
    lines += [
        "## GenRec per-level code accuracy (teacher-forced on the true prefix)",
        "",
        "| level | argmax accuracy | codebook size |",
        "|---|---|---|",
    ]
    for level, acc in enumerate(accuracy):
        lines.append(f"| {level + 1} | {acc:.4f} | {vocab.level_sizes[level]} |")
    lines += [
        "",
        f"All four correct if independent: {np.prod(accuracy):.5f} "
        "(not the retrieval rate -- levels are not independent -- but it bounds how much "
        "compounding costs).",
        "",
    ]

    out = Path("results/tables/genrec_diagnosis.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec-config", type=str, default="configs/sasrec_beauty.yaml")
    parser.add_argument("--genrec-config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    args = parser.parse_args()
    main(args.sasrec_config, args.genrec_config, args.k, args.alphas)
