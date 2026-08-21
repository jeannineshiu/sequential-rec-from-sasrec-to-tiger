"""Separate the representation change from the scorer change.

Week 6 found that GenRec's recommendations cover 7% of the catalogue against
SASRec's 76%, and proposed a mechanism: GenRec ranks by P(item | history), which
contains the popularity prior, while SASRec's dot product is unnormalized and
carries none. If that is right, dividing out the prior should recover tail
accuracy — and if it is wrong, it should not.

    score_alpha(item) = log P(item | history) - alpha * log P_prior(item)

alpha = 0 is the model as trained; alpha = 1 is pointwise mutual information,
fully removing the prior. The prior is add-one smoothed training frequency, so
items unseen in training get a small but finite prior rather than an infinite
bonus.

This also removes the beam approximation entirely: every one of the 12,101 items
is scored for every user, so GenRec is finally ranked exhaustively, exactly as
SASRec is. The alpha = 0 row is therefore the honest answer to "how much was the
beam costing?" as well.

    uv run python -m scripts.debias_decoding          # ~30 min on laptop MPS

Writes results/tables/debias_decoding.md.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.eval.cold_start import DEFAULT_BUCKETS, bucketed_metrics, item_train_frequency
from src.eval.generative import exhaustive_ranks, log_prior
from src.eval.metrics import summarize
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import pick_device
from src.utils import load_processed
from scripts.compare_atomic_vs_semantic import load_genrec


def main(genrec_config: str, alphas: list[float], k: int, limit: int | None) -> None:
    device = pick_device()
    with open(genrec_config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    n_items = meta["n_items"]
    vocab = SemanticIdVocab.from_data_dir(data_dir)
    extra = {u: [valid[u]] for u in test}
    if limit:
        keep = sorted(test)[:limit]
        test = {u: test[u] for u in keep}

    model = load_genrec(
        cfg, vocab, device, Path("results/checkpoints") / f"{cfg['mlflow']['run_name']}.pt"
    )
    frequency = item_train_frequency(train, n_items)
    prior = log_prior(frequency)

    users, ranks, topk = exhaustive_ranks(
        model, vocab, train, test, cfg["model"]["maxlen"], device, alphas, prior, extra
    )

    lines = [
        "# Popularity-debiased decoding — GenRec, Amazon Beauty, exhaustive full ranking",
        "",
        "Every one of the catalogue's items scored for every user, so there is no beam",
        "approximation here at all. `alpha` divides out the add-one-smoothed training-frequency",
        "prior: 0 is the model as trained, 1 is pointwise mutual information.",
        "",
        f"| alpha | HR@{k} | NDCG@{k} | " + " | ".join(f"{b} HR@{k}" for b, _, _ in DEFAULT_BUCKETS)
        + " | distinct items in top-10 |",
        "|" + "---|" * (4 + len(DEFAULT_BUCKETS)),
    ]
    for alpha in alphas:
        overall = summarize(ranks[alpha], k=k)
        rows = {
            row["bucket"]: row
            for row in bucketed_metrics(users, {"g": ranks[alpha]}, test, frequency, k=k)
        }
        cells = [f"{alpha:g}", f"{overall[f'HR@{k}']:.4f}", f"{overall[f'NDCG@{k}']:.4f}"]
        cells += [f"{rows[b]['g'][f'HR@{k}']:.4f}" for b, _, _ in DEFAULT_BUCKETS]
        cells.append(f"{len(np.unique(topk[alpha])):,}")
        lines.append("| " + " | ".join(cells) + " |")

    out = Path("results/tables/debias_decoding.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genrec-config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N users")
    args = parser.parse_args()
    main(args.genrec_config, args.alphas, args.k, args.limit)
