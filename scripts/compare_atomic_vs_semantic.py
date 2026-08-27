"""Main comparison: SASRec (atomic IDs) vs GenRec (semantic IDs).

Loads both trained checkpoints, scores the same test users under the same
protocol, and reports overall plus cold-start-bucketed metrics from *one* pass
so the two models are never compared across separately-summarised runs.

    uv run python -m scripts.compare_atomic_vs_semantic

Writes results/tables/atomic_vs_semantic.md and
results/figures/cold_start_buckets.png.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from scipy.stats import fisher_exact

from src.eval.cold_start import (
    DEFAULT_BUCKETS,
    bucketed_metrics,
    format_table,
    item_train_frequency,
    plot_buckets,
)
from src.eval.full_ranking import full_ranking_ranks
from src.eval.generative import exhaustive_ranks, generative_full_ranking_ranks, log_prior
from src.models.genrec import GenRec
from src.models.sasrec import SASRec
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import make_score_fn, pick_device
from src.utils import load_processed

ATOMIC = "SASRec (atomic)"
SEMANTIC = "GenRec (semantic)"


def significance_table(rows: list[dict], models: list[str], k: int) -> str:
    """Fisher exact tests for every bucket against the atomic baseline.

    Why this is here rather than in prose: the cold-start claim is the one place
    where the project leans on a hypothesis test, and until 2026-08-25 both of its
    p-values existed only as text in the README -- no script produced them, so
    nothing checked them when the numbers around them changed. One of the two did
    not reproduce. Every interval in the seed work comes from `seed_variance.py`;
    this is the same rule applied to the one test that had escaped it.

    The counts are recovered as HR x n_users, which is exact: HR@k in a bucket is
    hits/users by construction. `p(>)` is one-sided for the semantic model doing
    better -- the direction the cold-start hypothesis predicts -- and `p(2)` is
    two-sided, reported alongside so a reader is not handed only the friendlier
    of the two.
    """
    lines = [
        f"| bucket | users | model | hits@{k} | baseline hits | p(>) | p(2) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        n = row["n_users"]
        # An empty bucket has HR = nan, and there is no test to run on zero
        # users. Beauty fills every bucket; ML-1M is dense enough that `unseen`
        # is empty and `tail` holds two users, so this is reached the first time
        # the comparison runs on any dataset that is not Beauty.
        if not n:
            continue
        base = round(row[ATOMIC][f"HR@{k}"] * n)
        for model in models:
            if model == ATOMIC:
                continue
            hits = round(row[model][f"HR@{k}"] * n)
            table = [[hits, n - hits], [base, n - base]]
            greater = fisher_exact(table, alternative="greater")[1]
            two = fisher_exact(table, alternative="two-sided")[1]
            lines.append(
                f"| {row['bucket']} | {n} | {model} | {hits} | {base} | "
                f"{greater:.4f} | {two:.4f} |"
            )
    return "\n".join(lines)


def semantic_label(alpha: float) -> str:
    return SEMANTIC if not alpha else f"{SEMANTIC}, debiased a={alpha:g}"


def load_sasrec(cfg: dict, n_items: int, device: torch.device, checkpoint: Path) -> SASRec:
    model = SASRec(
        n_items=n_items,
        maxlen=cfg["model"]["maxlen"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        pos_emb_type=cfg["model"].get("pos_emb_type", "learnable"),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


def load_genrec(
    cfg: dict, vocab: SemanticIdVocab, device: torch.device, checkpoint: Path
) -> GenRec:
    model = GenRec(
        vocab_size=vocab.vocab_size,
        n_levels=vocab.n_levels,
        level_offsets=vocab.level_offsets,
        level_sizes=vocab.level_sizes,
        maxlen_items=cfg["model"]["maxlen"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
        num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
        pos_emb_type=cfg["model"].get("pos_emb_type", "learnable"),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.eval()


# Beauty was the only dataset this script ran on, so its outputs are unsuffixed
# and the README links them by those paths. Keeping that name is back-compat, not
# a default: every other dataset gets its own suffixed pair, so running ML-1M
# cannot silently overwrite the Beauty table the headline comparison rests on.
LEGACY_UNSUFFIXED_DATASET = "amazon-beauty"

DATASET_LABELS = {"amazon-beauty": "Amazon Beauty", "ml-1m": "MovieLens-1M"}


def output_stems(dataset: str, beam_size: int | None = None) -> tuple[str, str]:
    """(table stem, figure stem) for a dataset and ranking mode.

    Beam mode writes its own pair. It ranks a different way and produces numbers
    that are not comparable to the exhaustive ones -- that is the whole reason it
    is kept -- so letting it land on the exhaustive paths would overwrite the
    headline table with a superseded methodology under an identical filename.
    """
    table, figure = "atomic_vs_semantic", "cold_start_buckets"
    if dataset != LEGACY_UNSUFFIXED_DATASET:
        table, figure = f"{table}_{dataset}", f"{figure}_{dataset}"
    if beam_size is not None:
        table, figure = f"{table}_beam{beam_size}", f"{figure}_beam{beam_size}"
    return table, figure


def main(
    sasrec_config: str, genrec_config: str, k: int, beam_size: int, alphas: list[float], beam: bool
) -> None:
    device = pick_device()
    with open(sasrec_config) as f:
        sasrec_cfg = yaml.safe_load(f)
    with open(genrec_config) as f:
        genrec_cfg = yaml.safe_load(f)

    dataset = sasrec_cfg["dataset"]
    if genrec_cfg["dataset"] != dataset:
        raise SystemExit(
            f"configs disagree on dataset: {dataset} vs {genrec_cfg['dataset']} -- "
            "the two models must be scored on the same data"
        )
    table_stem, figure_stem = output_stems(dataset, beam_size if beam else None)
    label = DATASET_LABELS.get(dataset, dataset)

    data_dir = Path(sasrec_cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    n_items = meta["n_items"]
    vocab = SemanticIdVocab.from_data_dir(data_dir)
    extra = {u: [valid[u]] for u in test}

    sasrec = load_sasrec(
        sasrec_cfg,
        n_items,
        device,
        Path("results/checkpoints") / f"{sasrec_cfg['mlflow']['run_name']}.pt",
    )
    genrec = load_genrec(
        genrec_cfg,
        vocab,
        device,
        Path("results/checkpoints") / f"{genrec_cfg['mlflow']['run_name']}.pt",
    )

    print("scoring SASRec (exhaustive full ranking) ...")
    users_a, ranks_a = full_ranking_ranks(
        make_score_fn(sasrec, device, mode="full"),
        train,
        test,
        n_items=n_items,
        maxlen=sasrec_cfg["model"]["maxlen"],
        extra_history=extra,
        exclude_extra=extra,
    )

    frequency = item_train_frequency(train, n_items)
    ranks_by_model = {ATOMIC: ranks_a}

    if beam:
        # Kept for reproducing the superseded numbers; see the log for why beam
        # ranking flatters the generative model rather than only costing it.
        print(f"scoring GenRec (constrained beam search, beam={beam_size}) ...")
        users_b, ranks_b = generative_full_ranking_ranks(
            genrec,
            vocab,
            train,
            test,
            maxlen_items=genrec_cfg["model"]["maxlen"],
            device=device,
            extra_history=extra,
            exclude_extra=extra,
            k=k,
            beam_size=beam_size,
        )
        ranks_by_model[f"{SEMANTIC}, beam {beam_size}"] = ranks_b
    else:
        print(f"scoring GenRec (exhaustive, alphas={alphas}) ...")
        users_b, ranks, _ = exhaustive_ranks(
            genrec,
            vocab,
            train,
            test,
            maxlen_items=genrec_cfg["model"]["maxlen"],
            device=device,
            alphas=alphas,
            prior=log_prior(frequency),
            extra_history=extra,
        )
        for alpha in alphas:
            ranks_by_model[semantic_label(alpha)] = ranks[alpha]

    assert users_a == users_b, "the two models were scored on different user orderings"

    rows = bucketed_metrics(users_a, ranks_by_model, test, frequency, k=k)
    models = list(ranks_by_model)
    table = format_table(rows, models, k=k)
    print("\n" + table)

    figure = plot_buckets(rows, models, Path(f"results/figures/{figure_stem}.png"), k=k)

    significance = significance_table(rows, models, k=k)
    print("\nFisher exact vs the atomic baseline:\n" + significance)

    bucket_desc = ", ".join(
        f"{label}: {low}" + (f"–{high}" if high is not None else "+")
        for label, low, high in DEFAULT_BUCKETS
    )
    if beam:
        ranking_note = (
            f"SASRec ranks exhaustively; GenRec is ranked by constrained beam search "
            f"(beam={beam_size}), so a target the beam drops counts as a miss no matter how\n"
            "the model scores it. These numbers are a beam approximation and are not\n"
            "comparable to the exhaustive table.\n"
        )
    else:
        ranking_note = (
            "Both models rank exhaustively over the whole catalogue, so no beam approximation is\n"
            "involved on either side. `debiased a=1` subtracts the log training-frequency prior.\n"
        )
    out = Path(f"results/tables/{table_stem}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Atomic vs. semantic IDs — {label}, full ranking, test set\n\n"
        f"Buckets by the target item's training-split frequency ({bucket_desc}).\n"
        + ranking_note
        + "\n"
        + table
        + "\n\n## Significance vs the atomic baseline\n\n"
        "Fisher exact on hits/misses per bucket. `p(>)` is one-sided for the semantic model\n"
        "being better (the cold-start prediction); `p(2)` is two-sided.\n\n"
        + significance
        + f"\n\n![cold start buckets](../figures/{figure_stem}.png)\n"
    )
    print(f"\nWrote {out}\nWrote {figure}")

    summary = {
        row["bucket"]: {model: row[model] for model in models} | {"n_users": row["n_users"]}
        for row in rows
    }
    Path(f"results/tables/{table_stem}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec-config", type=str, default="configs/sasrec_beauty.yaml")
    parser.add_argument("--genrec-config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=20)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument(
        "--beam", action="store_true", help="rank GenRec by beam search instead of exhaustively"
    )
    args = parser.parse_args()
    main(args.sasrec_config, args.genrec_config, args.k, args.beam_size, args.alphas, args.beam)
