"""Week 6 main experiment: SASRec (atomic IDs) vs GenRec (semantic IDs).

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

from src.eval.cold_start import (
    DEFAULT_BUCKETS,
    bucketed_metrics,
    format_table,
    item_train_frequency,
    plot_buckets,
)
from src.eval.full_ranking import full_ranking_ranks
from src.eval.generative import generative_full_ranking_ranks
from src.models.genrec import GenRec
from src.models.sasrec import SASRec
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import make_score_fn, pick_device
from src.utils import load_processed

ATOMIC = "SASRec (atomic)"
SEMANTIC = "GenRec (semantic)"


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


def load_genrec(cfg: dict, vocab: SemanticIdVocab, device: torch.device, checkpoint: Path) -> GenRec:
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


def main(sasrec_config: str, genrec_config: str, k: int, beam_size: int) -> None:
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

    sasrec = load_sasrec(
        sasrec_cfg, n_items, device, Path("results/checkpoints") / f"{sasrec_cfg['mlflow']['run_name']}.pt"
    )
    genrec = load_genrec(
        genrec_cfg, vocab, device, Path("results/checkpoints") / f"{genrec_cfg['mlflow']['run_name']}.pt"
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
    assert users_a == users_b, "the two models were scored on different user orderings"

    frequency = item_train_frequency(train, n_items)
    rows = bucketed_metrics(
        users_a,
        {ATOMIC: ranks_a, SEMANTIC: ranks_b},
        test,
        frequency,
        k=k,
    )

    models = [ATOMIC, SEMANTIC]
    table = format_table(rows, models, k=k)
    print("\n" + table)

    figure = plot_buckets(rows, models, Path("results/figures/cold_start_buckets.png"), k=k)

    bucket_desc = ", ".join(
        f"{label}: {low}" + (f"–{high}" if high is not None else "+") for label, low, high in DEFAULT_BUCKETS
    )
    out = Path("results/tables/atomic_vs_semantic.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "# Atomic vs. semantic IDs — Amazon Beauty, full ranking, test set\n\n"
        f"Buckets by the target item's training-split frequency ({bucket_desc}).\n"
        f"SASRec ranks exhaustively; GenRec ranks by constrained beam search (beam {beam_size}),\n"
        "which can only cost the generative side — see the beam-sensitivity table in the log.\n\n"
        + table
        + "\n\n![cold start buckets](../figures/cold_start_buckets.png)\n"
    )
    print(f"\nWrote {out}\nWrote {figure}")

    summary = {
        row["bucket"]: {model: row[model] for model in models} | {"n_users": row["n_users"]}
        for row in rows
    }
    Path("results/tables/atomic_vs_semantic.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sasrec-config", type=str, default="configs/sasrec_beauty.yaml")
    parser.add_argument("--genrec-config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=20)
    args = parser.parse_args()
    main(args.sasrec_config, args.genrec_config, args.k, args.beam_size)
