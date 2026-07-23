"""Export all MLflow runs into a master results table and the A3
sampled-vs-full-ranking scatter plot. Run after training jobs finish:

    uv run python -m src.export_results
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def load_runs(experiment: str = "sequential-rec") -> pd.DataFrame:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    return mlflow.search_runs(experiment_names=[experiment])


def _coalesce(runs: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    """Merge same-meaning columns that differ only by the 'test_' prefix
    (SASRec runs log test_sampled_HR_at_10; baseline runs log sampled_HR_at_10)."""
    present = [c for c in candidates if c in runs.columns]
    if not present:
        return None
    merged = runs[present[0]]
    for c in present[1:]:
        merged = merged.fillna(runs[c])
    return merged


def build_master_table(runs: pd.DataFrame, out_path: Path) -> None:
    single_cols = {
        "tags.mlflow.runName": "run",
        "params.dataset": "dataset",
        "params.model_maxlen": "maxlen",
        "params.model_pos_emb_type": "pos_emb_type",
        "params.train_neg_sampling": "neg_sampling",
        "metrics.avg_epoch_time_sec": "avg_epoch_time_sec",
        "metrics.epochs_trained": "epochs_trained",
    }
    coalesced_cols = {
        "sampled_HR@10": ["metrics.test_sampled_HR_at_10", "metrics.sampled_HR_at_10"],
        "sampled_NDCG@10": ["metrics.test_sampled_NDCG_at_10", "metrics.sampled_NDCG_at_10"],
        "full_HR@10": ["metrics.test_full_HR_at_10", "metrics.full_HR_at_10"],
        "full_NDCG@10": ["metrics.test_full_NDCG_at_10", "metrics.full_NDCG_at_10"],
    }

    df = pd.DataFrame(index=runs.index)
    for src, dst in single_cols.items():
        if src in runs.columns:
            df[dst] = runs[src]
    for dst, candidates in coalesced_cols.items():
        merged = _coalesce(runs, candidates)
        if merged is not None:
            df[dst] = merged

    df = df.sort_values("run")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Master results table\n\n")
        f.write("Generated from MLflow (`sqlite:///mlflow.db`, experiment `sequential-rec`) ")
        f.write("via `uv run python -m src.export_results` -- do not hand-edit.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(f"Wrote {out_path}")


def build_sampled_vs_full_scatter(runs: pd.DataFrame, out_path: Path) -> None:
    sampled = _coalesce(runs, ["metrics.test_sampled_HR_at_10", "metrics.sampled_HR_at_10"])
    full = _coalesce(runs, ["metrics.test_full_HR_at_10", "metrics.full_HR_at_10"])
    name_col = "tags.mlflow.runName"
    if sampled is None or full is None:
        print("Missing sampled/full HR@10 columns, skipping scatter plot")
        return

    sampled_col, full_col = "sampled_HR@10", "full_HR@10"
    plot_df = runs[[name_col]].copy()
    plot_df[sampled_col] = sampled
    plot_df[full_col] = full
    plot_df = plot_df.dropna()
    if plot_df.empty:
        print("No runs with both sampled and full HR@10, skipping scatter plot")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(plot_df[sampled_col], plot_df[full_col], s=60)
    for _, row in plot_df.iterrows():
        ax.annotate(
            row[name_col],
            (row[sampled_col], row[full_col]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("sampled HR@10 (1 pos + 100 uniform negatives)")
    ax.set_ylabel("full-ranking HR@10 (entire catalog, excl. history)")
    ax.set_title("Sampled vs. full-ranking metrics across every trained model")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-out", type=str, default="results/tables/master.md")
    parser.add_argument("--figure-out", type=str, default="results/figures/sampled_vs_full.png")
    args = parser.parse_args()

    runs = load_runs()
    build_master_table(runs, Path(args.table_out))
    build_sampled_vs_full_scatter(runs, Path(args.figure_out))
