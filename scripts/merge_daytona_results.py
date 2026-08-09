"""Merge a Daytona sandbox's results db into the main MLflow tracking db.

Each detached Week 4 sandbox writes its own `mlflow_daytona_week4_<MODEL>.db` and
pushes it to the repo. Those runs have to land in `mlflow.db` before
`src.export_results` can pull them into results/tables/master.md -- the master
table is generated from MLflow, never hand-edited, and that guarantee is only
worth something if this merge is scripted too.

    uv run python -m scripts.merge_daytona_results mlflow_daytona_week4_SASRec.db

(Run it as a module, not a path -- it imports src.utils for the shared MLflow helper.)

Metric renaming: RecBole was configured with `mode: uni100` (1 positive + 100
uniform negatives), which IS this project's sampled protocol, so its
test_HR_at_10 / test_NDCG_at_10 are recorded under the sampled_* names the rest
of the repo uses. RecBole logged no full-ranking metrics, so those cells stay
empty rather than being filled with a sampled number wearing a full-ranking label.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import mlflow

from src.utils import log_run

# RecBole metric name -> this repo's name. uni100 == our sampled protocol.
METRIC_RENAME = {
    "test_HR_at_10": "test_sampled_HR_at_10",
    "test_NDCG_at_10": "test_sampled_NDCG_at_10",
    "valid_HR_at_10": "valid_HR_at_10",
    "valid_NDCG_at_10": "valid_NDCG_at_10",
}


def read_runs(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    runs = []
    for run_uuid, name, status in con.execute("select run_uuid, name, status from runs"):
        params = dict(con.execute("select key, value from params where run_uuid=?", (run_uuid,)))
        metrics = dict(con.execute("select key, value from metrics where run_uuid=?", (run_uuid,)))
        runs.append({"name": name, "status": status, "params": params, "metrics": metrics})
    con.close()
    return runs


def existing_run_names(experiment: str) -> set[str]:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    try:
        df = mlflow.search_runs(experiment_names=[experiment])
    except Exception:
        return set()
    if df.empty or "tags.mlflow.runName" not in df.columns:
        return set()
    return set(df["tags.mlflow.runName"].dropna())


def merge(db_path: Path, experiment: str, force: bool) -> None:
    runs = read_runs(db_path)
    if not runs:
        print(
            f"{db_path} contains 0 runs -- the sandbox produced nothing. Check its run.log "
            "before trusting any 'sweep complete' marker.",
            file=sys.stderr,
        )
        sys.exit(1)

    already = existing_run_names(experiment)
    for run in runs:
        if run["status"] != "FINISHED":
            print(f"skipping {run['name']}: status={run['status']}")
            continue
        if run["name"] in already and not force:
            print(f"skipping {run['name']}: already in mlflow.db (use --force to add anyway)")
            continue

        src = run["metrics"]
        metrics = {new: src[old] for old, new in METRIC_RENAME.items() if old in src}

        # The master table reports per-epoch cost; RecBole logs only the total.
        epochs = src.get("epochs_budget")
        if epochs and "train_time_sec" in src:
            metrics["avg_epoch_time_sec"] = src["train_time_sec"] / epochs
        if epochs:
            metrics["epochs_trained"] = epochs

        print(f"merging {run['name']} ({run['params'].get('model')}, {run['params'].get('epochs')} epochs)")
        log_run(
            experiment=experiment,
            run_name=run["name"],
            params=run["params"],
            metrics=metrics,
        )

    print("\nNow regenerate the table: uv run python -m src.export_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path, help="mlflow_daytona_week4_<MODEL>.db to merge")
    parser.add_argument("--experiment", default="sequential-rec")
    parser.add_argument("--force", action="store_true", help="merge even if the run name already exists")
    args = parser.parse_args()

    if not args.db.exists():
        print(f"{args.db} not found", file=sys.stderr)
        sys.exit(1)
    merge(args.db, args.experiment, args.force)
