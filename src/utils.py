"""Small shared utilities: git hash lookup and MLflow run helpers."""

import subprocess
from pathlib import Path
from typing import Any

import mlflow


def get_git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def log_run(
    experiment: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_hash", get_git_hash())
        mlflow.log_params(params)
        safe_metrics = {k.replace("@", "_at_"): v for k, v in metrics.items()}
        mlflow.log_metrics(safe_metrics)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


def load_processed(data_dir: str | Path) -> tuple[dict, dict, dict, dict]:
    import json

    data_dir = Path(data_dir)
    train = {int(k): v for k, v in json.load(open(data_dir / "train.json")).items()}
    valid = {int(k): v for k, v in json.load(open(data_dir / "valid.json")).items()}
    test = {int(k): v for k, v in json.load(open(data_dir / "test.json")).items()}
    meta = json.load(open(data_dir / "meta.json"))
    return train, valid, test, meta
