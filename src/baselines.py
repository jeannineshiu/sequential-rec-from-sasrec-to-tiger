"""Popularity and BPR-MF baselines, evaluated with the same fixed negatives
and evaluators SASRec will use later -- these numbers are the floor every
subsequent model must beat.
"""

import argparse
from pathlib import Path

import numpy as np
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

from src.eval.baseline_eval import evaluate_baseline_full_ranking, evaluate_baseline_sampled
from src.eval.sampled import load_negatives
from src.utils import load_processed, log_run


def popularity_score_fn(train: dict[int, list[int]], n_items: int):
    counts = np.zeros(n_items + 1, dtype=np.float64)
    for seq in train.values():
        for item in seq:
            counts[item] += 1
    return lambda user: counts


def bpr_score_fn(train: dict[int, list[int]], n_users: int, n_items: int, seed: int = 42):
    rows, cols = [], []
    for user, seq in train.items():
        for item in seq:
            rows.append(user - 1)
            cols.append(item - 1)
    data = np.ones(len(rows), dtype=np.float32)
    user_items = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

    model = BayesianPersonalizedRanking(
        factors=64, iterations=100, learning_rate=0.01, regularization=0.01, random_state=seed
    )
    model.fit(user_items)

    item_factors = model.item_factors  # [n_items, f]
    user_factors = model.user_factors  # [n_users, f]

    def score_for_user(user: int) -> np.ndarray:
        vec = user_factors[user - 1] @ item_factors.T
        full = np.empty(n_items + 1, dtype=np.float64)
        full[0] = -np.inf
        full[1:] = vec
        return full

    return score_for_user


def run(data_dir: str, k: int = 10) -> None:
    train, valid, test = None, None, None
    train, valid, test, meta = load_processed(data_dir)
    n_items = meta["n_items"]
    n_users = meta["n_users"]
    negatives = load_negatives(Path(data_dir) / "negatives.json")

    exclude_extra_test = {u: [valid[u]] for u in test}

    print("=== Popularity baseline ===")
    pop_fn = popularity_score_fn(train, n_items)
    pop_sampled = evaluate_baseline_sampled(pop_fn, test, negatives, k=k)
    pop_full = evaluate_baseline_full_ranking(
        pop_fn, train, test, exclude_extra=exclude_extra_test, k=k
    )
    log_run(
        experiment="sequential-rec",
        run_name="popularity_ml1m",
        params={"model": "popularity", "dataset": "ml-1m"},
        metrics={f"sampled_{k}": v for k, v in pop_sampled.items()}
        | {f"full_{k}": v for k, v in pop_full.items()},
    )

    print("=== BPR-MF baseline ===")
    bpr_fn = bpr_score_fn(train, n_users, n_items)
    bpr_sampled = evaluate_baseline_sampled(bpr_fn, test, negatives, k=k)
    bpr_full = evaluate_baseline_full_ranking(
        bpr_fn, train, test, exclude_extra=exclude_extra_test, k=k
    )
    log_run(
        experiment="sequential-rec",
        run_name="bpr_mf_ml1m",
        params={"model": "bpr-mf", "dataset": "ml-1m", "factors": 64, "iterations": 100},
        metrics={f"sampled_{k}": v for k, v in bpr_sampled.items()}
        | {f"full_{k}": v for k, v in bpr_full.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed/ml-1m")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run(args.data_dir, k=args.k)
