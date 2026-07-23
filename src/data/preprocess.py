"""Generic sequential-rec preprocessing: k-core filtering + leave-one-out split.

Shared by ML-1M (Week 1) and Amazon Beauty (Week 3) so both datasets go through
the identical pipeline -- the only thing that should differ between datasets is
the raw-file loader.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_ml1m_ratings(ml1m_dir: Path) -> pd.DataFrame:
    """Load ratings.dat as implicit feedback: all ratings count as positives."""
    path = Path(ml1m_dir) / "ratings.dat"
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"],
        encoding="latin-1",
    )
    return df[["user", "item", "timestamp"]]


def load_beauty_ratings(beauty_dir: Path) -> pd.DataFrame:
    """Load the Amazon 'Beauty' ratings-only CSV as implicit feedback.

    Columns (no header): userId, productId (ASIN), rating, timestamp.
    """
    path = Path(beauty_dir) / "ratings_Beauty.csv"
    df = pd.read_csv(path, names=["user", "item", "rating", "timestamp"])
    return df[["user", "item", "timestamp"]]


def k_core_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Iteratively drop users/items with fewer than k interactions until stable."""
    while True:
        user_counts = df["user"].value_counts()
        item_counts = df["item"].value_counts()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        new_df = df[df["user"].isin(keep_users) & df["item"].isin(keep_items)]
        if len(new_df) == len(df):
            return new_df
        df = new_df


def reindex_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map user/item ids to contiguous ints. Item ids start at 1 (0 = padding)."""
    user_ids = sorted(df["user"].unique())
    item_ids = sorted(df["item"].unique())
    user_map = {u: i + 1 for i, u in enumerate(user_ids)}
    item_map = {it: i + 1 for i, it in enumerate(item_ids)}
    df = df.copy()
    df["user"] = df["user"].map(user_map)
    df["item"] = df["item"].map(item_map)
    return df, user_map, item_map


def build_sequences(df: pd.DataFrame) -> dict[int, list[int]]:
    """Group by user, sort by timestamp, return user -> chronological item list."""
    df_sorted = df.sort_values(["user", "timestamp"], kind="mergesort")
    sequences: dict[int, list[int]] = {}
    for user, group in df_sorted.groupby("user"):
        sequences[int(user)] = group["item"].tolist()
    return sequences


def leave_one_out_split(
    sequences: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    """Last item -> test, second-to-last -> valid, rest -> train.

    Requires each user's sequence to have length >= 3 (guaranteed by 5-core).
    """
    train, valid, test = {}, {}, {}
    for user, seq in sequences.items():
        if len(seq) < 3:
            continue
        train[user] = seq[:-2]
        valid[user] = seq[-2]
        test[user] = seq[-1]
    return train, valid, test


def print_stats(df: pd.DataFrame, sequences: dict[int, list[int]]) -> None:
    n_users = len(sequences)
    n_items = df["item"].nunique()
    lengths = [len(s) for s in sequences.values()]
    avg_len = sum(lengths) / len(lengths)
    print(f"users={n_users} items={n_items} interactions={len(df)}")
    print(f"avg sequence length={avg_len:.2f} min={min(lengths)} max={max(lengths)}")


def run_pipeline(df: pd.DataFrame, out_dir: Path, k: int = 5) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = k_core_filter(df, k=k)
    df, user_map, item_map = reindex_ids(df)
    sequences = build_sequences(df)
    print_stats(df, sequences)

    train, valid, test = leave_one_out_split(sequences)

    # sanity: no leakage -- test/valid items must not appear in that user's train seq
    for u in test:
        assert test[u] not in train[u], f"leakage: user {u} test item in train"
        assert valid[u] not in train[u], f"leakage: user {u} valid item in train"

    with open(out_dir / "train.json", "w") as f:
        json.dump(train, f)
    with open(out_dir / "valid.json", "w") as f:
        json.dump(valid, f)
    with open(out_dir / "test.json", "w") as f:
        json.dump(test, f)
    with open(out_dir / "meta.json", "w") as f:
        json.dump({"n_users": len(user_map), "n_items": len(item_map)}, f)

    print(f"Wrote splits to {out_dir}")


def run_ml1m_pipeline(raw_dir: Path, out_dir: Path, k: int = 5) -> None:
    run_pipeline(load_ml1m_ratings(raw_dir), out_dir, k=k)


def run_beauty_pipeline(raw_dir: Path, out_dir: Path, k: int = 5) -> None:
    run_pipeline(load_beauty_ratings(raw_dir), out_dir, k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "beauty"], default="ml-1m")
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    raw_dir = args.raw_dir or f"data/raw/{'ml-1m' if args.dataset == 'ml-1m' else 'beauty'}"
    out_dir = args.out_dir or f"data/processed/{'ml-1m' if args.dataset == 'ml-1m' else 'beauty'}"

    if args.dataset == "ml-1m":
        run_ml1m_pipeline(Path(raw_dir), Path(out_dir), k=args.k)
    else:
        run_beauty_pipeline(Path(raw_dir), Path(out_dir), k=args.k)
