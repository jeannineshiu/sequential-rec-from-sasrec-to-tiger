"""Rescore an exported RecBole run with THIS repo's evaluation protocol.

Why this exists
---------------
The RecBole runs (SASRec and BERT4Rec) were evaluated by RecBole's own
evaluator in `uni100` mode. That leaves two holes in the master table, both
flagged in README.md's methodology notes:

1. **Different negatives.** Every non-RecBole model is scored against the fixed
   `data/processed/ml-1m/negatives.json` (seed 42) so comparisons are
   apples-to-apples. RecBole's evaluator draws its *own* 1+100 uniform negatives
   instead -- same protocol shape, different draw -- while the margins under
   discussion are 1-6%.
2. **No full-ranking numbers.** RecBole was run uni100-only, so the full-ranking
   cells for both RecBole rows are blank, even though README elsewhere warns
   against trusting sampled metrics alone.

`src/recbole_run.export_scores` dumps the raw test-set item-score matrix, which
is enough to close both holes offline, on a laptop, for free -- no GPU rerun.

The id mapping
--------------
The two pipelines index items differently, so the score matrix has to be
permuted into this repo's id space before any of its metrics mean anything:

- This repo (`src/data/preprocess.reindex_ids`): 5-core filter, then
  `sorted(unique ids)` -> 1..N, 0 reserved for padding.
- RecBole: its own 5-core filter over the same raw ratings, then its own
  internal index, with token 0 = `[PAD]`. `field2id_token` recovers the original
  ML-1M id for each internal index.

Both run the same recipe on the same raw file, so the surviving user/item *sets*
should be identical and the mapping is a pure relabelling -- which this script
asserts rather than assumes. The map is recomputed from the raw ratings (it is
deterministic) rather than loaded, since preprocess.py does not persist it.

Metrics are computed with `src.eval.metrics`, the same helpers both evaluators
use, so rank convention and tie-breaking match exactly. The evaluators
themselves are not called: they are model-driven (`score_fn(input_batch)`), and
here the scores are already computed, so the ranking/exclusion logic is
mirrored directly against precomputed rows.

Usage
-----
    uv run python -m scripts.rescore_recbole results/scores/sasrec_recbole_1x.npz

(run as a module, not a path -- it imports from `src`, which needs the repo root
on sys.path)
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.preprocess import k_core_filter, load_ml1m_ratings, reindex_ids
from src.eval.metrics import summarize


def build_maps(raw_dir: Path, k: int = 5) -> tuple[dict, dict]:
    """Recompute this repo's original-id -> internal-id maps (deterministic)."""
    df = load_ml1m_ratings(raw_dir)
    df = k_core_filter(df, k=k)
    _, user_map, item_map = reindex_ids(df)
    return user_map, item_map


def load_split(processed_dir: Path) -> tuple[dict, dict, dict, dict, int]:
    def _load(name):
        with open(processed_dir / name) as fh:
            return {int(k): v for k, v in json.load(fh).items()}

    train, valid, test = _load("train.json"), _load("valid.json"), _load("test.json")
    negatives = _load("negatives.json")
    with open(processed_dir / "meta.json") as fh:
        n_items = json.load(fh)["n_items"]
    return train, valid, test, negatives, n_items


def permute_scores(npz, user_map: dict, item_map: dict, n_items: int) -> dict[int, np.ndarray]:
    """RecBole score matrix -> {this-repo user id: score vector indexed by this-repo item id}.

    Column 0 of the output is the padding slot and is left at -inf so it can
    never be recommended, matching the evaluators' `scores[:, 0] = -inf`.
    """
    scores = npz["scores"].astype(np.float32)  # exported as float16; widen for arithmetic
    item_tokens = npz["item_tokens"]
    user_tokens = npz["user_tokens"]
    user_index = npz["user_index"]

    # RecBole internal item index -> this repo's item id. Index 0 is [PAD] on both
    # sides. A token RecBole kept but our 5-core dropped (or vice versa) would mean
    # the two pipelines disagree on the item set, which would invalidate the whole
    # cross-check -- so fail loudly instead of silently dropping the column.
    n_cols = scores.shape[1]
    col_to_item = np.zeros(n_cols, dtype=np.int64)
    missing = []
    for col in range(1, min(n_cols, len(item_tokens))):
        token = item_tokens[col]
        key = int(token) if str(token).lstrip("-").isdigit() else token
        if key not in item_map:
            missing.append(token)
        else:
            col_to_item[col] = item_map[key]
    if missing:
        raise SystemExit(
            f"{len(missing)} RecBole item tokens are absent from this repo's item map "
            f"(e.g. {missing[:5]}). The two 5-core filters disagree on the item set, so "
            "the score matrix cannot be aligned. Investigate before trusting any number."
        )

    out: dict[int, np.ndarray] = {}
    for row, uidx in enumerate(user_index):
        token = user_tokens[uidx]
        key = int(token) if str(token).lstrip("-").isdigit() else token
        if key not in user_map:
            raise SystemExit(f"RecBole user token {token!r} absent from this repo's user map")
        vec = np.full(n_items + 1, -np.inf, dtype=np.float32)
        cols = np.arange(1, n_cols)
        vec[col_to_item[cols]] = scores[row, cols]
        out[user_map[key]] = vec
    return out


def rescore(npz_path: Path, raw_dir: Path, processed_dir: Path, k: int = 10) -> dict[str, dict]:
    npz = np.load(npz_path, allow_pickle=True)
    user_map, item_map = build_maps(raw_dir)
    train, valid, test, negatives, n_items = load_split(processed_dir)
    user_scores = permute_scores(npz, user_map, item_map, n_items)

    users = [u for u in test if u in user_scores]
    if len(users) < len(test):
        print(
            f"  [warn] {len(test) - len(users)} of {len(test)} test users have no exported "
            "scores; metrics are over the users present in both.",
        )

    # --- sampled protocol: ground truth + this repo's fixed 100 negatives ---
    sampled_rows = np.stack(
        [np.concatenate(([user_scores[u][test[u]]], user_scores[u][negatives[u]])) for u in users]
    )
    # Column 0 is the ground truth, matching src.eval.metrics' convention.
    sampled_ranks = (sampled_rows[:, 1:] > sampled_rows[:, 0:1]).sum(axis=1)

    # --- full ranking: whole catalog, minus everything the user already saw ---
    full_ranks = np.empty(len(users), dtype=np.int64)
    for i, user in enumerate(users):
        vec = user_scores[user].copy()
        vec[0] = -np.inf
        # Exclude train history and the valid item, exactly as full_ranking.py does
        # when scoring test (the valid item is history by then, not a candidate).
        exclude = set(train.get(user, [])) | {valid[user]}
        exclude.discard(test[user])
        for item in exclude:
            vec[item] = -np.inf
        target = vec[test[user]]
        vec[test[user]] = -np.inf
        full_ranks[i] = int((vec > target).sum())

    return {
        "sampled": summarize(sampled_ranks, k=k),
        "full": summarize(full_ranks, k=k),
        "n_users": len(users),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="results/scores/<run_name>.npz")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ml-1m"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/ml-1m"))
    # Default naming is "<npz stem>_ourprotocol", which puts a seeded training run at
    # "<...>_seed1_ourprotocol". scripts/seed_variance.py collects a family as
    # "<prefix>" plus "<prefix>_seed<N>", so a seeded rescore has to be logged as
    # "<...>_ourprotocol_seed1" to be picked up. Override the name rather than
    # rename the training run, which is not "our protocol" in any sense.
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name for the rescored run (default: '<npz stem>_ourprotocol')",
    )
    parser.add_argument(
        "--log-mlflow",
        action="store_true",
        help="Log the rescored metrics to MLflow as a SEPARATE run named "
        "'<run>_ourprotocol', so results/tables/master.md stays script-generated. It is "
        "deliberately not merged into the original run's row: that row's uni100 numbers "
        "are what the other RecBole runs can be compared against, and overwriting them "
        "with differently-drawn negatives would silently break those comparisons.",
    )
    args = parser.parse_args()

    out = rescore(args.npz, args.raw_dir, args.processed_dir)
    print(f"\n{args.npz.stem}  (n_users={out['n_users']})")
    print(
        f"  sampled (fixed negatives): HR@10 {out['sampled']['HR@10']:.4f}  "
        f"NDCG@10 {out['sampled']['NDCG@10']:.4f}"
    )
    print(
        f"  full ranking:              HR@10 {out['full']['HR@10']:.4f}  "
        f"NDCG@10 {out['full']['NDCG@10']:.4f}"
    )

    if args.log_mlflow:
        from src.utils import log_run

        log_run(
            experiment="sequential-rec",
            run_name=args.run_name or f"{args.npz.stem}_ourprotocol",
            params={
                "model": "SASRec",
                "dataset": "ml-1m",
                "framework": "recbole",
                "rescored_from": args.npz.stem,
                "protocol": "this-repo evaluator (fixed negatives + full ranking)",
            },
            metrics={
                "test_sampled_HR_at_10": out["sampled"]["HR@10"],
                "test_sampled_NDCG_at_10": out["sampled"]["NDCG@10"],
                "test_full_HR_at_10": out["full"]["HR@10"],
                "test_full_NDCG_at_10": out["full"]["NDCG@10"],
            },
        )
