"""How much of every generative number is the seed?

GenRec was the last model family in this repo reported from a single training
run. Every other configuration has been re-run at seeds 1 and 2 and printed with
its own spread; the generative side carried `borrowed` floors and, worse, a
hypothesis test that controls the wrong source of variance. Fisher exact on the
unseen bucket asks whether *this* model's hit rate could come from user
sampling. It says nothing about whether a differently-seeded GenRec produces the
same count -- and at ten hits in a hundred and thirty-eight, the training noise
is plausibly the wider of the two.

This scores every seed's checkpoint through the same exhaustive pass the
headline table uses, so the spread it reports is on the *published* protocol.
One pass covers three of the four single-seed generative claims: the overall
margins, the cold-start hit counts, and the diversity collapse -- the last one
free, because catalogue coverage is read off the same top-k matrices the ranks
come from.
The mlflow `test_full_*` metrics cannot answer this: they are beam-ranked, and
beam ranking flatters the generative model (0.0329 against the exhaustive 0.0250
on Beauty), so their spread is a spread of a different number.

    uv run python -m scripts.genrec_seed_spread                    # Beauty, 3 seeds
    uv run python -m scripts.genrec_seed_spread \
        --genrec-config configs/genrec_ml1m.yaml \
        --run-names genrec_ml1m genrec_ml1m_seed1 genrec_ml1m_seed2

Writes results/tables/genrec_seed_spread_<dataset>.{md,json}.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.eval.cold_start import DEFAULT_BUCKETS, bucketed_metrics, item_train_frequency
from src.eval.generative import exhaustive_ranks, log_prior
from src.eval.metrics import summarize
from src.semantic_ids.vocab import SemanticIdVocab
from src.train import pick_device
from src.utils import load_processed
from scripts.compare_atomic_vs_semantic import DATASET_LABELS, semantic_label
from scripts.diagnose_genrec import popularity_profile

# The stem the comparison writes for a dataset, so the reader can be pointed at the
# single-seed table these numbers put an interval around.
DATASET_STEMS = {"amazon-beauty": "", "ml-1m": "_ml-1m"}


def checkpoint_path(run_name: str) -> Path:
    return Path("results/checkpoints") / f"{run_name}.pt"


def fingerprint(checkpoint: Path) -> dict:
    """Enough to tell one checkpoint from another without hashing 450KB.

    A resumed run has to distinguish "this seed is already scored" from "this
    seed was scored, and then retrained" -- reusing the second silently would
    publish a spread over checkpoints that no longer exist.
    """
    stat = checkpoint.stat()
    return {"path": str(checkpoint), "size": stat.st_size, "mtime": stat.st_mtime}


def score_run(cfg: dict, run_name: str, alphas: list[float], k: int, limit: int | None) -> dict:
    """One checkpoint, scored exhaustively: overall metrics plus per-bucket hits."""
    from scripts.compare_atomic_vs_semantic import load_genrec

    device = pick_device()
    data_dir = Path(cfg["data_dir"])
    train, valid, test, meta = load_processed(data_dir)
    vocab = SemanticIdVocab.from_data_dir(data_dir)
    extra = {u: [valid[u]] for u in test}
    if limit:
        test = {u: test[u] for u in sorted(test)[:limit]}

    checkpoint = checkpoint_path(run_name)
    if not checkpoint.exists():
        raise SystemExit(f"no checkpoint for {run_name!r} at {checkpoint}")
    model = load_genrec(cfg, vocab, device, checkpoint)

    frequency = item_train_frequency(train, meta["n_items"])
    users, ranks, topk = exhaustive_ranks(
        model,
        vocab,
        train,
        test,
        cfg["model"]["maxlen"],
        device,
        alphas,
        log_prior(frequency),
        extra,
        topk=k,
    )

    labels = {alpha: semantic_label(alpha) for alpha in alphas}
    rows = bucketed_metrics(users, {labels[a]: ranks[a] for a in alphas}, test, frequency, k=k)
    out = {
        "overall": {
            labels[a]: {m: float(v) for m, v in summarize(ranks[a], k=k).items()} for a in alphas
        },
        # The diversity collapse is a single-seed number too, and the top-k
        # matrices it is read off come out of the pass already scored -- so it
        # gets an interval here for free rather than in a second run.
        "diversity": {labels[a]: popularity_profile(labels[a], topk[a], frequency) for a in alphas},
        "buckets": {},
        "checkpoint": fingerprint(checkpoint),
    }
    for row in rows:
        n = row["n_users"]
        bucket = {"n_users": n}
        for alpha in alphas:
            metrics = row[labels[alpha]]
            hr = metrics[f"HR@{k}"]
            bucket[labels[alpha]] = {
                **{m: float(v) for m, v in metrics.items()},
                # The counts are what the cold-start claim is quoted as, and what any
                # test of it runs on. HR x n is exact -- HR@k in a bucket is hits/users.
                "hits": int(round(hr * n)) if n and np.isfinite(hr) else 0,
            }
        out["buckets"][row["bucket"]] = bucket
    return out


def spread(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "rel_std": float(v.std(ddof=1) / v.mean() * 100) if len(v) > 1 and v.mean() else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "n": int(len(v)),
    }


def build_tables(
    results: dict[str, dict], alphas: list[float], k: int, label: str
) -> tuple[str, dict]:
    as_trained = semantic_label(0.0)
    runs = list(results)

    lines = [
        f"# GenRec seed spread — {label}, exhaustive full ranking",
        "",
        "Every row is one training run of the same configuration, differing only in",
        "`train.seed`, scored through the exhaustive pass the headline table uses. The",
        "evaluation negatives and the frequency buckets are frozen, so what moves here is",
        "training noise and nothing else.",
        "",
        "| run | "
        + " | ".join(f"{semantic_label(a)} HR@{k}" for a in alphas)
        + " | "
        + " | ".join(f"{semantic_label(a)} NDCG@{k}" for a in alphas)
        + " |",
        "|" + "---|" * (1 + 2 * len(alphas)),
    ]
    for run in runs:
        overall = results[run]["overall"]
        cells = [f"`{run}`"]
        cells += [f"{overall[semantic_label(a)][f'HR@{k}']:.4f}" for a in alphas]
        cells += [f"{overall[semantic_label(a)][f'NDCG@{k}']:.4f}" for a in alphas]
        lines.append("| " + " | ".join(cells) + " |")

    spreads: dict[str, dict] = {"full": {}}
    lines += ["", "| metric | mean | std | rel. std | min | max |", "|" + "---|" * 6]
    for metric in (f"HR@{k}", f"NDCG@{k}"):
        s = spread([results[r]["overall"][as_trained][metric] for r in runs])
        spreads["full"][metric] = s
        # One run has no spread, and printing 0.00% for it would read as a
        # measurement rather than as the absence of one.
        std = f"{s['std']:.4f}" if s["n"] > 1 else "—"
        rel = f"{s['rel_std']:.2f}%" if s["n"] > 1 else "—"
        lines.append(
            f"| full {metric} | {s['mean']:.4f} | {std} | "
            f"{rel} | {s['min']:.4f} | {s['max']:.4f} |"
        )

    # The cold-start claim, per seed. This is the point of the exercise: the p-value
    # published next to these counts is a statement about user sampling, and the
    # column below is the quantity it does not describe.
    buckets = [b for b, _, _ in DEFAULT_BUCKETS]
    lines += [
        "",
        "## Hits per bucket, per seed",
        "",
        "Counts, not rates, because the cold-start claim is quoted as counts and the",
        "Fisher tests run on them.",
        "",
        "| run | model | "
        + " | ".join(f"{b} ({results[runs[0]]['buckets'][b]['n_users']})" for b in buckets)
        + " |",
        "|" + "---|" * (2 + len(buckets)),
    ]
    for run in runs:
        for alpha in alphas:
            model = semantic_label(alpha)
            cells = [f"`{run}`", model]
            cells += [str(results[run]["buckets"][b][model]["hits"]) for b in buckets]
            lines.append("| " + " | ".join(cells) + " |")

    spreads["buckets"] = {}
    for bucket in buckets:
        n = results[runs[0]]["buckets"][bucket]["n_users"]
        entry = {"n_users": n}
        for alpha in alphas:
            model = semantic_label(alpha)
            counts = [results[r]["buckets"][bucket][model]["hits"] for r in runs]
            entry[model] = {"hits": counts, "min": min(counts), "max": max(counts)}
        spreads["buckets"][bucket] = entry

    # What the model recommends, per seed. The catalogue coverage and the unseen
    # share are read off the same top-k matrices the ranks came from, so a seed
    # that ranks like the others but recommends a different slice of the
    # catalogue would show up here and nowhere else.
    lines += [
        "",
        "## What gets recommended, per seed",
        "",
        "Distinct items across every user's top-10, and the share of those",
        "recommendations that are items the training split never contained.",
        "",
        "| run | model | distinct items in top-10 | % unseen |",
        "|---|---|---|---|",
    ]
    for run in runs:
        for alpha in alphas:
            model = semantic_label(alpha)
            profile = results[run]["diversity"][model]
            lines.append(
                f"| `{run}` | {model} | {profile['distinct_items_recommended']:,} | "
                f"{profile['share_unseen']:.2%} |"
            )

    spreads["diversity"] = {}
    for alpha in alphas:
        model = semantic_label(alpha)
        spreads["diversity"][model] = {
            field: spread([results[r]["diversity"][model][field] for r in runs])
            for field in ("distinct_items_recommended", "share_unseen")
        }

    return "\n".join(lines) + "\n", spreads


def main(
    genrec_config: str,
    run_names: list[str],
    alphas: list[float],
    k: int,
    limit: int | None,
    resume: bool = True,
) -> None:
    with open(genrec_config) as f:
        cfg = yaml.safe_load(f)
    dataset = cfg["dataset"]
    label = DATASET_LABELS.get(dataset, dataset)

    # A subsampled pass must not land on the published path. The Beauty table
    # takes an hour a seed to produce, and a two-minute `--limit` smoke test
    # writing over it under the same filename is the failure `output_stems` in
    # `compare_atomic_vs_semantic` already exists to prevent.
    stem = f"genrec_seed_spread_{dataset}" + (f"_limit{limit}" if limit else "")
    md_path = Path(f"results/tables/{stem}.md")
    json_path = Path(f"results/tables/{stem}.json")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    def publish(results: dict) -> None:
        table, spreads = build_tables(results, alphas, k, label)
        md_path.write_text(table)
        json_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "k": k,
                    "alphas": alphas,
                    "runs": results,
                    "spread": spreads,
                    "comparison_table": (
                        f"atomic_vs_semantic{DATASET_STEMS.get(dataset, '_' + dataset)}"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        return table

    results = reusable(json_path, run_names) if resume else {}
    for run_name in run_names:
        if run_name in results:
            print(f"=== {run_name}: reusing the scored pass in {json_path} ===")
            continue
        print(f"=== scoring {run_name} (exhaustive, alphas={alphas}) ===")
        results[run_name] = score_run(cfg, run_name, alphas, k, limit)
        # Written after every seed, not once at the end. An hour of scoring per
        # seed is long enough that something will eventually interrupt it, and
        # a partial artifact is an honest table of the seeds that finished --
        # `--resume` then starts from there instead of from nothing.
        publish(results)
        print(f"  -> wrote {md_path} with {len(results)} of {len(run_names)} seeds")

    table = publish({name: results[name] for name in run_names})
    print("\n" + table)
    print(f"\nWrote {md_path} and {json_path}")


def reusable(json_path: Path, run_names: list[str]) -> dict:
    """Scored passes in an earlier (possibly partial) artifact that still match
    the checkpoints on disk. Anything else is rescored."""
    if not json_path.exists():
        return {}
    stored = json.loads(json_path.read_text()).get("runs", {})
    keep = {}
    for name in run_names:
        entry = stored.get(name)
        if not entry or "checkpoint" not in entry:
            continue
        path = checkpoint_path(name)
        if path.exists() and entry["checkpoint"] == fingerprint(path):
            keep[name] = entry
    return keep


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genrec-config", type=str, default="configs/genrec_beauty.yaml")
    parser.add_argument(
        "--run-names",
        nargs="+",
        default=["genrec_beauty", "genrec_beauty_seed1", "genrec_beauty_seed2"],
        help="checkpoint stems under results/checkpoints, one per seed",
    )
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N users")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rescore every seed even if a matching pass is already in the artifact",
    )
    args = parser.parse_args()
    main(
        args.genrec_config,
        args.run_names,
        args.alphas,
        args.k,
        args.limit,
        resume=not args.no_resume,
    )
