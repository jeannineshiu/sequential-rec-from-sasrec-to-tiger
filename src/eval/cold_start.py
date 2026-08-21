"""Cold-start bucketing: does the semantic ID pay off where it should?

Semantic IDs are supposed to help exactly where atomic IDs are weakest. An item
seen twice in training has an embedding trained by two gradient updates, and one
never seen in training has an embedding that is still at its initialization --
whereas its *semantic* ID is composed of codes that thousands of other items
share, so a generative model can reach it from content alone.

That predicts a specific shape: the generative model loses on head items and
closes the gap, or wins, on the tail. An overall loss does not settle it, which
is why this exists as a separate measurement rather than a footnote.

Buckets are by the target item's frequency in the *training* split. 5-core
filtering guarantees 5 interactions overall, not 5 in train, so `unseen` is a
real and non-empty bucket: those items appear only in some user's valid/test
position.
"""

from pathlib import Path

import numpy as np

from src.eval.metrics import summarize

# (label, lower bound inclusive, upper bound inclusive) on training frequency.
DEFAULT_BUCKETS = [
    ("unseen", 0, 0),
    ("tail", 1, 4),
    ("torso", 5, 19),
    ("head", 20, None),
]


def item_train_frequency(train: dict[int, list[int]], n_items: int) -> np.ndarray:
    """-> [n_items + 1] counts of each item in the training sequences."""
    counts = np.zeros(n_items + 1, dtype=np.int64)
    for seq in train.values():
        np.add.at(counts, np.asarray(seq, dtype=np.int64), 1)
    counts[0] = 0
    return counts


def bucket_of(frequency: int, buckets=DEFAULT_BUCKETS) -> str:
    for label, low, high in buckets:
        if frequency >= low and (high is None or frequency <= high):
            return label
    raise ValueError(f"no bucket for frequency {frequency}")


def bucket_users(
    users: list[int],
    targets: dict[int, int],
    frequency: np.ndarray,
    buckets=DEFAULT_BUCKETS,
) -> dict[str, np.ndarray]:
    """-> bucket label -> indices into `users`."""
    labels = np.array([bucket_of(int(frequency[targets[u]]), buckets) for u in users])
    return {label: np.nonzero(labels == label)[0] for label, _, _ in buckets}


def bucketed_metrics(
    users: list[int],
    ranks_by_model: dict[str, np.ndarray],
    targets: dict[int, int],
    frequency: np.ndarray,
    k: int = 10,
    buckets=DEFAULT_BUCKETS,
) -> list[dict]:
    """One row per bucket, with every model's metrics on that bucket's users."""
    indices = bucket_users(users, targets, frequency, buckets)
    rows = []
    for label, _, _ in buckets:
        idx = indices[label]
        row = {"bucket": label, "n_users": int(len(idx))}
        for model, ranks in ranks_by_model.items():
            row[model] = (
                summarize(ranks[idx], k=k)
                if len(idx)
                else {f"HR@{k}": float("nan"), f"NDCG@{k}": float("nan")}
            )
        rows.append(row)

    overall = {"bucket": "overall", "n_users": len(users)}
    for model, ranks in ranks_by_model.items():
        overall[model] = summarize(ranks, k=k)
    rows.append(overall)
    return rows


def format_table(rows: list[dict], models: list[str], k: int = 10) -> str:
    """Markdown table; models[0] is the baseline every other column is relative to."""
    columns = [f"{m} HR@{k}" for m in models] + [f"{m} vs {models[0]}" for m in models[1:]]
    lines = [
        "| bucket | users | " + " | ".join(columns) + " |",
        "|" + "---|" * (2 + len(columns)),
    ]
    for row in rows:
        cells = [row["bucket"], str(row["n_users"])]
        cells += [f"{row[m][f'HR@{k}']:.4f}" for m in models]
        base = row[models[0]][f"HR@{k}"]
        for model in models[1:]:
            rel = (row[model][f"HR@{k}"] - base) / base * 100 if base > 0 else float("nan")
            cells.append(f"{rel:+.1f}%" if np.isfinite(rel) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_buckets(rows: list[dict], models: list[str], out_path: Path, k: int = 10) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["bucket"] for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(4 + 1.6 * len(labels), 4.8))
    for i, model in enumerate(models):
        values = [row[model][f"HR@{k}"] for row in rows]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=model)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['bucket']}\n(n={row['n_users']})" for row in rows])
    ax.set_ylabel(f"full-ranking HR@{k}")
    ax.set_title("Amazon Beauty: accuracy by target-item training frequency")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
