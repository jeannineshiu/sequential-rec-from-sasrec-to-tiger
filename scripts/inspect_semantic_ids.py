"""Semantic-ID quality spot check (Week 5 Day 3-4).

The RQ-KMeans stats say the codebooks are healthy (no dead codes, low
collisions), but that is a statement about the quantizer, not about whether the
codes mean anything. This prints the actual item texts sharing a code prefix so
the semantic claim can be read rather than assumed, and measures it two ways:

  - level-1 prefix groups, sampled, printed for human reading
  - a numeric proxy: mean pairwise cosine similarity within a prefix group vs.
    between random item pairs. If prefixes carry no semantics the two are equal.

Writes a markdown report to results/tables/semantic_ids_<dataset>.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.semantic_ids.embed import load_embeddings
from src.semantic_ids.rq_kmeans import load_semantic_ids


def _normalized(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-8, None)


def prefix_coherence(
    codes: np.ndarray, embeddings: np.ndarray, depth: int, rng: np.random.Generator, n_pairs: int
) -> tuple[float, int]:
    """Mean cosine similarity between two distinct items sharing a depth-token prefix."""
    groups: dict[tuple, list[int]] = {}
    for row, code in enumerate(codes):
        groups.setdefault(tuple(int(c) for c in code[:depth]), []).append(row)
    eligible = [rows for rows in groups.values() if len(rows) >= 2]
    if not eligible:
        return float("nan"), 0

    sims = []
    for _ in range(n_pairs):
        rows = eligible[rng.integers(len(eligible))]
        i, j = rng.choice(len(rows), size=2, replace=False)
        sims.append(float(embeddings[rows[i]] @ embeddings[rows[j]]))
    return float(np.mean(sims)), len(eligible)


def random_pair_similarity(embeddings: np.ndarray, rng: np.random.Generator, n_pairs: int) -> float:
    i = rng.integers(len(embeddings), size=n_pairs)
    j = rng.integers(len(embeddings), size=n_pairs)
    keep = i != j
    return float(np.mean(np.sum(embeddings[i[keep]] * embeddings[j[keep]], axis=1)))


def sample_groups(
    codes: np.ndarray,
    item_ids: np.ndarray,
    texts: dict[str, str],
    depth: int,
    n_groups: int,
    max_items: int,
    rng: np.random.Generator,
) -> list[tuple[tuple, list[str]]]:
    groups: dict[tuple, list[int]] = {}
    for row, code in enumerate(codes):
        groups.setdefault(tuple(int(c) for c in code[:depth]), []).append(row)
    eligible = [(prefix, rows) for prefix, rows in groups.items() if len(rows) >= 3]
    eligible.sort()
    picked = rng.choice(len(eligible), size=min(n_groups, len(eligible)), replace=False)

    out = []
    for idx in sorted(picked):
        prefix, rows = eligible[idx]
        labels = [texts[str(int(item_ids[r]))] for r in rows[:max_items]]
        out.append((prefix, labels))
    return out


def report(dataset: str, data_dir: Path, depth: int, n_groups: int, seed: int) -> Path:
    item_ids, codes, _ = load_semantic_ids(data_dir)
    _, embeddings, _ = load_embeddings(data_dir / "semantic_ids" / "embeddings.npz")
    embeddings = _normalized(embeddings)
    with open(data_dir / "semantic_ids" / "item_texts.json") as f:
        texts = json.load(f)
    stats = json.loads((data_dir / "semantic_ids" / "semantic_ids_stats.json").read_text())

    rng = np.random.default_rng(seed)
    baseline = random_pair_similarity(embeddings, rng, n_pairs=20000)
    coherence = {
        d: prefix_coherence(codes, embeddings, d, rng, n_pairs=20000)
        for d in range(1, stats["n_levels"] + 1)
    }

    lines = [
        f"# Semantic ID spot check -- {dataset}",
        "",
        f"{stats['n_items']} items, {stats['n_levels']} levels x {stats['n_codes']} codes, "
        f"seed {stats['seed']}, L2-normalized embeddings: {stats['normalize']}.",
        "",
        "## Are prefixes semantically coherent?",
        "",
        "Mean cosine similarity between two items drawn from the same code prefix, against"
        " two items drawn at random. Same distribution would mean the codes carry nothing.",
        "",
        "| prefix depth | groups with >=2 items | mean within-group cosine | vs. random pair |",
        "|---|---|---|---|",
    ]
    for depth_i, (sim, n_groups_i) in coherence.items():
        lines.append(
            f"| {depth_i} token{'s' if depth_i > 1 else ''} | {n_groups_i} | {sim:.3f} | "
            f"{sim - baseline:+.3f} |"
        )
    lines += [
        f"| random pairs | -- | {baseline:.3f} | -- |",
        "",
        "## Codebook health",
        "",
        "| level | codes used | dead codes | median items/code | max items/code |",
        "|---|---|---|---|---|",
    ]
    for u in stats["codebook_usage"]:
        lines.append(
            f"| {u['level']} | {u['codes_used']}/{stats['n_codes']} | {u['dead_codes']} | "
            f"{u['median_items_per_code']:.0f} | {u['max_items_per_code']} |"
        )
    c = stats["collisions"]
    lines += [
        "",
        f"Collisions: **{c['n_colliding_items']}/{stats['n_items']} items "
        f"({c['collision_rate']:.2%})** share a full {stats['n_levels']}-token code and are"
        f" separated by the disambiguation token; largest colliding group is"
        f" {c['max_group_size']}, so that token needs a vocabulary of at least"
        f" {c['max_group_size']}.",
        "",
        f"Residual norm {stats['residual_norms'][0]:.3f} -> {stats['residual_norms'][-1]:.3f}"
        f" ({stats['fraction_explained']:.1%} of the embedding norm explained by"
        f" {stats['n_levels']} tokens).",
        "",
        f"## Sampled depth-{depth} prefix groups",
        "",
    ]
    for prefix, labels in sample_groups(
        codes, item_ids, texts, depth, n_groups, max_items=6, rng=rng
    ):
        lines.append(f"**prefix {list(prefix)}**")
        lines.append("")
        for label in labels:
            lines.append(f"- {label[:150]}")
        lines.append("")

    out_path = Path("results/tables") / f"semantic_ids_{dataset}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:24]))
    print(f"\nWrote {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "beauty"], required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--depth", type=int, default=2, help="prefix depth for sampled groups")
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report(
        dataset=args.dataset,
        data_dir=Path(args.data_dir or f"data/processed/{args.dataset}"),
        depth=args.depth,
        n_groups=args.groups,
        seed=args.seed,
    )
