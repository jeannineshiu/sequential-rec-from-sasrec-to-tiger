"""RQ-KMeans: content embeddings -> semantic IDs.

Residual quantization, the cheap deterministic version of TIGER's RQ-VAE:
level 1 runs KMeans on the embeddings, level 2 on what level 1 could not
explain, level 3 on what levels 1-2 could not explain. An item's semantic ID is
its tuple of centroid indices, which is a coarse-to-fine description of its
content -- items sharing a prefix should be semantically related, and that is
the property the generative model will exploit.

Two items can still land on the identical 3-level code. TIGER's fix, followed
here: append a 4th token that just counts collisions within a code group, so
every item has a unique ID while items differing only in that token are exactly
the ones the content signal could not tell apart.

Output `semantic_ids.npz`:
    item_ids  int32 [n_items]        internal ids, ascending
    codes     int32 [n_items, L+1]   L quantizer levels + disambiguation token
    codebooks float32 [L, K, D]      centroids per level
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from src.semantic_ids.embed import load_embeddings


def fit_rq_kmeans(
    embeddings: np.ndarray,
    n_levels: int = 3,
    n_codes: int = 256,
    seed: int = 42,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """-> (codes [n, n_levels], codebooks [n_levels, n_codes, dim], residual norms).

    The residual norm after each level is returned as the quality trace: it
    should drop monotonically, and how much it drops says how much of the
    content signal each extra token is actually buying.
    """
    residual = embeddings.astype(np.float32).copy()
    codes = np.zeros((len(embeddings), n_levels), dtype=np.int32)
    codebooks = np.zeros((n_levels, n_codes, embeddings.shape[1]), dtype=np.float32)
    residual_norms = [float(np.linalg.norm(residual, axis=1).mean())]

    for level in range(n_levels):
        kmeans = KMeans(n_clusters=n_codes, random_state=seed, n_init=n_init)
        assignments = kmeans.fit_predict(residual)
        centroids = kmeans.cluster_centers_.astype(np.float32)

        codes[:, level] = assignments
        codebooks[level] = centroids
        residual = residual - centroids[assignments]
        residual_norms.append(float(np.linalg.norm(residual, axis=1).mean()))

        print(
            f"  level {level + 1}: {len(np.unique(assignments))}/{n_codes} codes used, "
            f"mean residual norm {residual_norms[-1]:.4f}"
        )

    return codes, codebooks, residual_norms


def add_disambiguation(codes: np.ndarray) -> tuple[np.ndarray, dict]:
    """Append a 4th token counting items within each identical code prefix.

    Order is by ascending row index, so the token is deterministic given the
    item ordering (which is ascending internal id).
    """
    seen: dict[tuple, int] = {}
    extra = np.zeros((len(codes), 1), dtype=np.int32)
    for row, code in enumerate(codes):
        key = tuple(int(c) for c in code)
        extra[row, 0] = seen.get(key, 0)
        seen[key] = extra[row, 0] + 1

    group_sizes = np.array(list(seen.values()))
    stats = {
        "n_unique_prefixes": int(len(seen)),
        "n_colliding_items": int((extra[:, 0] > 0).sum()),
        "collision_rate": float((extra[:, 0] > 0).mean()),
        "max_group_size": int(group_sizes.max()),
        "n_groups_with_collision": int((group_sizes > 1).sum()),
    }
    return np.concatenate([codes, extra], axis=1), stats


def codebook_usage(codes: np.ndarray, n_codes: int) -> list[dict]:
    """Per level: how many codes are used, dead-code count, load imbalance."""
    usage = []
    for level in range(codes.shape[1]):
        counts = np.bincount(codes[:, level], minlength=n_codes)
        used = counts[counts > 0]
        usage.append(
            {
                "level": level + 1,
                "codes_used": int((counts > 0).sum()),
                "dead_codes": int((counts == 0).sum()),
                "min_items_per_code": int(used.min()) if len(used) else 0,
                "max_items_per_code": int(counts.max()),
                "median_items_per_code": float(np.median(used)) if len(used) else 0.0,
            }
        )
    return usage


def build(
    dataset: str,
    data_dir: str | Path,
    n_levels: int = 3,
    n_codes: int = 256,
    seed: int = 42,
    normalize: bool = True,
) -> Path:
    sem_dir = Path(data_dir) / "semantic_ids"
    item_ids, embeddings, has_text = load_embeddings(sem_dir / "embeddings.npz")
    print(f"{dataset}: {embeddings.shape[0]} items x {embeddings.shape[1]} dims")

    if normalize:
        # MiniLM is trained for cosine similarity, so its Euclidean geometry --
        # which is what KMeans minimizes -- is only meaningful on the unit
        # sphere. Without this, clusters partly track text length.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)

    codes, codebooks, residual_norms = fit_rq_kmeans(
        embeddings, n_levels=n_levels, n_codes=n_codes, seed=seed
    )
    usage = codebook_usage(codes, n_codes)
    codes, collision_stats = add_disambiguation(codes)

    print(
        f"  collisions: {collision_stats['n_colliding_items']}/{len(codes)} items "
        f"({collision_stats['collision_rate']:.2%}) share a {n_levels}-level code; "
        f"largest group {collision_stats['max_group_size']}"
    )
    explained = 1 - residual_norms[-1] / residual_norms[0]
    print(
        f"  residual norm {residual_norms[0]:.4f} -> {residual_norms[-1]:.4f} ({explained:.1%} explained)"
    )

    out_path = sem_dir / "semantic_ids.npz"
    np.savez_compressed(out_path, item_ids=item_ids, codes=codes, codebooks=codebooks)
    stats = {
        "dataset": dataset,
        "n_items": int(len(item_ids)),
        "n_levels": n_levels,
        "n_codes": n_codes,
        "seed": seed,
        "normalize": normalize,
        "items_without_text": int((~has_text).sum()),
        "residual_norms": residual_norms,
        "fraction_explained": explained,
        "codebook_usage": usage,
        "collisions": collision_stats,
    }
    with open(sem_dir / "semantic_ids_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {out_path} codes shape={codes.shape}")
    return out_path


def load_semantic_ids(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (item_ids, codes, codebooks)."""
    data = np.load(Path(data_dir) / "semantic_ids" / "semantic_ids.npz")
    return data["item_ids"], data["codes"], data["codebooks"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "beauty"], required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--codes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="quantize raw embeddings instead of L2-normalized ones",
    )
    args = parser.parse_args()

    build(
        dataset=args.dataset,
        data_dir=args.data_dir or f"data/processed/{args.dataset}",
        n_levels=args.levels,
        n_codes=args.codes,
        seed=args.seed,
        normalize=not args.no_normalize,
    )
