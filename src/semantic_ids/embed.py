"""Content embeddings for semantic IDs.

Item text -> all-MiniLM-L6-v2 -> 384-dim vector, one row per internal item id.
The output feeds RQ-KMeans quantization (`src.semantic_ids.rq_kmeans`).

Output `embeddings.npz` holds:
    item_ids   int32  [n_items]        internal ids, ascending (1..n_items)
    embeddings float32 [n_items, 384]  row i is the embedding of item_ids[i]
    has_text   bool   [n_items]        False where metadata gave no text at all

Embeddings are stored unnormalized; normalization is a quantization-time
decision and is made in rq_kmeans, not here.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.semantic_ids.text import build_texts

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embed_texts(
    texts: list[str], model_name: str = DEFAULT_MODEL, batch_size: int = 256
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return vectors.astype(np.float32)


def build(
    dataset: str,
    raw_dir: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 256,
) -> Path:
    texts_by_id = build_texts(dataset, raw_dir, data_dir)
    item_ids = np.array(sorted(texts_by_id), dtype=np.int32)
    texts = [texts_by_id[int(i)] for i in item_ids]
    has_text = np.array([bool(t) for t in texts], dtype=bool)

    n_empty = int((~has_text).sum())
    print(f"{dataset}: {len(item_ids)} items, {n_empty} with no metadata text")
    lengths = [len(t) for t in texts if t]
    if lengths:
        print(
            f"  text length chars: min={min(lengths)} mean={np.mean(lengths):.0f} max={max(lengths)}"
        )
    for i in item_ids[:3]:
        print(f"  item {int(i)}: {texts_by_id[int(i)][:110]!r}")

    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        item_ids=item_ids,
        embeddings=embeddings,
        has_text=has_text,
    )
    # Kept alongside the vectors so the quality spot-checks (and any later
    # debugging of a weird-looking code cluster) can read the actual strings.
    with open(out_dir / "item_texts.json", "w") as f:
        json.dump({str(int(i)): texts_by_id[int(i)] for i in item_ids}, f)

    size_mb = npz_path.stat().st_size / 1e6
    print(f"Wrote {npz_path} shape={embeddings.shape} ({size_mb:.1f} MB)")
    return npz_path


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (item_ids, embeddings, has_text)."""
    data = np.load(path)
    return data["item_ids"], data["embeddings"], data["has_text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ml-1m", "beauty"], required=True)
    parser.add_argument("--raw-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    build(
        dataset=args.dataset,
        raw_dir=args.raw_dir or f"data/raw/{args.dataset}",
        data_dir=args.data_dir or f"data/processed/{args.dataset}",
        out_dir=args.out_dir or f"data/processed/{args.dataset}/semantic_ids",
        model_name=args.model,
        batch_size=args.batch_size,
    )
