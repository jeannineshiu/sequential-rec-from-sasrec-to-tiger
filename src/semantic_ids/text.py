"""Item side-info -> one content string per internal item id.

Week 5 Day 1-2. Everything downstream (embeddings, RQ codes) is keyed by the
*internal* contiguous item id produced by preprocessing, so the join through
`data/processed/<dataset>/id_maps.json` happens here and nowhere else.

Coverage is reported rather than silently patched: an item with no usable text
still gets a row (an empty string), and the caller decides what to do about it.
"""

import ast
import gzip
import json
from pathlib import Path


def load_item_map(data_dir: str | Path) -> dict[str, int]:
    """raw item id (str) -> internal item id (1-based)."""
    with open(Path(data_dir) / "id_maps.json") as f:
        return json.load(f)["item_map"]


def ml1m_texts(raw_dir: str | Path, item_map: dict[str, int]) -> dict[int, str]:
    """movies.dat -> "<title>. Genres: <g1>, <g2>".

    Titles carry the release year ("Toy Story (1995)"), which is kept: it is
    real content signal for a movie catalogue.
    """
    texts: dict[int, str] = {}
    path = Path(raw_dir) / "movies.dat"
    with open(path, encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) != 3:
                continue
            raw_id, title, genres = parts
            internal = item_map.get(raw_id)
            if internal is None:
                continue  # dropped by 5-core filtering
            genre_str = ", ".join(g for g in genres.split("|") if g)
            texts[internal] = f"{title}. Genres: {genre_str}" if genre_str else title
    return texts


def beauty_texts(raw_dir: str | Path, item_map: dict[str, int]) -> dict[int, str]:
    """meta_Beauty.json.gz -> "<title>. Category: <leaf path>. Brand: <brand>".

    The file is one Python literal per line (single-quoted), not JSON -- the
    SNAP-era Amazon format -- so it needs ast.literal_eval. `categories` is a
    list of paths; the longest path is used as the most specific one, with the
    leading "Beauty" dropped since it is constant across the dataset.
    """
    texts: dict[int, str] = {}
    path = Path(raw_dir) / "meta_Beauty.json.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            record = ast.literal_eval(line)
            internal = item_map.get(record.get("asin"))
            if internal is None:
                continue
            texts[internal] = _beauty_text(record)
    return texts


def _beauty_text(record: dict) -> str:
    fields = []
    title = (record.get("title") or "").strip()
    if title:
        fields.append(title)

    paths = record.get("categories") or []
    if paths:
        longest = max(paths, key=len)
        trimmed = [c for c in longest if c != "Beauty"]
        if trimmed:
            fields.append("Category: " + " > ".join(trimmed))

    brand = (record.get("brand") or "").strip()
    if brand:
        fields.append(f"Brand: {brand}")

    return ". ".join(fields)


BUILDERS = {"ml-1m": ml1m_texts, "beauty": beauty_texts}


def build_texts(dataset: str, raw_dir: str | Path, data_dir: str | Path) -> dict[int, str]:
    """internal item id -> content string, for every item in the 5-core catalogue.

    Items the metadata does not cover get an empty string so that the returned
    dict always has exactly n_items entries and row order is never ambiguous.
    """
    item_map = load_item_map(data_dir)
    texts = BUILDERS[dataset](raw_dir, item_map)
    return {internal: texts.get(internal, "") for internal in sorted(item_map.values())}
