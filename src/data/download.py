"""Download and extract the MovieLens-1M and Amazon Beauty datasets."""

import argparse
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"

BEAUTY_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv"


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download_ml1m(dest_dir: Path) -> Path:
    """Download MovieLens-1M into dest_dir/ml-1m/ and return that path."""
    dest_dir = Path(dest_dir)
    out_dir = dest_dir / "ml-1m"
    ratings_file = out_dir / "ratings.dat"
    if ratings_file.exists():
        print(f"Already present: {ratings_file}")
        return out_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL) as resp:
        raw = resp.read()

    digest = _md5(raw)
    if digest != ML1M_MD5:
        raise ValueError(f"MD5 mismatch: expected {ML1M_MD5}, got {digest}")

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        zf.extractall(dest_dir)

    print(f"Extracted to {out_dir}")
    return out_dir


def download_beauty(dest_dir: Path) -> Path:
    """Download the Amazon 'Beauty' ratings-only CSV into dest_dir/beauty/ and
    return that directory. No official checksum is published for this file;
    presence + row-count sanity checks happen downstream in preprocess.py.
    """
    dest_dir = Path(dest_dir)
    out_dir = dest_dir / "beauty"
    ratings_file = out_dir / "ratings_Beauty.csv"
    if ratings_file.exists():
        print(f"Already present: {ratings_file}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {BEAUTY_URL} ...")
    with urllib.request.urlopen(BEAUTY_URL) as resp, open(ratings_file, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"Saved to {ratings_file}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="data/raw")
    parser.add_argument("--dataset", choices=["ml-1m", "beauty", "all"], default="all")
    args = parser.parse_args()
    if args.dataset in ("ml-1m", "all"):
        download_ml1m(Path(args.dest))
    if args.dataset in ("beauty", "all"):
        download_beauty(Path(args.dest))
