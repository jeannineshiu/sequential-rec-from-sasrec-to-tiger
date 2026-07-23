"""Download and extract the MovieLens-1M dataset."""

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="data/raw")
    args = parser.parse_args()
    download_ml1m(Path(args.dest))
