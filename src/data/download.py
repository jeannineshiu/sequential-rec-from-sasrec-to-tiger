"""Download and extract the MovieLens-1M and Amazon Beauty datasets."""

import argparse
import hashlib
import io
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_MD5 = "c4d9eecfca2ab87c1945afe126590906"

BEAUTY_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv"
BEAUTY_META_URL = (
    "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
)

MAX_RETRIES = 5
RETRY_BACKOFF_SEC = 5

# Some servers (files.grouplens.org observed doing this from a Daytona GPU
# sandbox) reset the connection when they see Python urllib's default User-
# Agent ("Python-urllib/3.x"), which reads as a bot/scraper. A browser-like
# UA fixed it -- 5/5 retries had failed identically with the default UA,
# which pointed at something structural rather than a transient network blip.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _urlopen_with_retry(url: str):
    """urllib.request.urlopen (with a browser User-Agent) and retries."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return urllib.request.urlopen(request)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_error = e
            print(f"  download attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
    raise last_error


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
    with _urlopen_with_retry(ML1M_URL) as resp:
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
    with _urlopen_with_retry(BEAUTY_URL) as resp, open(ratings_file, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"Saved to {ratings_file}")
    return out_dir


def download_beauty_meta(dest_dir: Path) -> Path:
    """Download the Amazon 'Beauty' item metadata (~99MB gzip) into
    dest_dir/beauty/ and return that directory.

    Only needed for semantic IDs -- the SASRec/BERT4Rec work uses interactions
    alone. Taken from the same categoryFiles/ directory as
    ratings_Beauty.csv, so the ASINs line up with the ratings file (verified:
    all 12,101 5-core items are present in the metadata).
    """
    dest_dir = Path(dest_dir)
    out_dir = dest_dir / "beauty"
    meta_file = out_dir / "meta_Beauty.json.gz"
    if meta_file.exists():
        print(f"Already present: {meta_file}")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {BEAUTY_META_URL} ...")
    with _urlopen_with_retry(BEAUTY_META_URL) as resp, open(meta_file, "wb") as f:
        shutil.copyfileobj(resp, f)

    print(f"Saved to {meta_file}")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=str, default="data/raw")
    parser.add_argument("--dataset", choices=["ml-1m", "beauty", "all"], default="all")
    parser.add_argument(
        "--with-meta",
        action="store_true",
        help="also fetch Amazon Beauty item metadata (~99MB), needed for semantic IDs",
    )
    args = parser.parse_args()
    if args.dataset in ("ml-1m", "all"):
        download_ml1m(Path(args.dest))
    if args.dataset in ("beauty", "all"):
        download_beauty(Path(args.dest))
        if args.with_meta:
            download_beauty_meta(Path(args.dest))
