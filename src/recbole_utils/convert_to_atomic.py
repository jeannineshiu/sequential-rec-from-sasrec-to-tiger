"""Convert the raw MovieLens-1M ratings file into RecBole's atomic .inter format.

We hand RecBole the *raw* ratings (not our own train/valid/test split) and let it
run its own 5-core + leave-one-out pipeline via config (see
configs/recbole/ml1m_base.yaml). This isn't byte-identical to our split, but it's
the same published dataset processed with the same standard recipe -- enough for
a protocol-equivalence cross-check, without the fragility of forcing RecBole's
sequential dataloader to consume an externally pre-split benchmark file.
"""

import argparse
from pathlib import Path

import pandas as pd


def convert_ml1m(raw_dir: Path, out_dir: Path) -> None:
    ratings_path = Path(raw_dir) / "ratings.dat"
    df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id:token", "item_id:token", "rating:float", "timestamp:float"],
        encoding="latin-1",
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ml-1m.inter"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path} ({len(df)} interactions)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default="data/raw/ml-1m")
    parser.add_argument("--out-dir", type=str, default="dataset/ml-1m")
    args = parser.parse_args()
    convert_ml1m(Path(args.raw_dir), Path(args.out_dir))
