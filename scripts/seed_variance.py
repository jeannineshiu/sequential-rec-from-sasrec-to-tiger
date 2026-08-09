"""Summarize seed-to-seed variance for this repo's SASRec on ML-1M, and use it as
a noise floor for the margins reported elsewhere in the project.

Why this exists
---------------
Every headline number in this repo is a single seed (42), while conclusions are
drawn from margins as small as 1.5%. Without knowing how much a metric moves when
only the seed changes, there is no way to tell which of those margins are signal.
This is the cheapest possible way to find out: the model trains in ~25 minutes on
a laptop GPU, so N extra seeds cost nothing but wall-clock.

What varies and what does not
-----------------------------
Only `train.seed` changes, which drives weight initialization and the *training*
negative sampler. The *evaluation* negatives come from the frozen
`negatives.json` (seed 42) for every run, so what is measured here is training
noise, not evaluation-sampling noise. Those are different quantities and mixing
them would overstate the floor.

Caveat on the ablation comparison below: the ablations ran at 100 epochs and the
seed runs at 200 (each config's own protocol, unchanged). The comparison is
therefore indicative rather than exact -- a 100-epoch noise floor could differ.
It is reported because an ablation margin far *inside* the 200-epoch floor is
still a clear warning, and margins far outside it are still safe.

Usage
-----
    uv run python -m scripts.seed_variance
"""

import argparse

import mlflow
import numpy as np

METRICS = [
    ("sampled HR@10", "test_sampled_HR_at_10"),
    ("sampled NDCG@10", "test_sampled_NDCG_at_10"),
    ("full HR@10", "test_full_HR_at_10"),
    ("full NDCG@10", "test_full_NDCG_at_10"),
]

# Margins asserted elsewhere in the repo, to be read against the measured floor.
# (label, relative % difference, which protocol it was measured on, where it is claimed)
#
# The protocol matters: full-ranking metrics are roughly 4x noisier than sampled
# ones here, so a single global floor would be far too strict for sampled margins
# and far too loose for full-ranking ones.
CLAIMED_MARGINS = [
    ("Week 4: dropout 0.5 -> 0.2, HR@10", 3.71, "sampled", "README headline"),
    ("Week 4: dropout 0.5 -> 0.2, NDCG@10", 6.33, "sampled", "README headline"),
    ("Week 4: BERT4Rec vs default SASRec, HR@10", 3.39, "sampled", "controversy doc"),
    ("Week 4: SASRec vs BERT4Rec at matched dropout, HR@10", 0.31, "sampled", "controversy doc"),
    ("M4 cross-check: RecBole vs this repo, HR@10", 0.61, "sampled", "M4 criterion"),
    ("M4 cross-check: RecBole vs this repo, NDCG@10", 7.41, "sampled", "M4 criterion"),
    ("Loss effect?: RecBole vs this repo, full HR@10", 40.1, "full", "controversy doc"),
    # Week 3 ablations, all measured against the 100-epoch baseline
    # (`ablation_ml1m_baseline_100ep`) so both sides share a budget. The table
    # originally compared them against the 200-epoch headline run, which inflated
    # every delta -- by +0.47% on sampled HR@10 but +5.36% on full HR@10, enough to
    # flip two conclusions. See REPRODUCTION_LOG.md.
    ("A1: sinusoidal vs learnable pos emb", -0.06, "sampled", "Week 3 ablation"),
    ("A1: none vs learnable pos emb", -1.05, "sampled", "Week 3 ablation"),
    ("A2: maxlen 100 vs 200", -1.15, "sampled", "Week 3 ablation"),
    ("A2: maxlen 50 vs 200", -3.61, "sampled", "Week 3 ablation"),
    ("A4: popularity vs uniform negatives", -7.51, "sampled", "Week 3 ablation"),
    ("A1: sinusoidal vs learnable pos emb, full", -7.11, "full", "Week 3 ablation"),
    ("A1: none vs learnable pos emb, full", -2.47, "full", "Week 3 ablation"),
    ("A2: maxlen 100 vs 200, full", -0.13, "full", "Week 3 ablation"),
    ("A2: maxlen 50 vs 200, full", -13.45, "full", "Week 3 ablation"),
    ("A4: popularity vs uniform negatives, full", -20.35, "full", "Week 3 ablation"),
]

# A claimed margin is a difference between two runs that were each measured once.
# The standard deviation of that difference is sqrt(2) x the per-run standard
# deviation, not 1x -- both sides carry independent noise. Two of those combined
# standard deviations is the usual rough bar for "not obviously noise".
# Stated as a rule of thumb, not a significance test: five seeds estimate a
# standard deviation only loosely, and nothing here is a formal hypothesis test.
FLOOR_MULTIPLIER = 2 * np.sqrt(2)


def collect(tracking_uri: str, prefix: str) -> dict[str, np.ndarray]:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("sequential-rec")
    runs = client.search_runs([exp.experiment_id])

    # seed 42 is the original `sasrec_ml1m` run; the rest are `sasrec_ml1m_seed<N>`.
    wanted = {r.data.tags.get("mlflow.runName") for r in runs}
    names = sorted(n for n in wanted if n and (n == prefix or n.startswith(f"{prefix}_seed")))

    out: dict[str, list] = {key: [] for _, key in METRICS}
    used = []
    for name in names:
        match = [r for r in runs if r.data.tags.get("mlflow.runName") == name]
        if not match:
            continue
        run = max(match, key=lambda r: r.info.start_time)
        if not all(key in run.data.metrics for _, key in METRICS):
            continue
        used.append(name)
        for _, key in METRICS:
            out[key].append(run.data.metrics[key])
    return {k: np.asarray(v) for k, v in out.items()}, used


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--prefix", default="sasrec_ml1m")
    args = parser.parse_args()

    vals, used = collect(args.tracking_uri, args.prefix)
    print(f"runs included ({len(used)}): {', '.join(used)}\n")

    worst: dict[str, float] = {"sampled": 0.0, "full": 0.0}
    print(f"{'metric':<20} {'mean':>8} {'std':>8} {'rel.std':>8} {'min':>8} {'max':>8} {'rel.range':>10}")
    for label, key in METRICS:
        v = vals[key]
        if len(v) < 2:
            print(f"{label:<20} {'(need >=2 runs)':>8}")
            continue
        rel_std = v.std(ddof=1) / v.mean() * 100
        rel_range = (v.max() - v.min()) / v.mean() * 100
        protocol = "full" if label.startswith("full") else "sampled"
        worst[protocol] = max(worst[protocol], rel_std)
        print(
            f"{label:<20} {v.mean():>8.4f} {v.std(ddof=1):>8.4f} {rel_std:>7.2f}% "
            f"{v.min():>8.4f} {v.max():>8.4f} {rel_range:>9.2f}%"
        )

    floors = {p: FLOOR_MULTIPLIER * s for p, s in worst.items()}
    print("\nNoise floors (2*sqrt(2) x worst per-run relative std, per protocol):")
    for protocol, floor in floors.items():
        print(f"  {protocol:<8} {floor:.2f}%")
    print()
    print(f"{'claimed margin':<52} {'size':>7} {'proto':>8}  verdict")
    for label, margin, protocol, where in CLAIMED_MARGINS:
        floor = floors[protocol]
        verdict = "above floor" if abs(margin) > floor else "INSIDE NOISE"
        print(f"{label:<52} {margin:>6.2f}% {protocol:>8}  {verdict}  ({where})")
