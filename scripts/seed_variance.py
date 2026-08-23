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
    ("Dropout default 0.5 -> 0.2, HR@10", 3.71, "sampled", "README headline"),
    ("Dropout default 0.5 -> 0.2, NDCG@10", 6.33, "sampled", "README headline"),
    ("BERT4Rec vs default SASRec, HR@10", 3.39, "sampled", "controversy doc"),
    ("SASRec vs BERT4Rec at matched dropout, HR@10", 0.31, "sampled", "controversy doc"),
    ("M4 cross-check: RecBole vs this repo, HR@10", 0.61, "sampled", "M4 criterion"),
    ("M4 cross-check: RecBole vs this repo, NDCG@10", 7.41, "sampled", "M4 criterion"),
    ("Loss effect?: RecBole vs this repo, full HR@10", 40.07, "full", "controversy doc"),
    # The loss-only ablation (configs/ablation/sasrec_ml1m_loss_ce.yaml): the same
    # model, seed, schedule and data as `sasrec_ml1m`, trained with a full-catalog
    # softmax instead of BCE against one sampled negative. The residual rows are
    # what RecBole still has left over CE, i.e. the part the objective does NOT
    # explain -- architecture (d=64 vs 50, 2 heads vs 1, inner 256) and batch size.
    ("CE vs BCE loss, sampled HR@10", -0.38, "sampled", "loss ablation"),
    ("CE vs BCE loss, sampled NDCG@10", 3.10, "sampled", "loss ablation"),
    ("CE vs BCE loss, full HR@10", 22.54, "full", "loss ablation"),
    ("CE vs BCE loss, full NDCG@10", 32.29, "full", "loss ablation"),
    ("Residual: RecBole vs CE, full HR@10", 14.30, "full", "loss ablation"),
    ("Residual: RecBole vs CE, full NDCG@10", 16.06, "full", "loss ablation"),
    # The three architecture arms, each one field against the CE control.
    #
    # These rows are checked against the BLANKET floor below, which is measured on
    # five seeds of the BCE baseline. For the CE configurations it is a proxy and a
    # bad one: three seeds of the CE control put its full HR@10 spread at 0.18%,
    # not 1.19%, and width64's effect is significant (p=0.03) despite reading as
    # "INSIDE NOISE" here. The rows are kept because the blanket verdict is what
    # the rest of the repo quotes, and seeing it disagree with the measured result
    # is the point -- see "The architecture residual" in README.md.
    ("P1b width64 vs CE, sampled NDCG@10", 1.14, "sampled", "architecture arms"),
    ("P1b batch19 vs CE, full HR@10", -0.33, "full", "architecture arms (1 seed)"),
    ("P1b batch19 vs CE, full NDCG@10", -0.46, "full", "architecture arms (1 seed)"),
    ("P1b width64 vs CE, full HR@10", 3.33, "full", "architecture arms (1 seed)"),
    ("P1b width64 vs CE, full NDCG@10", 2.57, "full", "architecture arms (1 seed)"),
    ("P1b heads2 vs CE, full HR@10", -0.16, "full", "architecture arms (1 seed)"),
    ("P1b heads2 vs CE, full NDCG@10", 0.69, "full", "architecture arms (1 seed)"),
    # Seeded, 3 per arm, tested against the CE control's own measured spread
    # rather than the blanket floor. Reported by --arm-seeds.
    ("P1b width64 vs CE, full HR@10 (3 seeds)", 2.72, "full", "seeded, p=0.034"),
    ("P1b width64 vs CE, full NDCG@10 (3 seeds)", 2.74, "full", "seeded, p=0.006"),
    # Ablations, all measured against the 100-epoch baseline
    # (`ablation_ml1m_baseline_100ep`) so both sides share a budget. The table
    # originally compared them against the 200-epoch headline run, which inflated
    # every delta -- by +0.47% on sampled HR@10 but +5.36% on full HR@10, enough to
    # flip two conclusions. See REPRODUCTION_LOG.md.
    ("A1: sinusoidal vs learnable pos emb", -0.06, "sampled", "ablation table"),
    ("A1: none vs learnable pos emb", -1.05, "sampled", "ablation table"),
    ("A2: maxlen 100 vs 200", -1.15, "sampled", "ablation table"),
    ("A2: maxlen 50 vs 200", -3.61, "sampled", "ablation table"),
    ("A4: popularity vs uniform negatives", -7.51, "sampled", "ablation table"),
    ("A1: sinusoidal vs learnable pos emb, full", -7.11, "full", "ablation table"),
    ("A1: none vs learnable pos emb, full", -2.47, "full", "ablation table"),
    ("A2: maxlen 100 vs 200, full", -0.13, "full", "ablation table"),
    ("A2: maxlen 50 vs 200, full", -13.45, "full", "ablation table"),
    ("A4: popularity vs uniform negatives, full", -20.35, "full", "ablation table"),
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


def arm_seeds(tracking_uri: str, arm: str, control: str) -> None:
    """Compare two seeded arms using their OWN measured spread, not the blanket floor.

    The blanket floor is five seeds of the BCE baseline. Applying it to a CE
    configuration understated CE's reproducibility badly enough to hide a real
    effect: width64 read as "INSIDE NOISE" at 3.33% against a 3.37% floor, while
    the CE control's actual full HR@10 spread is 0.18%. Anything compared here is
    tested against the variance of the configurations being compared.
    """
    from scipy import stats

    a, a_used = collect(tracking_uri, arm)
    c, c_used = collect(tracking_uri, control)
    print(f"arm     ({len(a_used)}): {', '.join(a_used)}")
    print(f"control ({len(c_used)}): {', '.join(c_used)}\n")
    if len(a_used) < 2 or len(c_used) < 2:
        print("need >=2 seeds per arm")
        return

    print(f"{'metric':<20}{'arm mean':>10}{'ctl mean':>10}{'arm rsd':>9}{'ctl rsd':>9}{'delta':>9}{'p':>9}")
    for label, key in METRICS:
        x, y = a[key], c[key]
        d = (x.mean() - y.mean()) / y.mean() * 100
        _, pv = stats.ttest_ind(x, y, equal_var=False)
        print(
            f"{label:<20}{x.mean():>10.4f}{y.mean():>10.4f}"
            f"{x.std(ddof=1) / x.mean() * 100:>8.2f}%{y.std(ddof=1) / y.mean() * 100:>8.2f}%"
            f"{d:>8.2f}%{pv:>9.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--prefix", default="sasrec_ml1m")
    parser.add_argument(
        "--arm-seeds",
        nargs=2,
        metavar=("ARM", "CONTROL"),
        help="compare two seeded run families against their own spread, e.g. "
        "--arm-seeds ablation_ml1m_ce_width64 ablation_ml1m_loss_ce",
    )
    args = parser.parse_args()

    if args.arm_seeds:
        arm_seeds(args.tracking_uri, *args.arm_seeds)
        raise SystemExit(0)

    vals, used = collect(args.tracking_uri, args.prefix)
    print(f"runs included ({len(used)}): {', '.join(used)}\n")

    worst: dict[str, float] = {"sampled": 0.0, "full": 0.0}
    print(
        f"{'metric':<20} {'mean':>8} {'std':>8} {'rel.std':>8} {'min':>8} {'max':>8} {'rel.range':>10}"
    )
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
