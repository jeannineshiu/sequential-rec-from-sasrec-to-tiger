"""Summarize seed-to-seed variance for one configuration family -- `--prefix` picks
it, defaulting to this repo's SASRec on ML-1M -- and use the measured spreads as
noise floors for the margins reported elsewhere in the project.

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
import json
from pathlib import Path
from typing import NamedTuple

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
# Each margin names the two CONFIGURATIONS it compares, so it can be judged against
# their own measured spread instead of one blanket number.
#
# Why this replaces a single floor: the blanket 0.96%/3.37% figures are five seeds of
# the BCE baseline, and applying them elsewhere is wrong in BOTH directions. On the CE
# control the true full HR@10 spread is 0.19%, so the blanket floor was far too wide
# and hid width64's real effect for a day. On RecBole's dropout-0.2 runs it is 1.83%,
# so the blanket floor was too NARROW and would wave through a full-ranking margin that
# its own seeds cannot support. There is no constant that fixes both.
#
# side_a / side_b name a family in FAMILIES, or None where that configuration has no
# seeds of its own -- printed as "borrowed" rather than quietly using a proxy.
# (label, relative % difference, protocol, where it is claimed, side_a, side_b)
CLAIMED_MARGINS = [
    (
        "Dropout default 0.5 -> 0.2, HR@10",
        3.71,
        "sampled",
        "README headline",
        "recbole_d02_uni100",
        None,
    ),
    (
        "Dropout default 0.5 -> 0.2, NDCG@10",
        6.33,
        "sampled",
        "README headline",
        "recbole_d02_uni100",
        None,
    ),
    ("BERT4Rec vs default SASRec, HR@10", 3.39, "sampled", "controversy doc", None, None),
    (
        "SASRec vs BERT4Rec at matched dropout, HR@10",
        0.31,
        "sampled",
        "controversy doc",
        "recbole_d02_uni100",
        None,
    ),
    (
        "M4 cross-check: RecBole vs this repo, HR@10",
        0.61,
        "sampled",
        "M4 criterion",
        "recbole_d02",
        "bce",
    ),
    (
        "M4 cross-check: RecBole vs this repo, NDCG@10",
        7.41,
        "sampled",
        "M4 criterion",
        "recbole_d02",
        "bce",
    ),
    (
        "Loss effect?: RecBole vs this repo, full HR@10",
        40.07,
        "full",
        "controversy doc",
        "recbole_d02",
        "bce",
    ),
    # The loss-only ablation (configs/ablation/sasrec_ml1m_loss_ce.yaml): the same
    # model, seed, schedule and data as `sasrec_ml1m`, trained with a full-catalog
    # softmax instead of BCE against one sampled negative.
    ("CE vs BCE loss, sampled HR@10", -0.38, "sampled", "loss ablation", "ce", "bce"),
    ("CE vs BCE loss, sampled NDCG@10", 3.10, "sampled", "loss ablation", "ce", "bce"),
    ("CE vs BCE loss, full HR@10", 22.54, "full", "loss ablation", "ce", "bce"),
    ("CE vs BCE loss, full NDCG@10", 32.29, "full", "loss ablation", "ce", "bce"),
    # The residual RecBole still has over CE, i.e. the part the objective does NOT
    # explain -- architecture (d=64 vs 50, 2 heads vs 1, inner 256) and batch size.
    # Restated on 3-seed means once RecBole had seeds of its own: seed 42 turned out
    # to be the lowest of the three on full HR@10, so the single-seed residual
    # (+14.30% / +16.06%) understated it.
    (
        "Residual: RecBole vs CE, full HR@10 (3-seed means)",
        15.87,
        "full",
        "loss ablation",
        "recbole_d02",
        "ce",
    ),
    (
        "Residual: RecBole vs CE, full NDCG@10 (3-seed means)",
        17.16,
        "full",
        "loss ablation",
        "recbole_d02",
        "ce",
    ),
    # The three architecture arms, each one field against the CE control. Judged
    # against each arm's OWN spread: heads2's is seven times the control's, which is
    # why three seeds cannot settle it.
    (
        "P1b width64 vs CE, full HR@10 (3 seeds)",
        2.72,
        "full",
        "seeded, p=0.034",
        "ce_width64",
        "ce",
    ),
    (
        "P1b width64 vs CE, full NDCG@10 (3 seeds)",
        2.74,
        "full",
        "seeded, p=0.006",
        "ce_width64",
        "ce",
    ),
    (
        "P1b batch19 vs CE, full HR@10 (3 seeds)",
        0.36,
        "full",
        "seeded, p=0.474 (null)",
        "ce_batch19",
        "ce",
    ),
    (
        "P1b batch19 vs CE, full NDCG@10 (3 seeds)",
        0.02,
        "full",
        "seeded, p=0.975 (null)",
        "ce_batch19",
        "ce",
    ),
    (
        "P1b heads2 vs CE, full HR@10 (3 seeds)",
        1.11,
        "full",
        "seeded, p=0.285 (undecided)",
        "ce_heads2",
        "ce",
    ),
    (
        "P1b heads2 vs CE, full NDCG@10 (3 seeds)",
        0.99,
        "full",
        "seeded, p=0.304 (undecided)",
        "ce_heads2",
        "ce",
    ),
    # Ablations, all measured against the 100-epoch baseline
    # (`ablation_ml1m_baseline_100ep`) so both sides share a budget. The table
    # originally compared them against the 200-epoch headline run, which inflated
    # every delta -- by +0.47% on sampled HR@10 but +5.36% on full HR@10, enough to
    # flip two conclusions. See REPRODUCTION_LOG.md. Both sides are the BCE
    # configuration, but at 100 epochs rather than the 200 the seeds were run at.
    ("A1: sinusoidal vs learnable pos emb", -0.06, "sampled", "ablation table", "bce", "bce"),
    ("A1: none vs learnable pos emb", -1.05, "sampled", "ablation table", "bce", "bce"),
    ("A2: maxlen 100 vs 200", -1.15, "sampled", "ablation table", "bce", "bce"),
    ("A2: maxlen 50 vs 200", -3.61, "sampled", "ablation table", "bce", "bce"),
    ("A4: popularity vs uniform negatives", -7.51, "sampled", "ablation table", "bce", "bce"),
    ("A1: sinusoidal vs learnable pos emb, full", -7.11, "full", "ablation table", "bce", "bce"),
    ("A1: none vs learnable pos emb, full", -2.47, "full", "ablation table", "bce", "bce"),
    ("A2: maxlen 100 vs 200, full", -0.13, "full", "ablation table", "bce", "bce"),
    ("A2: maxlen 50 vs 200, full", -13.45, "full", "ablation table", "bce", "bce"),
    ("A4: popularity vs uniform negatives, full", -20.35, "full", "ablation table", "bce", "bce"),
    # Amazon Beauty, atomic vs semantic IDs. Both sides are seeded as of 2026-08-28:
    # the SASRec side by P5(b), the generative side by N2. Beauty is a different
    # dataset from every other family in this file, which is why borrowing ML-1M's
    # spread for it was the largest proxy in the repo until the SASRec side was run,
    # and why the generative half was the last one left.
    (
        "Beauty: GenRec vs SASRec, sampled HR@10",
        -28.95,
        "sampled",
        "atomic-vs-semantic table",
        "beauty_sasrec",
        "beauty_genrec",
    ),
    (
        "Beauty: GenRec vs SASRec, sampled NDCG@10",
        -35.27,
        "sampled",
        "atomic-vs-semantic table",
        "beauty_sasrec",
        "beauty_genrec",
    ),
    (
        "Beauty: GenRec vs SASRec, full HR@10",
        -57.83,
        "full",
        "atomic-vs-semantic table",
        "beauty_sasrec",
        "beauty_genrec",
    ),
    (
        "Beauty: GenRec vs SASRec, full NDCG@10",
        -56.63,
        "full",
        "atomic-vs-semantic table",
        "beauty_sasrec",
        "beauty_genrec",
    ),
    # ML-1M, the same comparison in the dense regime. The SASRec side is the
    # five-seed BCE baseline; the generative side was seeded by N2 alongside
    # Beauty's, so the second deliverable no longer has a borrowed side on
    # either dataset.
    (
        "ML-1M: GenRec vs SASRec, sampled HR@10",
        -23.57,
        "sampled",
        "atomic-vs-semantic table",
        "bce",
        "ml1m_genrec",
    ),
    (
        "ML-1M: GenRec vs SASRec, sampled NDCG@10",
        -32.80,
        "sampled",
        "atomic-vs-semantic table",
        "bce",
        "ml1m_genrec",
    ),
    (
        "ML-1M: GenRec vs SASRec, full HR@10",
        -52.98,
        "full",
        "atomic-vs-semantic table",
        "bce",
        "ml1m_genrec",
    ),
    (
        "ML-1M: GenRec vs SASRec, full NDCG@10",
        -54.11,
        "full",
        "atomic-vs-semantic table",
        "bce",
        "ml1m_genrec",
    ),
]

# Configuration families that have seeds of their own. Anything not listed here has
# no measured spread and is reported as borrowed rather than silently proxied.
# uni100 families carry only the sampled pair -- RecBole never ranked the full catalog.
SAMPLED_ONLY = METRICS[:2]


class Family(NamedTuple):
    """One configuration whose spread is measured rather than borrowed.

    `metrics` names the metrics its MLflow runs are required to carry. `exhaustive`
    points at a `results/tables/*.json` that measures the full-ranking spread
    outside MLflow, for a family whose logged full-ranking metric is on a
    different protocol from the margin it has to judge -- see the GenRec entries.
    """

    prefix: str
    desc: str
    metrics: list[tuple[str, str]] = METRICS
    exhaustive: str | None = None


# The generative families take their sampled spread from MLflow and their
# full-ranking spread from `genrec_seed_spread`, and the split is not
# bookkeeping. `train_genrec` logs `test_full_*` from a beam-20 decode, while
# every full-ranking margin on the page comes from the exhaustive pass -- 0.0329
# against 0.0250 on Beauty, because the beam flatters the model. Using the logged
# spread as the floor for the exhaustive margin would be a proxy of exactly the
# kind this file exists to remove. The sampled pair has no such problem: it is the
# same `evaluate_sampled` for both models.
FAMILIES = {
    "bce": Family("sasrec_ml1m", "this repo's SASRec, BCE, 200ep"),
    "ce": Family("ablation_ml1m_loss_ce", "this repo's SASRec, CE control"),
    "ce_width64": Family("ablation_ml1m_ce_width64", "CE + hidden_dim 64"),
    "ce_batch19": Family("ablation_ml1m_ce_batch19", "CE + batch_size 19"),
    "ce_heads2": Family("ablation_ml1m_ce_heads2", "CE + 2 heads"),
    "recbole_d02": Family(
        "sasrec_recbole_1x_dropout02_ourprotocol",
        "RecBole SASRec dropout 0.2, rescored here",
    ),
    "beauty_sasrec": Family("sasrec_beauty", "this repo's SASRec on Beauty, BCE"),
    "recbole_d02_uni100": Family(
        "sasrec_recbole_1x_dropout02",
        "RecBole SASRec dropout 0.2, RecBole's own uni100",
        SAMPLED_ONLY,
    ),
    "beauty_genrec": Family(
        "genrec_beauty",
        "this repo's GenRec on Beauty, semantic IDs",
        SAMPLED_ONLY,
        "results/tables/genrec_seed_spread_amazon-beauty.json",
    ),
    "ml1m_genrec": Family(
        "genrec_ml1m",
        "this repo's GenRec on ML-1M, semantic IDs",
        SAMPLED_ONLY,
        "results/tables/genrec_seed_spread_ml-1m.json",
    ),
}

# A claimed margin is a difference between two runs that were each measured once.
# The standard deviation of that difference is sqrt(2) x the per-run standard
# deviation, not 1x -- both sides carry independent noise. Two of those combined
# standard deviations is the usual rough bar for "not obviously noise".
# Stated as a rule of thumb, not a significance test: five seeds estimate a
# standard deviation only loosely, and nothing here is a formal hypothesis test.
FLOOR_MULTIPLIER = 2 * np.sqrt(2)


def collect(
    tracking_uri: str, prefix: str, require: list[tuple[str, str]] | None = None
) -> dict[str, np.ndarray]:
    """Gather every seed of one configuration family: `<prefix>` plus `<prefix>_seed<N>`.

    `require` names the metrics a run must carry to be counted, defaulting to all
    four. RecBole's own uni100 runs log only the sampled pair (they were never
    evaluated against the full catalog inside RecBole), so a family measured on
    that protocol passes just those two -- without this they are silently dropped
    and the dropout headline has no measured floor of its own.
    """
    require = require or METRICS
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("sequential-rec")
    runs = client.search_runs([exp.experiment_id])

    # seed 42 is the original `sasrec_ml1m` run; the rest are `sasrec_ml1m_seed<N>`.
    wanted = {r.data.tags.get("mlflow.runName") for r in runs}
    names = sorted(n for n in wanted if n and (n == prefix or n.startswith(f"{prefix}_seed")))

    out: dict[str, list] = {key: [] for _, key in require}
    used = []
    for name in names:
        match = [r for r in runs if r.data.tags.get("mlflow.runName") == name]
        if not match:
            continue
        run = max(match, key=lambda r: r.info.start_time)
        if not all(key in run.data.metrics for _, key in require):
            continue
        used.append(name)
        for _, key in require:
            out[key].append(run.data.metrics[key])
    return {k: np.asarray(v) for k, v in out.items()}, used


def family_spreads(tracking_uri: str) -> dict[str, tuple[int, dict[str, float]]]:
    """Per-run relative standard deviation for every configuration family that has
    seeds, keyed by family then protocol. This is what replaces the single blanket
    floor: the spread is a property of the configuration, not of the repo."""
    out: dict[str, tuple[int, dict[str, float]]] = {}
    for key, family in FAMILIES.items():
        vals, used = collect(tracking_uri, family.prefix, require=family.metrics)
        per: dict[str, float] = {}
        if len(used) >= 2:
            for label, metric_key in family.metrics:
                v = vals[metric_key]
                if len(v) < 2:
                    continue
                protocol = "full" if label.startswith("full") else "sampled"
                per[protocol] = max(per.get(protocol, 0.0), v.std(ddof=1) / v.mean() * 100)
        n_seeds = len(used)
        if family.exhaustive:
            measured = exhaustive_spread(family.exhaustive)
            if measured:
                n_exhaustive, rel_std = measured
                per["full"] = rel_std
                n_seeds = max(n_seeds, n_exhaustive)
        if per:
            out[key] = (n_seeds, per)
    return out


def exhaustive_spread(path: str) -> tuple[int, float] | None:
    """(seeds, worst relative std) on the full-ranking protocol, from a
    `genrec_seed_spread` artifact. None when it has not been generated, which is
    the honest state before the seeds have run -- the family then reports only
    what MLflow measures and its full-ranking rows stay marked borrowed."""
    file = Path(path)
    if not file.exists():
        return None
    spread = json.loads(file.read_text())["spread"]["full"]
    entries = [v for v in spread.values() if v["n"] >= 2]
    if not entries:
        return None
    return min(v["n"] for v in entries), max(v["rel_std"] for v in entries)


def margin_floor(
    spreads: dict[str, tuple[int, dict[str, float]]],
    protocol: str,
    side_a: str | None,
    side_b: str | None,
    fallback: float,
) -> tuple[float, str]:
    """Floor for one claimed margin: 2 * sqrt(sd_a^2 + sd_b^2) over the two
    configurations being compared, in relative %.

    A side with no seeds of its own contributes its partner's spread, and the row is
    marked "borrowed" so the reader can see which verdicts still rest on a proxy
    rather than a measurement. If neither side is measured on this protocol the
    blanket floor is used and the row says so.
    """
    sds = []
    borrowed = 0
    for side in (side_a, side_b):
        got = spreads.get(side) if side else None
        sd = got[1].get(protocol) if got else None
        if sd is None:
            borrowed += 1
            sds.append(None)
        else:
            sds.append(sd)
    measured = [x for x in sds if x is not None]
    if not measured:
        return fallback, "; floor borrowed from the BCE baseline"
    filled = [x if x is not None else max(measured) for x in sds]
    floor = 2 * float(np.sqrt(filled[0] ** 2 + filled[1] ** 2))
    note = "" if borrowed == 0 else f"; {borrowed} side(s) borrowed"
    return floor, note


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

    header = (
        f"{'metric':<20}{'arm mean':>10}{'ctl mean':>10}{'arm rsd':>9}{'ctl rsd':>9}"
        f"{'delta':>9}{'95% CI':>18}{'p':>9}"
    )
    print(header)
    for label, key in METRICS:
        x, y = a[key], c[key]
        d = (x.mean() - y.mean()) / y.mean() * 100
        _, pv = stats.ttest_ind(x, y, equal_var=False)
        # Welch CI on the same relative scale as `delta`, so the interval and the
        # point estimate in the README tables come from one place.
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        df = se**4 / (
            (x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1)
            + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1)
        )
        half = stats.t.ppf(0.975, df) * se / y.mean() * 100
        ci = f"[{d - half:+.2f}%, {d + half:+.2f}%]"
        print(
            f"{label:<20}{x.mean():>10.4f}{y.mean():>10.4f}"
            f"{x.std(ddof=1) / x.mean() * 100:>8.2f}%{y.std(ddof=1) / y.mean() * 100:>8.2f}%"
            f"{d:>8.2f}%{ci:>18}{pv:>9.4f}"
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
    print("\nBlanket floors from THIS family (2*sqrt(2) x worst per-run relative std):")
    for protocol, floor in floors.items():
        print(f"  {protocol:<8} {floor:.2f}%")

    print("\nMeasured spread per configuration family (worst per-run rel. std, per protocol):")
    spreads = family_spreads(args.tracking_uri)
    for key, family in FAMILIES.items():
        got = spreads.get(key)
        if not got:
            print(f"  {key:<20} (no seeds found under {family.prefix!r})")
            continue
        n, per = got
        cells = "  ".join(f"{p}={v:.2f}%" for p, v in sorted(per.items()))
        print(f"  {key:<20} {n} seeds  {cells:<28} {family.desc}")

    # Each margin against the two configurations it actually compares. sd of a
    # difference is sqrt(sd_a^2 + sd_b^2), which only collapses to sqrt(2)*sd when
    # both sides have the same spread -- and the whole point of the table above is
    # that they do not.
    print()
    header = f"{'claimed margin':<52} {'size':>7} {'proto':>8} {'floor':>7}  verdict"
    print(header)
    for label, margin, protocol, where, side_a, side_b in CLAIMED_MARGINS:
        floor, note = margin_floor(spreads, protocol, side_a, side_b, floors[protocol])
        verdict = "above floor" if abs(margin) > floor else "INSIDE NOISE"
        print(
            f"{label:<52} {margin:>6.2f}% {protocol:>8} {floor:>6.2f}%  "
            f"{verdict}  ({where}{note})"
        )
