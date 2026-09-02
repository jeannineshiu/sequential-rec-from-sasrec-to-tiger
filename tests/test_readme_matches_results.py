"""Every table on the README is transcribed by hand from `results/` and
`mlflow.db`, and CI only ran tests, lint and format -- so nothing mechanically
checked that the page still says what the artifacts say. A p-value of 0.059 stood
here for weeks and reproduced under no test; it was caught by reading, not by CI.

These tests parse the README's tables and assert every cell against the thing
that generated it: the JSON and markdown reports in `results/tables/`, the runs in
`mlflow.db`, and -- for the derived columns, which are where a corrected number is
easiest to leave stale -- arithmetic recomputed from those at full precision.

Two rules the checks follow. A cell is compared at the precision the README
printed it, so 0.0251 has to agree to four decimals and −57.7% to one. And a
missing table fails loudly rather than passing vacuously: if a header moves, the
test says which constant to update instead of quietly checking nothing.
"""

import json
import re

import pytest
from scipy.stats import fisher_exact

from scripts.seed_variance import METRICS, SAMPLED_ONLY, arm_stats, collect, family_spreads
from tests.readme_tables import README, ROOT, TABLES, Checker, artifact_table, number, table

TRACKING_URI = "sqlite:///mlflow.db"

# README metric label -> the metric key MLflow carries. Baseline runs (Popularity,
# BPR-MF) log the same numbers without the `test_` prefix, which `Runs.metric` falls
# back to, exactly as `export_results` coalesces them for master.md.
METRIC_KEYS = {
    "sampled HR@10": "sampled_HR_at_10",
    "sampled NDCG@10": "sampled_NDCG_at_10",
    "full HR@10": "full_HR_at_10",
    "full NDCG@10": "full_NDCG_at_10",
}
# The same four, keyed the way `seed_variance.collect` returns them.
SEED_KEYS = dict(METRICS)


class Runs:
    """Every reportable run in the tracking DB, keyed by name, at full precision.

    Loaded through `export_results`' own filters so that what a test can see is
    exactly what master.md reports: FINISHED runs, smoke and probe runs dropped.
    Two runs can share a name -- `sasrec_beauty` was re-run -- and the newest wins,
    which is the row the README quotes.
    """

    def __init__(self):
        from src.export_results import drop_debris, load_runs

        frame = drop_debris(load_runs())
        by_name: dict[str, list] = {}
        for _, row in frame.iterrows():
            by_name.setdefault(row["tags.mlflow.runName"], []).append(row)
        self._runs = {
            name: max(rows, key=lambda r: r["start_time"]) for name, rows in by_name.items()
        }

    def metric(self, run: str, label: str) -> float:
        assert run in self._runs, f"no run named {run} in mlflow.db"
        key = METRIC_KEYS.get(label, label)
        for column in (f"metrics.test_{key}", f"metrics.{key}"):
            value = self._runs[run].get(column)
            if value is not None and value == value:  # NaN means the run has no such metric
                return float(value)
        raise AssertionError(f"run {run} has no metric {key}")

    def param(self, run: str, key: str) -> str:
        return self._runs[run][f"params.{key}"]


@pytest.fixture(scope="module")
def runs() -> Runs:
    return Runs()


def rel(new: float, base: float) -> float:
    """Relative difference in percent, the form every Δ column on the page uses."""
    return (new - base) / base * 100.0


# == Results: SASRec reproduction — ML-1M, sampled protocol ==============

REPRODUCTION_HEADER = "| Model | HR@10 | NDCG@10 | Note |"
# The paper's own row is literature, not a run here, so it has no artifact to
# check -- every other row does, and the test asserts the split is exactly this.
REPRODUCTION_PAPER_ROW = "SASRec — Kang & McAuley (2018)"
REPRODUCTION_RUNS = {
    "Popularity": "popularity_ml1m",
    "BPR-MF": "bpr_mf_ml1m",
    "SASRec (this repo)": "sasrec_ml1m",
    "SASRec (RecBole, dropout 0.5 — its default)": "sasrec_recbole_1x",
    "SASRec (RecBole, dropout 0.2)": "sasrec_recbole_1x_dropout02",
    "SASRec (RecBole, dropout 0.2, rescored on frozen negatives)": (
        "sasrec_recbole_1x_dropout02_ourprotocol"
    ),
    "BERT4Rec (RecBole)": "bert4rec_recbole_1x",
}


def test_readme_reproduction_table_matches_mlflow(runs):
    rows = table(REPRODUCTION_HEADER, what="SASRec reproduction table")
    assert set(rows) - {REPRODUCTION_PAPER_ROW} == set(REPRODUCTION_RUNS), (
        "the reproduction table's rows changed; a row with no entry in "
        "REPRODUCTION_RUNS would go unchecked"
    )

    check = Checker("mlflow.db")
    for label, run in REPRODUCTION_RUNS.items():
        for column, metric in enumerate(["sampled HR@10", "sampled NDCG@10"]):
            check.check(f"{label} / {metric}", rows[label][column], runs.metric(run, metric))

    # The rescored row's note carries the comparison that opens key finding 2.
    note = rows["SASRec (RecBole, dropout 0.2, rescored on frozen negatives)"][2]
    rescored, ours = "sasrec_recbole_1x_dropout02_ourprotocol", "sasrec_ml1m"
    for quoted, metric in zip(note.split(";"), ["sampled HR@10", "sampled NDCG@10"], strict=True):
        check.check(
            f"rescored note / {metric}",
            quoted.split("%")[0] + "%",
            rel(runs.metric(rescored, metric), runs.metric(ours, metric)),
        )
    check.done()


# == Results: the dropout default =======================================

DROPOUT_HEADER = "| Comparison (protocol- and budget-matched) | HR@10 | NDCG@10 | Winner |"
# Each row is a relative difference between two runs. Nothing generates this
# table, so what is checked is that its arithmetic still holds at full precision --
# the four rows are the whole of key finding 1.
DROPOUT_PAIRS = {
    "BERT4Rec vs RecBole SASRec (dropout 0.5, default)": (
        "bert4rec_recbole_1x",
        "sasrec_recbole_1x",
    ),
    "BERT4Rec vs this repo's SASRec": ("bert4rec_recbole_1x", "sasrec_ml1m"),
    "BERT4Rec vs RecBole SASRec (dropout 0.2)": (
        "bert4rec_recbole_1x",
        "sasrec_recbole_1x_dropout02",
    ),
    "effect of the dropout default alone": (
        "sasrec_recbole_1x_dropout02",
        "sasrec_recbole_1x",
    ),
}


def test_readme_dropout_table_is_arithmetic_on_the_runs(runs):
    rows = table(DROPOUT_HEADER, what="dropout default table")
    assert set(rows) == set(DROPOUT_PAIRS), "the dropout table's rows changed"

    check = Checker("mlflow.db")
    for label, (a, b) in DROPOUT_PAIRS.items():
        for column, metric in enumerate(["sampled HR@10", "sampled NDCG@10"]):
            check.check(
                f"{label} / {metric}",
                rows[label][column],
                rel(runs.metric(a, metric), runs.metric(b, metric)),
            )
    check.done()


# == Results: sampled vs full-catalog ranking ===========================

FULL_RANKING_HEADING = "#### 2.1 Sampled versus full-catalog ranking"
FULL_RANKING_HEADER = "| Model | HR@10 | NDCG@10 |"
FULL_RANKING_RUNS = {
    "Popularity": "popularity_ml1m",
    "BPR-MF": "bpr_mf_ml1m",
    "SASRec (this repo)": "sasrec_ml1m",
    "SASRec (RecBole, dropout 0.2, rescored)": "sasrec_recbole_1x_dropout02_ourprotocol",
}


def test_readme_full_ranking_table_matches_mlflow(runs):
    rows = table(
        FULL_RANKING_HEADER, after=FULL_RANKING_HEADING, what="sampled-vs-full ranking table"
    )
    assert set(rows) == set(FULL_RANKING_RUNS), "the full-ranking table's rows changed"

    check = Checker("mlflow.db")
    for label, run in FULL_RANKING_RUNS.items():
        for column, metric in enumerate(["full HR@10", "full NDCG@10"]):
            check.check(f"{label} / {metric}", rows[label][column], runs.metric(run, metric))
    check.done()


# == Results: the training objective, isolated ==========================

OBJECTIVE_HEADER = (
    "| ML-1M, test, k=10 | BCE (control) | CE | Δ | RecBole | residual (RecBole vs CE) |"
)
OBJECTIVE_BCE, OBJECTIVE_CE = "sasrec_ml1m", "ablation_ml1m_loss_ce"
OBJECTIVE_RECBOLE = "sasrec_recbole_1x_dropout02_ourprotocol"


def test_readme_objective_table_matches_mlflow(runs):
    """The three measured columns against their runs, and the two derived columns
    recomputed -- 56% and 60% of the cross-framework gap are read off Δ."""
    rows = table(OBJECTIVE_HEADER, what="training objective table")
    assert set(rows) == set(METRIC_KEYS), "the objective table's rows changed"

    check = Checker("mlflow.db")
    for metric in rows:
        bce = runs.metric(OBJECTIVE_BCE, metric)
        ce = runs.metric(OBJECTIVE_CE, metric)
        recbole = runs.metric(OBJECTIVE_RECBOLE, metric)
        check.check(f"{metric} / BCE", rows[metric][0], bce)
        check.check(f"{metric} / CE", rows[metric][1], ce)
        check.check(f"{metric} / Δ", rows[metric][2], rel(ce, bce))
        check.check(f"{metric} / RecBole", rows[metric][3], recbole)
        check.check(f"{metric} / residual", rows[metric][4], rel(recbole, ce))
    check.done()


# == Results: the architecture residual =================================

ARCHITECTURE_HEADER = "| ML-1M, test, k=10 | CE (control) | batch19 | width64 | heads2 | RecBole |"
ARCHITECTURE_COLUMNS = [
    "ablation_ml1m_loss_ce",
    "ablation_ml1m_ce_batch19",
    "ablation_ml1m_ce_width64",
    "ablation_ml1m_ce_heads2",
    "sasrec_recbole_1x_dropout02_ourprotocol",
]
# The RecBole column's metrics come from the rescoring run, which does no training
# and logs no epoch count; its 200 epochs are RecBole's own training run.
ARCHITECTURE_EPOCH_RUNS = ARCHITECTURE_COLUMNS[:4] + ["sasrec_recbole_1x_dropout02"]


def test_readme_architecture_table_matches_mlflow(runs):
    rows = table(ARCHITECTURE_HEADER, what="architecture residual table")
    assert set(rows) == set(METRIC_KEYS) | {
        "epochs trained"
    }, "the architecture table's rows changed"

    check = Checker("mlflow.db")
    for metric in METRIC_KEYS:
        for column, run in enumerate(ARCHITECTURE_COLUMNS):
            check.check(f"{metric} / {run}", rows[metric][column], runs.metric(run, metric))
    for column, run in enumerate(ARCHITECTURE_EPOCH_RUNS):
        check.check(
            f"epochs trained / {run}",
            rows["epochs trained"][column],
            runs.metric(run, "epochs_trained"),
        )
    check.done()


THREE_SEED_HEADER = "| full ranking, 3 seeds each | CE control | batch19 | width64 | heads2 |"
THREE_SEED_COLUMNS = ARCHITECTURE_COLUMNS[:4]


def test_readme_three_seed_spread_table_matches_mlflow():
    """The row that carries "the spread is a property of the configuration": four
    means and four relative standard deviations, three seeds each."""
    rows = table(THREE_SEED_HEADER, what="three-seed full-ranking table")
    check = Checker("mlflow.db")
    for column, prefix in enumerate(THREE_SEED_COLUMNS):
        values, used = collect(TRACKING_URI, prefix)
        assert len(used) == 3, f"{prefix} has {len(used)} seeds, not the 3 the table claims"
        for metric in ("HR@10", "NDCG@10"):
            seeds = values[SEED_KEYS[f"full {metric}"]]
            check.check(f"{metric} mean / {prefix}", rows[f"{metric} mean"][column], seeds.mean())
            check.check(
                f"{metric} rel. std / {prefix}",
                rows[f"{metric} rel. std"][column],
                seeds.std(ddof=1) / seeds.mean() * 100,
            )
    check.done()


ARM_HEADERS = {
    "ablation_ml1m_ce_width64": "| width64 vs CE control, 3 seeds each | Δ | 95% CI | p (Welch) |",
    "ablation_ml1m_ce_batch19": "| batch19 vs CE control, 3 seeds each | Δ | 95% CI | p (Welch) |",
    "ablation_ml1m_ce_heads2": "| heads2 vs CE control, 3 seeds each | Δ | 95% CI | p (Welch) |",
}


@pytest.mark.parametrize("arm", list(ARM_HEADERS))
def test_readme_arm_tables_match_seed_variance(arm):
    """Δ, interval and p come from `seed_variance.arm_stats`, not from Welch
    recomputed here: the point is that the page agrees with the script that
    printed it, and a test carrying its own copy of the formula would only agree
    with the formula. `width64` is significant, `batch19` null and `heads2`
    unresolved -- the distinction rests entirely on these three cells per row."""
    rows = table(ARM_HEADERS[arm], what=f"{arm} vs control table")
    stats, arm_used, control_used = arm_stats(TRACKING_URI, arm, "ablation_ml1m_loss_ce")
    assert len(arm_used) == 3 and len(control_used) == 3, "the arm tables claim three seeds each"
    assert set(rows) == {label for label, _ in METRICS}, f"the {arm} table's rows changed"

    check = Checker("scripts/seed_variance.py")
    for label in rows:
        stat = stats[label]
        check.check(f"{label} / Δ", rows[label][0], stat.delta)
        low, high = rows[label][1].strip("[]").split()
        check.check(f"{label} / CI low", low, stat.ci_low)
        check.check(f"{label} / CI high", high, stat.ci_high)
        check.check(f"{label} / p", rows[label][2], stat.p)
    check.done()


# == Results: atomic vs semantic IDs, both datasets ======================

ATOMIC_HEADER = "| | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 | parameters |"
# The two datasets' tables share a header, so the section heading disambiguates.
# Their halves have different sources: the sampled pair is logged by training, the
# full pair comes from the exhaustive rescoring pass, and only GenRec's parameter
# count is recorded anywhere (SASRec's is not logged, so the `relative` cell is
# checked as arithmetic on the two printed counts).
ATOMIC_TABLES = {
    "beauty": (
        "#### 3.1 Accuracy, compression, and parameters — Amazon Beauty",
        "sasrec_beauty",
        "genrec_beauty",
        "atomic_vs_semantic.json",
    ),
    "ml-1m": (
        "#### 3.6 The dense regime — ML-1M",
        "sasrec_ml1m",
        "genrec_ml1m",
        "atomic_vs_semantic_ml-1m.json",
    ),
}


@pytest.mark.parametrize("dataset", list(ATOMIC_TABLES))
def test_readme_atomic_vs_semantic_table(runs, dataset):
    """The project's second deliverable, both datasets. The full-ranking columns
    are the ones the NaN bug got wrong, and the `relative` row is arithmetic on
    the two rows above it -- recomputed rather than trusted, because that is the
    cell a corrected number is easiest to leave stale."""
    heading, atomic_run, semantic_run, artifact = ATOMIC_TABLES[dataset]
    overall = json.loads((TABLES / artifact).read_text())["overall"]
    rows = table(ATOMIC_HEADER, after=heading, what=f"{dataset} atomic-vs-semantic table")
    assert set(rows) == {"SASRec (atomic)", "GenRec (semantic)", "relative"}

    check = Checker(f"mlflow.db and results/tables/{artifact}")
    measured = {}
    for column, metric in enumerate(["sampled HR@10", "sampled NDCG@10"]):
        measured[column] = (runs.metric(atomic_run, metric), runs.metric(semantic_run, metric))
    for column, metric in enumerate(["HR@10", "NDCG@10"], start=2):
        measured[column] = (
            overall["SASRec (atomic)"][metric],
            overall["GenRec (semantic)"][metric],
        )

    for column, (atomic, semantic) in measured.items():
        check.check(f"SASRec / column {column}", rows["SASRec (atomic)"][column], atomic)
        check.check(f"GenRec / column {column}", rows["GenRec (semantic)"][column], semantic)
        check.check(f"relative / column {column}", rows["relative"][column], rel(semantic, atomic))

    # The parameter column: GenRec's count is logged, SASRec's is not, and the
    # relative cell is a ratio rather than a difference -- it is the compression
    # claim, so it is checked against what the two printed counts actually say.
    check.check(
        "GenRec / parameters",
        rows["GenRec (semantic)"][4],
        float(runs.param(semantic_run, "n_params")),
    )
    check.check(
        "relative / parameters",
        rows["relative"][4],
        number(rows["GenRec (semantic)"][4]) / number(rows["SASRec (atomic)"][4]) * 100,
    )
    check.done()


BEAUTY_SASREC_HEADER = (
    "| Beauty SASRec, 3 seeds | mean | rel. std | min | max | borrowed ML-1M proxy |"
)


def test_readme_beauty_sasrec_seed_table_matches_mlflow():
    """Beauty's own noise floor, and the ML-1M figures it replaced."""
    rows = table(BEAUTY_SASREC_HEADER, what="Beauty SASRec seed table")
    beauty, used = collect(TRACKING_URI, "sasrec_beauty")
    ml1m, _ = collect(TRACKING_URI, "sasrec_ml1m")
    assert len(used) == 3, f"the table claims three Beauty seeds, mlflow has {len(used)}"
    assert set(rows) == {label for label, _ in METRICS}

    check = Checker("mlflow.db")
    for label in rows:
        seeds = beauty[SEED_KEYS[label]]
        check.check(f"{label} / mean", rows[label][0], seeds.mean())
        check.check(f"{label} / rel. std", rows[label][1], seeds.std(ddof=1) / seeds.mean() * 100)
        check.check(f"{label} / min", rows[label][2], seeds.min())
        check.check(f"{label} / max", rows[label][3], seeds.max())

        # The borrowed column quotes ML-1M's five-seed spread: per metric where the
        # README names one, and the protocol figure -- the worse of that protocol's
        # two metrics, which is what `seed_variance` actually borrows -- for the
        # sampled pair. Either is a real measurement; anything else is not.
        protocol = "full" if label.startswith("full") else "sampled"
        candidates = {
            float(f"{ml1m[key].std(ddof=1) / ml1m[key].mean() * 100:.2f}")
            for name, key in SEED_KEYS.items()
            if name == label or name.startswith(protocol)
        }
        proxy = number(rows[label][4])
        if proxy not in candidates:
            check.mismatches.append(
                f"{label} / borrowed proxy: README {proxy}% is not an ML-1M spread {candidates}"
            )
    check.done()


GENREC_SEED_TABLES = {
    "amazon-beauty": (
        "| Beauty GenRec, 3 seeds | mean | rel. std | min | max |",
        "genrec_beauty",
    ),
    "ml-1m": ("| ML-1M GenRec, 3 seeds | mean | rel. std | min | max |", "genrec_ml1m"),
}


@pytest.mark.parametrize("dataset", list(GENREC_SEED_TABLES))
def test_readme_genrec_seed_tables(dataset):
    """GenRec's spread comes from two places and the split matters: `train_genrec`
    logs a beam-20 full-ranking number, while every full-ranking margin on the page
    is exhaustive, so those rows come from the seed-spread artifact and only the
    sampled pair from MLflow. Reading the logged spread as the exhaustive one would
    be the proxy this section exists to remove."""
    header, prefix = GENREC_SEED_TABLES[dataset]
    rows = table(header, what=f"{dataset} GenRec seed table")
    spread = json.loads((TABLES / f"genrec_seed_spread_{dataset}.json").read_text())["spread"]

    check = Checker(f"mlflow.db and results/tables/genrec_seed_spread_{dataset}.json")
    for label, cells in rows.items():
        if label.endswith("exhaustive"):
            stat = spread["full"][label.split(",")[0].replace("full ", "")]
            values = (stat["mean"], stat["rel_std"], stat["min"], stat["max"])
        else:
            seeds, used = collect(TRACKING_URI, prefix, require=SAMPLED_ONLY)
            assert len(used) == 3, f"the table claims three seeds, mlflow has {len(used)}"
            s = seeds[SEED_KEYS[label]]
            values = (s.mean(), s.std(ddof=1) / s.mean() * 100, s.min(), s.max())
        for column, (name, value) in enumerate(
            zip(["mean", "rel. std", "min", "max"], values, strict=True)
        ):
            check.check(f"{label} / {name}", cells[column], value)
    check.done()


# == Results: cold start ================================================

COLD_START_JSON = TABLES / "atomic_vs_semantic.json"
COLD_START_HEADER = "| bucket | users | SASRec | GenRec | GenRec debiased α=1 |"
# The README labels buckets with their frequency range; the JSON keys them by name.
BUCKET_ALIASES = {
    "unseen (0)": "unseen",
    "tail (1–4)": "tail",
    "torso (5–19)": "torso",
    "head (20+)": "head",
    "overall": "overall",
}
# Column order in the README table, after `bucket` and `users`.
COLUMN_MODELS = [
    "SASRec (atomic)",
    "GenRec (semantic)",
    "GenRec (semantic), debiased a=1",
]


@pytest.fixture(scope="module")
def cold_start():
    if not COLD_START_JSON.exists():
        pytest.skip("results/tables/atomic_vs_semantic.json not generated")
    return json.loads(COLD_START_JSON.read_text())


def test_readme_cold_start_table_matches_generated_json(cold_start):
    rows = table(COLD_START_HEADER, what="cold-start table")
    assert set(rows) == set(
        BUCKET_ALIASES
    ), f"README buckets {sorted(rows)} do not match {sorted(BUCKET_ALIASES)}"

    check = Checker("results/tables/atomic_vs_semantic.json")
    for label, cells in rows.items():
        bucket = cold_start[BUCKET_ALIASES[label]]
        check.check(f"{label} / users", cells[0], bucket["n_users"])
        for column, model in enumerate(COLUMN_MODELS, start=1):
            check.check(f"{label} / {model}", cells[column], bucket[model]["HR@10"])
    check.done()


def test_readme_unseen_hit_counts_match_generated_json(cold_start):
    """The reach claim is quoted as counts, not rates, and the counts are what the
    Fisher tests run on. HR x n_users is exact: HR@k in a bucket is hits/users."""
    unseen = cold_start["unseen"]
    n = unseen["n_users"]
    debiased = round(unseen["GenRec (semantic), debiased a=1"]["HR@10"] * n)
    as_trained = round(unseen["GenRec (semantic)"]["HR@10"] * n)

    text = README.read_text()
    for count, phrase in (
        (debiased, f"{debiased} hits in {n}"),
        (as_trained, f"{as_trained} hit in {n}"),
    ):
        assert phrase in text, (
            f"README should quote '{phrase}' for the unseen bucket; the generated counts are "
            f"{debiased} debiased and {as_trained} as trained out of {n}"
        )


UNSEEN_SEED_HEADER = "| unseen bucket, 138 users | seed 42 | seed 1 | seed 2 |"
UNSEEN_SEED_ROWS = {
    "debiased α=1 hits": "GenRec (semantic), debiased a=1",
    "as trained hits": "GenRec (semantic)",
}


def test_readme_unseen_seed_table_matches_seed_spread():
    """The three-seed version of the reach claim. The p-values are recomputed from
    the stored hit counts with the same Fisher call `compare_atomic_vs_semantic`
    makes -- they were never written to an artifact, which is how "10 hits" reached
    the headline as if it were the result rather than the top of a 7-10 range."""
    rows = table(UNSEEN_SEED_HEADER, what="per-seed unseen bucket table")
    spread = json.loads((TABLES / "genrec_seed_spread_amazon-beauty.json").read_text())["spread"]
    unseen = spread["buckets"]["unseen"]
    n = unseen["n_users"]
    assert f"{n} users" in UNSEEN_SEED_HEADER, "the header quotes the bucket size"
    assert set(rows) == set(UNSEEN_SEED_ROWS) | {"one-sided p vs SASRec's 0"}

    check = Checker("results/tables/genrec_seed_spread_amazon-beauty.json")
    for label, model in UNSEEN_SEED_ROWS.items():
        hits = unseen[model]["hits"]
        for column, count in enumerate(hits):
            check.check(f"{label} / seed column {column}", rows[label][column], count)

    # SASRec retrieves nothing in this bucket under any seed, so every test is
    # `hits in 138` against `0 in 138`, one-sided in GenRec's favour.
    for column, count in enumerate(unseen["GenRec (semantic), debiased a=1"]["hits"]):
        p = fisher_exact([[count, n - count], [0, n]], alternative="greater")[1]
        check.check(
            f"one-sided p / seed column {column}", rows["one-sided p vs SASRec's 0"][column], p
        )
    check.done()


# == Results: mechanism -- diversity collapse ===========================
#
# Added 2026-08-27, after a padding token spent weeks inside these numbers: the
# debiased row read 1,976 items and 11.6% unseen, and nothing compared it to the
# artifact it was transcribed from. This table has no JSON companion, so the
# generated markdown is the source of truth and is parsed directly.

DIVERSITY_HEADER = (
    "| model | distinct items across all top-10s | median train freq | "
    "% head | % torso | % tail | % unseen |"
)
# README label -> the label the generating script writes.
DIVERSITY_MODELS = {
    "SASRec (atomic)": "SASRec (atomic)",
    "GenRec (semantic)": "GenRec (semantic)",
    "GenRec debiased α=1": "GenRec (semantic), debiased a=1",
}
# README column -> its index in the generated row, which carries an extra `mean`.
DIVERSITY_COLUMNS = {"distinct": 0, "median": 1, "head": 3, "torso": 4, "tail": 5, "unseen": 6}


def test_readme_diversity_table_matches_generated_table():
    rows = table(DIVERSITY_HEADER, what="diversity table")
    generated = artifact_table(TABLES / "genrec_diagnosis.md", width=8)
    assert set(rows) == set(
        DIVERSITY_MODELS
    ), f"README diversity rows {sorted(rows)} do not match {sorted(DIVERSITY_MODELS)}"

    check = Checker("results/tables/genrec_diagnosis.md")
    for label, cells in rows.items():
        source = generated[DIVERSITY_MODELS[label]]
        for column, (name, index) in enumerate(DIVERSITY_COLUMNS.items()):
            check.check(f"{label} / {name}", cells[column], number(source[index]))
    check.done()


# == Results: what the first code is worth ==============================

ORACLE_HEADER = "| oracle depth | median candidates | HR@10 | NDCG@10 |"
# The README shortens the depth-0 label; the artifact carries a fifth column
# (`vs d=0`) the README leaves out.
ORACLE_ROWS = {"0 (as it runs)": "0 (as trained)", "1": "1", "2": "2", "3": "3"}

LEVEL1_HEADER = "| the model's level-1 code | users | unaided HR@10 |"
LEVEL1_ROWS = {
    "top-1 correct": "top-1 correct",
    "in its top-10": "in top-10",
    "in its top-64": "in top-64",
    "outside top-64": "outside top-64",
}


def test_readme_oracle_table_matches_first_code_ceiling():
    rows = table(ORACLE_HEADER, what="oracle depth table")
    generated = artifact_table(TABLES / "first_code_ceiling.md", width=5)
    assert set(rows) == set(ORACLE_ROWS), "the oracle table's rows changed"

    check = Checker("results/tables/first_code_ceiling.md")
    for label, source_label in ORACLE_ROWS.items():
        source = generated[source_label]
        for column, name in enumerate(["median candidates", "HR@10", "NDCG@10"]):
            check.check(f"depth {label} / {name}", rows[label][column], number(source[column]))
    check.done()


def test_readme_level1_split_table_matches_first_code_ceiling():
    rows = table(LEVEL1_HEADER, what="level-1 split table")
    generated = artifact_table(TABLES / "first_code_ceiling.md", width=3)
    assert set(rows) == set(LEVEL1_ROWS), "the level-1 split table's rows changed"

    check = Checker("results/tables/first_code_ceiling.md")
    for label, source_label in LEVEL1_ROWS.items():
        source = generated[source_label]
        for column, name in enumerate(["users", "unaided HR@10"]):
            check.check(f"{label} / {name}", rows[label][column], number(source[column]))
    check.done()


# == Results: ablations =================================================

ABLATION_HEADER = "| Ablation | sampled HR@10 | Δ | full HR@10 | Δ | avg s/epoch |"
ABLATION_BASELINE = "Baseline (learnable pos emb, maxlen 200)"
ABLATION_RUNS = {
    ABLATION_BASELINE: "ablation_ml1m_baseline_100ep",
    "positional embedding = none": "ablation_ml1m_posemb_none",
    "positional embedding = sinusoidal": "ablation_ml1m_posemb_sinusoidal",
    "maxlen = 50": "ablation_ml1m_maxlen50",
    "maxlen = 100": "ablation_ml1m_maxlen100",
    "negative sampling = popularity-weighted": "ablation_ml1m_negsampling_popularity",
}


def test_readme_ablation_table_matches_mlflow(runs):
    """Both Δ columns are recomputed against the baseline run rather than read.
    They are the whole content of the section -- two of these deltas already
    reversed once, when the arms were rebaselined onto a matched budget."""
    rows = table(ABLATION_HEADER, what="ablations table")
    assert set(rows) == set(ABLATION_RUNS), "the ablations table's rows changed"

    check = Checker("mlflow.db")
    base = ABLATION_RUNS[ABLATION_BASELINE]
    for label, run in ABLATION_RUNS.items():
        for value_column, metric in ((0, "sampled HR@10"), (2, "full HR@10")):
            value = runs.metric(run, metric)
            check.check(f"{label} / {metric}", rows[label][value_column], value)
            if label != ABLATION_BASELINE:
                check.check(
                    f"{label} / {metric} Δ",
                    rows[label][value_column + 1],
                    rel(value, runs.metric(base, metric)),
                )
        check.check(
            f"{label} / avg s/epoch", rows[label][4], runs.metric(run, "avg_epoch_time_sec")
        )
    check.done()


# == Results: noise floor ===============================================

NOISE_FLOOR_HEADER = "| Metric | mean | rel. std | range |"


def test_readme_noise_floor_table_matches_mlflow():
    """Five seeds of the BCE baseline: the numbers the rest of the page stopped
    borrowing from, and still quotes as the floor for this one configuration."""
    rows = table(NOISE_FLOOR_HEADER, what="noise floor table")
    values, used = collect(TRACKING_URI, "sasrec_ml1m")
    assert len(used) == 5, f"the table claims five seeds, mlflow has {len(used)}: {used}"
    assert set(rows) == {label for label, _ in METRICS}

    check = Checker("mlflow.db")
    for label, cells in rows.items():
        seeds = values[SEED_KEYS[label]]
        check.check(f"{label} / mean", cells[0], seeds.mean())
        check.check(f"{label} / rel. std", cells[1], seeds.std(ddof=1) / seeds.mean() * 100)
        check.check(f"{label} / range", cells[2], (seeds.max() - seeds.min()) / seeds.mean() * 100)
    check.done()


SPREADS_HEADER = "| configuration | seeds | sampled | full |"
# README row -> the family key `seed_variance` measures it under. Every family with
# seeds is listed, which is what the sentence above the table claims.
SPREAD_FAMILIES = {
    "SASRec, BCE, 200ep (the floor above)": "bce",
    "SASRec, CE control": "ce",
    "CE + hidden_dim 64": "ce_width64",
    "CE + batch_size 19": "ce_batch19",
    "CE + 2 heads": "ce_heads2",
    "RecBole SASRec, dropout 0.2, rescored here": "recbole_d02",
    "RecBole SASRec, dropout 0.2, RecBole's own uni100": "recbole_d02_uni100",
    "SASRec on Beauty, BCE": "beauty_sasrec",
    "GenRec on Beauty, semantic IDs": "beauty_genrec",
    "GenRec on ML-1M, semantic IDs": "ml1m_genrec",
}


def test_readme_per_configuration_spread_table_matches_seed_variance():
    """The table that replaced the blanket floor. Each cell is one family's worst
    relative standard deviation on that protocol, straight from `family_spreads` --
    the same call `seed_variance` judges every claimed margin against."""
    rows = table(SPREADS_HEADER, what="per-configuration spread table")
    spreads = family_spreads(TRACKING_URI)
    assert set(rows) == set(SPREAD_FAMILIES), "the spread table's rows changed"
    # The sentence above the table says every configuration with seeds carries its
    # own spread, so a newly seeded family has to appear here rather than only in
    # the prose -- which is what happened to the two GenRec families for a day.
    assert set(SPREAD_FAMILIES.values()) == set(spreads), (
        "these seeded configurations are missing from the README table: "
        f"{sorted(set(spreads) - set(SPREAD_FAMILIES.values()))}"
    )

    check = Checker("scripts/seed_variance.py")
    for label, family in SPREAD_FAMILIES.items():
        assert family in spreads, f"{family} has no measured spread; the table lists it"
        n_seeds, per_protocol = spreads[family]
        check.check(f"{label} / seeds", rows[label][0], n_seeds)
        for column, protocol in enumerate(["sampled", "full"], start=1):
            cell = rows[label][column]
            if cell.strip() == "—":
                assert (
                    protocol not in per_protocol
                ), f"{label} now has a measured {protocol} spread; the table still prints —"
                continue
            check.check(f"{label} / {protocol}", cell, per_protocol[protocol])
    check.done()


# == Results: semantic ID quality =======================================

SEMANTIC_HEADER = "| | ML-1M | Beauty |"
SEMANTIC_REPORTS = {"ML-1M": "semantic_ids_ml-1m.md", "Beauty": "semantic_ids_beauty.md"}


def _semantic_report(name: str) -> dict[str, float]:
    """The five figures the README quotes, pulled out of one spot-check report."""
    import re

    text = (TABLES / name).read_text()
    codebook = artifact_table(TABLES / name, width=5)
    prefixes = artifact_table(TABLES / name, width=4)
    collisions = re.search(r"\(([\d.]+)%\)", text)
    largest = re.search(r"largest colliding group is (\d+)", text)
    explained = re.search(r"\(([\d.]+)% of the embedding norm", text)
    assert collisions and largest and explained, f"{name} no longer states its collision figures"
    return {
        "dead codes (any level)": sum(
            float(row[1]) for level, row in codebook.items() if level.isdigit()
        ),
        "collision rate on the 3-token code": float(collisions.group(1)),
        "largest colliding group": float(largest.group(1)),
        "embedding norm explained by 3 tokens": float(explained.group(1)),
        "within-prefix cosine @ depth 3 (vs random pair)": float(prefixes["3 tokens"][1]),
        "random pair": float(prefixes["random pairs"][1]),
    }


def test_readme_semantic_id_quality_table_matches_reports():
    """Zero dead codes is why RQ-VAE was skipped, and 11.78% collisions is the
    constraint the Beauty results are read against; both live only in these
    reports."""
    rows = table(SEMANTIC_HEADER, what="semantic ID quality table")
    check = Checker("results/tables/semantic_ids_*.md")
    for column, (dataset, report) in enumerate(SEMANTIC_REPORTS.items()):
        source = _semantic_report(report)
        assert set(rows) == set(source) - {"random pair"}, "the semantic ID table's rows changed"
        for label, cells in rows.items():
            check.check(f"{dataset} / {label}", cells[column], source[label])
        # The cosine row carries the random-pair baseline in parentheses; without
        # it the number above means nothing.
        cell = rows["within-prefix cosine @ depth 3 (vs random pair)"][column]
        check.check(
            f"{dataset} / random pair", cell.split("(")[1].rstrip(")"), source["random pair"]
        )
    check.done()


# == What's in the repo =================================================

COMPONENT_HEADER = "| Component | Path | Notes |"


def test_readme_component_table_paths_exist():
    """The one table on the page whose cells are paths rather than numbers. It is
    the reader's map of the repo, and a moved module breaks it silently -- the
    Markdown is not a link, so nothing else would notice."""
    lines = README.read_text().splitlines()
    assert COMPONENT_HEADER in lines, "the 'What's in the repo' table header moved"

    missing = []
    for line in lines[lines.index(COMPONENT_HEADER) + 2 :]:
        if not line.startswith("|"):
            break
        component, paths = (c.strip() for c in line.strip().strip("|").split("|")[:2])
        # A cell can name more than one path, and one of them is a glob.
        for quoted in re.findall(r"`([^`]+)`", paths):
            found = list(ROOT.glob(quoted)) if "*" in quoted else [ROOT / quoted]
            if not found or not all(path.exists() for path in found):
                missing.append(f"{component}: {quoted}")
    assert not missing, "the component table points at paths that do not exist:\n  " + "\n  ".join(
        missing
    )


# == The margins `seed_variance` judges =================================
#
# `CLAIMED_MARGINS` is a hand-written copy of the figures this page claims, and
# the script prints "above floor" or "INSIDE NOISE" for each one. It is a
# transcription like any other: on 2026-08-29 it still carried +7.41%, the six
# pre-correction ablation deltas and a Beauty margin of −57.83% that its own
# artifact had stopped supporting. Verdicts did not move, but the script that
# judges the page had drifted from the page.


def _claimed_margin_sources(runs) -> dict[str, float]:
    """Every claimed margin recomputed from the runs and artifacts it summarizes.

    Magnitudes only: some entries are stored as the size of a difference and
    others keep its sign, and the direction is asserted against the README tables
    elsewhere in this file.
    """
    b4r, d05 = "bert4rec_recbole_1x", "sasrec_recbole_1x"
    uni100 = "sasrec_recbole_1x_dropout02"
    rescored, ours, ce = (
        "sasrec_recbole_1x_dropout02_ourprotocol",
        "sasrec_ml1m",
        ABLATION_RUNS[ABLATION_BASELINE],
    )
    loss_ce = OBJECTIVE_CE

    def m(run, metric):
        return runs.metric(run, metric)

    out = {
        "Dropout default 0.5 -> 0.2, HR@10": rel(
            m(uni100, "sampled HR@10"), m(d05, "sampled HR@10")
        ),
        "Dropout default 0.5 -> 0.2, NDCG@10": rel(
            m(uni100, "sampled NDCG@10"), m(d05, "sampled NDCG@10")
        ),
        "BERT4Rec vs default SASRec, HR@10": rel(m(b4r, "sampled HR@10"), m(d05, "sampled HR@10")),
        "SASRec vs BERT4Rec at matched dropout, HR@10": rel(
            m(b4r, "sampled HR@10"), m(uni100, "sampled HR@10")
        ),
        "M4 cross-check: RecBole vs this repo, HR@10": rel(
            m(rescored, "sampled HR@10"), m(ours, "sampled HR@10")
        ),
        "M4 cross-check: RecBole vs this repo, NDCG@10": rel(
            m(rescored, "sampled NDCG@10"), m(ours, "sampled NDCG@10")
        ),
        "Loss effect?: RecBole vs this repo, full HR@10": rel(
            m(rescored, "full HR@10"), m(ours, "full HR@10")
        ),
    }
    for label, metric in (
        ("sampled HR@10", "sampled HR@10"),
        ("sampled NDCG@10", "sampled NDCG@10"),
        ("full HR@10", "full HR@10"),
        ("full NDCG@10", "full NDCG@10"),
    ):
        out[f"CE vs BCE loss, {label}"] = rel(m(loss_ce, metric), m(ours, metric))

    # The residual rows are three-seed means on both sides, unlike the objective
    # table's seed-42 pair.
    recbole_seeds, _ = collect(TRACKING_URI, rescored)
    ce_seeds, _ = collect(TRACKING_URI, loss_ce)
    for metric in ("full HR@10", "full NDCG@10"):
        key = SEED_KEYS[metric]
        out[f"Residual: RecBole vs CE, {metric} (3-seed means)"] = rel(
            recbole_seeds[key].mean(), ce_seeds[key].mean()
        )

    for arm, name in (
        ("ablation_ml1m_ce_width64", "width64"),
        ("ablation_ml1m_ce_batch19", "batch19"),
        ("ablation_ml1m_ce_heads2", "heads2"),
    ):
        stats, _, _ = arm_stats(TRACKING_URI, arm, loss_ce)
        for metric in ("full HR@10", "full NDCG@10"):
            out[f"P1b {name} vs CE, {metric} (3 seeds)"] = stats[metric].delta

    ablation_labels = {
        "A1: sinusoidal vs learnable pos emb": "ablation_ml1m_posemb_sinusoidal",
        "A1: none vs learnable pos emb": "ablation_ml1m_posemb_none",
        "A2: maxlen 100 vs 200": "ablation_ml1m_maxlen100",
        "A2: maxlen 50 vs 200": "ablation_ml1m_maxlen50",
        "A4: popularity vs uniform negatives": "ablation_ml1m_negsampling_popularity",
    }
    for label, run in ablation_labels.items():
        out[label] = rel(m(run, "sampled HR@10"), m(ce, "sampled HR@10"))
        out[f"{label}, full"] = rel(m(run, "full HR@10"), m(ce, "full HR@10"))

    for prefix, (_, atomic_run, semantic_run, artifact) in (
        ("Beauty", ATOMIC_TABLES["beauty"]),
        ("ML-1M", ATOMIC_TABLES["ml-1m"]),
    ):
        for metric in ("sampled HR@10", "sampled NDCG@10"):
            out[f"{prefix}: GenRec vs SASRec, {metric}"] = rel(
                m(semantic_run, metric), m(atomic_run, metric)
            )
        overall = json.loads((TABLES / artifact).read_text())["overall"]
        for metric in ("HR@10", "NDCG@10"):
            out[f"{prefix}: GenRec vs SASRec, full {metric}"] = rel(
                overall["GenRec (semantic)"][metric], overall["SASRec (atomic)"][metric]
            )
    return out


def test_seed_variance_claimed_margins_match_their_sources(runs):
    from scripts.seed_variance import CLAIMED_MARGINS

    expected = _claimed_margin_sources(runs)
    unmapped = [label for label, *_ in CLAIMED_MARGINS if label not in expected]
    assert not unmapped, (
        "these claimed margins have no recomputation here and would go unchecked:\n  "
        + "\n  ".join(unmapped)
    )

    check = Checker("the runs and artifacts each margin summarizes", claimed="seed_variance")
    for label, size, *_ in CLAIMED_MARGINS:
        # Magnitudes: some entries store the size of a difference, others its sign.
        check.check(label, f"{abs(size):.2f}", abs(expected[label]))
    check.done()
