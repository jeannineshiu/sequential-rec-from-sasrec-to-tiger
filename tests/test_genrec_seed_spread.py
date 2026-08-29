"""The generative seed spread, and the one thing that could silently go wrong in it.

The full-ranking floor GenRec contributes has to come from the exhaustive pass,
because `train_genrec` logs a beam-20 `test_full_*` and the margins it judges are
exhaustive. Nothing about that is visible in a number: a beam-based floor would
print exactly like a real one. So it is asserted here instead.
"""

import json
import pathlib

import pytest

from scripts.genrec_seed_spread import build_tables, fingerprint, reusable, spread
from scripts.seed_variance import FAMILIES, SAMPLED_ONLY, exhaustive_spread

K = 10
AS_TRAINED = "GenRec (semantic)"
DEBIASED = "GenRec (semantic), debiased a=1"


def _run(
    hr: float, ndcg: float, unseen_hits: int, n_unseen: int = 138, distinct: int = 1749
) -> dict:
    """One scored checkpoint, in the shape `score_run` returns."""
    return {
        "overall": {
            AS_TRAINED: {f"HR@{K}": hr, f"NDCG@{K}": ndcg},
            DEBIASED: {f"HR@{K}": hr / 2, f"NDCG@{K}": ndcg / 2},
        },
        "diversity": {
            model: {
                "model": model,
                "distinct_items_recommended": distinct,
                "median_train_freq": 12.0,
                "mean_train_freq": 20.0,
                "share_unseen": 0.0725,
            }
            for model in (AS_TRAINED, DEBIASED)
        },
        "buckets": {
            bucket: {
                "n_users": n_unseen if bucket == "unseen" else 100,
                AS_TRAINED: {f"HR@{K}": 0.0, f"NDCG@{K}": 0.0, "hits": 0},
                DEBIASED: {
                    f"HR@{K}": unseen_hits / n_unseen if bucket == "unseen" else 0.0,
                    f"NDCG@{K}": 0.0,
                    "hits": unseen_hits if bucket == "unseen" else 0,
                },
            }
            for bucket in ("unseen", "tail", "torso", "head")
        },
    }


def test_spread_is_relative_and_unbiased():
    s = spread([0.020, 0.024, 0.022])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.022)
    # ddof=1: three runs estimate a population, they are not the population.
    assert s["std"] == pytest.approx(0.002)
    assert s["rel_std"] == pytest.approx(0.002 / 0.022 * 100)


def test_spread_of_one_run_is_not_zero_variance():
    """A single run has no measured spread. Reporting 0.00% would read as a
    measurement of perfect reproducibility, which is the opposite of the truth."""
    s = spread([0.020])
    assert s["n"] == 1
    table, _ = build_tables({"genrec_beauty": _run(0.020, 0.010, 10)}, [0.0, 1.0], K, "Beauty")
    assert "0.00%" not in table
    assert "| full HR@10 | 0.0200 | — | — |" in table


def test_table_carries_every_seed_and_its_spread():
    results = {
        "genrec_beauty": _run(0.0250, 0.0131, 10),
        "genrec_beauty_seed1": _run(0.0260, 0.0136, 14),
        "genrec_beauty_seed2": _run(0.0240, 0.0126, 6),
    }
    table, spreads = build_tables(results, [0.0, 1.0], K, "Amazon Beauty")

    for run in results:
        assert f"`{run}`" in table
    assert spreads["full"][f"HR@{K}"]["n"] == 3
    assert spreads["full"][f"HR@{K}"]["mean"] == pytest.approx(0.0250)

    # The cold-start claim is quoted as a count, so the count is what the spread
    # has to be reported on: 6 to 14 hits is the interval the published p-value
    # does not describe.
    unseen = spreads["buckets"]["unseen"][DEBIASED]
    assert unseen["hits"] == [10, 14, 6]
    assert (unseen["min"], unseen["max"]) == (6, 14)


def test_the_diversity_collapse_gets_an_interval_too():
    """Catalogue coverage is read off the same top-k matrices as the ranks, so
    the seed that ranks like the others but recommends a different slice of the
    catalogue is visible here and nowhere else."""
    results = {
        "genrec_beauty": _run(0.0250, 0.0131, 10, distinct=1749),
        "genrec_beauty_seed1": _run(0.0260, 0.0136, 14, distinct=1600),
        "genrec_beauty_seed2": _run(0.0240, 0.0126, 6, distinct=1900),
    }
    table, spreads = build_tables(results, [0.0, 1.0], K, "Amazon Beauty")
    assert "| 1,749 |" in table  # thousands separator, as the diagnosis table prints it

    coverage = spreads["diversity"][AS_TRAINED]["distinct_items_recommended"]
    assert (coverage["min"], coverage["max"]) == (1600, 1900)
    assert coverage["mean"] == pytest.approx(1749.667, abs=1e-3)


def test_generative_families_take_full_ranking_from_the_exhaustive_artifact():
    """`train_genrec` logs a beam-20 full-ranking metric; the margins it would be
    used to judge are exhaustive. The MLflow side of these families must therefore
    stay on the sampled pair, with `full` coming from the artifact."""
    for key in ("beauty_genrec", "ml1m_genrec"):
        family = FAMILIES[key]
        assert family.metrics == SAMPLED_ONLY, f"{key} must not take `full` from MLflow"
        assert family.exhaustive and family.exhaustive.endswith(".json")


def test_exhaustive_spread_is_absent_until_the_seeds_have_run(tmp_path):
    assert exhaustive_spread(str(tmp_path / "not-generated.json")) is None

    one_seed = tmp_path / "one.json"
    one_seed.write_text(json.dumps({"spread": {"full": {"HR@10": {"n": 1, "rel_std": 0.0}}}}))
    assert exhaustive_spread(str(one_seed)) is None


def test_exhaustive_spread_reports_the_worst_metric(tmp_path):
    """Same rule as the MLflow path: a family's floor is the widest of its metrics
    on that protocol, not the friendliest."""
    path = tmp_path / "spread.json"
    path.write_text(
        json.dumps(
            {
                "spread": {
                    "full": {
                        "HR@10": {"n": 3, "rel_std": 2.5},
                        "NDCG@10": {"n": 3, "rel_std": 4.1},
                    }
                }
            }
        )
    )
    assert exhaustive_spread(str(path)) == (3, 4.1)


# -- resuming an interrupted scoring pass --------------------------------
#
# Added after a scoring run was interrupted 40 minutes into its second seed and
# lost the first seed's finished hour with it. The artifact is now written after
# every seed and reused on the next attempt -- but only for a seed whose
# checkpoint has not changed underneath it, which is the part worth a test.


def _artifact(tmp_path, runs: dict) -> "pathlib.Path":
    import json as _json

    path = tmp_path / "genrec_seed_spread_amazon-beauty.json"
    path.write_text(_json.dumps({"runs": runs}))
    return path


def test_reusable_skips_a_seed_whose_checkpoint_moved(tmp_path, monkeypatch):
    import scripts.genrec_seed_spread as module

    checkpoint = tmp_path / "genrec_beauty.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(module, "checkpoint_path", lambda name: checkpoint)

    stored = {"genrec_beauty": {"overall": {}, "checkpoint": fingerprint(checkpoint)}}
    path = _artifact(tmp_path, stored)
    assert list(reusable(path, ["genrec_beauty"])) == ["genrec_beauty"]

    # Retrained since: same path, different bytes. Reusing this would publish a
    # spread over a checkpoint that no longer exists.
    checkpoint.write_bytes(b"different weights")
    assert reusable(path, ["genrec_beauty"]) == {}


def test_reusable_is_empty_without_an_artifact(tmp_path):
    assert reusable(tmp_path / "absent.json", ["genrec_beauty"]) == {}


def test_reusable_ignores_a_pass_stored_before_fingerprints_existed(tmp_path, monkeypatch):
    """An artifact from an older version of this script has no `checkpoint` key.
    Silently trusting it would reuse numbers whose provenance cannot be checked."""
    import scripts.genrec_seed_spread as module

    checkpoint = tmp_path / "genrec_beauty.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(module, "checkpoint_path", lambda name: checkpoint)

    path = _artifact(tmp_path, {"genrec_beauty": {"overall": {}}})
    assert reusable(path, ["genrec_beauty"]) == {}
