"""master.md is the artifact the README points a reader at to verify its numbers,
so two properties matter beyond it being generated: it must not carry smoke runs,
and regenerating it with no new runs must produce a byte-identical file.
"""

from pathlib import Path

import pandas as pd

from src.export_results import build_master_table, drop_debris


def _runs() -> pd.DataFrame:
    # Two runs sharing a name is the case that used to sort unstably: the older
    # LOCALSMOKE_dropout02 pair, and the real sasrec_beauty rerun.
    return pd.DataFrame(
        {
            "tags.mlflow.runName": [
                "sasrec_beauty",
                "probe_delete_me",
                "sasrec_beauty",
                "SMOKE_loss_ce",
                "genrec_beauty_smoke",
                "LOCALSMOKE_dropout02",
                "ablation_ml1m_loss_ce",
            ],
            "start_time": [200, 1, 100, 2, 3, 4, 300],
            "params.dataset": ["amazon-beauty"] * 6 + ["ml-1m"],
            "metrics.test_sampled_HR_at_10": [0.51, None, 0.32, None, None, 0.72, 0.82],
        }
    )


def test_debris_runs_are_dropped_by_name():
    kept = set(drop_debris(_runs())["tags.mlflow.runName"])
    assert kept == {"sasrec_beauty", "ablation_ml1m_loss_ce"}


def test_real_run_names_are_not_mistaken_for_debris():
    # The pattern is a substring match, so guard that it cannot eat a real arm.
    names = pd.DataFrame(
        {
            "tags.mlflow.runName": [
                "sasrec_recbole_1x_dropout02_ourprotocol_seed2",
                "ablation_ml1m_negsampling_popularity",
                "bert4rec_recbole_1x",
                "genrec_ml1m",
            ]
        }
    )
    assert len(drop_debris(names)) == 4


def test_same_named_rows_keep_a_deterministic_order(tmp_path: Path):
    runs = drop_debris(_runs())
    out = tmp_path / "master.md"
    build_master_table(runs, out)
    first = out.read_text()

    # Same runs, arriving from MLflow in a different order -- which is all that
    # separated two regenerations of the table before the sort had a tiebreak.
    build_master_table(runs.iloc[::-1].reset_index(drop=True), out)
    assert out.read_text() == first

    # Oldest run of a duplicated name comes first, so the pair cannot swap.
    body = [ln for ln in first.splitlines() if ln.startswith("| sasrec_beauty")]
    assert len(body) == 2
    assert "0.3200" in body[0] and "0.5100" in body[1]
