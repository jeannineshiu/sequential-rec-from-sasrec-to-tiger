"""Every table on the README is transcribed by hand from results/, and CI only
ran tests, lint and format -- so nothing mechanically checked that the page still
says what the artifacts say. A p-value of 0.059 stood on the page for weeks and
reproduced under no test; it was caught by reading, not by CI.

These tests parse the README's headline figures and assert them against the JSON
the generating script writes, so the next drift fails a build instead of waiting
to be noticed.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
COLD_START_JSON = ROOT / "results" / "tables" / "atomic_vs_semantic.json"

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
HEADER = "| bucket | users | SASRec | GenRec | GenRec debiased α=1 |"


def _cells(line: str) -> list[str]:
    # Bold marks emphasis on the winning column and thousands separators are for
    # reading; neither is part of the number.
    return [
        c.strip().replace("**", "").replace(",", "") for c in line.strip().strip("|").split("|")
    ]


def readme_cold_start_rows() -> dict[str, list[str]]:
    lines = README.read_text().splitlines()
    if HEADER not in lines:
        pytest.fail(
            f"cold-start table header not found in README.md. If the table was reformatted, "
            f"update HEADER in {Path(__file__).name} -- do not delete the check."
        )
    start = lines.index(HEADER) + 2  # skip the |---| separator
    rows = {}
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        cells = _cells(line)
        rows[cells[0]] = cells[1:]
    return rows


@pytest.fixture(scope="module")
def cold_start():
    if not COLD_START_JSON.exists():
        pytest.skip(f"{COLD_START_JSON.relative_to(ROOT)} not generated")
    return json.loads(COLD_START_JSON.read_text())


def test_readme_cold_start_table_matches_generated_json(cold_start):
    rows = readme_cold_start_rows()
    assert set(rows) == set(BUCKET_ALIASES), (
        f"README buckets {sorted(rows)} do not match {sorted(BUCKET_ALIASES)}"
    )

    mismatches = []
    for label, cells in rows.items():
        bucket = cold_start[BUCKET_ALIASES[label]]
        if int(cells[0]) != bucket["n_users"]:
            mismatches.append(f"{label}: users {cells[0]} != {bucket['n_users']}")
        for model, cell in zip(COLUMN_MODELS, cells[1:], strict=True):
            printed = float(cell)
            actual = round(bucket[model]["HR@10"], 4)
            if printed != actual:
                mismatches.append(
                    f"{label} / {model}: README {printed:.4f} != results {actual:.4f}"
                )
    assert not mismatches, "README disagrees with results/tables/:\n  " + "\n  ".join(mismatches)


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


# -- the recommendation-diversity table ----------------------------------
#
# Added 2026-08-27, after a padding token spent weeks inside these numbers: the
# debiased row read 1,976 items and 11.6% unseen, and nothing compared it to the
# artifact it was transcribed from. This table has no JSON companion, so the
# markdown is the source of truth and is parsed directly.

DIVERSITY_MD = ROOT / "results" / "tables" / "genrec_diagnosis.md"
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


def _diversity_cells(line: str) -> list[str]:
    """Like `_cells`, but the model label keeps its comma.

    `_cells` strips thousands separators, which also eats the one in
    "GenRec (semantic), debiased a=1".
    """
    cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
    return [cells[0]] + [c.replace(",", "") for c in cells[1:]]


def _readme_diversity_rows() -> dict[str, list[str]]:
    lines = README.read_text().splitlines()
    if DIVERSITY_HEADER not in lines:
        pytest.fail(
            f"diversity table header not found in README.md. If the table was reformatted, "
            f"update DIVERSITY_HEADER in {Path(__file__).name} -- do not delete the check."
        )
    rows = {}
    for line in lines[lines.index(DIVERSITY_HEADER) + 2 :]:
        if not line.startswith("|"):
            break
        cells = _diversity_cells(line)
        # "9,221 (76% of catalog)" -> "9221": the share is prose, the count is the claim.
        rows[cells[0]] = [c.split(" (")[0] for c in cells[1:]]
    return rows


def _generated_diversity_rows() -> dict[str, list[str]]:
    rows = {}
    for line in DIVERSITY_MD.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| model") or "---" in line:
            continue
        cells = _diversity_cells(line)
        if len(cells) == 8:  # model + 7 columns; the per-level table is narrower
            rows[cells[0]] = cells[1:]
    return rows


def test_readme_diversity_table_matches_generated_table():
    readme, generated = _readme_diversity_rows(), _generated_diversity_rows()
    assert set(readme) == set(DIVERSITY_MODELS), (
        f"README diversity rows {sorted(readme)} do not match {sorted(DIVERSITY_MODELS)}"
    )

    mismatches = []
    for label, cells in readme.items():
        source = generated[DIVERSITY_MODELS[label]]
        for name, index in DIVERSITY_COLUMNS.items():
            if cells[list(DIVERSITY_COLUMNS).index(name)] != source[index]:
                mismatches.append(
                    f"{label} / {name}: README "
                    f"{cells[list(DIVERSITY_COLUMNS).index(name)]} != results {source[index]}"
                )
    assert not mismatches, "README disagrees with results/tables/genrec_diagnosis.md:\n  " + (
        "\n  ".join(mismatches)
    )
