"""Locating and parsing the README's tables, shared by the tests that hold them
against the artifacts in `results/` and the runs in `mlflow.db`.

Every table on the page is transcribed by hand, so the parsing has to be strict
about *finding* the table -- a silently-missed header turns a check into a no-op,
which is worse than no check at all. Every lookup here fails loudly, naming the
constant to update, rather than returning nothing.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
TABLES = ROOT / "results" / "tables"


def cells(line: str) -> list[str]:
    """Split one markdown row into cells.

    Bold and italic mark emphasis on a winning column and thousands separators
    are for reading; neither is part of the number. The first cell is the row
    label and keeps its commas -- "GenRec (semantic), debiased a=1" is a name.
    """
    out = [c.strip().replace("**", "").replace("*", "") for c in line.strip().strip("|").split("|")]
    return [out[0]] + [c.replace(",", "") for c in out[1:]]


def number(cell: str) -> float:
    """A README cell as a number: `−28.95%` -> -28.95, `9,221 (76% of catalog)` -> 9221.

    The minus sign on the page is U+2212, which `float` does not accept, and `~`
    prefixes a delta the README marks as inside seed noise.
    """
    text = cell.split(" (")[0].strip().lstrip("~").strip()
    return float(text.replace("−", "-").replace("%", "").replace(",", ""))


def decimals(cell: str) -> int:
    """How many decimal places the README printed, which is the precision any
    check against it can honestly demand."""
    text = cell.split(" (")[0].strip().lstrip("~").strip().replace("%", "")
    return len(text.split(".")[1]) if "." in text else 0


def table(header: str, *, after: str | None = None, what: str = "") -> dict[str, list[str]]:
    """The rows of the README table whose header line is exactly `header`, keyed
    by first cell.

    `after` names a heading to start searching from, for the tables that share a
    header with another one -- Beauty's and ML-1M's atomic-vs-semantic tables.
    """
    lines = README.read_text().splitlines()
    start = 0
    if after is not None:
        if after not in lines:
            pytest.fail(
                f"heading '{after}' not found in README.md. If the section was renamed, update "
                "the constant in the calling test -- do not delete the check."
            )
        start = lines.index(after)
    for i in range(start, len(lines)):
        if lines[i] == header:
            break
    else:
        pytest.fail(
            f"{what or 'table'} header not found in README.md:\n  {header}\n"
            "If the table was reformatted, update the header constant in the calling test "
            "-- do not delete the check."
        )
    rows: dict[str, list[str]] = {}
    for line in lines[i + 2 :]:  # skip the |---| separator
        if not line.startswith("|"):
            break
        row = cells(line)
        rows[row[0]] = row[1:]
    return rows


def artifact_table(path: Path, width: int) -> dict[str, list[str]]:
    """Rows of every markdown table in a generated report that has `width` cells,
    keyed by first cell. A report holds several tables; the cell count picks one
    out without depending on its header wording."""
    rows: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("|") or "---" in line:
            continue
        row = cells(line)
        if len(row) == width:
            rows[row[0]] = row[1:]
    return rows


class Checker:
    """Collects every disagreement before failing, so one run reports the whole
    drift rather than the first cell of it."""

    def __init__(self, source: str, claimed: str = "README"):
        self.source = source
        self.claimed = claimed  # what the printed side is, for the failure message
        self.mismatches: list[str] = []

    def check(self, label: str, cell: str, actual: float) -> None:
        """Assert a README cell against a value, at the precision the cell printed."""
        printed = number(cell)
        if abs(printed - actual) > 0.5 * 10 ** -decimals(cell) + 1e-9:
            self.mismatches.append(f"{label}: {self.claimed} {cell.strip()} != {actual:.6g}")

    def equal(self, label: str, cell: str, actual: str) -> None:
        if cell.strip() != actual:
            self.mismatches.append(f"{label}: {self.claimed} {cell.strip()!r} != {actual!r}")

    def done(self) -> None:
        assert (
            not self.mismatches
        ), f"{self.claimed} disagrees with {self.source}:\n  " + "\n  ".join(self.mismatches)
