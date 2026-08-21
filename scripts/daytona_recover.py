"""Rescue results from a Daytona sandbox whose own push back to GitHub failed.

The detached Week 4 runner (scripts/daytona_remote_runner.sh) pushes its MLflow
results db to GitHub and then deletes its own sandbox. If that push fails -- an
expired PAT, a network blip, a rate limit -- the runner now deliberately leaves the
sandbox RUNNING rather than deleting it, because GPU sandboxes require
auto_delete_interval=0 and a stopped sandbox is gone instantly, disk and all.

That leaves a sandbox billing at ~$2.2/h with the only copy of a multi-hour result
on its disk. This script is the other half of that trade: pull the file down, then
delete the sandbox.

    uv run --extra daytona python scripts/daytona_recover.py <sandbox-id>
    uv run --extra daytona python scripts/daytona_recover.py --list

Nothing is deleted until the download is verified, and --keep skips deletion entirely.
"""

import argparse
import os
import sys
from pathlib import Path

from daytona import Daytona, DaytonaConfig

REPO_DIR = "/workspace/sequential-rec-from-sasrec-to-tiger"


def connect() -> Daytona:
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("Set DAYTONA_API_KEY first (app.daytona.io/dashboard/keys)", file=sys.stderr)
        sys.exit(1)
    return Daytona(DaytonaConfig(api_key=api_key))


def list_sandboxes(daytona: Daytona) -> None:
    sandboxes = list(daytona.list())
    if not sandboxes:
        print("No sandboxes exist. Nothing to recover, and nothing is billing.")
        return
    print(f"{len(sandboxes)} sandbox(es):")
    for sb in sandboxes:
        print(f"  {sb.id}  state={sb.state}")


def recover(daytona: Daytona, sandbox_id: str, keep: bool) -> None:
    sandbox = daytona.get(sandbox_id)
    print(f"sandbox {sandbox_id} state={sandbox.state}")

    listing = sandbox.process.exec(f"ls -la {REPO_DIR}/mlflow*.db", cwd=REPO_DIR, timeout=60)
    print(listing.result)

    # Take the runner's per-model results db if present, else the raw tracking db
    # the training process wrote (identical content; the copy step may not have run).
    found = [
        line.split()[-1] for line in listing.result.splitlines() if line.strip().endswith(".db")
    ]
    if not found:
        print(
            "No results db on the sandbox. The run probably died before logging "
            "anything -- check run.log before deleting:\n"
            f"  daytona ssh {sandbox_id}   then   tail -50 run.log",
            file=sys.stderr,
        )
        sys.exit(1)

    for remote in found:
        local = Path(Path(remote).name)
        if local.exists():
            local = local.with_name(f"recovered_{local.name}")
        print(f"downloading {remote} -> {local}")
        sandbox.fs.download_file(remote, str(local))
        size = local.stat().st_size
        print(f"  wrote {local} ({size} bytes)")
        if size == 0:
            print(
                "  WARNING: downloaded file is empty -- NOT deleting the sandbox", file=sys.stderr
            )
            keep = True

    if keep:
        print(f"\n--keep (or a bad download): sandbox {sandbox_id} left running -- still billing.")
        return

    print(f"\ndeleting sandbox {sandbox_id} (results are now local)")
    daytona.delete(sandbox)
    print("done -- billing stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sandbox_id", nargs="?", help="sandbox to recover results from")
    parser.add_argument("--list", action="store_true", help="list sandboxes and exit")
    parser.add_argument(
        "--keep", action="store_true", help="download but do not delete the sandbox"
    )
    args = parser.parse_args()

    daytona = connect()
    if args.list or not args.sandbox_id:
        list_sandboxes(daytona)
        if not args.sandbox_id:
            sys.exit(0)
    recover(daytona, args.sandbox_id, args.keep)
