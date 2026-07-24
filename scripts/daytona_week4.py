"""Provision a Daytona GPU sandbox and run the Week 4 RecBole training-budget
sweep (SASRec vs BERT4Rec at 1x/4x/10x epochs) end to end.

Why this exists: local Mac (CPU/MPS) testing of RecBole repeatedly stalled --
once on a 6.4GB dataloader-cache pickle nearly filling the disk, once on what
looked like an indefinite Metal/MPS shader-compilation hang. RecBole's native,
well-supported path is CUDA, so this offloads the actual training runs to a
rented GPU instead of continuing to fight the local MPS path.

Usage:
    export DAYTONA_API_KEY=...        # app.daytona.io/dashboard/keys
    uv run python scripts/daytona_week4.py

Safety notes -- READ BEFORE RUNNING (this spends real Daytona credits, billed
per second while the sandbox is running):

1. auto_delete_interval is set to -1 (never auto-delete). A positive value
   means "delete N minutes after the sandbox is stopped"; 0 means "delete
   immediately on stop" -- neither is what we want, since we need the sandbox
   to still exist after this script finishes so we can pull results off it.
   YOU must manually stop + delete the sandbox once you've confirmed the
   results download succeeded (instructions are printed at the end).

2. This script does NOT try to merge the sandbox's results into this repo's
   results/tables/master.md automatically. That file is regenerated from
   *all* MLflow runs found in mlflow.db, and the sandbox's mlflow.db only
   ever sees the 6 RecBole runs it just created -- running export_results.py
   inside the sandbox and pushing that back would silently overwrite/erase
   every Week 1-3 row. Instead, this script downloads the sandbox's mlflow.db
   as mlflow_daytona_week4.db in the repo root; bring that back to your main
   Claude Code session and it'll pull the 6 new rows out and merge them in.

3. If this script crashes partway through, the sandbox is deliberately left
   running (not cleaned up) so you don't lose a half-finished run. Check the
   Daytona dashboard (app.daytona.io) if this script doesn't reach the end.
"""

import os
import sys

from daytona import CreateSandboxFromImageParams, Daytona, DaytonaConfig, GpuType, Resources

REPO_URL = "https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger.git"
REPO_DIR = "sequential-rec-from-sasrec-to-tiger"

# (model, run_name, epochs) for the 1x/4x/10x training-budget sweep.
# 200 epochs matches our own SASRec headline run (configs/sasrec_ml1m.yaml);
# adjust here if you want different budgets before running.
EXPERIMENTS = [
    ("SASRec", "sasrec_recbole_1x", 200),
    ("SASRec", "sasrec_recbole_4x", 800),
    ("SASRec", "sasrec_recbole_10x", 2000),
    ("BERT4Rec", "bert4rec_recbole_1x", 200),
    ("BERT4Rec", "bert4rec_recbole_4x", 800),
    ("BERT4Rec", "bert4rec_recbole_10x", 2000),
]


def run(sandbox, command: str, cwd: str | None = None) -> None:
    print(f"\n$ {command}")
    result = sandbox.process.exec(
        command, cwd=cwd, timeout=0
    )  # 0 = no timeout (multi-hour training)
    print(result.result)
    if result.exit_code != 0:
        raise RuntimeError(f"command failed (exit {result.exit_code}): {command}")


def main() -> None:
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("Set DAYTONA_API_KEY first (app.daytona.io/dashboard/keys)", file=sys.stderr)
        sys.exit(1)

    daytona = Daytona(DaytonaConfig(api_key=api_key))

    print("Creating GPU sandbox (RTX 4090, falling back to RTX Pro 6000 / RTX 5090)...")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            resources=Resources(
                gpu=1,
                gpu_type=[GpuType.RTX_4090, GpuType.RTX_PRO_6000, GpuType.RTX_5090],
            ),
            auto_delete_interval=-1,  # never auto-delete -- see module docstring
        ),
        timeout=180,  # GPU provisioning can take longer than the 60s default
    )
    print(f"Sandbox created: id={sandbox.id}")

    try:
        run(sandbox, f"git clone {REPO_URL} {REPO_DIR}")
        run(sandbox, "curl -LsSf https://astral.sh/uv/install.sh | sh")
        run(sandbox, "uv sync", cwd=REPO_DIR)
        run(
            sandbox,
            "uv run python -m src.data.download --dest data/raw --dataset ml-1m",
            cwd=REPO_DIR,
        )
        run(sandbox, "uv run python -m src.recbole_utils.convert_to_atomic", cwd=REPO_DIR)

        for model, run_name, epochs in EXPERIMENTS:
            run(
                sandbox,
                f"uv run python -m src.recbole_run --model {model} "
                f"--epochs {epochs} --run-name {run_name}",
                cwd=REPO_DIR,
            )

        local_db_path = "mlflow_daytona_week4.db"
        sandbox.fs.download_file(f"{REPO_DIR}/mlflow.db", local_db_path)
        print(f"\nDownloaded sandbox's mlflow.db -> {local_db_path}")
        print("Bring this file back to your main Claude Code session to merge the")
        print("6 new RecBole runs into results/tables/master.md and REPRODUCTION_LOG.md.")

    finally:
        print("\n" + "=" * 70)
        print(f"Sandbox ID: {sandbox.id}")
        print("This sandbox is STILL RUNNING and billing per second.")
        print("Once you've confirmed mlflow_daytona_week4.db downloaded correctly:")
        print(f"  daytona stop {sandbox.id}")
        print(f"  daytona delete {sandbox.id}")
        print("=" * 70)


if __name__ == "__main__":
    main()
