"""Provision a Daytona GPU sandbox and run the Week 4 RecBole training-budget
sweep (SASRec vs BERT4Rec at 1x/4x/10x epochs) end to end.

Why this exists: local Mac (CPU/MPS) testing of RecBole repeatedly stalled --
once on a 6.4GB dataloader-cache pickle nearly filling the disk, once on what
looked like an indefinite Metal/MPS shader-compilation hang. RecBole's native,
well-supported path is CUDA, so this offloads the actual training runs to a
rented GPU instead of continuing to fight the local MPS path.

Usage:
    export DAYTONA_API_KEY=...        # app.daytona.io/dashboard/keys
    uv run python -m src.data.download --dest data/raw --dataset ml-1m  # local copy first, see below
    uv run python scripts/daytona_week4.py

Note: ML-1M is uploaded from a local copy rather than downloaded by the
sandbox -- files.grouplens.org resets the connection from inside Daytona's
sandboxes every time (confirmed 5/5 retries, with and without a browser
User-Agent), most likely blocking the cloud provider's outbound IP range
rather than anything fixable client-side. Run the download command above on
this machine once before running this script.

Safety notes -- READ BEFORE RUNNING (this spends real Daytona credits, billed
per second while the sandbox is running):

1. CORRECTED after actually hitting this: GPU sandboxes on Daytona are
   REQUIRED to use auto_delete_interval=0, which the API enforces
   server-side ("GPU sandboxes must be ephemeral") -- you cannot opt out.
   auto_delete_interval=0 means the sandbox is deleted THE INSTANT it stops,
   with everything on it. There is no "stopped but recoverable" state for a
   GPU sandbox, unlike regular sandboxes. Consequences:
     - auto_stop_interval is set to 0 (disables auto-stop-on-idle) so a long
       training run can't get silently stopped -> deleted by an inactivity
       timer while epochs are still running.
     - We download mlflow.db after EVERY experiment (not just at the end),
       so a crash partway through the sweep still leaves partial results on
       your local machine instead of losing everything.
     - NEVER call sandbox.stop()/delete() until you've confirmed the final
       download succeeded. Once stopped, it's gone -- there's no "oops, let
       me grab one more file" recovery.

2. This script does NOT try to merge the sandbox's results into this repo's
   results/tables/master.md automatically. That file is regenerated from
   *all* MLflow runs found in mlflow.db, and the sandbox's mlflow.db only
   ever sees the RecBole runs it creates -- running export_results.py inside
   the sandbox and pushing that back would silently overwrite/erase every
   Week 1-3 row. Instead, this script downloads the sandbox's mlflow.db as
   mlflow_daytona_week4.db in the repo root; bring that back to your main
   Claude Code session and it'll pull the new rows out and merge them in.

3. If this script crashes partway through, the sandbox is deliberately left
   running (not cleaned up) so you don't lose a half-finished run -- but see
   point 1: it's still one crash-then-idle-timeout away from deletion if
   auto_stop_interval weren't disabled, which is why that's set explicitly.
   Check the Daytona dashboard (app.daytona.io) if this script doesn't reach
   the end, and stop/delete the sandbox manually once you're done with it.
"""

import os
import sys
import time
from pathlib import Path

from daytona import (
    CreateSandboxFromImageParams,
    Daytona,
    DaytonaConfig,
    GpuType,
    Resources,
    SessionExecuteRequest,
)

# Line-buffer stdout/stderr so progress prints appear in a redirected log
# (`uv run python scripts/daytona_week4.py > week4_run.log`) as they happen,
# instead of sitting in Python's default block buffer until the process exits.
# This is not cosmetic: that block-buffering once made a normally-progressing
# run look frozen at the end of apt-get's output -- every later step's "$ cmd"
# marker was buffered and invisible -- which led to a live GPU training job
# being killed under the false belief that setup had hung. There was no hang;
# the log just wasn't being flushed.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except AttributeError:  # not a TextIOWrapper (unlikely); nothing to do
        pass

REPO_URL = "https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger.git"
# Absolute path on purpose: process.exec's default cwd is /workspace (the
# pytorch image's WORKDIR -- confirmed by an earlier run's traceback showing
# the clone at /workspace/...), but the FileSystem API (fs.upload_file /
# fs.download_file) resolves relative paths against the user's home dir,
# which for root is /root, NOT /workspace. Using one absolute path everywhere
# removes the ambiguity for both APIs.
REPO_DIR = "/workspace/sequential-rec-from-sasrec-to-tiger"

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


SETUP_TIMEOUT_SEC = 300  # setup steps should finish in minutes, not hours


def run(sandbox, command: str, cwd: str | None = None, timeout: float = SETUP_TIMEOUT_SEC) -> None:
    """Run a short setup command and block until it finishes.

    sandbox.process.exec() is blocking and returns the command's whole output
    only on completion, so nothing prints between the "$ cmd" marker and the
    command finishing. That's fine for setup steps (seconds to a couple of
    minutes). It is NOT fine for the multi-hour training runs -- those go
    through run_streaming() instead so their output shows up live. The bounded
    timeout guards against a setup step genuinely stalling; earlier runs that
    looked like they "hung after apt-get" were a stdout-buffering artifact
    (see the module-level reconfigure), not a real hang."""
    print(f"\n$ {command}")
    result = sandbox.process.exec(command, cwd=cwd, timeout=timeout)
    print(result.result)
    if result.exit_code != 0:
        raise RuntimeError(f"command failed (exit {result.exit_code}): {command}")


TRAIN_POLL_SEC = 15  # how often to poll a running training command for new output


def run_streaming(sandbox, session_id: str, command: str, cwd: str | None = None) -> None:
    """Run a long-running command asynchronously in a session and stream its
    output live by polling, so a multi-hour training run shows per-epoch
    progress instead of looking hung until it finishes.

    Setup uses the blocking run() above; training uses this because RecBole
    prints progress for the whole duration and we need to see it (and be able
    to tell training-in-progress apart from a real stall). Session commands
    have no cwd parameter, so cwd is applied with an explicit `cd`.

    PYTHONUNBUFFERED=1: the remote process's stdout is a pipe, not a tty, so
    Python block-buffers it inside the sandbox -- RecBole's per-epoch logger
    output (which goes to stdout) then sits in that remote buffer and never
    reaches get_session_command_logs until a full block fills or the process
    exits, making a live run look frozen after only its stderr warnings show.
    This is the same buffering trap as the module-level stdout reconfigure,
    one level down on the sandbox side; forcing unbuffered stdout there makes
    epoch progress stream out line by line."""
    env_command = f"PYTHONUNBUFFERED=1 {command}"
    full_cmd = f"cd {cwd} && {env_command}" if cwd else env_command
    print(f"\n$ {full_cmd}")
    resp = sandbox.process.execute_session_command(
        session_id, SessionExecuteRequest(command=full_cmd, run_async=True)
    )
    cmd_id = resp.cmd_id

    # get_session_command_logs returns the cumulative output each call; track
    # how much we've already printed and emit only the new tail.
    printed = 0
    while True:
        logs = sandbox.process.get_session_command_logs(session_id, cmd_id)
        output = logs.output or ""
        if len(output) > printed:
            sys.stdout.write(output[printed:])
            sys.stdout.flush()
            printed = len(output)

        cmd = sandbox.process.get_session_command(session_id, cmd_id)
        if cmd.exit_code is not None:
            if cmd.exit_code != 0:
                raise RuntimeError(f"command failed (exit {cmd.exit_code}): {command}")
            return
        time.sleep(TRAIN_POLL_SEC)


def main() -> None:
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("Set DAYTONA_API_KEY first (app.daytona.io/dashboard/keys)", file=sys.stderr)
        sys.exit(1)

    # Check the local data file BEFORE creating (and paying for) a sandbox.
    local_ratings = Path("data/raw/ml-1m/ratings.dat")
    if not local_ratings.exists():
        print(
            f"{local_ratings} not found locally -- run "
            "`uv run python -m src.data.download --dest data/raw --dataset ml-1m` "
            "on this machine first (and run this script from the repo root).",
            file=sys.stderr,
        )
        sys.exit(1)

    daytona = Daytona(DaytonaConfig(api_key=api_key))

    print("Creating GPU sandbox (RTX 4090, falling back to RTX Pro 6000 / RTX 5090)...")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            resources=Resources(
                cpu=4,
                memory=16,
                disk=30,  # GB -- explicit and modest: ML-1M + code + docker
                # layers don't need much. Not specifying disk risks a large
                # default that can push you over an org-wide storage cap
                # (we hit "Total disk limit exceeded. Maximum allowed:
                # 300GiB" on the first real run -- check app.daytona.io for
                # leftover sandboxes from earlier attempts/experiments if
                # you still hit this after adding an explicit disk size).
                gpu=1,
                gpu_type=[GpuType.RTX_4090, GpuType.RTX_PRO_6000, GpuType.RTX_5090],
            ),
            # Required by Daytona for GPU sandboxes ("must be ephemeral") --
            # deletes the sandbox the instant it stops. Not optional; see the
            # module docstring's safety notes for what this means in practice.
            auto_delete_interval=0,
            # Disables auto-stop-on-idle so a multi-hour training run can't
            # get stopped (-> immediately deleted, per the line above) by an
            # inactivity timer between epochs.
            auto_stop_interval=0,
        ),
        timeout=180,  # GPU provisioning can take longer than the 60s default
    )
    print(f"Sandbox created: id={sandbox.id}")

    # Explicit path to the installed uv binary rather than bare "uv": each
    # sandbox.process.exec() call may be a fresh non-login shell that never
    # sources the rc file the uv installer appends its PATH entry to.
    uv = "$HOME/.local/bin/uv"

    try:
        # -qq / --quiet everywhere below just keeps the log small. (An earlier
        # theory blamed large/\r-heavy output for "hanging" the run right after
        # apt-get; that turned out to be stdout block-buffering hiding the
        # progress, not a real hang -- see the module-level reconfigure. The
        # quiet flags are harmless to keep and do cut log noise.)
        run(sandbox, "apt-get update -qq && apt-get install -qq -y git curl")
        run(sandbox, f"git clone --quiet {REPO_URL} {REPO_DIR}")
        run(sandbox, "curl -LsSf https://astral.sh/uv/install.sh | sh")
        run(sandbox, f"{uv} sync -q", cwd=REPO_DIR)

        # files.grouplens.org resets the connection from inside this sandbox
        # every time (5/5 retries, with and without a browser User-Agent) --
        # looks like it's blocking the cloud provider's outbound IP range
        # rather than anything fixable client-side. Upload the already-
        # downloaded local copy instead of letting the sandbox fetch it.
        # (local_ratings existence was already checked before sandbox creation.)
        run(sandbox, f"mkdir -p {REPO_DIR}/data/raw/ml-1m")
        print(f"\n(uploading {local_ratings} -> sandbox:{REPO_DIR}/data/raw/ml-1m/ratings.dat)")
        sandbox.fs.upload_file(str(local_ratings), f"{REPO_DIR}/data/raw/ml-1m/ratings.dat")

        run(sandbox, f"{uv} run python -m src.recbole_utils.convert_to_atomic", cwd=REPO_DIR)

        local_db_path = "mlflow_daytona_week4.db"

        # One long-lived session for all training runs. Training goes through
        # run_streaming (not the blocking run()) so each multi-hour run prints
        # progress live -- otherwise the log looks frozen for hours and the run
        # is indistinguishable from a real stall.
        train_session = "week4-training"
        sandbox.process.create_session(train_session)

        for model, run_name, epochs in EXPERIMENTS:
            run_streaming(
                sandbox,
                train_session,
                f"{uv} run python -m src.recbole_run --model {model} "
                f"--epochs {epochs} --run-name {run_name}",
                cwd=REPO_DIR,
            )
            # Download after every experiment, not just at the end -- a GPU
            # sandbox is deleted the instant it stops (see docstring), so a
            # crash on run 5 of 6 must not cost us runs 1-4's results too.
            sandbox.fs.download_file(f"{REPO_DIR}/mlflow.db", local_db_path)
            print(f"  -> synced mlflow.db to {local_db_path} after {run_name}")

        print(f"\nFinal mlflow.db synced to {local_db_path}")
        print("Bring this file back to your main Claude Code session to merge the")
        print("new RecBole runs into results/tables/master.md and REPRODUCTION_LOG.md.")

    finally:
        print("\n" + "=" * 70)
        print(f"Sandbox ID: {sandbox.id}")
        print("This sandbox is STILL RUNNING and billing per second.")
        print("Stopping it deletes it immediately (GPU sandboxes are forced-ephemeral) --")
        print("only run this once you've confirmed mlflow_daytona_week4.db looks right:")
        print(f"  daytona stop {sandbox.id}")
        print("=" * 70)


if __name__ == "__main__":
    main()
