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

import argparse
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

# Explicit path to the installed uv binary rather than bare "uv": each
# sandbox.process.exec() call may be a fresh non-login shell that never sources
# the rc file the uv installer appends its PATH entry to.
UV = "$HOME/.local/bin/uv"

# CPU cores requested for the sandbox. Used both for the Resources request AND
# exported to the training process as RECBOLE_NUM_THREADS so it caps its CPU
# thread pools to this exact number. This has to be passed explicitly: inside
# the sandbox neither os.cpu_count() nor os.sched_getaffinity() sees the limit
# (Daytona's cpu=N is a CFS quota, not an affinity mask -- both report the ~96
# host cores), so letting the process auto-detect oversubscribes and livelocks.
#
# 16 is this account's per-GPU vCPU ceiling. The bottleneck is CPU-side: RecBole
# rebuilds the MAX_ITEM_LIST_LENGTH=200 sequence augmentation each run and feeds
# the GPU from a single dataloader, so epoch time is CPU-bound (275s/epoch at 4
# cores). More cores speeds up the augmentation and the OMP/MKL/numpy paths.
SANDBOX_CPUS = 16

# Per-process CPU-thread cap for the training command (exported as
# RECBOLE_NUM_THREADS). Deliberately lower than SANDBOX_CPUS: with RecBole
# worker=4 dataloader processes, the 16 cores are meant to be used by the 4
# parallel workers + main feeding the GPU, NOT by each process spawning 16
# OMP/MKL threads. Capping each process low keeps 4 workers + main from
# oversubscribing the 16 cores (the oversubscription that livelocked earlier).
TRAIN_THREADS = 4

# Per-model 1x/4x/10x training-budget milestones (epochs, run_name). Budgets
# match our own SASRec headline (200 = configs/sasrec_ml1m.yaml). src/recbole_run
# trains each model ONCE to the largest budget and recovers every smaller budget
# from that single trajectory (see its docstring), so this is 2 training runs of
# 2000 epochs, not 6 runs totalling 6000 epochs -- ~1/3 fewer epochs, no fidelity
# loss. Every budget must be divisible by eval_step (10) so a validation lands on
# each milestone.
BUDGETS = {
    "SASRec": [(200, "sasrec_recbole_1x"), (800, "sasrec_recbole_4x"), (2000, "sasrec_recbole_10x")],
    "BERT4Rec": [(200, "bert4rec_recbole_1x"), (800, "bert4rec_recbole_4x"), (2000, "bert4rec_recbole_10x")],
}

# Tiny budgets for a cheap end-to-end pipeline check (--smoke): validates the
# milestone-snapshot logic, worker=4 dataloading, and the batch/eval_step config
# on ~30 epochs before committing to the multi-day real sweep. Divisible by
# eval_step (10) so milestones land.
SMOKE_BUDGETS = {
    "SASRec": [(10, "sasrec_smoke_1x"), (20, "sasrec_smoke_2x")],
    "BERT4Rec": [(10, "bert4rec_smoke_1x"), (20, "bert4rec_smoke_2x")],
}


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
    # RECBOLE_NUM_THREADS: authoritative per-process CPU-thread cap (the sandbox
    # can't detect its quota itself -- see SANDBOX_CPUS/TRAIN_THREADS). Set to
    # TRAIN_THREADS, not SANDBOX_CPUS, so RecBole's 4 dataloader workers + main
    # don't each spawn 16 threads and oversubscribe. OMP_NUM_THREADS is set too
    # as a belt-and-suspenders in case a numeric lib imports before recbole_run.py
    # reads the env.
    env_command = (
        f"RECBOLE_NUM_THREADS={TRAIN_THREADS} OMP_NUM_THREADS={TRAIN_THREADS} "
        f"PYTHONUNBUFFERED=1 {command}"
    )
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


def _require_local_data() -> Path:
    """ML-1M ratings.dat must exist locally: grouplens blocks the sandbox's IP,
    so we upload our copy instead of letting the sandbox download it."""
    local_ratings = Path("data/raw/ml-1m/ratings.dat")
    if not local_ratings.exists():
        print(
            f"{local_ratings} not found locally -- run "
            "`uv run python -m src.data.download --dest data/raw --dataset ml-1m` "
            "on this machine first (and run this script from the repo root).",
            file=sys.stderr,
        )
        sys.exit(1)
    return local_ratings


def provision_sandbox(daytona, local_ratings: Path):
    """Create a GPU sandbox and get it ready to train: install git/curl/uv, clone
    the repo, uv sync, and upload the ML-1M data. Returns the sandbox. Shared by
    the local-driven (main) and autonomous (main_detached) paths."""
    print("Creating GPU sandbox (RTX 4090, falling back to RTX Pro 6000 / RTX 5090)...")
    sandbox = daytona.create(
        CreateSandboxFromImageParams(
            image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
            resources=Resources(
                cpu=SANDBOX_CPUS,
                memory=32,  # GB -- generous headroom (per-GPU cap is 192GB); the
                # maxlen-200 augmented sequences pushed RSS to ~8GB at 4 cores.
                disk=30,  # GB -- explicit and modest; not specifying risks a large
                # default that can trip an org-wide storage cap.
                gpu=1,
                gpu_type=[GpuType.RTX_4090, GpuType.RTX_PRO_6000, GpuType.RTX_5090],
            ),
            # Required by Daytona for GPU sandboxes ("must be ephemeral") -- deletes
            # the sandbox the instant it stops. auto_stop_interval=0 disables the
            # idle timer so a multi-hour run can't get stopped (-> deleted) between
            # epochs. NOTE for the detached path: idle here means no API/toolbox
            # activity, so a detached in-sandbox run MUST keep auto_stop at 0 or it
            # could be killed mid-training; it cleans up by self-deleting instead.
            auto_delete_interval=0,
            auto_stop_interval=0,
        ),
        timeout=180,  # GPU provisioning can take longer than the 60s default
    )
    print(f"Sandbox created: id={sandbox.id}")

    run(sandbox, "apt-get update -qq && apt-get install -qq -y git curl")
    run(sandbox, f"git clone --quiet {REPO_URL} {REPO_DIR}")
    run(sandbox, "curl -LsSf https://astral.sh/uv/install.sh | sh")
    # --extra daytona installs the daytona SDK (an optional-dependency extra, not a
    # default dep) so the detached runner can self-stop this sandbox at the end.
    # Without it the self-stop step dies with ModuleNotFoundError: No module 'daytona'.
    run(sandbox, f"{UV} sync -q --extra daytona", cwd=REPO_DIR)

    run(sandbox, f"mkdir -p {REPO_DIR}/data/raw/ml-1m")
    print(f"\n(uploading {local_ratings} -> sandbox:{REPO_DIR}/data/raw/ml-1m/ratings.dat)")
    sandbox.fs.upload_file(str(local_ratings), f"{REPO_DIR}/data/raw/ml-1m/ratings.dat")
    return sandbox


def budgets_arg_for(budget_map: dict, model: str) -> str:
    """One "epochs:run_name" token per budget, comma-joined (no spaces -> a single
    shell arg): e.g. "200:sasrec_recbole_1x,800:...,2000:...`."""
    return ",".join(f"{epochs}:{name}" for epochs, name in budget_map[model])


def main(models: list[str], budget_map: dict, local_db_path: str) -> None:
    """Local-driven path: this process stays connected, streams progress, and
    downloads mlflow.db. Needs the laptop awake for the whole run. See
    main_detached for the Mac-independent path."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    if not api_key:
        print("Set DAYTONA_API_KEY first (app.daytona.io/dashboard/keys)", file=sys.stderr)
        sys.exit(1)
    local_ratings = _require_local_data()
    daytona = Daytona(DaytonaConfig(api_key=api_key))
    sandbox = provision_sandbox(daytona, local_ratings)

    try:
        run(sandbox, f"{UV} run python -m src.recbole_utils.convert_to_atomic", cwd=REPO_DIR)

        train_session = "week4-training"
        sandbox.process.create_session(train_session)

        for model in models:
            run_streaming(
                sandbox,
                train_session,
                f"{UV} run python -m src.recbole_run --model {model} "
                f"--budgets {budgets_arg_for(budget_map, model)}",
                cwd=REPO_DIR,
            )
            sandbox.fs.download_file(f"{REPO_DIR}/mlflow.db", local_db_path)
            print(f"  -> synced mlflow.db to {local_db_path} after {model}")

        print(f"\nFinal mlflow.db synced to {local_db_path}")

    finally:
        print("\n" + "=" * 70)
        print(f"Sandbox ID: {sandbox.id}")
        print("This sandbox is STILL RUNNING and billing per second.")
        print(f"Stop it once you've confirmed {local_db_path} looks right:")
        print(f"  daytona stop {sandbox.id}")
        print("=" * 70)


def main_detached(models: list[str], budget_map: dict) -> None:
    """Autonomous, Mac-independent path. For each model: provision a sandbox, then
    launch scripts/daytona_remote_runner.sh DETACHED inside it (nohup + run_async)
    so training survives this laptop sleeping/closing. The runner pushes results to
    GitHub and self-stops its sandbox when done. This launcher exits right after
    firing both runners -- nothing here needs to stay running."""
    api_key = os.environ.get("DAYTONA_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    if not api_key:
        print("Set DAYTONA_API_KEY first (app.daytona.io/dashboard/keys)", file=sys.stderr)
        sys.exit(1)
    if not github_token:
        print(
            "--detached needs GITHUB_TOKEN (a PAT with contents:write) so the sandbox "
            "can push results back. Add it to ~/.zshenv and re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    local_ratings = _require_local_data()
    daytona = Daytona(DaytonaConfig(api_key=api_key))

    for model in models:
        print(f"\n########## provisioning autonomous sandbox for {model} ##########")
        sandbox = provision_sandbox(daytona, local_ratings)

        # Write the credentials + params to a file INSIDE the sandbox (via upload,
        # not an exec string) so secrets never appear in a command line or log.
        env_lines = (
            f"export MODEL={model}\n"
            f"export BUDGETS={budgets_arg_for(budget_map, model)}\n"
            # Cap CPU threads exactly like the local-driven path's run_streaming.
            # Without this the detached training process auto-detects 16 cores and,
            # with worker=4 dataloader processes, oversubscribes (4x16 threads).
            f"export RECBOLE_NUM_THREADS={TRAIN_THREADS}\n"
            f"export OMP_NUM_THREADS={TRAIN_THREADS}\n"
            f"export SANDBOX_ID={sandbox.id}\n"
            f"export DAYTONA_API_KEY={api_key}\n"
            f"export GITHUB_TOKEN={github_token}\n"
        )
        env_remote = f"{REPO_DIR}/.week4_env"
        sandbox.fs.upload_file(env_lines.encode(), env_remote)

        # Fire the runner detached: run_async returns immediately and the command
        # keeps running server-side after we disconnect; nohup + redirect makes it
        # independent of the session too. We source the env file, then delete it
        # (the exported vars persist into the nohup'd runner) so the token doesn't
        # linger on disk during the multi-day run.
        launch = (
            f"cd {REPO_DIR} && chmod +x scripts/daytona_remote_runner.sh && "
            f"set -a && . {env_remote} && set +a && rm -f {env_remote} && "
            f"nohup bash scripts/daytona_remote_runner.sh > run.log 2>&1 &"
        )
        session = f"week4-launch-{model}"
        sandbox.process.create_session(session)
        sandbox.process.execute_session_command(
            session, SessionExecuteRequest(command=launch, run_async=True)
        )
        print(f"  -> {model} runner launched DETACHED in sandbox {sandbox.id}")

    print("\n" + "=" * 70)
    print("Autonomous runs launched. This laptop is no longer needed -- you can")
    print("close it. Each sandbox trains, pushes results to GitHub, and self-stops.")
    print("Watch progress: `git pull` and look for mlflow_daytona_week4_<MODEL>.db")
    print("and WEEK4_<MODEL>_DONE commits, or `daytona ssh <id>` + `tail -f run.log`.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Week 4 RecBole training-budget sweep on a Daytona GPU sandbox.")
    parser.add_argument(
        "--model",
        choices=list(BUDGETS.keys()),
        default=None,
        help="Run only this model (its own sandbox + its own mlflow db file). Launch "
        "the script twice, once per model, to run the two models in parallel across two "
        "sandboxes (halves wall-clock; same credits). Default: both models, one sandbox.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use tiny budgets (~30 epochs) to validate the pipeline end-to-end cheaply "
        "before committing to the multi-day real sweep.",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Autonomous, Mac-independent mode: provision one sandbox per model, launch "
        "the runner detached inside it (training survives this laptop sleeping/closing), and "
        "exit. Each sandbox pushes results to GitHub and self-stops. Needs GITHUB_TOKEN set.",
    )
    args = parser.parse_args()

    budget_map = SMOKE_BUDGETS if args.smoke else BUDGETS
    models = [args.model] if args.model else list(budget_map.keys())

    if args.detached:
        # Autonomous path: one sandbox per model, fire-and-forget. Results come back
        # via GitHub (mlflow_daytona_week4_<MODEL>.db), not a local download.
        main_detached(models, budget_map)
    else:
        # Local-driven path: distinct db filename per variant so parallel per-model
        # runs (and smoke runs) don't clobber each other's downloads.
        suffix = (f"_{args.model}" if args.model else "") + ("_smoke" if args.smoke else "")
        db_path = f"mlflow_daytona_week4{suffix}.db"
        main(models, budget_map, db_path)
