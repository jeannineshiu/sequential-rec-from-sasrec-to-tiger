#!/bin/bash
# Fully autonomous, Mac-independent Week 4 runner. Runs INSIDE a Daytona GPU
# sandbox, detached, so training survives your laptop sleeping / closing / going
# offline -- nothing on your machine drives it. When done it pushes results to
# GitHub and STOPS (deletes) its own sandbox so there is no idle billing.
#
# It is normally started for you by `scripts/daytona_week4.py --detached`, which
# provisions the sandbox, uploads the ML-1M data (grouplens blocks the sandbox's
# IP, so it can't self-download), injects the env vars below, and launches this
# under nohup, then exits. You do not run this by hand.
#
# Required env (injected by the launcher):
#   MODEL            SASRec | BERT4Rec         (one model per sandbox, for parallelism)
#   BUDGETS          "200:sasrec_recbole_1x,800:..._4x,2000:..._10x"  (recbole_run --budgets)
#   GITHUB_TOKEN     PAT with contents:write   (push results back to the repo)
#   DAYTONA_API_KEY  Daytona key               (self-stop at the end)
#   SANDBOX_ID       this sandbox's id         (self-stop target)
# Optional:
#   CONFIGS          comma-separated RecBole config files, layered left-to-right
#                    (default: configs/recbole/ml1m_base.yaml)
#   RUN_TAG          names the results file and DONE marker (default: $MODEL)
#   SEED             training seed passed to recbole_run (default: 42). Every RecBole
#                    number in the repo is seed 42; seeded repeats exist to measure
#                    this framework's own spread instead of borrowing one.
#
# Results land in the repo as mlflow_daytona_week4_<RUN_TAG>.db on branch main, plus
# a WEEK4_<RUN_TAG>_DONE marker commit. Watch progress by pulling the repo, or SSH
# in and `tail -f run.log`.

set -uo pipefail

UV="$HOME/.local/bin/uv"
REPO_DIR="$(pwd)"
MODEL="${MODEL:?MODEL not set}"
BUDGETS="${BUDGETS:?BUDGETS not set}"
CONFIGS="${CONFIGS:-configs/recbole/ml1m_base.yaml}"
# A variant run (e.g. SASRec at dropout 0.2) is still MODEL=SASRec, so keying the
# results file on MODEL alone would overwrite the previous SASRec run's db and
# DONE marker in the repo. RUN_TAG separates them.
RUN_TAG="${RUN_TAG:-$MODEL}"
SEED="${SEED:-42}"
LOCAL_DB="mlflow_daytona_week4_${RUN_TAG}.db"

echo "=== Week 4 autonomous runner: model=${MODEL} tag=${RUN_TAG} budgets=${BUDGETS} ==="
echo "=== configs: ${CONFIGS} seed: ${SEED} ==="

# Tracks whether the results db is safely off this sandbox. self_stop refuses to
# delete the sandbox unless this is 1 -- see the comment on self_stop.
PUSH_OK=0

# Authenticate git push if a token was injected (results safety net).
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger.git"
  git config user.email "week4-runner@daytona.local"
  git config user.name "week4-daytona-runner"
else
  echo "WARNING: GITHUB_TOKEN not set -- results will stay only in this sandbox."
fi

# Prove we can actually reach the remote with these credentials BEFORE spending
# hours of GPU time. A 2026-08-08 SASRec run trained all 200 epochs (~4.7 GPU-h,
# ~$12) and only then discovered the injected PAT had expired: the push failed and
# the results were destroyed by the unconditional self-stop below. A dead token is
# knowable in one second at startup, so fail here instead -- nothing is lost yet.
if [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "=== pre-flight: verifying push credentials ==="
  if git ls-remote origin >/dev/null 2>&1; then
    echo "  -> remote reachable with the injected token"
  else
    echo "FATAL: cannot reach origin with the injected GITHUB_TOKEN (expired/revoked?)."
    echo "Aborting BEFORE training so no GPU time is wasted. Refresh the PAT and relaunch."
    # Nothing has been computed yet, so deleting the sandbox here loses nothing.
    PUSH_OK=1
    self_stop_now=1
  fi
fi

push_results() {
  local label="$1"
  cp mlflow.db "$LOCAL_DB" 2>/dev/null || { echo "  (no mlflow.db yet to push)"; return; }
  [ -z "${GITHUB_TOKEN:-}" ] && return
  # -f: .gitignore has `mlflow*.db`, so a plain `git add` would silently skip the
  # results db and push nothing useful. Force-add this specific results file.
  git add -f "$LOCAL_DB"
  # Raw test score matrices (results/scores/*.npz) let the metrics be recomputed
  # locally with this repo's own evaluator -- fixed negatives and full ranking --
  # so they ride back with the db rather than dying with the sandbox. Not ignored,
  # so a plain add suffices; the || true keeps a run with no export from failing here.
  git add results/scores 2>/dev/null || true
  git commit -m "Week 4 (Daytona ${RUN_TAG}): ${label}" >/dev/null 2>&1
  # Retry: a single transient failure (network blip, rate limit, a racing push from
  # the sibling sandbox) must not be the difference between keeping and losing a
  # multi-hour result. Re-pull before each attempt so a rejected non-fast-forward
  # gets a fresh base.
  local attempt
  for attempt in 1 2 3 4 5; do
    # Rebase-pull first: the two per-model sandboxes push the same branch, so land
    # each other's commits instead of racing/rejecting.
    git pull --rebase --autostash origin main >/dev/null 2>&1 || true
    if git push origin main; then
      echo "  -> pushed ${LOCAL_DB} to GitHub (${label}, attempt ${attempt})"
      PUSH_OK=1
      return
    fi
    echo "  -> push attempt ${attempt}/5 failed (${label}); retrying in $((attempt * 30))s"
    sleep $((attempt * 30))
  done
  PUSH_OK=0
  echo "  -> ERROR: all 5 push attempts failed (${label}); ${LOCAL_DB} exists ONLY on this sandbox"
}

self_stop() {
  # Delete this sandbox from within itself so there's no idle billing -- but ONLY
  # when the results are provably safe on GitHub.
  #
  # This used to run unconditionally, which made the failure mode catastrophic
  # rather than merely annoying: GPU sandboxes require auto_delete_interval=0
  # ("must be ephemeral"), so a stopped sandbox is deleted instantly and its disk
  # goes with it. On 2026-08-08 an expired PAT made the push fail; this function
  # then deleted the only copy of a completed 200-epoch run. Idle billing (~$2.2/h,
  # and visible in the dashboard) is strictly cheaper than re-running (~$12 + 4.7h),
  # so when the push failed we keep the sandbox ALIVE and shout about it instead.
  if [ "$PUSH_OK" -ne 1 ]; then
    echo "=================================================================="
    echo "NOT self-stopping: results were never pushed."
    echo "Sandbox ${SANDBOX_ID:-<unknown>} is being LEFT RUNNING (~\$2.2/h) so"
    echo "${LOCAL_DB} can still be recovered from ${REPO_DIR}/${LOCAL_DB}."
    echo "Recover it, THEN stop the sandbox manually:"
    echo "  uv run python scripts/daytona_recover.py ${SANDBOX_ID:-<id>}"
    echo "=================================================================="
    return
  fi
  if [ -n "${DAYTONA_API_KEY:-}" ] && [ -n "${SANDBOX_ID:-}" ]; then
    echo "=== Self-stopping sandbox ${SANDBOX_ID} (results confirmed pushed) ==="
    "$UV" run python - <<'PY'
import os
from daytona import Daytona, DaytonaConfig
d = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
d.delete(d.get(os.environ["SANDBOX_ID"]))
print("sandbox delete requested")
PY
  else
    echo "DAYTONA_API_KEY/SANDBOX_ID not set -- NOT self-stopping. Stop it manually."
  fi
}

# Pre-flight said the credentials are dead: bail out before burning GPU hours.
if [ "${self_stop_now:-0}" = "1" ]; then
  self_stop
  exit 1
fi

# Data was uploaded by the launcher (grouplens blocks the sandbox IP); just build
# the RecBole atomic files.
echo "=== convert_to_atomic ==="
"$UV" run python -m src.recbole_utils.convert_to_atomic

# One trajectory to the largest budget; recbole_run logs every budget milestone.
echo "=== training: ${MODEL} (${BUDGETS}) ==="
if "$UV" run python -m src.recbole_run --model "$MODEL" --budgets "$BUDGETS" --config "$CONFIGS" --seed "$SEED"; then
  echo "  -> ${MODEL} training complete"
else
  echo "  -> WARNING: ${MODEL} training FAILED (exit $?) -- pushing whatever synced, then stopping"
fi

push_results "final results"

# Completion marker so a watcher can tell this model is done from GitHub alone.
# Only emitted once the results themselves are safely pushed -- an earlier version
# committed this marker on the failure path too, so "sweep complete" appeared in
# the history for runs whose results never arrived.
if [ "$PUSH_OK" -eq 1 ] && [ -n "${GITHUB_TOKEN:-}" ]; then
  touch "WEEK4_${RUN_TAG}_DONE"
  git add "WEEK4_${RUN_TAG}_DONE"
  git commit -m "Week 4 (Daytona ${RUN_TAG}): sweep complete" >/dev/null 2>&1
  git pull --rebase --autostash origin main >/dev/null 2>&1 || true
  git push origin main || echo "  -> WARNING: marker push failed (results themselves are safe)"
fi

self_stop
echo "=== Done. ==="
