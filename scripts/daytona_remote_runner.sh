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
#
# Results land in the repo as mlflow_daytona_week4_<MODEL>.db on branch main, plus
# a WEEK4_<MODEL>_DONE marker commit. Watch progress by pulling the repo, or SSH
# in and `tail -f run.log`.

set -uo pipefail

UV="$HOME/.local/bin/uv"
REPO_DIR="$(pwd)"
MODEL="${MODEL:?MODEL not set}"
BUDGETS="${BUDGETS:?BUDGETS not set}"
LOCAL_DB="mlflow_daytona_week4_${MODEL}.db"

echo "=== Week 4 autonomous runner: model=${MODEL} budgets=${BUDGETS} ==="

# Authenticate git push if a token was injected (results safety net).
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger.git"
  git config user.email "week4-runner@daytona.local"
  git config user.name "week4-daytona-runner"
else
  echo "WARNING: GITHUB_TOKEN not set -- results will stay only in this sandbox."
fi

push_results() {
  local label="$1"
  cp mlflow.db "$LOCAL_DB" 2>/dev/null || { echo "  (no mlflow.db yet to push)"; return; }
  [ -z "${GITHUB_TOKEN:-}" ] && return
  # -f: .gitignore has `mlflow*.db`, so a plain `git add` would silently skip the
  # results db and push nothing useful. Force-add this specific results file.
  git add -f "$LOCAL_DB"
  git commit -m "Week 4 (Daytona ${MODEL}): ${label}" >/dev/null 2>&1
  # Rebase-pull first: the two per-model sandboxes push the same branch, so land
  # each other's commits instead of racing/rejecting.
  git pull --rebase --autostash origin main >/dev/null 2>&1 || true
  if git push origin main; then
    echo "  -> pushed ${LOCAL_DB} to GitHub (${label})"
  else
    echo "  -> WARNING: push failed (${label}); ${LOCAL_DB} is still on the sandbox disk"
  fi
}

self_stop() {
  # Delete this sandbox from within itself so there's no idle billing. Results are
  # already on GitHub, so even if this fails nothing is lost -- the sandbox just
  # keeps billing until stopped manually / by the recover tool.
  if [ -n "${DAYTONA_API_KEY:-}" ] && [ -n "${SANDBOX_ID:-}" ]; then
    echo "=== Self-stopping sandbox ${SANDBOX_ID} (results already pushed) ==="
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

# Data was uploaded by the launcher (grouplens blocks the sandbox IP); just build
# the RecBole atomic files.
echo "=== convert_to_atomic ==="
"$UV" run python -m src.recbole_utils.convert_to_atomic

# One trajectory to the largest budget; recbole_run logs every budget milestone.
echo "=== training: ${MODEL} (${BUDGETS}) ==="
if "$UV" run python -m src.recbole_run --model "$MODEL" --budgets "$BUDGETS"; then
  echo "  -> ${MODEL} training complete"
else
  echo "  -> WARNING: ${MODEL} training FAILED (exit $?) -- pushing whatever synced, then stopping"
fi

push_results "final results"

# Completion marker so a watcher can tell this model is done from GitHub alone.
touch "WEEK4_${MODEL}_DONE"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git add "WEEK4_${MODEL}_DONE"
  git commit -m "Week 4 (Daytona ${MODEL}): sweep complete" >/dev/null 2>&1
  git pull --rebase --autostash origin main >/dev/null 2>&1 || true
  git push origin main || echo "  -> WARNING: marker push failed"
fi

self_stop
echo "=== Done. ==="
