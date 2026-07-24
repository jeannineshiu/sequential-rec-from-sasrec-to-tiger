#!/bin/bash
# Runs the Week 4 RecBole training-budget sweep FROM INSIDE a Daytona GPU
# sandbox, pushing results to GitHub after every single experiment.
#
# Why this exists instead of relying solely on scripts/daytona_week4.py:
# that script drives the sandbox from your local machine over a long-lived
# connection (sandbox.process.exec calls, several hours total across 6
# experiments). If your laptop sleeps, loses wifi, or the terminal closes
# mid-run, the whole thing can get interrupted -- this happened earlier in
# this project on local MPS training (a 10-hour laptop sleep silently paused
# a background job). Running this script INSIDE the sandbox under tmux makes
# progress independent of your local machine staying connected: SSH in once,
# kick it off, detach, and it keeps running server-side even if you close
# your laptop.
#
# Usage (from your local machine):
#   daytona ssh <sandbox-id>
#   cd sequential-rec-from-sasrec-to-tiger && git pull
#   export GITHUB_TOKEN=ghp_...          # PAT with repo write access
#   chmod +x scripts/daytona_remote_runner.sh
#   tmux new -d -s week4 'bash scripts/daytona_remote_runner.sh 2>&1 | tee run.log'
#   # now safe to disconnect -- check back later with:
#   #   daytona ssh <sandbox-id>
#   #   tmux attach -t week4          (or: tail -f sequential-rec-from-sasrec-to-tiger/run.log)
#
# Keep the EXPERIMENTS list here in sync with scripts/daytona_week4.py's
# EXPERIMENTS list if you change epoch budgets -- there's no shared config
# file between the Python driver and this shell script.

set -uo pipefail

UV="$HOME/.local/bin/uv"
REPO_DIR="$(pwd)"
LOCAL_DB="mlflow_daytona_week4.db"

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN is not set -- results will run locally on the sandbox but"
  echo "won't be pushed to GitHub after each experiment. Set it and re-run if"
  echo "you want the incremental push-after-each-run safety net."
fi

if [ -n "${GITHUB_TOKEN:-}" ]; then
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger.git"
fi

sync_results() {
  local run_name="$1"
  cp mlflow.db "$LOCAL_DB" 2>/dev/null || { echo "  (no mlflow.db yet to sync)"; return; }
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    git add "$LOCAL_DB"
    git commit -m "Week 4 (Daytona GPU): sync after ${run_name}" >/dev/null 2>&1
    if git push origin main; then
      echo "  -> pushed $LOCAL_DB to GitHub after ${run_name}"
    else
      echo "  -> WARNING: push failed after ${run_name} (network blip?); mlflow.db is still safe on disk locally"
    fi
  fi
}

echo "=== Downloading data ==="
"$UV" run python -m src.data.download --dest data/raw --dataset ml-1m
"$UV" run python -m src.recbole_utils.convert_to_atomic

# (model, run_name, epochs) -- keep in sync with scripts/daytona_week4.py
EXPERIMENTS=(
  "SASRec sasrec_recbole_1x 200"
  "SASRec sasrec_recbole_4x 800"
  "SASRec sasrec_recbole_10x 2000"
  "BERT4Rec bert4rec_recbole_1x 200"
  "BERT4Rec bert4rec_recbole_4x 800"
  "BERT4Rec bert4rec_recbole_10x 2000"
)

for exp in "${EXPERIMENTS[@]}"; do
  read -r model run_name epochs <<< "$exp"
  echo ""
  echo "=== $run_name ($model, $epochs epochs) ==="
  if "$UV" run python -m src.recbole_run --model "$model" --epochs "$epochs" --run-name "$run_name"; then
    echo "  -> $run_name finished"
  else
    echo "  -> WARNING: $run_name FAILED (exit $?) -- continuing to the next experiment"
  fi
  sync_results "$run_name"
done

echo ""
echo "=== All experiments attempted. Final sync + marker file. ==="
touch WEEK4_SWEEP_DONE
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git add WEEK4_SWEEP_DONE
  git commit -m "Week 4 (Daytona GPU): sweep complete" >/dev/null 2>&1
  git push origin main
fi
echo "Done. Check run.log / mlflow_daytona_week4.db / the GitHub repo for results."
echo "Remember: stopping this GPU sandbox deletes it immediately. Only stop it"
echo "once you've confirmed results are safe (pushed to GitHub, or downloaded)."
