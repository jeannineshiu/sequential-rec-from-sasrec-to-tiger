# Reproduction Log

Format: `date / hypothesis / change / result / next step`. Written as work happens, not
after the fact — this is raw material for the Medium post's "how I actually debugged this"
section.

---

## Week 1 — Data pipeline, evaluators, baselines

**2026-07-23**

- **Data pipeline sanity check.** Ran `src/data/download.py` + `src/data/preprocess.py`
  (5-core filtering, leave-one-out split) on MovieLens-1M.
  - Result: `users=6040 items=3416 interactions=999611`, `avg sequence length=165.50
    (min=18, max=2277)`.
  - This matches Kang & McAuley (2018) Table 1 essentially exactly (paper: 6040 users,
    3416 items, avg actions/user 165.5). Strong signal the 5-core + reindexing logic is
    correct before any model code is written.
  - No-leakage assertions (`test[u] not in train[u]`, `valid[u] not in train[u]`) pass
    for all users.

- **Evaluator unit tests.** Built a 3-user hand-computed toy dataset for both the sampled
  (1 pos + 100 neg) and full-ranking evaluators (`tests/test_eval.py`). Both match manual
  HR@k / NDCG@k calculations exactly. Evaluator protocol is now frozen — no changes to
  `src/eval/metrics.py`, `sampled.py`, or `full_ranking.py` without re-running every
  affected experiment.

- **Fixed negative sampling.** Generated one `negatives.json` (seed=42, 100 negatives/user,
  excluding each user's train+valid+test items) so every model — self-implemented SASRec,
  RecBole SASRec/BERT4Rec, and the TIGER-style model — gets scored against the identical
  negative set. This is what makes the Week 4 cross-validation and Week 6 comparisons
  meaningful rather than apples-to-oranges.

- **Baselines (ML-1M, test set, k=10):**

  | Model | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 |
  |---|---|---|---|---|
  | Popularity | 0.4363 | 0.2401 | 0.0369 | 0.0180 |
  | BPR-MF (implicit, factors=64, iters=100) | 0.5745 | 0.3357 | 0.0671 | 0.0333 |

  - BPR-MF clears Popularity on every metric, as expected — sanity check that the
    pipeline + evaluators aren't silently broken in a way that makes baselines
    indistinguishable.
  - The gap between sampled and full-ranking numbers (e.g. Popularity 0.44 vs 0.037
    HR@10) is the Krichene & Rendle (2020) sampled-metric inflation effect, visible
    even at the baseline stage. Worth a callout in the README methodology section.
  - Logged to MLflow (`sqlite:///mlflow.db`, experiment `sequential-rec`, runs
    `popularity_ml1m` / `bpr_mf_ml1m`).

- **Still open before Week 1 is fully closed:** run SASRec via RecBole (or the official
  repo) once to record a third-party reference number for Week 2's triangulation step.

---

## Week 2 — SASRec from scratch

**2026-07-23**

- **Model implementation (`src/models/sasrec.py`).** Followed Kang & McAuley (2018)
  closely: item embedding scaled by `sqrt(hidden_dim)` + learnable positional embedding
  indexed by absolute slot position in the padded window (not relative-to-content),
  2 causal self-attention blocks (pre-LN: LayerNorm → MHA → residual) each followed by
  a point-wise conv FFN, output layer shares weights with the input item embedding.

- **Guarded against the known SASRec footguns up front** (`tests/test_sasrec.py`, 5 tests):
  - Causal mask direction: changing the last token of a sequence must not change the
    encoder output at any earlier position. Verified directly — encoding two sequences
    that differ only in the final slot gives identical hidden states everywhere except
    that slot.
  - Padding excluded from attention: `key_padding_mask` construction unit-tested against
    `input_seqs == 0` directly.
  - Positional embedding range: `pos_emb` sized `maxlen + 1` (slot 0 reserved for
    padding_idx); tested with a full-length, no-padding sequence to make sure indexing
    `maxlen` doesn't throw.
  - `model.score()` (candidate-subset scoring, used by the sampled evaluator) and
    `model.score_full_catalog()` (used by full-ranking) tested for numerical agreement
    on overlapping items — a divergence here would have meant sampled and full-ranking
    numbers for the same model aren't actually comparable.

- **Training loop (`src/train.py`).** Adam (lr=1e-3, β2=0.98), BCE-with-logits over
  per-position pos/neg pairs masked to non-padding positions, early stopping on sampled
  valid NDCG@10 (patience 20), MLflow logging per epoch, checkpoint saved to
  `results/checkpoints/`.

- **Smoke test (2 epochs)** before committing to a full run: loss 1.18 → 1.00, no
  crashes, ~7-14s/epoch on Apple Silicon MPS. Confirmed feasible to run the full 200
  epochs locally (~25 min) rather than needing cloud compute for this phase.

- **Full run, ML-1M, 200 epochs (no early stop triggered — valid NDCG kept slowly
  improving through epoch 200):**

  | Metric | Paper (Kang & McAuley 2018) | This repo | In range? |
  |---|---|---|---|
  | sampled HR@10 | 0.80–0.83 | **0.8190** | ✅ |
  | sampled NDCG@10 | 0.57–0.60 | **0.5948** | ✅ |
  | full HR@10 | — | 0.2475 | (no paper reference; far above BPR-MF's 0.0671) |
  | full NDCG@10 | — | 0.1322 | (no paper reference; far above BPR-MF's 0.0333) |

  **M2 milestone met on the first full training run** — no debugging iteration needed
  against the known footguns list, which suggests writing the 5 targeted unit tests
  before training paid off (would otherwise have been the Day 6-7 "align with paper"
  debugging slog per EXECUTION_PLAN.md).

- **Operational note:** accidentally deleted `mlflow.db` right before kicking off the
  full training run (ran `rm -f mlflow.db` to get a clean tracking db, forgetting it
  also held the Week 1 Popularity/BPR-MF run records). The numbers themselves were safe
  in this log and in README.md, so just re-ran `src/baselines.py` afterward to restore
  the MLflow entries (identical results, fully deterministic seeds). Lesson: don't blow
  away shared tracking state for a single new run — MLflow experiments should be
  additive, not reset per run.

- **Still open before Week 2 is fully closed:** run SASRec via RecBole for the
  three-way triangulation (paper vs. RecBole vs. this repo) mentioned in
  EXECUTION_PLAN.md Day 6-7 — deferred since the repo's own number already lands
  inside the target range, but the cross-validation still adds credibility for the
  eventual writeup.

---
