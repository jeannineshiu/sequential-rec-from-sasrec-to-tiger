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

## Week 3 — Amazon Beauty + ablations

**2026-07-23**

- **Beauty pipeline.** Downloaded the Amazon "Beauty" ratings-only CSV (snap.stanford.edu
  mirror, 82MB, no official checksum published so just sanity-checked via row/user/item
  counts) and ran it through the same generic `k_core_filter` / `leave_one_out_split`
  pipeline as ML-1M (refactored `preprocess.py` to share that logic — only the raw-file
  loader differs between datasets now).
  - Result: `users=22363 items=12101 interactions=198502`, avg sequence length 8.88
    (min 5, max 204) — matches Kang & McAuley (2018) Table 1 almost exactly (paper:
    22363 users, 12101 items, 198502 actions). Second dataset in a row where the stats
    line up before any model code touches it.
  - This dataset is sparse (avg 8.9 actions/user) vs. ML-1M's dense 165.5 — the
    contrast itself is the point of running both (per EXECUTION_PLAN.md).

- **Beauty SASRec run** (`configs/sasrec_beauty.yaml`: maxlen 50, hidden_dim 64,
  dropout 0.5, 200 epochs):

  | Metric | Paper (~0.4854, ±2pp accepted) | This repo | In range? |
  |---|---|---|---|
  | sampled HR@10 | 0.4654–0.5054 | **0.5097** | ⚠️ +0.43pp over the accepted band |
  | sampled NDCG@10 | — | 0.3453 | (no paper reference given in EXECUTION_PLAN.md) |
  | full HR@10 | — | 0.0594 | |
  | full NDCG@10 | — | 0.0303 | |

  Slightly outside the ±2pp band, but only barely, and Beauty reproductions are known
  to be more hyperparameter-sensitive than ML-1M across published SASRec ports (sparse
  data, short sequences) — not chasing this further given the small margin; reporting
  as-is rather than tuning until it lands inside the band, per the project's honesty
  principle over cherry-picked numbers.

- **Model/dataset extended for the ablations** (`src/models/sasrec.py`,
  `src/data/dataset.py`), each covered by new unit tests before any training was run:
  - `pos_emb_type`: `learnable` (existing) / `none` (skip positional embedding
    entirely) / `sinusoidal` (fixed Vaswani et al. 2017 encoding, registered as a
    buffer, not a parameter — confirmed via `named_buffers()` vs `named_parameters()`).
  - `neg_sampling`: `uniform` (existing) / `popularity` (weighted by training-set
    item frequency, sampled via a precomputed CDF + `searchsorted` rather than
    re-normalizing on every call).
  - 10 new tests (27 total): positional-embedding invariance under `none`, buffer-vs-
    parameter check and zero-padding-row check for `sinusoidal`, popularity sampling
    never returning a user's own history and empirically favoring frequent items over
    rare ones (500-sample frequency check).

- **Ablation runs use `max_epochs=100`** (vs. 200 for the headline SASRec/Beauty
  numbers above) — a deliberate compute-saving choice since ablations are about
  *relative* comparison between configs, not re-chasing the exact paper-aligned peak
  for each variant. Noting this explicitly since it's a deviation from the main-result
  protocol.

- **All 5 ablation runs completed** (100/100 epochs each, no early stopping triggered).
  Full table: `results/tables/master.md` (generated via `uv run python -m src.export_results`,
  never hand-copied). Results (ML-1M, sampled/full HR@10 & NDCG@10, test set):

  | Ablation | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 | avg s/epoch |
  |---|---|---|---|---|---|
  | Baseline (learnable pos emb, maxlen 200, uniform neg) | 0.8190 | 0.5948 | 0.2475 | 0.1322 | ~7.0 |
  | **A1** pos_emb = none | 0.8066 | 0.5707 | 0.2291 | 0.1222 | 7.47 |
  | **A1** pos_emb = sinusoidal | 0.8147 | 0.5763 | 0.2182 | 0.1134 | 6.91 |
  | **A2** maxlen = 50 | 0.7858 | 0.5539 | 0.2033 | 0.1080 | 1.53 |
  | **A2** maxlen = 100 | 0.8058 | 0.5762 | 0.2346 | 0.1228 | 2.85 |
  | **A4** neg_sampling = popularity | 0.7540 | 0.5225 | 0.1871 | 0.0995 | 6.93 |

  **A1 (positional embedding):** learnable > sinusoidal > none on sampled NDCG@10, but
  the gap is smaller than expected (none only ~2.4pp NDCG below learnable) — on ML-1M's
  dense sequences (avg 165 actions/user), the model apparently recovers a fair amount of
  order information from the causal mask structure alone. This is a *softer* result
  than the paper's framing that learnable positional embeddings matter a lot on dense
  data; worth flagging honestly rather than overstating the effect.

  **A2 (maxlen):** clear monotonic trend, HR/NDCG improve from maxlen 50 → 100 → 200
  (baseline), while per-epoch time roughly follows the expected quadratic-ish scaling
  in attention cost (1.53s → 2.85s → ~7.0s, i.e. maxlen 200 costs ~4.6x maxlen 50 for
  ~13x the sequence length — sub-quadratic in practice, likely because much of the
  per-epoch cost is embedding/FFN overhead that's linear in maxlen, not just attention).
  Textbook sequence-length-vs-quality-vs-cost tradeoff curve.

  **A4 (negative sampling):** popularity-weighted training negatives *hurt* rather than
  helped (0.754 vs. 0.819 sampled HR@10) — the biggest single drop of any ablation.
  Plausible explanation: popularity-weighted negatives are "harder" (the model spends
  more capacity distinguishing popular-but-wrong items from the true next item) but at
  only 100 epochs the model hasn't had time to fully exploit that harder signal, so it
  just looks like slower/worse convergence. Flagging as an open question rather than a
  settled conclusion — a longer run might change this ranking, which is itself the
  point of running ablations at reduced epoch budget: real but budget-sensitive effects
  need to be labeled as such, not presented as final.

- **A3 (sampled vs. full-ranking correlation):** scatter plot at
  `results/figures/sampled_vs_full.png`, built from every trained model so far (Week 2
  ML-1M, Week 3 Beauty, both baselines, all 5 ablation variants) rather than per-epoch
  checkpoints of a single run — treating each *trained model* as one (sampled, full)
  data point. Finding: within ML-1M, the two protocols correlate cleanly (all 6 ML-1M
  variants + baselines fall on a clear upward trend). But the relationship is not
  universal across datasets/catalog sizes — Beauty's full-ranking HR@10 (0.059) sits
  far below where its sampled HR@10 (0.510) would predict if extrapolating the ML-1M
  trend, because full-ranking difficulty scales with catalog size (12101 items vs.
  3416) independent of model quality. Takeaway for the eventual writeup: sampled-vs-full
  correlation is real but dataset-conditional — comparing full-ranking numbers *across*
  datasets needs catalog-size normalization, comparing *within* a dataset is fine.

---

## Week 4 — BERT4Rec via RecBole, and a GPU-infrastructure detour

**2026-07-24 → 2026-07-25**

Analysis of what the resulting numbers do and don't support lives in
[`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md). This section is the trail of
how they were produced — which, honestly, was mostly not about recommendation.

### Protocol matching (the only part that determines whether the comparison means anything)

- `configs/recbole/ml1m_base.yaml` mirrors this repo's own protocol field by field: 5-core
  on both users and items, `LS: valid_and_test` leave-one-out grouped by user ordered by
  timestamp, `MAX_ITEM_LIST_LENGTH: 200`, `mode: uni100` (1 positive + 100 uniform
  negatives), Hit/NDCG@10 with NDCG@10 as the valid metric.
- **Gap I did not close:** `uni100` makes RecBole draw its *own* negatives instead of
  consuming the frozen `negatives.json` this repo shares across every other model. The
  EXECUTION_PLAN fallback (export RecBole's raw scores, rescore through this repo's
  evaluator) was never implemented. Same protocol shape, different draw — recorded rather
  than hidden, because 1.6pp is what separates the two models.
- Both RecBole models run `loss_type: CE` with `train_neg_sample_args: ~` (full softmax over
  the catalog), while this repo's SASRec trains with BCE against one sampled negative. So the
  eventual comparison varies architecture, framework, and loss simultaneously.

### Cloud GPU detour (Daytona) — where the week actually went

ML-1M + maxlen 200 + RecBole was too slow on the Mac, so the sweep moved to a Daytona GPU
sandbox. Roughly a day and a half went into making that work. The failures were worth
recording because none of them were model bugs:

- **Thread oversubscription livelock, twice.** RecBole training wedged at epoch 1: GPU at
  0–1%, all allotted cores pegged, ~228 threads, main thread in `futex_wait`. Inside a
  cgroup-limited container OpenMP/MKL/BLAS read the *host* core count and each spawn that
  many threads. First fix capped threads to `os.sched_getaffinity(0)` — which **reproduced
  the bug**, because Daytona's `cpu=N` is a CFS quota, not an affinity mask, so affinity
  still reported ~96 host cores and the "cap" set 96 threads. Real fix
  (`_effective_cpu_limit()` in `src/recbole_run.py`): honor an explicit launcher override
  first, then read the cgroup CFS quota, then fall back to affinity. Lesson: in a container,
  "how many CPUs do I have" has at least three different wrong answers.
- **A "hang" that wasn't.** A run was killed three times for hanging after `apt-get`. It was
  stdout block-buffering: setup had completed and training was progressing invisibly.
  Fixed with line-buffered stdout, `PYTHONUNBUFFERED=1`, and `init_logger(config)` — RecBole's
  low-level API path leaves the logger handler-less, so a perfectly healthy run prints
  *nothing* for its entire duration. Diagnosis cost more than the bug: I killed working runs
  because I couldn't see them working.
- **Path base mismatch.** `process.exec` resolves relative paths against `/workspace`, the
  FileSystem API against `/root`. The uploaded `ratings.dat` would have landed somewhere the
  converter never looked — caught on review before it triggered.
- **`torch.load` `weights_only`.** PyTorch 2.6 flipped the default to `True`, which can't
  unpickle RecBole checkpoints (they carry full config/optimizer state). Latent the whole
  time; only surfaced at the smoke test because every earlier run was killed mid-training and
  never reached post-training evaluation.
- **Detached mode.** Final design runs training *inside* the sandbox under `nohup`, pushes
  its results db to GitHub, and self-deletes — so nothing depends on the laptop staying awake
  and no idle GPU billing accrues.

### Speedups: two worked, one was a no-op

Measured rather than assumed, which is the only reason I know one of them was wasted effort:

| Change | Verdict |
|---|---|
| One trajectory per model, freezing best-valid checkpoints at each budget milestone via `fit`'s `callback_fn` | ✅ 6000 → 4000 epochs across the planned sweep |
| `eval_step` 1 → 10 (validation is ~16s) | ✅ ~90% of evaluation time removed |
| `train_batch_size` 128 → 2048 + `worker=4` | ❌ **no measurable effect** — throughput saturated by 1024; workers neutral |
| `--model` per sandbox for cross-model parallelism | ✅ halves wall-clock at identical credit cost |

The batch/worker change came from a real observation (GPU at 20–25% utilization, dataloader-
starved) and still did nothing. Keeping the negative result visible: the profiling was right
about the bottleneck and the fix still didn't move the number.

### Result, and the scope cut

- **BERT4Rec (RecBole, 200 epochs, CUDA):** test sampled HR@10 **0.8031**, NDCG@10 **0.6036**;
  valid HR@10 0.8291, NDCG@10 0.6304. 104.1 s/epoch, 20823 s total (~5.8 GPU-hours). MLflow
  run `bert4rec_recbole_1x`.
- **Against this repo's SASRec** (0.8190 / 0.5948, ~7.0 s/epoch on MPS, ~25 min): a tie.
  SASRec +1.6pp HR@10, BERT4Rec +0.9pp NDCG@10. Under a matched protocol and a matched epoch
  budget, BERT4Rec's decisive win over SASRec does not reproduce.
- **Only the 1x budget was run.** The sweep infrastructure supports 1x/4x/10x from a single
  trajectory and `BUDGETS` in `scripts/daytona_week4.py` still lists all three, but at
  104 s/epoch the full 2000-epoch run is ~58 GPU-hours *per model*. Cut to the 1x point on
  cost. There is therefore no `training_budget.png` and no budget curve — the scaling claim
  at the centre of the controversy is untested here, not answered.
- **No full-ranking metrics for either RecBole run** — RecBole eval was uni100-only, so those
  cells are blank in `results/tables/master.md`.

---

## Week 4 (cont.) — the RecBole SASRec run, lost and re-run

**2026-08-08 → 2026-08-09**

- **First attempt: trained fine, results destroyed.** 200 epochs completed (~4.7 GPU-hours),
  then the push back to GitHub failed on an expired PAT and the runner's unconditional
  self-delete removed the sandbox — results db and log with it. Root cause and the five
  process fixes are in the (gitignored) notes vault as
  `20260808-daytona-sasrec-結果全損事故-postmortem.md`; the code fixes are commit `40ad41f`
  (pre-flight credential validation on both sides, push retries, `PUSH_OK`-gated self-stop,
  and `scripts/daytona_recover.py`).
- **Second attempt reproduced the first exactly.** With `seed=42`, epoch 19 came back at
  train loss 2721.4938 vs. the lost run's 2721.4937 and identical valid HR/NDCG. Worth
  recording as an unplanned determinism check: the RecBole path is reproducible across
  sandboxes, so the loss cost money and time but no information.

**Result — RecBole SASRec, 200 epochs, ML-1M, sampled protocol:** test HR@10 **0.7768**,
NDCG@10 **0.5702** (valid 0.8101 / 0.6009), 84.2 s/epoch. MLflow run `sasrec_recbole_1x`,
merged into `mlflow.db` via the new `scripts/merge_daytona_results.py` so the master table
stays script-generated rather than hand-edited.

**This did not do what it was supposed to do, and did something more interesting instead.**

- **M4's `<2%` criterion failed**: −5.16% HR@10, −4.14% NDCG@10 against this repo's SASRec.
  But the criterion assumed the two runs differ only by implementation. They do not — RecBole
  defaults are d=64, 2 heads, dropout 0.5, CE loss, batch 2048, against this repo's d=50,
  1 head, dropout 0.2, BCE+1neg, batch 128. The number measures configuration, not
  correctness, so it neither convicts nor exonerates the implementation. The real
  cross-validation (RecBole SASRec with *this repo's* hyperparameters) is still unrun.
- **The same-framework comparison inverts the headline.** Against RecBole's own SASRec,
  BERT4Rec wins both metrics (+3.39% HR@10, +5.86% NDCG@10); against this repo's SASRec the
  same BERT4Rec run merely ties. Same BERT4Rec number, opposite conclusion, and the only
  thing that changed is the baseline.
- **The likely mechanism, untested:** RecBole ships `hidden_dropout_prob: 0.5` for SASRec and
  `0.2` for BERT4Rec. On dense ML-1M that is a real handicap — this repo's own ML-1M config
  uses 0.2 and reserves 0.5 for sparse Beauty. Flagging as a hypothesis: no dropout ablation
  was run, and the settling experiment is one ~4.7 GPU-hour rerun at dropout 0.2.
- **The methodological lesson, which is the actual Week 4 finding:** this project spent its
  whole design effort matching the *evaluation* protocol across models and never checked that
  the *model* hyperparameters were comparable. `configs/recbole/ml1m_base.yaml` sets split,
  negatives, maxlen, budget and metrics identically, then silently inherits per-model
  defaults that are not symmetric. A protocol-controlled comparison is not automatically a
  fair one — which is precisely the failure mode the BERT4Rec reproducibility literature
  describes, reproduced here by accident.

**M4 still not met.** The signature training-budget figure and a like-for-like
cross-validation are both outstanding. Cheapest path, in cost order: RecBole SASRec at
dropout 0.2 (~4.7 GPU-hours / ~$10 — tests the hypothesis *and* serves as the real
cross-check) → rescore RecBole predictions through this repo's evaluator (CPU-only; removes
the negative-draw caveat and yields the missing full-ranking numbers) → one 4x point per
model (~23 GPU-hours) for an actual budget curve.

---
