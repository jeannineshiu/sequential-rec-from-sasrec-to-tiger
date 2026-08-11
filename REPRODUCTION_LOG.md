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

## Week 4 (cont.) — the dropout hypothesis, tested and confirmed

**2026-08-09**

### The setup, and why it is a single-variable test

Before spending anything, checked what actually separates RecBole's two models by reading
`recbole/properties/model/{SASRec,BERT4Rec}.yaml`. They are identical on every architectural
default — `n_layers: 2`, `n_heads: 2`, `hidden_size: 64`, `inner_size: 256`,
`loss_type: CE` — and differ on exactly one: dropout, 0.5 for SASRec against 0.2 for
BERT4Rec. That is stronger than assumed. A dropout-only rerun is not an approximation of a
single-variable experiment, it *is* one.

`configs/recbole/ml1m_sasrec_dropout02.yaml` is an overlay rather than a forked base config,
so the protocol settings stay defined in exactly one place; `--config` now takes a
comma-separated list applied left-to-right. Verified locally that the two files resolve to
dropout 0.2 with every other key untouched, before launching.

### Two things caught before spending money, and one after

**The score export wrote 101 rows per user.** `export_scores` was added to dump the test
score matrix so RecBole runs could be rescored offline. A 1-epoch local smoke run produced a
610,040 x 3,417 matrix for 6,040 users — under `eval_args.mode: uni100` the test loader is a
`NegSampleEvalDataLoader` that emits one row per positive-plus-negative candidate, all
sharing an input sequence, so `full_sort_predict` returned 101 bit-identical rows each. At
68MB compressed that was close enough to GitHub's 100MB hard reject to risk the sandbox's
results push — the exact failure that destroyed the 2026-08-08 run. Deduplicated to one row
per user: 37MB. The lesson is narrow and worth keeping: "the file was written" is not
verification; check its shape.

**A dropout config that silently did not apply would have been invisible for 4.7 hours.**
The previous two SASRec runs agreed at epoch 19 to four decimal places (train loss
2721.4937 / 2721.4938), which makes that epoch a free assertion. This run came back at
**2495.6744** — a 8.3% lower training loss, in the direction less regularization predicts.
Confirmed the overlay was live 27 minutes in rather than at the end.

**A false pass on M4, caught while writing it up.** The obvious cross-validation number —
this repo's SASRec against the new run's headline figures — is −1.64% HR@10 / +1.93%
NDCG@10, inside the ±2% criterion on both metrics. It is also wrong: this repo's sampled
numbers use the frozen `negatives.json` and RecBole's use its own `uni100` draw, so that
comparison differences two different negative sets. Recomputed with both sides on the frozen
negatives it is **+0.61% HR@10 (passes) / +7.41% NDCG@10 (fails)**. The mixed-draw version is
what falls out of reading the master table's headline columns straight across, and it happens
to look like a clean pass. Recorded in `docs/bert4rec-controversy.md` §4 precisely because it
is easy to make.

### Result — RecBole SASRec, dropout 0.2, 200 epochs, ML-1M

Test sampled HR@10 **0.8056**, NDCG@10 **0.6063** (valid 0.8316 / 0.6394), 84.2 s/epoch,
identical to the default-dropout run's throughput. MLflow run `sasrec_recbole_1x_dropout02`.

| RecBole SASRec | HR@10 | NDCG@10 |
|---|---|---|
| dropout 0.5 (default) | 0.7768 | 0.5702 |
| dropout 0.2 | 0.8056 | 0.6063 |
| **effect of dropout alone** | **+3.71%** | **+6.33%** |

BERT4Rec's win over the default-configured SASRec was +3.39% / +5.86%. **The dropout default
alone more than accounts for it.** With the asymmetry removed the comparison does not
collapse to a tie, it reverses: SASRec ahead by +0.31% / +0.45%, which is itself small enough
to read as a tie. The hypothesis from the previous section is confirmed, and the Week 4
headline — BERT4Rec beats SASRec on ML-1M — is a baseline-configuration artifact.

The valid curve was still climbing at epoch 189 (0.6394, a new best) while the run was cut at
200, so this configuration had not converged. The default-dropout run had plateaued. That
matters for the budget curve: the 4x point is unlikely to be flat.

### What the offline rescoring added

`scripts/rescore_recbole.py` maps the exported score matrix back into this repo's id space
(recomputing `reindex_ids` deterministically from the raw ratings, asserting the two 5-core
filters agree on the item set rather than assuming it) and rescores with `src/eval/metrics`.

Two findings, both about protocol rather than models:

- **The negative draw is worth +2.28% HR@10 / +5.38% NDCG@10** on *identical predictions*
  (uni100 0.8056/0.6063 vs. frozen negatives 0.8240/0.6389). The caveat this project has been
  flagging all along now has a number, and it is the same order as every margin in the Week 4
  analysis.
- **The two SASRec implementations diverge far more than the sampled protocol suggests.**
  Against this repo's SASRec: +0.61% sampled HR@10, +7.41% sampled NDCG@10, **+40.1% full
  HR@10, +53.5% full NDCG@10**. The divergence grows monotonically with how much the metric
  cares about position. Cross-entropy over the full catalog versus BCE with one sampled
  negative is the obvious suspect and fits the pattern, but no loss-only ablation was run —
  this is now the largest unexplained effect in the project.

That last row is the more useful result of the two. "Agrees within 2%" was always a statement
about one metric under the protocol this repo elsewhere argues is inflationary, and here is a
case where two implementations agree almost exactly under it while differing by half on the
protocol it recommends.

### M4 status

| Criterion | Status |
|---|---|
| RecBole SASRec within 2% of this repo's | **partially met** — HR@10 +0.61% ✅, NDCG@10 +7.41% ❌ |
| Signature training-budget figure | **not met** — still only the 1x point |

Remaining work, cheapest first: seed-variance study (~14 GPU-hours; nothing here is
established above seed noise) → loss-only ablation to test the full-ranking divergence
(~4.7) → retrain the other two RecBole runs with score export so the comparison can be
repeated on full ranking (~10.5) → one 4x point per model for the budget curve (~23).

---

## Seed variance — the noise floor, and what it invalidates

**2026-08-09**

The seed study above was costed at ~14 GPU-hours because it was scoped to the RecBole runs.
That was the wrong scope for a first pass. This repo's own SASRec trains in ~20 minutes on
laptop MPS, so five seeds of it cost nothing but wall-clock and answer the question that
actually blocks everything else: **how large does a margin have to be here before it means
anything?** If that number had come back at ±3%, the Week 4 headline would have been in
trouble and a GPU-based seed study would have been worth buying. It did not, so it isn't.

Five seeds (42, 1, 2, 3, 4), varying only `train.seed` — weight init and the *training*
negative sampler. Evaluation negatives stay frozen for every run, so this measures training
noise and not evaluation-sampling noise. Added `--seed` / `--run-name` overrides to
`src/train.py` rather than creating five near-duplicate config files.

| Metric | mean | rel. std | range |
|---|---|---|---|
| sampled HR@10 | 0.8188 | 0.28% | 0.71% |
| sampled NDCG@10 | 0.5925 | 0.34% | 0.83% |
| full HR@10 | 0.2453 | 1.19% | 2.97% |
| full NDCG@10 | 0.1305 | 1.08% | 2.45% |

**Full-ranking metrics are ~4x noisier than sampled ones.** That is a useful result on its
own and not one I expected to be so clean: ranking against 3,416 items is far more sensitive
to initialization than ranking against 101. Every full-ranking comparison in this repo needs
a wider band than its sampled counterpart, and a single global noise floor would be wrong in
both directions.

Two details of the floor's construction, both of which I got wrong on the first pass:

- **Per-protocol, not global.** The first version took the worst relative std across all four
  metrics (1.19%, from full HR@10) and applied it everywhere. That is four times too strict
  for sampled comparisons and would have wrongly declared two live Week 3 results dead.
- **√2, not 1.** Every claimed margin is a difference between two runs each measured once, so
  the difference carries both runs' noise: σ_diff = √2·σ. The floor is 2·√2·σ, giving
  **0.96% sampled / 3.37% full**.

`scripts/seed_variance.py` prints every margin claimed in this repo against these floors.

### Week 4 survives; two Week 3 conclusions do not

Week 4 is clean. The dropout effect (+3.71% / +6.33%) is 4–6x the sampled floor. The residual
SASRec-over-BERT4Rec margin at matched dropout (+0.31%) is inside noise, confirming as a
measurement what was already reported as a tie. The M4 cross-check's HR@10 agreement (+0.61%)
is *also* inside noise — which is a stronger result than "passes <2%": on sampled HR@10 the
two independent implementations are indistinguishable. Its NDCG@10 disagreement (+7.41%) and
the full-ranking divergence (+40.1%, against a 3.37% floor) are both real signal.

Week 3 did not fare as well, for a reason unrelated to seeds. **The ablation table was using
the 200-epoch headline run as its baseline while every ablation ran at 100 epochs**, so each
delta was also being charged for 100 fewer epochs of training. The README's own limitation
section said ablations "compare configs against each other" — the table did not do that.

Ran the baseline config at exactly 100 epochs (`ablation_ml1m_baseline_100ep`, ~10 minutes,
free) to get the missing row. The budget effect it exposes is small on sampled HR@10 (+0.47%,
itself inside noise) but large on full HR@10 (**+5.36%**), and correcting for it reverses two
conclusions:

| Ablation, full HR@10 | vs 200-epoch baseline (wrong) | vs 100-epoch baseline (right) | verdict |
|---|---|---|---|
| A1 posemb = none | −7.43% | −2.47% | **inside noise** |
| A2 maxlen = 100 | −5.21% | −0.13% | **inside noise** |
| A1 posemb = sinusoidal | −11.84% | −7.11% | real |
| A2 maxlen = 50 | −17.86% | −13.45% | real |
| A4 neg = popularity | −24.40% | −20.35% | real |

So **maxlen 100 and 200 are indistinguishable on full ranking at this budget**, at less than
half the per-epoch cost (2.85s vs 5.85s) — the opposite of what the table implied. And
dropping positional embeddings entirely is a ~1% sampled regression with no detectable
full-ranking cost, while *sinusoidal* embeddings are the variant that clearly hurts. The
original table made learnable-vs-none look like the important comparison; it is
learnable-vs-sinusoidal.

The general lesson is the same one Week 4 produced, in a different costume: a comparison is
only as good as its baseline. Week 4's baseline carried an unexamined framework default;
Week 3's carried an unexamined epoch budget. Both were introduced while carefully controlling
something else.

---
## Week 5 — Semantic IDs (Day 1–4: content embeddings + RQ-KMeans)

**2026-08-10**

### The prerequisite nobody wrote down: the id map was never saved

Content embeddings need to join item side-info (movie titles, Amazon ASINs) onto the
internal contiguous item ids that everything downstream speaks. `preprocess.py` built that
map and threw it away — `meta.json` kept only `n_users` / `n_items`. Four weeks of results
are keyed to ids whose provenance existed only inside a function scope.

Recovering it is trivial (`reindex_ids` sorts the raw ids, so it is deterministic), but
"trivial to recompute" is exactly the assumption worth testing before building on it. Added
`id_maps.json` to the pipeline output and re-ran both datasets into a scratch directory:
`train.json`, `valid.json`, `test.json`, `meta.json` came back **byte-identical** to the
files on disk for both ML-1M and Beauty. So the maps are the ones the existing results were
produced with, and no Week 1–4 number is disturbed.

### Beauty item metadata

`ratings_Beauty.csv` is ratings-only. The matching `meta_Beauty.json.gz` (99 MB) is still
live in the same SNAP `categoryFiles/` directory, which matters — it is the same ASIN
namespace as the ratings file, so the join needs no fuzzy matching. It is one Python literal
per line (single-quoted), not JSON, so it wants `ast.literal_eval`.

Coverage, over the 12,101 5-core items: **12,101 matched (100%)**. Fields present: categories
100%, title 99.9%, brand 82.7%, description 92.2%. Description is excluded from the embedding
text — it runs to paragraphs of marketing copy and would swamp title/category/brand in a
384-dim mean-pooled encoder. ML-1M is 3,416/3,416 from `movies.dat`.

So neither dataset has an items-without-text problem. `has_text` is stored in the npz anyway,
because discovering that at model-training time would be much more expensive than checking.

### Embeddings

`all-MiniLM-L6-v2`, 384-dim, unnormalized on disk (normalization is a quantizer decision).
ML-1M text is `"<title>. Genres: ..."`, Beauty is `"<title>. Category: <deepest path>. Brand:
<brand>"`. Both datasets encode in under 30 s on laptop MPS — 4.9 MB and 17.2 MB npz.

### RQ-KMeans, 3 levels × 256 codes

Embeddings are L2-normalized before quantization. MiniLM is trained for cosine similarity, so
the Euclidean geometry KMeans minimizes is only meaningful on the unit sphere; without it the
clusters partly track text length, which for Beauty means they track *marketing verbosity*.

| | ML-1M | Beauty |
|---|---|---|
| items | 3,416 | 12,101 |
| codes used / level | 256, 256, 256 | 256, 256, 256 |
| dead codes | 0 | 0 |
| collision rate (identical 3-token code) | 1.46% | 11.78% |
| largest colliding group | 3 | 12 |
| residual norm explained by 3 tokens | 55.7% | 48.7% |

No dead codes at any level on either dataset, which is the failure mode RQ-VAE exists to fix —
so the stretch goal (RQ-VAE with EMA updates) has nothing to fix here and stays skipped.
Beauty's 11.78% collision rate is the number that matters for the model: the disambiguation
token needs a vocabulary of at least 12, and ~1 item in 8 is distinguished *only* by a token
that carries no content signal at all. That is a ceiling on what semantic IDs can do for the
Beauty cold-start story, and it is worth stating before running the experiment rather than
after.

### Do the codes mean anything?

Codebook health is a statement about the quantizer, not about semantics. Measured it directly
— mean cosine similarity between two items sharing a code prefix vs. two random items:

| prefix depth | ML-1M | Beauty |
|---|---|---|
| 1 token | 0.641 (+0.202) | 0.605 (+0.318) |
| 2 tokens | 0.722 (+0.282) | 0.738 (+0.451) |
| 3 tokens | 0.753 (+0.313) | 0.871 (+0.583) |
| random pair | 0.439 | 0.288 |

Coherence rises monotonically with depth on both, which is the coarse-to-fine property the
whole idea depends on. Sampled groups read correctly: Beauty produces a hair-dryer prefix, a
Joico shampoo-duo prefix, a China Glaze nail-polish prefix. Full reports:
`results/tables/semantic_ids_{ml-1m,beauty}.md`.

### What ML-1M's codes actually cluster on

Reading the ML-1M groups, they look less like genres than like *years* — "1997 war films",
"1994 dramas", "1986 comedies". Measured it, since the title string carries the release year:

| | mean abs. year gap | genre Jaccard |
|---|---|---|
| same 1-token prefix | 5.36 | 0.538 |
| same 2-token prefix | **1.68** | 0.737 |
| same 3-token prefix | 2.19 | 0.844 |
| random pair | 15.85 | 0.164 |

Two items sharing a 2-token prefix are on average **1.68 years apart against a 15.85-year
baseline**. Genre is captured strongly too, so this is not the year *instead of* content — the
ML-1M semantic ID is roughly an era × genre address, where Beauty's is category × brand.

This is a modelling choice that was made by accident, in the sense that nobody decided "the
release year should be a primary axis of the semantic ID" — it arrived because MovieLens
titles happen to embed the year in the string. No split leakage is involved (no interaction
data touches the embedding), but it is the same shape of unexamined default as Week 4's
dropout and Week 3's epoch budget, caught this time before it decided a result rather than
after. Whether stripping the year helps or hurts is a one-line change and a cheap rerun; left
as an ablation to run against the Week 6 comparison rather than settled by taste now.

**Next:** Day 5–7 — `src/models/genrec.py`, the generative model over semantic ID token
sequences, in the two de-risking steps from the plan (greedy + post-filter first, then
Trie-constrained beam search).

---

## Week 5 (cont.) — the generative model (Day 5–7)

**2026-08-10**

### The architecture decision, and why it is not TIGER's

TIGER is a T5 encoder-decoder over semantic ID tokens. This repo's GenRec is
SASRec's backbone — the same causal blocks, the same pre-LN ordering, the same tied output
head — with one thing changed: an item is a sequence of 4 semantic tokens instead of one
atomic embedding.

That is a deliberate departure from the paper, and the reason is Week 4. Swapping in a
different transformer stack *alongside* the different item representation would produce a
"TIGER vs SASRec" number that is a mixture of both changes, and Week 4 is the story of what
happens when a comparison quietly acquires a second variable. With a shared backbone, Week 6's
headline compares atomic IDs against semantic IDs and nothing else. The cost is that a
negative result cannot be blamed on T5; the benefit is that a result of either sign means
something.

A second consequence, unplanned but useful: because the backbone is causal and shared,
GenRec gets SASRec's training efficiency — every one of the ~200 token positions in a window
is a supervised next-token prediction, so a user still yields one sample per epoch rather than
one sample per prefix as TIGER's seq2seq formulation would.

### Scoring, so the generative model can be compared at all

A generative model naturally produces a *list*, not a score for an arbitrary item, which
would have made it incomparable with every sampled-protocol number in this repo.
`score_item_tokens` fixes that: a candidate's score is the log-probability the model assigns
to generating its code sequence, teacher-forced over all 4 levels in one pass. The existing
`evaluate_sampled` then runs unchanged, against the same frozen `negatives.json` every other
model uses.

Full ranking cannot work that way — scoring 12,101 items per user autoregressively is not one
matrix product — so it uses constrained beam search, TIGER's own protocol. That carries an
approximation the dot-product models do not have: an item the beam never reaches is a miss,
even if exhaustive scoring would have put it in the top 10. Noted here so the Week 6 table can
be read honestly; beam width bounds it.

### Two bugs the tests caught, both about consistency rather than crashes

**The two paths disagreed.** Beam search and candidate scoring appended tokens to the history
differently, so each hit the positional-embedding limit at a different length and trimmed a
different amount. Both "worked"; they just returned different numbers for the same item —
`beam scores agree with direct scoring` failed by up to 0.28 nats. The fix is to reserve the
final item slot of the window for the item being generated, so neither path ever overflows and
both see an identical context. A silent 0.28-nat disagreement between the ranker and the
scorer is precisely the class of bug that would have produced a confidently wrong Week 6
table.

**The probability test was wrong, not the code.** The first version asserted that the scores
of all items exp-sum to 1. They do not, and should not: the per-level softmax spreads mass
over every *code combination*, and only some combinations are real items. The corrected test
asserts the sum over the full code space is 1 and the sum over items is strictly less — and
that gap is exactly the probability mass constrained decoding redistributes and post-hoc
filtering throws away.

### The performance problem, and the fix that removed a methodological compromise

The first end-to-end run took 11 minutes without finishing one epoch. Not a hang: scoring
re-encoded the 196-token history once per candidate, so 101 candidates meant 101 redundant
encodes to read 4 numbers.

Caching the history's per-layer keys/values and pushing only the candidate's own tokens
through attention:

| | uncached | cached |
|---|---|---|
| sampled scoring, 40 users x 101 candidates | 1057 ms | 24.6 ms |
| full sampled eval, all 22,363 users | ~591 s | **13.8 s** |
| beam search (width 20), all 22,363 users | — | 11.7 s |

**43x.** The cache is only trustworthy if it changes nothing, so the test asserts cached and
uncached scores match to 1e-5 across 1 and 2 attention heads, an empty history, and a
completely full window.

The interesting part is what the speedup bought back. The config originally validated on a
2,000-user subsample because full validation was unaffordable — a real deviation from the
SASRec harness, and the kind of asymmetry that quietly makes two runs incomparable. At 13.8 s
it is affordable, so GenRec now validates on all 22,363 users every epoch, exactly as SASRec
does. An epoch costs 34 s end to end.

### First measurement: the Trie is load-bearing

From the 2-epoch smoke run, unconstrained greedy decoding produced a legal item only **32.8%**
of the time. So constrained decoding is not a tidiness measure here — without it, two thirds
of the model's first-choice recommendations would not exist. Whether that rate improves with
training is a real question, and both numbers are logged every run.

**Next:** the 200-epoch Beauty run, then Week 6's atomic-vs-semantic comparison and the
cold-start bucket analysis.

---

## Week 5 (cont.) — the first generative result, and it is a loss

**2026-08-10**

200-epoch config, early stopped at epoch 100 (patience 20), 37.7 s/epoch on laptop MPS.
Same frozen `negatives.json`, same evaluator, same leave-one-out split as every SASRec row.

| Amazon Beauty, test, k=10 | sampled HR | sampled NDCG | full HR | full NDCG |
|---|---|---|---|---|
| SASRec (atomic IDs, this repo) | **0.5097** | **0.3453** | **0.0594** | **0.0303** |
| GenRec (semantic IDs, same backbone) | 0.3621 | 0.2235 | 0.0329 | 0.0168 |
| relative | −28.96% | −35.27% | −44.6% | −44.6% |

Nowhere near the seed-noise floor (0.96% sampled / 3.37% full), so this is signal, not
variance. Reported as measured.

### Ruling out the beam first

Full ranking is the one number carrying an approximation the baselines do not have — an item
the beam never reaches is a miss. Before differencing anything, that had to be bounded:

| beam | full HR@10 | full NDCG@10 | sec |
|---|---|---|---|
| 10 | 0.0296 | 0.0157 | 13 |
| 20 | **0.0329** | **0.0168** | 17 |
| 50 | 0.0327 | 0.0168 | 38 |
| 100 | 0.0328 | 0.0169 | 74 |
| 200 | 0.0327 | 0.0168 | 143 |

Flat from 20 to 200 — a 10x wider beam recovers nothing. Beam 10 is genuinely too narrow
(seen-item filtering eats into the list), 20 is saturated. The gap to SASRec is not a decoding
artifact. `scripts/beam_sensitivity.py` reproduces this.

### The number that reframes the result

| | SASRec | GenRec |
|---|---|---|
| item / token table | 774,528 | 50,048 |
| everything else | 53,824 | 63,424 |
| **total** | **828,352** | **113,472** |

GenRec runs on **13.7% of the parameters**, because 12,101 item embeddings collapse into 782
token embeddings — a 15.5x smaller table. So the honest statement is not "semantic IDs are
28-45% worse". It is "semantic IDs lose 29% of sampled HR@10 and 45% of full HR@10 while
using an eighth of the parameters", and the per-parameter comparison points the other way.

This is also a confound the Week 6 headline has to name out loud, because it cannot be
controlled away: matching parameter counts would mean either crippling SASRec's item table or
inflating GenRec's hidden dimension, and neither is the method being tested. The compression
*is* the method. Week 4's lesson applies in reverse here — the second variable is intrinsic,
so it gets stated rather than eliminated.

### Candidate explanations, none tested yet

- **Beauty's 11.78% collision rate.** For ~1 item in 8, the only thing distinguishing it from
  a catalogue neighbour is a disambiguation token carrying no content signal at all. The model
  cannot learn to emit that token from content, only from co-occurrence.
- **Capacity**, as above.
- **Four sequential decisions instead of one.** An item is only retrieved if all four codes
  are right; errors compound along the levels. Worth measuring per-level accuracy in Week 6.

### Greedy legality: the Trie stays load-bearing

Unconstrained greedy decoding was legal 32.8% of the time after 2 epochs and **81.8%** after
100. So the model does learn the code manifold — but nearly 1 in 5 of its unconstrained
first choices is still not an item. Constrained decoding is doing real work, not tidying up.

**Next:** Week 6 — the atomic-vs-semantic comparison table, cold-start bucketing (where the
semantic ID story is supposed to pay off, and where an overall loss does not preclude a
tail win), and per-level decode accuracy to test the error-compounding explanation.

---

## Week 6 — the cold-start experiment answers the wrong way

**2026-08-10**

The hypothesis this project was built to test: semantic IDs should help exactly where atomic
IDs are weakest. An item seen twice in training has an embedding shaped by two gradient
updates; an item never seen in training has an embedding still at its initialization. Its
*semantic* ID, by contrast, is made of codes thousands of other items share, so a generative
model ought to reach it from content alone. Prediction: GenRec loses on the head and closes
the gap — or wins — on the tail.

Both models scored in one pass, same users, same order (asserted), so no cross-run
summarisation is involved. Buckets are by the target item's frequency in the *training* split.

| bucket | users | SASRec HR@10 | GenRec HR@10 | relative |
|---|---|---|---|---|
| unseen (0) | 138 | 0.0000 | 0.0000 | — |
| tail (1–4) | 4,594 | 0.0185 | 0.0022 | **−88.2%** |
| torso (5–19) | 9,539 | 0.0427 | 0.0085 | **−80.1%** |
| head (20+) | 8,092 | 0.1033 | 0.0797 | −22.8% |
| overall | 22,363 | 0.0594 | 0.0329 | −44.6% |

**The gap widens monotonically as items get rarer — the exact opposite of the prediction.**
Semantic IDs are worst precisely where they were supposed to help.

Checked the obvious confound first, since beam search would plausibly drop low-probability
(rare) items before high-probability ones: re-ran every bucket at beam 200 instead of 20. Tail
went −88.2% → −89.4%, torso −80.1% → −83.8%. A 10x wider beam does not rescue the tail; if
anything it sharpens the effect. The finding is not a decoding artifact.

The `unseen` bucket is honestly uninformative: 138 users, both models at zero. SASRec's zero is
structural (untrained embedding). GenRec's zero is consistent with any true rate below ~2.2%
(rule of three), so it neither confirms nor refutes the content-generalisation claim. It is
reported as measured and not spun.

### Why: the model recommends 839 items out of 12,101

| model | distinct items across all top-10s | median train freq | % head | % torso | % tail |
|---|---|---|---|---|---|
| SASRec (atomic) | **9,221** | 22 | 54.2% | 42.1% | 3.7% |
| GenRec (semantic) | **839** | 84 | 84.7% | 13.1% | 2.2% |

GenRec's entire recommendation output covers **7% of the catalogue**, against SASRec's 76%.
Generation has collapsed onto a small set of high-probability code sequences, and the median
recommended item is nearly 4x more popular. The tail result is a symptom of that collapse.

This is a known property of MAP-style decoding rather than something specific to semantic IDs:
the model ranks by P(item | history), which contains the popularity prior, while SASRec's dot
product is unnormalized and carries no such prior. That asymmetry is arguably the more
important finding here — swapping atomic IDs for semantic IDs also silently swaps an
unnormalized scorer for a normalized one, and the second change is doing much of the damage.
Nobody sets out to change the objective's relationship to popularity; it arrives with the
architecture, exactly like Week 4's dropout arrived with the framework.

### Per-level accuracy: compounding is real but not the main story

Teacher-forced on the true prefix, so each level is measured independently of the previous
level's mistakes:

| level | argmax accuracy | codebook |
|---|---|---|
| 1 | 0.0976 | 256 |
| 2 | 0.1788 | 256 |
| 3 | 0.2243 | 256 |
| 4 | 0.8627 | 12 |

Accuracy *rises* with depth, because conditioning on the true prefix narrows the choice — the
first code is both the hardest and the most consequential decision. Level 4, the
content-free disambiguation token, is nearly free at 86%, so Beauty's 11.78% collision rate is
not the bottleneck it looked like. The product (0.0034) is not the retrieval rate, since the
levels are not independent, but it shows how little slack there is: four sequential decisions
where the first is right 10% of the time.

### Where this leaves the project

The headline is a negative result, and it is a real one rather than a broken implementation:
beam saturation, per-level accuracy, and the coverage collapse all point the same way, and the
sampled-protocol numbers (which involve no beam at all) agree.

The honest summary is three claims, in decreasing order of confidence:

1. On Beauty, a semantic-ID generative recommender underperforms an atomic-ID SASRec at
   matched backbone, protocol, and training budget — by 29% sampled HR@10, 45% full HR@10.
2. It underperforms *most* on rare items, which contradicts the cold-start motivation.
3. The mechanism is a collapse in recommendation diversity (7% catalogue coverage vs 76%),
   which is at least partly attributable to ranking by a normalized probability rather than by
   an unnormalized score — a change that rides along with the architecture and is not the part
   anyone intends to test.

Not tested, and the obvious next experiments: GenRec at matched *parameter count* rather than
matched hidden dim; popularity-debiased decoding (dividing by an item prior) to separate the
scorer change from the representation change; and whether ML-1M, with 3.5x fewer items and far
longer sequences, shows the same shape.

---

## Week 6 (cont.) — a correction: beam search was flattering the generative model

**2026-08-11**

### What was wrong

The Week 6 entry above states that GenRec's full-ranking numbers come from beam search,
that widening the beam 20 → 200 changes nothing, and therefore that the approximation
"can only cost the generative side". **The last part is false, and false in the direction
that mattered.** Beam ranking was inflating GenRec's scores, not depressing them.

Found it by scoring every catalogue item for every user — which the history KV cache makes
affordable — and comparing the two rankings on the same 1,500 users, same model, same masking:

| | HR@10 |
|---|---|
| beam 20 | 0.0407 |
| exhaustive | 0.0240 |

| of the 61 users where beam reports a top-10 hit | |
|---|---|
| exhaustive agrees it is a hit | 27 |
| beam's rank is better than the true rank | 39 |
| **mean true rank of a beam-reported "hit"** | **166.9** |

Beam search calls an item top-10 when its true rank is ~167th. The mechanism is pruning at
level 1: the beam keeps 20 of 256 first codes, and every high-scoring item behind a discarded
prefix vanishes from the returned list. Those vanished items are exactly the competitors that
should have pushed the target down, so the target's position in a 20-item list flatters it.
Beam also misses genuine hits (9 of 36), but that effect is smaller.

### Why the beam-width sweep did not catch it

Widening the beam does two opposing things at once: it finds more true targets (helps HR@10)
and it finds more competitors that outrank them (hurts HR@10). The two roughly cancel, so a
flat HR@10-vs-beam-width curve reads as "saturated" when it is really two biases in balance.
The sweep measured the wrong quantity — it tested whether the beam finds the target, never
whether the beam's *ranking* is faithful. The right test is the one that should have been run
first: compare against exhaustive scoring.

This is the third time in this project that a control turned out to be measuring something
adjacent to the thing it was supposed to control (Week 3's epoch budget, Week 4's dropout
default, now this). The pattern is the same each time: the check was cheap, plausible, and
answered a slightly different question than the one being asked.

### Corrected numbers, and what actually holds

All rows exhaustive over the full catalogue, all 22,363 test users, k=10. SASRec's numbers are
unchanged — it was always ranked exhaustively.

| Amazon Beauty, full ranking | overall | unseen (138) | tail (4,594) | torso (9,539) | head (8,092) |
|---|---|---|---|---|---|
| SASRec (atomic) | **0.0594** | 0.0000 | **0.0185** | **0.0427** | **0.1033** |
| GenRec (semantic) | 0.0251 | 0.0072 | 0.0026 | 0.0060 | 0.0608 |
| GenRec, debiased α=1 | 0.0117 | **0.0725** | 0.0141 | 0.0080 | 0.0137 |

GenRec's overall loss is **−57.7%**, not the −44.6% reported from the beam. Every superseded
figure moves the same way: the generative model is worse than the first pass said.

### The debiasing sweep, and a partial vindication

`score_α(item) = log P(item | history) − α · log P_prior(item)`, add-one smoothed training
frequency, exhaustive:

| α | HR@10 | unseen | tail | torso | head | distinct items in top-10 |
|---|---|---|---|---|---|---|
| 0 | 0.0251 | 0.0072 | 0.0026 | 0.0060 | 0.0608 | 1,749 |
| 0.25 | 0.0232 | 0.0072 | 0.0048 | 0.0073 | 0.0526 | 1,838 |
| 0.5 | 0.0198 | 0.0145 | 0.0065 | 0.0089 | 0.0402 | 1,974 |
| 0.75 | 0.0155 | 0.0362 | 0.0122 | 0.0084 | 0.0255 | 2,021 |
| 1 | 0.0117 | **0.0725** | 0.0141 | 0.0080 | 0.0137 | 1,976 |

Two results, in opposite directions.

**The cold-start claim survives, in its narrowest form.** On items never seen in training, the
debiased generative model retrieves 7.25% at HR@10 where SASRec retrieves 0.00% — 10 hits in
138 against none (Fisher exact, one-sided, **p = 0.0008**). On tail items the debiased model is
statistically indistinguishable from SASRec (~65 vs ~85 hits in 4,594, p = 0.059). So semantic
IDs *can* reach items an atomic embedding table structurally cannot, which is the thing the
architecture was supposed to buy. The price is more than half the overall accuracy
(0.0251 → 0.0117), and it is paid on the head, where most of the traffic is.

**The proposed mechanism is only partly right.** If the popularity prior were the whole story,
removing it should restore diversity. Recommendation coverage goes 1,749 → 1,976 distinct
items out of 12,101 — a 13% improvement on a number that needs to grow 5x to match SASRec's
9,221. Debiasing moves probability mass from popular items to rare ones without making the
model more *discriminative*; it trades head accuracy for tail accuracy along a fixed frontier
rather than expanding it. The collapse is therefore not merely a scoring-time artifact that a
better decoder fixes — something about training the model to emit 4 codes leaves it with far
less resolution over the catalogue than 12,101 free embeddings have.

### Status of the remaining Week 6 work

Done: the main comparison, cold-start bucketing, the diversity/per-level diagnosis, the beam
correction, the debiasing sweep, and the serving demo.

In flight: a single-pass version of the corrected comparison table (SASRec + GenRec α=0 +
GenRec α=1 scored in one run, for the figure). The numbers above come from two completed
full-catalogue runs rather than one, which is a provenance wrinkle worth closing but not a
correctness problem — both used identical users, masking, and metric code.

Not done: README rewrite against the corrected numbers, and the Medium/interview write-ups.

---
