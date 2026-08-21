# From SASRec to TIGER

[![CI](https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger/actions/workflows/ci.yml/badge.svg)](https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

A controlled study of sequential recommendation, from the 2018 self-attentive baseline to a
TIGER-style generative retriever over RQ-quantized semantic IDs — one codebase, one evaluation
harness, one set of frozen negatives, so that every number on this page is differenced against
something measured the same way.

The repo has two deliverables. The first is a **faithful, verified SASRec reproduction** on
MovieLens-1M and Amazon Beauty. The second is a **generative recommender built on the same
backbone**, where items are emitted as sequences of semantic tokens rather than looked up in an
embedding table — and a measurement of exactly what that representation buys and what it costs.

Both are evaluated under two protocols side by side (sampled and full-catalog ranking), against a
measured seed-noise floor, with negative and mixed results reported as they came out.

---

## Key findings

**1. A framework's default hyperparameter outweighed the architectural effect it was used to
demonstrate.** RecBole's SASRec and BERT4Rec configs are identical on every architectural default
except dropout (0.5 vs 0.2). Aligning that single line moves the SASRec–BERT4Rec comparison by
+3.71% HR@10 / +6.33% NDCG@10 — larger than the entire margin the comparison was meant to explain,
and enough to flip the winner. Same BERT4Rec run, three conclusions.

**2. Agreement under the sampled protocol is not agreement.** Two SASRec implementations that match
to +0.61% on sampled HR@10 diverge by **+40% HR@10 / +53% NDCG@10** under full-catalog ranking. Any
model selection done on the sampled protocol alone is selecting on a metric that does not preserve
ordering.

**3. Semantic IDs trade accuracy for compression and reach, not for accuracy.** On Beauty, the
generative model reaches 71% of SASRec's sampled HR@10 while running on **13.7% of the parameters**
(12,101 item embeddings collapse into 782 token embeddings). It also retrieves items that are
structurally unreachable for an atomic embedding table: on items never seen in training, 7.25%
HR@10 against SASRec's 0.00% (Fisher exact, one-sided *p* = 0.0008).

**4. Swapping item representations silently swaps scoring rules — and that is where the damage
is.** A generative model ranks by `P(item | history)`, which carries a popularity prior; a dot
product does not. Constrained beam search compounds it: beam-20 reports HR@10 0.0407 where
exhaustive scoring gives 0.0240, because the mean true rank of a beam-reported hit is 167. Both
effects arrived with the architecture, uninvited, in the same way the dropout default arrived with
the framework.

---

## What's in the repo

| Component | Path | Notes |
|---|---|---|
| SASRec | `src/models/sasrec.py` | From-scratch PyTorch; causal self-attention, BCE against one sampled negative per position, per Kang & McAuley (2018) |
| GenRec (TIGER-style) | `src/models/genrec.py` | Same backbone, decoder over semantic tokens; constrained (Trie) decoding, KV-cached |
| Semantic ID construction | `src/semantic_ids/` | Item text → `all-MiniLM-L6-v2` → residual K-Means, 3×256 codes + a disambiguation token |
| Baselines | `src/baselines.py` | Popularity, BPR-MF (`implicit`) |
| Evaluation | `src/eval/` | Sampled (1+100), full-catalog, cold-start bucketing, generative scoring |
| Cross-framework runs | `src/recbole_run.py`, `scripts/rescore_recbole.py` | RecBole SASRec/BERT4Rec + offline rescoring onto this repo's frozen negatives |
| Analysis | `scripts/` | Seed variance, ablations, atomic-vs-semantic, decoding diagnostics, popularity debiasing |
| Serving demo | `serving/app.py` | FastAPI; both models side by side with decoded semantic IDs |
| Remote GPU orchestration | `scripts/daytona_*.py` | Detached, host-independent sweeps with result recovery |

Datasets after 5-core filtering and leave-one-out splitting: **ML-1M** — 6,040 users / 3,416 items;
**Amazon Beauty** — 22,363 users / 12,101 items.

---

## Evaluation protocol

Everything below follows the same three rules; deviations are marked at the point of use.

- **Two protocols, always reported together.** The *sampled* protocol (1 positive + 100 uniform
  negatives) matches the original SASRec paper. The *full* protocol ranks against the entire
  catalog excluding history. Sampled metrics are known to be non-order-preserving
  (Krichene & Rendle, 2020) and this repo measures how badly, rather than citing it.
- **Frozen negatives.** `data/processed/*/negatives.json` is drawn once (seed 42) and reused by
  every model. *Exception:* RecBole's evaluator draws its own `uni100` sample. That is not a
  hand-wave — rescoring identical predictions both ways puts the effect at +2.28% HR@10 / +5.38%
  NDCG@10, the same order as the margins under discussion, so rows from different draws are never
  differenced against each other.
- **Leave-one-out split**, 5-core filtered, per user: last item → test, second-to-last → valid, rest
  → train. Absence of leakage is asserted at preprocessing time.

---

## Results

### SASRec reproduction — ML-1M, sampled protocol, test, k=10

| Model | HR@10 | NDCG@10 | Note |
|---|---|---|---|
| Popularity | 0.4363 | 0.2401 | floor |
| BPR-MF | 0.5745 | 0.3357 | floor |
| SASRec — Kang & McAuley (2018) | 0.8245 | 0.5905 | target |
| **SASRec (this repo)** | **0.8190** | **0.5948** | ✅ inside the accepted band (0.80–0.83 / 0.57–0.60) |
| SASRec (RecBole, dropout 0.5 — its default) | 0.7768 | 0.5702 | measures the dropout default, not the implementation |
| SASRec (RecBole, dropout 0.2) | 0.8056 | **0.6063** | dropout-only rerun |
| SASRec (RecBole, dropout 0.2, rescored on frozen negatives) | 0.8240 | 0.6389 | +0.61% HR@10 vs this repo; +7.41% NDCG@10 |
| BERT4Rec (RecBole) | 0.8031 | 0.6036 | see below |

All RecBole rows are 200 epochs.

### The dropout default

| Comparison (protocol- and budget-matched) | HR@10 | NDCG@10 | Winner |
|---|---|---|---|
| BERT4Rec vs RecBole SASRec (**dropout 0.5**, default) | +3.39% | +5.86% | BERT4Rec, on both |
| BERT4Rec vs this repo's SASRec | −1.94% | +1.48% | tie |
| BERT4Rec vs RecBole SASRec (**dropout 0.2**) | −0.31% | −0.45% | SASRec, on both |
| *effect of the dropout default alone* | *+3.71%* | *+6.33%* | — |

The residual SASRec-vs-BERT4Rec margin (+0.31% / +0.45%) sits inside the measured noise floor and
should be read as a tie. The dropout effect is 4–6× the floor.

This repo walked into the failure mode by matching the *evaluation* protocol across models with
care and never checking that the *model* hyperparameters were comparable — which is precisely what
the BERT4Rec reproducibility literature describes. Claim-by-claim analysis:
[`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md).

### Sampled vs full-catalog ranking — ML-1M, test, k=10

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0369 | 0.0180 |
| BPR-MF | 0.0671 | 0.0333 |
| SASRec (this repo) | 0.2475 | 0.1322 |
| SASRec (RecBole, dropout 0.2, rescored) | **0.3467** | **0.2029** |

+40% / +53% for a pair that agrees to +0.61% on sampled HR@10. The divergence grows monotonically
with how much the metric cares about *where* in the ranking the target lands, which points at the
training objective — full-catalog cross-entropy (RecBole) vs BCE against one sampled negative (this
repo, per the paper). Consistent with the pattern, but untested: no loss-only ablation was run.
This is the largest unexplained effect in the project.

### Atomic vs semantic IDs — Amazon Beauty, test, k=10

Same backbone, same protocol, same frozen negatives. The only variable is whether an item is one
embedding or a sequence of four semantic tokens. Both full-ranking columns score the catalog
exhaustively.

| | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 | parameters |
|---|---|---|---|---|---|
| SASRec (atomic) | **0.5097** | **0.3453** | **0.0594** | **0.0303** | 828,352 |
| GenRec (semantic) | 0.3621 | 0.2235 | 0.0250 | 0.0131 | **113,472** |
| relative | −28.96% | −35.27% | −57.8% | −56.6% | **13.7%** |

Every margin is far outside the seed-noise floor.

The parameter column is what makes this a trade rather than a loss. The comparison **cannot** be
parameter-matched: equalizing would mean crippling SASRec's item table or inflating GenRec's hidden
dimension, and the compression *is* the method under test. It is stated, not controlled away.

Unconstrained greedy decoding produces a valid item 81.8% of the time after training (32.8% after
two epochs), so the Trie constraint is doing real work — without it, nearly one in five top-1
recommendations would not be an item that exists.

### Cold start: the hypothesis fails in the direction it was supposed to win

Semantic IDs should help where atomic IDs are weakest — a rare item has a barely-trained embedding,
but its semantic ID is built from codes thousands of items share. Prediction: GenRec loses on the
head, closes the gap on the tail.

![cold-start buckets](results/figures/cold_start_buckets.png)

Full-catalog HR@10, bucketed by the target's training frequency. `debiased α=1` subtracts the log
training-frequency prior at ranking time — same weights, different scoring rule. All three columns
come from a single scoring run over identical users.

| bucket | users | SASRec | GenRec | GenRec debiased α=1 |
|---|---|---|---|---|
| unseen (0) | 138 | 0.0000 | 0.0072 | **0.0725** |
| tail (1–4) | 4,594 | **0.0185** | 0.0026 | 0.0144 |
| torso (5–19) | 9,539 | **0.0427** | 0.0060 | 0.0081 |
| head (20+) | 8,092 | **0.1033** | 0.0606 | 0.0138 |
| overall | 22,363 | **0.0594** | 0.0250 | 0.0118 |

**As trained, the gap widens as items get rarer — the opposite of the prediction.**

**Debiased, the claim survives in its narrowest form.** On never-seen items the generative model
retrieves 7.25% HR@10 where SASRec retrieves 0.00% — 10 hits in 138 against none, Fisher exact
one-sided *p* = 0.0008. On tail items it becomes statistically indistinguishable from SASRec
(*p* = 0.059). Semantic IDs really do reach items an atomic table structurally cannot. The price is
more than half the overall accuracy, paid on the head where the traffic is.

### Mechanism: generation collapses onto a small set of code sequences

| model | distinct items across all top-10s | median train freq | % head | % torso | % tail | % unseen |
|---|---|---|---|---|---|---|
| SASRec (atomic) | **9,221** (76% of catalog) | 22 | 54.2% | 42.1% | 3.7% | 0.0% |
| GenRec (semantic) | **1,749** (14%) | 63 | 74.1% | 21.3% | 4.6% | 0.0% |
| GenRec debiased α=1 | **1,976** (16%) | 5 | 7.0% | 44.8% | 36.6% | 11.6% |

The debiased row pins the mechanism down, and it cuts against the tidy explanation. Debiasing
changes *what* is recommended enormously — head share 74.1% → 7.0%, median recommended item from 63
training appearances to 5, 11.6% of slots to never-seen items. Yet coverage moves only 1,749 →
1,976: a 13% gain on a number that needs 5× to match SASRec.

So the popularity prior is not the whole story. Removing it slides the model along a fixed frontier,
trading head accuracy for tail accuracy — which is why overall HR@10 halves — without making it
more *discriminative*. Learning to emit four codes leaves the model with far less resolution over
the catalog than 12,101 free embeddings have, and that is not a scoring-time artifact a better
decoder repairs.

Per-level code accuracy (teacher-forced on the true prefix) is 9.8% / 17.9% / 22.4% / 86.3%.
Accuracy *rises* with depth as the prefix narrows the choice, and the content-free disambiguation
token is nearly free — so Beauty's 11.78% collision rate is not the bottleneck it appears to be.
**The binding constraint is the first code.**

### Beam search is not a faithful ranker

An earlier iteration ranked GenRec by constrained beam search and argued the approximation "can
only cost the generative side," on the evidence that widening the beam 20 → 200 changed nothing.
That reasoning was wrong, and the numbers throughout this README have been regenerated with
exhaustive scoring.

On the same 1,500 users, beam-20 reports HR@10 0.0407 where exhaustive scoring gives 0.0240, and
the **mean true rank of a beam-reported top-10 hit is 167**. Beam pruning discards 236 of 256 first
codes, so the high-scoring items that should have outranked the target never enter the returned
list. The width sweep missed this because widening the beam finds more targets *and* more
competitors simultaneously; the two cancel, and a flat curve reads as convergence. The sweep tested
whether the beam finds the target, never whether its ranking is faithful.

Superseded beam figures: −44.6% overall (vs −57.8% exhaustive), 839 distinct items / 84.7% head
(vs 1,749 / 74.1%). The collapse is real either way, and about half as severe as the beam made it
look.

### Ablations — ML-1M, test, k=10, all rows at 100 epochs

Deltas against the 100-epoch baseline and marked against the measured noise floor (sampled 0.96%,
full 3.37%). `~` means inside seed noise.

| Ablation | sampled HR@10 | Δ | full HR@10 | Δ | avg s/epoch |
|---|---|---|---|---|---|
| **Baseline (learnable pos emb, maxlen 200)** | **0.8152** | — | **0.2349** | — | 5.85 |
| positional embedding = none | 0.8066 | −1.05% | 0.2291 | ~ −2.47% | 7.47 |
| positional embedding = sinusoidal | 0.8147 | ~ −0.06% | 0.2182 | −7.11% | 6.91 |
| maxlen = 50 | 0.7858 | −3.61% | 0.2033 | −13.45% | 1.53 |
| maxlen = 100 | 0.8058 | −1.15% | 0.2346 | ~ −0.13% | 2.85 |
| negative sampling = popularity-weighted | 0.7540 | −7.51% | 0.1871 | −20.35% | 6.93 |

Two notes on reading this table. First, it is baselined at a matched budget: an earlier version
charged every ablation for 100 fewer epochs than its baseline, which is small on sampled HR@10
(+0.47%) but large on full HR@10 (+5.36%) and reversed two conclusions — pos-emb-none and
maxlen-100 both looked like real full-ranking regressions and are in fact inside noise. Read
literally, **maxlen 100 and 200 are indistinguishable on full ranking at this budget**, at less
than half the per-epoch cost.

Second, positional embeddings are the interesting row: dropping them costs ~1% sampled and nothing
detectable on full ranking, while *sinusoidal* is the one variant that clearly hurts full ranking
(−7.11%). Learnable-vs-none is nearly a wash; learnable-vs-sinusoidal is not.

### Noise floor — ML-1M SASRec, 5 seeds (42, 1–4)

Only the training seed varies (weight init + training negative sampler); evaluation negatives stay
frozen, so this is training noise alone.

| Metric | mean | rel. std | range |
|---|---|---|---|
| sampled HR@10 | 0.8188 | 0.28% | 0.71% |
| sampled NDCG@10 | 0.5925 | 0.34% | 0.83% |
| full HR@10 | 0.2453 | 1.19% | 2.97% |
| full NDCG@10 | 0.1305 | 1.08% | 2.45% |

**Full-ranking metrics are ~4× noisier than sampled ones** — separating 3,416 items is far more
sensitive to initialization than separating 101. Comparisons here are between two runs each
measured once, so the relevant floor is 2·√2·σ: **0.96% sampled, 3.37% full**.
`uv run python -m scripts.seed_variance` re-checks every margin claimed in this repo against it.
Five seeds estimate σ loosely — this is a sanity floor, not a significance test.

### Semantic ID quality

Item text → `all-MiniLM-L6-v2` (384-d) → residual K-Means, 3 levels × 256 codes, plus a 4th token
disambiguating items that land on an identical 3-token code. ML-1M text is title + genres; Beauty is
title + deepest category path + brand. Metadata coverage is 100% on both.

| | ML-1M | Beauty |
|---|---|---|
| dead codes (any level) | 0 | 0 |
| collision rate on the 3-token code | 1.46% | 11.78% |
| largest colliding group | 3 | 12 |
| embedding norm explained by 3 tokens | 55.7% | 48.7% |
| within-prefix cosine @ depth 3 (vs random pair) | 0.753 (0.439) | 0.871 (0.288) |

No dead codes on either dataset — the codebook-collapse failure mode RQ-VAE exists to fix does not
appear with residual K-Means here, so RQ-VAE was not needed. Prefix coherence rises monotonically
with depth, which is the coarse-to-fine property the generative decoder depends on.

Two properties that constrain the downstream results: **Beauty collides at 11.78%**, so for ~1 item
in 8 the only separator from a catalog neighbour is a content-free token; and **ML-1M's codes encode
release year at least as strongly as genre** — items sharing a 2-token prefix are 1.68 years apart
on average against a 15.85-year baseline — because MovieLens titles embed the year in the string.
Nobody chose that; it came in with the text format.

Reports: [`results/tables/semantic_ids_ml-1m.md`](results/tables/semantic_ids_ml-1m.md) ·
[`results/tables/semantic_ids_beauty.md`](results/tables/semantic_ids_beauty.md).

Script-generated master table (never hand-edited):
[`results/tables/master.md`](results/tables/master.md).

---

## Serving demo

```bash
uv run uvicorn serving.app:app --reload   # http://127.0.0.1:8000/
```

FastAPI service exposing `/recommend` and `/random_user`, running both models on the same input
sequence and returning each recommendation's decoded semantic ID, training frequency, and
popularity bucket. The point is not a leaderboard — the offline tables settled that — but that the
*shape* of the disagreement is legible per request: the generative model's list is measurably more
popular and less varied, and items sharing a code prefix look related.

The demo ranks GenRec by beam search, because scoring 12,101 items per request is not a serving-time
option. Its list is therefore more flattering than the tables above, by the margin quantified in
the beam-search section.

---

## Reproduce

```bash
uv sync

# Data
uv run python -m src.data.download --dest data/raw --dataset all
uv run python -m src.data.preprocess --dataset ml-1m  --out-dir data/processed/ml-1m
uv run python -m src.data.preprocess --dataset beauty --out-dir data/processed/beauty

# Baselines + SASRec
uv run python -m src.baselines --data-dir data/processed/ml-1m
uv run python -m src.train --config configs/sasrec_ml1m.yaml
uv run python -m src.train --config configs/sasrec_beauty.yaml

# Semantic IDs (Beauty metadata is an extra ~99MB)
uv run python -m src.data.download --dest data/raw --dataset beauty --with-meta
uv run python -m src.semantic_ids.embed     --dataset ml-1m
uv run python -m src.semantic_ids.rq_kmeans --dataset ml-1m
uv run python -m scripts.inspect_semantic_ids --dataset ml-1m   # quality report

# Generative model
uv run python -m src.train_genrec --config configs/genrec_beauty.yaml

# Analysis. Each of these scores all 12,101 items for all 22,363 test users:
# budget ~55 min apiece on an M-series GPU. Add --beam to the first for the
# superseded beam-ranked numbers.
uv run python -m scripts.compare_atomic_vs_semantic  # bucket table + cold-start figure
uv run python -m scripts.diagnose_genrec             # diversity + per-level accuracy
uv run python -m scripts.debias_decoding             # popularity-debiasing α sweep
uv run python -m scripts.seed_variance               # noise floor + margin re-check

uv run python -m src.export_results   # rebuild results/tables/master.md + figures
uv run pytest tests/
```

Experiment tracking is MLflow (`sqlite:///mlflow.db`, experiment `sequential-rec`); the master
table is generated from it rather than transcribed. Long GPU sweeps run detached on remote
sandboxes via `scripts/daytona_*.py`, with result recovery for interrupted runs.

---

## Limitations and open questions

Kept current, on the principle that a mixed result reported is worth more than a clean result
implied.

- **The full-ranking divergence between the two SASRecs is unexplained.** +40% HR@10 / +53% NDCG@10
  from implementations agreeing to +0.61% on sampled HR@10. Full-catalog CE vs sampled BCE is the
  obvious suspect and fits the pattern, but no loss-only ablation was run.
- **No training-budget curve.** Only the 1× (200-epoch) point exists; 4× and 10× were cut on cost
  (~58 GPU-hours per model for the full trajectory). The scaling claim at the heart of the BERT4Rec
  controversy is therefore untested here.
- **BERT4Rec comparisons rest on the sampled protocol.** Only one RecBole run has full-ranking
  numbers, and for the one pair where both exist they disagree by 40–53%. Nothing here establishes
  that the SASRec-vs-BERT4Rec ordering survives full-catalog evaluation. The cross-framework
  comparison also varies loss, batch size and architecture at once. Full accounting in
  [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md) §4.
- **Every RecBole number is a single seed.** The 0.96% / 3.37% floors come from five seeds of *this
  repo's* SASRec on ML-1M and are applied to RecBole runs and to Beauty as a proxy — different
  models, frameworks, datasets. Indicative there, not measured.
- **Ablations are a statement about 100-epoch training.** Deliberate compute saving; both sides of
  every delta share the budget, but the popularity-negatives result in particular may be a
  convergence-speed effect rather than a final ranking.
- **Beauty SASRec sits 0.43pp above the accepted reproduction band** (0.5097 vs 0.4654–0.5054).
  Reported as-is rather than tuned until it lands inside.
- **The atomic-vs-semantic comparison is not parameter-matched** and cannot be, by construction.
- **The serving demo's GenRec list is beam-ranked** and therefore more optimistic than every table
  on this page.

---

## References

- Kang & McAuley, *Self-Attentive Sequential Recommendation* (ICDM 2018) — SASRec
- Sun et al., *BERT4Rec* (CIKM 2019); Petrov & Macdonald, *A Systematic Review and Replicability
  Study of BERT4Rec* (RecSys 2022)
- Rajput et al., *Recommender Systems with Generative Retrieval* (NeurIPS 2023) — TIGER
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020)

Supporting documents:
[`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md) — claim-by-claim analysis of what
this repo's data does and does not establish ·
[`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md) — the full debugging and decision trail, including
every result that was later corrected ·
[`docs/original-plan.md`](docs/original-plan.md) and
[`docs/execution-plan.md`](docs/execution-plan.md) — the pre-registered plan and acceptance
criteria, written before any experiment ran and kept unedited (several predictions did not
survive the data).
