# From SASRec to TIGER

[![CI](https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger/actions/workflows/ci.yml/badge.svg)](https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

A controlled study of sequential recommendation, from the 2018 self-attentive baseline to a
generative retriever that emits items as sequences of quantized semantic tokens — one codebase, one
evaluation harness, one set of frozen negatives, so that every number on this page is differenced
against something measured the same way.

The repo has two deliverables. The first is a **faithful, verified SASRec reproduction** on
MovieLens-1M and Amazon Beauty. The second is a **generative recommender built on the same
backbone**, where items are emitted as sequences of semantic tokens rather than looked up in an
embedding table — and a measurement of exactly what that representation buys and what it costs.

**What the second deliverable is, precisely.** It tests *semantic IDs as an item representation*,
under a single-variable ablation. It is not a TIGER reproduction. TIGER pairs semantic IDs with a T5
encoder–decoder, an RQ-VAE quantizer, and a user token; none of those are here. The generative model
runs on SASRec's own transformer stack, and that is the point: swapping the backbone *and* the item
representation at once would produce exactly the kind of comparison this repo exists to document —
one where the measured effect belongs to something nobody chose to test. Holding the backbone fixed
is what makes "atomic vs. semantic ID" a clean single variable. RQ-VAE was skipped on measured
evidence rather than on budget: residual K-Means produces **zero dead codes on both datasets**
([below](#semantic-id-quality)), so the codebook-collapse failure mode RQ-VAE exists to fix does not
arise here. That is a finding, not a shortcut. The title names the direction of travel; the
experiments are on the item representation.

Both are evaluated under two protocols side by side (sampled and full-catalog ranking), against a
measured seed-noise floor, with negative and mixed results reported as they came out.

---

## Key findings

**1. A framework's default hyperparameter outweighed the architectural effect it was used to
demonstrate.** RecBole's SASRec and BERT4Rec configs are identical on every architectural default
except dropout (0.5 vs 0.2). Aligning that single line moves the SASRec–BERT4Rec comparison by
+3.71% HR@10 / +6.33% NDCG@10 — larger than the entire margin the comparison was meant to explain,
and enough to flip the winner. Same BERT4Rec run, three conclusions. This puts the *premise* of the
BERT4Rec controversy in question rather than settling it: the budget claim at its centre is
[untested here](#the-dropout-default).

**2. Agreement under the sampled protocol is not agreement.** Two SASRec implementations that match
to +0.61% on sampled HR@10 diverge by **+40% HR@10 / +53% NDCG@10** under full-catalog ranking. Any
model selection done on the sampled protocol alone is selecting on a metric that does not preserve
ordering.

**3. Most of that divergence is the training objective, and the sampled protocol is blind to it.**
A loss-only ablation — full-catalog softmax instead of BCE against one sampled negative, nothing
else touched — recovers **+22.54% full HR@10 / +32.29% full NDCG@10**, accounting for 56% and 60%
of the cross-framework gap. On sampled HR@10 the same change measures **−0.38%, inside seed noise**.
The objective is worth a fifth of the full-ranking score and is invisible to the protocol the
original paper evaluated with.

**4. Semantic IDs trade accuracy for compression and reach, not for accuracy.** On Beauty, the
generative model reaches 71% of SASRec's sampled HR@10 while running on **13.7% of the parameters**
(12,101 item embeddings collapse into 782 token embeddings). It also retrieves items that are
structurally unreachable for an atomic embedding table: on items never seen in training, 7.25%
HR@10 against SASRec's 0.00% (Fisher exact, one-sided *p* = 0.0008).

**5. Swapping item representations silently swaps scoring rules — and that is where the damage
is.** A generative model ranks by `P(item | history)`, which carries a popularity prior; a dot
product does not. Constrained beam search compounds it: beam-20 reports HR@10 0.0407 where
exhaustive scoring gives 0.0240, because the mean true rank of a beam-reported hit is 167. Both
effects arrived with the architecture, uninvited, in the same way the dropout default arrived with
the framework.

**6. The generative model's deficit is not localized to one repairable stage.** Handing it the
target's true first semantic code multiplies HR@10 by 12.5x — and still leaves **69% of targets
outside the top 10 with a median of 51 candidates remaining**. Its level-1 prediction is not lost
either (median rank 26 of 256 against 128 for chance). An earlier reading of this repo's per-level
accuracies called the first code the binding constraint; measured in retrieval terms it is
expensive but not binding, and no better decoder recovers the rest.

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

**What this does and does not settle.** The BERT4Rec reproducibility literature's central claim is
about *training budget* — that BERT4Rec's reported wins need far more training than the original
comparisons gave it. **This repo does not test that claim.** Only the 1× (200-epoch) point ran; 4×
and 10× were cut on cost (~58 GPU-hours per model for the full trajectory), so there is no budget
curve here and nothing on this page adjudicates the controversy on its own terms. What the table
above establishes is something upstream of it: at a *fixed* budget, the SASRec–BERT4Rec comparison
is decided by a hyperparameter default neither paper is about, and the winner flips three ways
depending on which SASRec you pick. That makes the controversy's premise — that these two models
were ever compared like for like — the thing in question, which is a different and, taken on its
own, arguably more useful finding. Read it that way, not as a verdict.

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
repo, per the paper). The next section tests that directly.

### The training objective, isolated

The two frameworks' SASRecs differ in the loss *and* in width (64 vs 50), heads (2 vs 1), inner size
and update granularity, so the cross-framework gap cannot attribute anything on its own.

A note on that last one, because the nominal numbers invert it. RecBole's batch size is 2048 against
this repo's 128, which reads as RecBole taking the larger steps. It does not: RecBole augments one
row per sequence position and takes the loss only at each row's last position, so 2048 rows is 2048
target positions per update, 480 updates per epoch. This repo emits one sample per user and takes
the loss at every valid position, so 128 sequences is **13,488** target positions per update and 48
updates per epoch. RecBole's step is 6.6× *smaller* and it takes 10× more of them. Any ablation on
"batch size" that copies the number 2048 across would move this repo further from RecBole, not
closer.
This ablation changes the objective and nothing else: same model, same seed, same 200-epoch
schedule, same data, same frozen evaluation negatives, with `train.loss_type: ce` swapping BCE for
a softmax over the full catalog.

| ML-1M, test, k=10 | BCE (control) | CE | Δ | RecBole | residual (RecBole vs CE) |
|---|---|---|---|---|---|
| sampled HR@10 | 0.8190 | 0.8159 | ~ −0.38% | 0.8240 | +0.99% |
| sampled NDCG@10 | 0.5948 | 0.6133 | +3.10% | 0.6389 | +4.17% |
| full HR@10 | 0.2475 | **0.3033** | **+22.54%** | 0.3467 | +14.30% |
| full NDCG@10 | 0.1322 | **0.1749** | **+32.29%** | 0.2029 | +16.06% |

`~` marks a difference inside the seed-noise floor. **The objective alone reproduces the whole
shape of the cross-framework gap**: nothing on sampled HR@10, a large gain on full ranking, and
NDCG gaining more than HR — the same monotone-in-rank-sensitivity pattern, from a single changed
line. It accounts for **56% of the full HR@10 gap and 60% of the full NDCG@10 gap**.

It does not account for all of it. The residual (+14.30% / +16.06%) is four to five times the
full-ranking noise floor, so it is real, and what remains on the table is architecture and batch
size. The suspect named in earlier versions of this README was the right one, but it was never the
only one.

Two things worth taking from this beyond the attribution. **Sampled HR@10 cannot see this at all** —
the two objectives tie on it (−0.38%, inside noise) while diverging by 22.54% on full ranking.
A model selected on sampled HR@10 would call these interchangeable. And **CE converges far faster**:
it reaches BCE's best-over-200-epochs validation NDCG@10 at **epoch 36**, a 5.6x saving, at 16.9
s/epoch against 6.9 (the full-catalog softmax is ~2.4x more expensive per epoch, so the saving is
real but roughly 2.3x rather than 5.6x in wall-clock).

Caveat on budget: both arms are still improving at epoch 200 (best validation at epoch 195 for BCE,
198 for CE), so this is a matched-budget comparison, not a converged one. Given CE's faster
trajectory, a longer budget would if anything favor BCE by letting it catch up — the measured gap
is the conservative direction.

Reproduce: `uv run python -m src.train --config configs/ablation/sasrec_ml1m_loss_ce.yaml`
(~56 min on an M-series GPU).

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
An earlier version of this README concluded from those four numbers that the binding constraint is
the first code. The next section measures what that constraint is actually worth, and the
conclusion does not survive.

### What the first code is worth, and why fixing it would not be enough

Per-level accuracy is a statement about logits. This is the same question asked in retrieval terms:
hand the model the target's true first *d* codes for free, restrict scoring to the items sharing
that prefix, and let the model's own scores rank what is left.

| oracle depth | median candidates | HR@10 | NDCG@10 |
|---|---|---|---|
| 0 (as it runs) | 12,096 | 0.0250 | 0.0131 |
| 1 | 51 | **0.3119** | 0.1627 |
| 2 | 2 | 0.9879 | 0.7765 |
| 3 | 1 | 1.0000 | 0.9457 |

Not comparable to SASRec's 0.0594 — SASRec gets no oracle. Read it as a decomposition of where the
probability mass goes wrong. Depth 0 reproduces the reported GenRec row exactly (0.0250) and the
level-1 top-1 rate below reproduces the teacher-forced 9.8%, from an independent code path.

The first code is expensive: handing it over multiplies HR@10 by **12.5x**. But it is not the
binding constraint, because handing it over does not rescue the model — **with the right region and
a median of 51 candidates left, 69% of targets still miss the top 10.** Retrieval only becomes
reliable at depth 2, where the median candidate set is 2.

Splitting users by how well the model placed level 1 on its own, and then measuring **unaided**
HR@10, separates "cannot find the region" from "finds it and cannot rank inside it":

| the model's level-1 code | users | unaided HR@10 |
|---|---|---|
| top-1 correct | 2,182 | 0.0903 |
| in its top-10 | 5,291 | 0.0556 |
| in its top-64 | 8,154 | 0.0083 |
| outside top-64 | 6,736 | 0.0001 |

**Even when the model's own first choice of level-1 code is correct, unaided HR@10 is 0.0903** —
so the residual loss sits inside the region, not in reaching it. And the model is not lost:
the true level-1 code has a median rank of **26 of 256** against 128 for chance, and is in the
model's top-25 for 49.1% of users. It localizes the neighbourhood well and rarely commits to it.

Both halves are weak, which is consistent with the diversity result above and against the tidier
story: a better first-code predictor — a wider level-1 codebook, a two-stage decoder, more beam at
level 1 — would move 0.0250 toward 0.3119 at best, and 0.3119 is still a model that misses two
targets in three with fifty candidates in front of it.

Reproduce: `uv run python -m scripts.first_code_ceiling` (~55 min; one exhaustive pass serves every
depth). Full report: [`results/tables/first_code_ceiling.md`](results/tables/first_code_ceiling.md).

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

# Loss-only ablation: full-catalog softmax instead of BCE-with-one-negative (~56 min)
uv run python -m src.train --config configs/ablation/sasrec_ml1m_loss_ce.yaml

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
uv run python -m scripts.first_code_ceiling          # oracle-prefix ladder + level-1 localization
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

- **40% of the full-ranking divergence between the two SASRecs is still unattributed.** The
  loss-only ablation accounts for 56% of the HR@10 gap and 60% of the NDCG@10 gap; the residual
  (+14.30% / +16.06%) is four to five times the noise floor and therefore real. Width, head count,
  inner size and update granularity all remain uncontrolled, and no ablation separates them. Configs
  for the first three arms are written (`configs/ablation/sasrec_ml1m_ce_{batch19,width64,heads2}`
  .yaml), each changing one field against the CE run; none has been trained.
- **The loss ablation is matched-budget, not converged.** Both arms are still improving at epoch
  200. CE's advantage is measured where BCE has not finished training, which understates BCE — the
  conservative direction for the claim being made, but not a converged comparison.
- **The loss ablation is one seed, and CE ran at a learning rate tuned for BCE.** Both arms share
  seed 42 and lr 0.001. CE winning at a rate it was never tuned for makes the result a lower bound
  rather than an artifact, but no lr sweep or second seed was run.
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
