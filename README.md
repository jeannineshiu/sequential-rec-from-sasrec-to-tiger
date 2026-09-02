# From SASRec to TIGER

**A verified SASRec reproduction and a semantic-ID generative recommender on one backbone, one
evaluation harness, and one set of frozen negatives — with every reported margin judged against a
measured seed-noise floor.**

[![CI](https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger/actions/workflows/ci.yml/badge.svg)](https://github.com/jeannineshiu/sequential-rec-from-sasrec-to-tiger/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Datasets](https://img.shields.io/badge/datasets-ML--1M%20%7C%20Amazon%20Beauty-lightgrey)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Summary

Two deliverables on one codebase: a **verified SASRec reproduction** (ML-1M, Amazon Beauty), and a
**generative recommender on the same backbone**, which emits items as sequences of quantized
semantic tokens instead of looking them up in an embedding table. Six results, in the order they
would change what someone does.

**1. A framework's default hyperparameter outweighed the architecture it was used to demonstrate.**
RecBole's SASRec and BERT4Rec configs are identical on every architectural default except dropout
(0.5 vs 0.2). Aligning that single line moves the SASRec–BERT4Rec comparison by +3.71% HR@10 /
+6.33% NDCG@10 — larger than the margin the comparison was meant to explain, and enough to flip the
winner three ways. This puts the *premise* of the BERT4Rec controversy in question rather than
settling it: the budget claim at its centre is untested here. → [§1.2](#12-the-dropout-default)

**2. Agreement under the sampled protocol is not agreement.** Two SASRec implementations matching to
+0.61% on sampled HR@10 diverge by **+40% HR@10 / +53% NDCG@10** under full-catalog ranking. Any
model selection done on the sampled protocol alone is selecting on a metric that does not preserve
ordering. → [§2.1](#21-sampled-versus-full-catalog-ranking)

**3. Most of that divergence is the training objective, and the sampled protocol is blind to it.**
A loss-only ablation — full-catalog softmax instead of BCE against one sampled negative, nothing
else touched — recovers **+22.54% full HR@10 / +32.29% full NDCG@10**, accounting for 56% and 60%
of the cross-framework gap. The same change measures **−0.38% on sampled HR@10, inside seed noise**.
→ [§2.2](#22-training-objective)

**4. Semantic IDs buy compression and reach, not accuracy.** On Beauty the generative model reaches
71% of SASRec's sampled HR@10 on **13.7% of the parameters**, and retrieves items an atomic
embedding table structurally cannot: 7.25% HR@10 on never-seen items against SASRec's 0.00%
(Fisher exact, one-sided *p* = 0.0008). But that reach requires a popularity debiasing that costs
more than half of the model's own accuracy, and on ML-1M the accuracy verdict repeats while the
compression does not (51.8%) — the saving scales with the catalog, not with the method.
→ [§3.1](#31-accuracy-compression-and-parameters--amazon-beauty), [§3.6](#36-the-dense-regime--ml-1m)

**5. Swapping item representations silently swaps scoring rules.** A generative model ranks by
`P(item | history)`, which carries a popularity prior; a dot product does not. Constrained beam
search compounds it: beam-20 reports HR@10 0.0407 where exhaustive scoring gives 0.0240, because
the mean true rank of a beam-reported hit is 167. Both effects arrived with the architecture,
uninvited, in the same way the dropout default arrived with the framework.
→ [§3.3](#33-recommendation-diversity-and-the-popularity-prior), [§3.5](#35-beam-search-as-a-ranker)

**6. The generative model's deficit is not localized to one repairable stage.** Handing it the
target's true first semantic code multiplies HR@10 by 12.4× — and still leaves **69% of targets
outside the top 10 with a median of 51 candidates remaining**. Its level-1 prediction is not lost
either (median rank 26 of 256 against 128 for chance). No better decoder recovers the rest.
→ [§3.4](#34-oracle-prefix-decomposition)

Every margin above is judged against a seed-noise floor measured on the configurations it compares,
ten configurations deep, and negative and mixed results are reported as they came out — including
the one where the generative model is the noisiest thing here, on the protocol that carries the
argument. Every table on this page is asserted in CI against the run that produced it, so a figure
that drifts from its source fails the build instead of waiting to be noticed by a reader.

---

## Scope

This repo tests *semantic IDs as an item representation*, under a single-variable ablation. **It is
not a TIGER reproduction.** TIGER pairs semantic IDs with a T5 encoder–decoder, an RQ-VAE
quantizer, and a user token; none of those are here. The generative model runs on SASRec's own
transformer stack, and that is the point: swapping the backbone *and* the item representation at
once would produce exactly the kind of comparison this repo exists to document — one where the
measured effect belongs to something nobody chose to test. Holding the backbone fixed is what makes
"atomic vs. semantic ID" a clean single variable.

RQ-VAE was skipped on measured evidence rather than on budget: residual K-Means produces **zero dead
codes on both datasets** ([§4.3](#43-semantic-id-quality)), so the codebook-collapse failure mode
RQ-VAE exists to fix does not arise here. That is a finding, not a shortcut.

The title names the direction of travel; the experiments are on the item representation.

---

## Contents

- [Repository layout](#repository-layout)
- [Data and evaluation protocol](#data-and-evaluation-protocol)
- [Results](#results)
  - [1. SASRec reproduction and the cross-framework gap](#1-sasrec-reproduction-and-the-cross-framework-gap)
    — [1.1 Reproduction against published numbers](#11-reproduction-against-published-numbers)
    · [1.2 The dropout default](#12-the-dropout-default)
  - [2. What the sampled protocol hides](#2-what-the-sampled-protocol-hides)
    — [2.1 Sampled versus full-catalog ranking](#21-sampled-versus-full-catalog-ranking)
    · [2.2 Training objective](#22-training-objective)
    · [2.3 Architecture: width, update granularity, head count](#23-architecture-width-update-granularity-head-count)
  - [3. Semantic IDs versus atomic IDs](#3-semantic-ids-versus-atomic-ids)
    — [3.1 Accuracy, compression, and parameters — Amazon Beauty](#31-accuracy-compression-and-parameters--amazon-beauty)
    · [3.2 Cold start by item frequency](#32-cold-start-by-item-frequency)
    · [3.3 Recommendation diversity and the popularity prior](#33-recommendation-diversity-and-the-popularity-prior)
    · [3.4 Oracle-prefix decomposition](#34-oracle-prefix-decomposition)
    · [3.5 Beam search as a ranker](#35-beam-search-as-a-ranker)
    · [3.6 The dense regime — ML-1M](#36-the-dense-regime--ml-1m)
  - [4. Supporting measurements](#4-supporting-measurements)
    — [4.1 Hyperparameter ablations](#41-hyperparameter-ablations)
    · [4.2 Seed-noise floors](#42-seed-noise-floors)
    · [4.3 Semantic ID quality](#43-semantic-id-quality)
- [Serving demo](#serving-demo)
- [Reproduce](#reproduce)
- [Limitations and open questions](#limitations-and-open-questions)
- [References](#references) · [Supporting documents](#supporting-documents)
- [License and dataset terms](#license-and-dataset-terms)

This page reports current state. The dated trail of how it got there — including every result later
corrected — is [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md).

---

## Repository layout

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

Experiment tracking is MLflow (`sqlite:///mlflow.db`, experiment `sequential-rec`). Timings quoted
throughout are for an Apple M-series GPU (MPS) unless stated otherwise.

---

## Data and evaluation protocol

| Dataset | Users | Items | Preprocessing |
|---|---|---|---|
| MovieLens-1M | 6,040 | 3,416 | 5-core filtering, leave-one-out split |
| Amazon Beauty | 22,363 | 12,101 | 5-core filtering, leave-one-out split |

Neither dataset is redistributed here; `src.data.download` fetches both from their original hosts,
and `data/` is gitignored. Both carry their own terms — see
[License and dataset terms](#license-and-dataset-terms).

Every result below follows the same three rules; deviations are marked at the point of use.

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

### 1. SASRec reproduction and the cross-framework gap

#### 1.1 Reproduction against published numbers

*ML-1M, sampled protocol, test, k=10.*

| Model | HR@10 | NDCG@10 | Note |
|---|---|---|---|
| Popularity | 0.4363 | 0.2401 | floor |
| BPR-MF | 0.5745 | 0.3357 | floor |
| SASRec — Kang & McAuley (2018) | 0.8245 | 0.5905 | target |
| **SASRec (this repo)** | **0.8190** | **0.5948** | inside the accepted band (0.80–0.83 / 0.57–0.60) |
| SASRec (RecBole, dropout 0.5 — its default) | 0.7768 | 0.5702 | measures the dropout default, not the implementation |
| SASRec (RecBole, dropout 0.2) | 0.8056 | **0.6063** | dropout-only rerun |
| SASRec (RecBole, dropout 0.2, rescored on frozen negatives) | 0.8240 | 0.6389 | +0.61% HR@10 vs this repo; +7.40% NDCG@10 |
| BERT4Rec (RecBole) | 0.8031 | 0.6036 | see below |

All RecBole rows are 200 epochs.

#### 1.2 The dropout default

**One hyperparameter default decides the SASRec–BERT4Rec comparison, and the winner flips three
ways depending on which SASRec is used.** RecBole's two configs are identical on every
architectural default except dropout; the rows below are protocol- and budget-matched, so the only
thing separating them is that one line.

| Comparison (protocol- and budget-matched) | HR@10 | NDCG@10 | Winner |
|---|---|---|---|
| BERT4Rec vs RecBole SASRec (**dropout 0.5**, default) | +3.39% | +5.86% | BERT4Rec, on both |
| BERT4Rec vs this repo's SASRec | −1.95% | +1.47% | tie |
| BERT4Rec vs RecBole SASRec (**dropout 0.2**) | −0.31% | −0.45% | SASRec, on both |
| *effect of the dropout default alone* | *+3.71%* | *+6.33%* | — |

The residual SASRec-vs-BERT4Rec margin (+0.31% / +0.45%) should be read as a tie. Against the
dropout-0.2 configuration's own measured floors — 0.67% on HR@10 and 0.38% on NDCG@10, each with one
side borrowed from a single-seed BERT4Rec — it is inside the floor on HR@10 and 1.2× it on NDCG@10.
The dropout effect clears those same floors by 5.5× and 17×.

This repo walked into the failure mode by matching the *evaluation* protocol across models with
care and never checking that the *model* hyperparameters were comparable — which is precisely what
the BERT4Rec reproducibility literature describes. Claim-by-claim analysis:
[`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md).

**What this does and does not settle.** The BERT4Rec reproducibility literature's central claim is
about *training budget* — that BERT4Rec's reported wins need far more training than the original
comparisons gave it. **This repo does not test that claim, and deliberately will not.** Only the
1× (200-epoch) point ran; 4× and 10× were cut on cost, and
the limitations section below carries the full costing and reasoning. There is no budget curve here
and nothing on this page adjudicates the controversy on its own terms. What the table
above establishes is something upstream of it: at a *fixed* budget, the SASRec–BERT4Rec comparison
is decided by a hyperparameter default neither paper is about, and the winner flips three ways
depending on which SASRec you pick. That makes the controversy's premise — that these two models
were ever compared like for like — the thing in question, which is a different and, taken on its
own, arguably more useful finding. Read it that way, not as a verdict.

### 2. What the sampled protocol hides

#### 2.1 Sampled versus full-catalog ranking

**Two implementations that agree to +0.61% on the sampled protocol diverge by +40% / +53% under
full-catalog ranking.**

*ML-1M, test, k=10.*

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0369 | 0.0180 |
| BPR-MF | 0.0671 | 0.0333 |
| SASRec (this repo) | 0.2475 | 0.1322 |
| SASRec (RecBole, dropout 0.2, rescored) | **0.3467** | **0.2029** |

Both rows are seed 42, so the comparison is like for like. RecBole's row has since been run at three
seeds: full HR@10 comes out at 0.3507 mean (0.3467 / 0.3474 / 0.3581), and **seed 42 is the lowest
of the three**, so the gap below is if anything understated.

+40% / +53% for a pair that agrees to +0.61% on sampled HR@10. The divergence grows monotonically
with how much the metric cares about *where* in the ranking the target lands, which points at the
training objective — full-catalog cross-entropy (RecBole) vs BCE against one sampled negative (this
repo, per the paper). The next section tests that directly.

![sampled versus full-catalog ranking](results/figures/sampled_vs_full.png)

#### 2.2 Training objective

**The loss function alone reproduces the whole shape of the cross-framework gap, and the sampled
protocol cannot see it.**

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

It does not account for all of it. The residual is +14.30% / +16.06% seed-42-to-seed-42, and
**+15.86% / +17.15%** on three-seed means now that RecBole has seeds of its own. Either way it
clears the floor for this particular comparison — 3.84%, built from RecBole's measured 1.83%
full-ranking spread and CE's 0.58% rather than from one blanket number — so it is real, and what
remains on the table is architecture and batch size. Width was the right suspect, but it was never
the only one. The next section tests the remaining config-visible suspects one at a time, three seeds
each: width is real and worth about a sixth of the residual, batch size is null, and head count is
too noisy an arm for three seeds to settle.

Two things worth taking from this beyond the attribution. **Sampled HR@10 cannot see this at all** —
the two objectives tie on it (−0.38%, inside noise) while diverging by 22.54% on full ranking.
A model selected on sampled HR@10 would call these interchangeable. And **CE converges far faster**:
it reaches BCE's best-over-200-epochs validation NDCG@10 at **epoch 36**, a 5.6× saving, at 16.9
s/epoch against 6.9 (the full-catalog softmax is ~2.4× more expensive per epoch, so the saving is
real but roughly 2.3× rather than 5.6× in wall-clock).

Caveat on budget: both arms are still improving at epoch 200 (best validation at epoch 195 for BCE,
198 for CE), so this is a matched-budget comparison, not a converged one. Given CE's faster
trajectory, a longer budget would if anything favor BCE by letting it catch up — the measured gap
is the conservative direction.

Reproduce: `uv run python -m src.train --config configs/ablation/sasrec_ml1m_loss_ce.yaml`
(~56 min on an M-series GPU).

#### 2.3 Architecture: width, update granularity, head count

**Width is a real effect and a small one; update granularity is a measured null; head count is
underpowered and left undecided.**

The objective left +14.30% HR@10 / +16.06% NDCG@10 on the table against RecBole. The named suspects
were width (64 vs 50), head count (2 vs 1) and update granularity. Each gets one arm, single-variable
on top of CE, against the same control (`ablation_ml1m_loss_ce`) rather than against the BCE baseline.

On granularity the arm is `batch_size: 19`, not 2048. Matching RecBole means matching *target
positions per optimizer step*, and by that measure RecBole's step is 6.6× smaller than this repo's,
not 16× larger (the note above works through the counting). 2048/107.2 = 19.1, so 19 is the number
that lands this repo on RecBole's granularity; copying 2048 across would move away from it and OOM
besides.

| ML-1M, test, k=10 | CE (control) | batch19 | width64 | heads2 | RecBole |
|---|---|---|---|---|---|
| sampled HR@10 | 0.8159 | 0.8166 | 0.8199 | 0.8169 | 0.8240 |
| sampled NDCG@10 | 0.6133 | 0.6114 | 0.6203 | 0.6140 | 0.6389 |
| full HR@10 | 0.3033 | 0.3023 | 0.3134 | 0.3028 | 0.3467 |
| full NDCG@10 | 0.1749 | 0.1741 | 0.1794 | 0.1761 | 0.2029 |
| epochs trained | 200 | 116 | 146 | 200 | 200 |

The RecBole column is seed 42; at three seeds its full-ranking means are 0.3507 / 0.2037, so the
share-of-residual figures below are computed against a target that is itself ±1.83% per run.

One seed per arm is not enough to read these deltas, and neither is a borrowed floor. Judged against
the repo's blanket 3.37% full-ranking floor, every delta above reads as noise — but that floor is
2·√2·σ measured on five seeds of the *BCE baseline*, and it does not describe CE runs. Every arm and
the control therefore have three seeds of their own:

| full ranking, 3 seeds each | CE control | batch19 | width64 | heads2 |
|---|---|---|---|---|
| HR@10 mean | 0.3027 | 0.3038 | 0.3109 | 0.3061 |
| HR@10 rel. std | **0.19%** | 0.71% | 0.92% | **1.32%** |
| NDCG@10 mean | 0.1739 | 0.1739 | 0.1787 | 0.1756 |
| NDCG@10 rel. std | 0.58% | 0.58% | 0.61% | 1.23% |

The CE control's seed-to-seed spread on full HR@10 is **0.19%, not 1.19%** — CE is roughly six times
more reproducible there than BCE is. The blanket floor was too wide by enough to hide an effect.

But read the row across, not just the control column: the spread is **a property of the
configuration, not a constant of the repo**. On full HR@10 it runs 0.19% → 0.71% → 0.92% → 1.32%
across four configurations that differ by one field each. That is the deeper reason a blanket floor
cannot work, and it is stronger than the original "BCE is six times looser than CE" — there is no
single number to substitute, because each arm has to carry its own.

| width64 vs CE control, 3 seeds each | Δ | 95% CI | p (Welch) |
|---|---|---|---|
| sampled HR@10 | +0.22% | [−0.24%, +0.68%] | 0.245 |
| sampled NDCG@10 | +0.82% | [+0.31%, +1.32%] | 0.017 |
| full HR@10 | **+2.72%** | [+0.48%, +4.96%] | 0.034 |
| full NDCG@10 | **+2.74%** | [+1.37%, +4.11%] | 0.005 |

**Width is a real effect and a small one.** At three seeds it clears significance on both
full-ranking metrics, and the point estimate drops from the single-seed +3.33% to +2.72% — the one
seed was on the lucky side. Against the residual it buys **18.7% of the full HR@10 gap and 16.4% of
the NDCG@10 gap**: real, and nowhere near enough. Note also that sampled HR@10 cannot see it
(+0.22%, p=0.25) while full HR@10 can — the same blindness the loss ablation ran into, reappearing
for a change of a different kind.

| batch19 vs CE control, 3 seeds each | Δ | 95% CI | p (Welch) |
|---|---|---|---|
| sampled HR@10 | −0.18% | [−0.77%, +0.41%] | 0.431 |
| sampled NDCG@10 | −0.13% | [−0.89%, +0.62%] | 0.551 |
| full HR@10 | +0.36% | [−1.27%, +2.00%] | 0.474 |
| full NDCG@10 | +0.02% | [−1.30%, +1.33%] | 0.975 |

**Update granularity is null.** All four intervals straddle zero and no p is below 0.43. Matching
RecBole's target-positions-per-optimizer-step changes nothing measurable here, which is the cleanest
negative in this section: the suspect was specific, the arm was built to isolate it, and it is not
the answer. Note this is a null *at this granularity ratio*, not a demonstration that batch size is
irrelevant in general — 19 was chosen to match RecBole, and the sweep between 19 and 128 is untested.

| heads2 vs CE control, 3 seeds each | Δ | 95% CI | p (Welch) |
|---|---|---|---|
| sampled HR@10 | −0.09% | [−0.54%, +0.35%] | 0.567 |
| sampled NDCG@10 | +0.06% | [−1.16%, +1.28%] | 0.854 |
| full HR@10 | +1.11% | [−2.12%, +4.34%] | 0.285 |
| full NDCG@10 | +0.99% | [−1.61%, +3.59%] | 0.304 |

**Head count stays unresolved, and this time the reason is measured.** Both full-ranking point
estimates are positive and about 40% the size of width64's, but neither is significant, because the
arm's own spread is the widest of the four configurations — 1.32% on full HR@10, seven times the
control's. The three seeds land at 0.30281 / 0.31060 / 0.30480 against a control spanning
0.30215–0.30331; the arm is simply less reproducible, and one of its seeds also early-stopped at 140
epochs while the other two ran the full 200.

Three seeds cannot settle an effect that size against that spread. Using the measured pooled
standard deviation, detecting +1.11% on full HR@10 at 80% power would take roughly **12 seeds per
arm, and 15 for NDCG@10** — 20-plus GPU-hours to resolve something that, if real, is smaller than
width64's already-small contribution. That is not worth spending, so this arm is left explicitly
undecided.

The distinction matters and is the same one width64 turned on: `batch19` is **null** (measured, tight,
centered on zero), while `heads2` is **unresolved** (measured, wide, centered off zero). Recording the
second as a null would repeat exactly the error that hid width64 — discarding a positive point
estimate because an underpowered test failed to confirm it.

What stays uncontrolled is the FFN inner size (RecBole 256, here tied to hidden_dim) and RecBole's
per-position sequence augmentation, which yields 981,491 training targets per epoch against this
repo's 647,430 — a difference in what an epoch *is*. Neither is a config field to flip, and after
width they are the larger remaining suspects.

Two cautions:

- **Three seeds is a small sample and the Welch df are 2-4.** The CIs are correspondingly wide: full
  HR@10's lower bound is +0.48%, so "real" here means the sign is established, not the magnitude.
- **No arm is RecBole.** RecBole runs 2 heads *at* d=64; width64 changes width at 1 head, heads2
  changes heads at d=50. The combination is untested, so the arms cannot be summed and the 18.7%
  share is not a budget the remaining differences must fit inside.
- **Only one of the three arms is a settled negative.** `batch19` is null; `heads2` is undecided and
  left that way. Reading this section as "width is the only architectural effect" overstates it.

Reproduce: `uv run python -m src.train --config configs/ablation/sasrec_ml1m_ce_{batch19,width64,heads2}.yaml`
(~30-55 min each), and for the seed check add `--seed N --run-name <name>_seedN` (the run name must
be overridden alongside the seed, or each seed overwrites the previous checkpoint). Each seeded
comparison is regenerated by, e.g.,
`uv run python -m scripts.seed_variance --arm-seeds ablation_ml1m_ce_width64 ablation_ml1m_loss_ce`
(substitute `batch19` or `heads2`); the printed table carries the means, per-arm relative standard
deviations, Welch CIs and p-values quoted above.

### 3. Semantic IDs versus atomic IDs

#### 3.1 Accuracy, compression, and parameters — Amazon Beauty

**Semantic IDs cost 57.7% of full-catalog HR@10 and buy a 7.3× parameter reduction. The trade is
real, and it is not a wash.**

*Amazon Beauty, test, k=10.*

Same backbone, same protocol, same frozen negatives. The only variable is whether an item is one
embedding or a sequence of four semantic tokens. Both full-ranking columns score the catalog
exhaustively.

| | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 | parameters |
|---|---|---|---|---|---|
| SASRec (atomic) | **0.5097** | **0.3453** | **0.0594** | **0.0303** | 828,352 |
| GenRec (semantic) | 0.3621 | 0.2235 | 0.0251 | 0.0131 | **113,472** |
| relative | −28.95% | −35.27% | −57.7% | −56.6% | **13.7%** |

Every margin is far outside the seed-noise floor, and that floor is Beauty's own rather than a
proxy: the SASRec row was re-run at seeds 1 and 2 (200 epochs each, ~22 min on the laptop GPU)
specifically so that no Beauty margin is judged against a spread measured on ML-1M.

| Beauty SASRec, 3 seeds | mean | rel. std | min | max | borrowed ML-1M proxy |
|---|---|---|---|---|---|
| sampled HR@10 | 0.5116 | 0.64% | 0.5097 | 0.5154 | 0.34% |
| sampled NDCG@10 | 0.3466 | 0.78% | 0.3448 | 0.3497 | 0.34% |
| full HR@10 | 0.0611 | 2.53% | 0.0594 | 0.0624 | 1.19% |
| full NDCG@10 | 0.0314 | **3.73%** | 0.0303 | 0.0326 | 1.08% |

Beauty SASRec was the noisiest configuration in the repo when it was seeded, and the borrowed floor
was too narrow on every one of the four metrics — 2× on the sampled pair, 2.1× and 3.5× on full
ranking. That is the same direction of error the RecBole seeds found (1.83% against a borrowed
1.19%) and the opposite of the CE family's — a third independent demonstration that a single noise
floor cannot be right for every configuration.

The generative side is seeded too, so neither half of this comparison rests on a single run:

| Beauty GenRec, 3 seeds | mean | rel. std | min | max |
|---|---|---|---|---|
| sampled HR@10 | 0.3616 | 0.76% | 0.3586 | 0.3641 |
| sampled NDCG@10 | 0.2226 | 0.98% | 0.2202 | 0.2243 |
| full HR@10, exhaustive | 0.0236 | **13.57%** | 0.0199 | 0.0258 |
| full NDCG@10, exhaustive | 0.0124 | **11.28%** | 0.0108 | 0.0133 |

**The generative model is by far the noisiest configuration measured here — but only on the protocol
that carries the argument.** Its sampled spread, 0.98%, is ordinary: comparable to Beauty SASRec's
0.78% and to every other family on the page. Its exhaustive full-ranking spread is fourteen times
that, and 3.6× the widest previously measured (Beauty SASRec's 3.73%). The two protocols do not
merely differ in width; they disagree about which run is best. `genrec_beauty_seed1` has the highest
sampled HR@10 of the three and the lowest full HR@10 by 20%, and it trained longest doing it —
early stopping ran 184 epochs against seed 42's 100 and seed 2's 62, a threefold spread in training
length that moved the sampled metrics by under 1%. A 101-negative protocol reports this model as
reproducible to a percent while full ranking on the same three checkpoints reports 13.6%. That is
the same sentence this repo spends its BERT4Rec chapter on, turned on its own model.

The margins survive, less comfortably than a borrowed floor would suggest. Against measured floors
of 2.49% sampled and **28.15%** full — the borrowed pair was 2.20% and 10.55% — the four relatives
clear by 11.6×, 14.2×, **2.05× and 2.01×**. Measuring the generative side rather than borrowing a
spread for it roughly halved the full-ranking headroom. It is still a margin no plausible reading
closes, and it is no longer an assumed one.


The parameter column is what makes this a trade rather than a loss. The comparison **cannot** be
parameter-matched: equalizing would mean crippling SASRec's item table or inflating GenRec's hidden
dimension, and the compression *is* the method under test. It is stated, not controlled away.

Unconstrained greedy decoding produces a valid item 81.8% of the time after training (32.8% after
two epochs), so the Trie constraint is doing real work — without it, nearly one in five top-1
recommendations would not be an item that exists.

#### 3.2 Cold start by item frequency

**The hypothesis fails in the direction it was supposed to win, and survives only in its narrowest
form.**

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
| tail (1–4) | 4,594 | **0.0185** | 0.0026 | 0.0141 |
| torso (5–19) | 9,539 | **0.0427** | 0.0060 | 0.0080 |
| head (20+) | 8,092 | **0.1033** | 0.0608 | 0.0137 |
| overall | 22,363 | **0.0594** | 0.0251 | 0.0117 |

**As trained, the gap widens as items get rarer — the opposite of the prediction.**

**Debiased, the claim survives in its narrowest form.** On never-seen items the generative model
retrieves 7.25% HR@10 where SASRec retrieves 0.00% — 10 hits in 138 against none, Fisher exact
one-sided *p* = 0.0008 (two-sided 0.0016). Undebiased it is 1 hit in 138, *p* = 0.50: the reach
claim rests on the debiased scoring rule, not on the model as trained. Every p-value here is
printed by `scripts/compare_atomic_vs_semantic.py` and written into
[`results/tables/atomic_vs_semantic.md`](results/tables/atomic_vs_semantic.md). On tail items it
does not separate from SASRec (65 hits against 85 in 4,594; two-sided *p* = 0.118, and 0.059
one-sided in SASRec's favour).

Read that 0.059 as one significant figure, not three: the bucket count behind it sits one user from
a top-10 boundary, and it has moved between 0.059 and 0.070 across regenerations of the artifact.
[`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md) records which regenerations moved it and why.

Semantic IDs really do reach items an atomic table structurally cannot. The price is
more than half the overall accuracy, paid on the head where the traffic is.

**The p-value controls the wrong source of variance, and now both are measured.** Fisher exact on
10 hits in 138 asks whether *this* model's hit rate could have come from user sampling. It says
nothing about whether a differently-seeded GenRec produces 10 hits, or 3, or 20 — and at counts
this small the training noise is the interval a reader is more likely to care about. Two more seeds
put a number on it:

| unseen bucket, 138 users | seed 42 | seed 1 | seed 2 |
|---|---|---|---|
| debiased α=1 hits | **10** | 7 | 8 |
| one-sided *p* vs SASRec's 0 | 0.0008 | 0.0072 | 0.0035 |
| as trained hits | 1 | 0 | 0 |

The claim holds at every seed — SASRec retrieves nothing in this bucket under any of them, so the
direction is not in doubt — but the published 10 is the top of the range, and the honest statement
of the reach result is 7 to 10 hits in 138, *p* ≤ 0.0072. The undebiased "1 hit in 138" is likewise
1 in the luckiest seed and 0 in the other two; it was already reported as not significant
(*p* = 0.50), and three seeds make it a rounding artifact rather than a hit. Per-seed counts are in
[`results/tables/genrec_seed_spread_amazon-beauty.md`](results/tables/genrec_seed_spread_amazon-beauty.md).

#### 3.3 Recommendation diversity and the popularity prior

**Generation collapses onto a small set of code sequences, and removing the popularity prior does
not undo it.**

| model | distinct items across all top-10s | median train freq | % head | % torso | % tail | % unseen |
|---|---|---|---|---|---|---|
| SASRec (atomic) | **9,221** (76% of catalog) | 22 | 54.2% | 42.1% | 3.7% | 0.0% |
| GenRec (semantic) | **1,749** (14%) | 63 | 74.1% | 21.3% | 4.6% | 0.0% |
| GenRec debiased α=1 | **2,084** (17%) | 5 | 8.1% | 50.3% | 39.8% | 1.8% |

The distinct-item counts are reproducible only to about ±15 (0.8%): they are set statistics over
223,630 slots, and MPS float noise at the top-10 boundary moves them. Neither the size of the
collapse nor any conclusion below turns on that margin. (The debiased row also once carried a
padding-token artifact that inflated its coverage and unseen share; the ranked metrics were never
affected. See [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md).)

The debiased row pins the mechanism down, and it cuts against the tidy explanation. Debiasing
changes *what* is recommended enormously — head share 74.1% → 8.1%, torso 21.3% → 50.3%, tail
4.5% → 39.8%, median recommended item from 63 training appearances to 5. Yet coverage moves only
1,749 → 2,084: a 19% gain on a number that needs 5× to match SASRec.

So the popularity prior is not the whole story. Removing it slides the model along a fixed frontier,
trading head accuracy for tail accuracy — which is why overall HR@10 halves — without making it
more *discriminative*. Learning to emit four codes leaves the model with far less resolution over
the catalog than 12,101 free embeddings have, and that is not a scoring-time artifact a better
decoder repairs.

Per-level code accuracy (teacher-forced on the true prefix) is 9.8% / 17.9% / 22.4% / 86.3%.
Accuracy *rises* with depth as the prefix narrows the choice, and the content-free disambiguation
token is nearly free — so Beauty's 11.78% collision rate is not the bottleneck it appears to be.
It is tempting to read those four numbers as making the first code the binding constraint. The next
section measures what that constraint is actually worth in retrieval terms, and it is not.

#### 3.4 Oracle-prefix decomposition

**The first code is expensive but not binding: fixing it would not rescue the model.**

Per-level accuracy is a statement about logits. This is the same question asked in retrieval terms:
hand the model the target's true first *d* codes for free, restrict scoring to the items sharing
that prefix, and let the model's own scores rank what is left.

| oracle depth | median candidates | HR@10 | NDCG@10 |
|---|---|---|---|
| 0 (as it runs) | 12,096 | 0.0251 | 0.0131 |
| 1 | 51 | **0.3117** | 0.1627 |
| 2 | 2 | 0.9880 | 0.7767 |
| 3 | 1 | 1.0000 | 0.9458 |

Not comparable to SASRec's 0.0594 — SASRec gets no oracle. Read it as a decomposition of where the
probability mass goes wrong. Depth 0 reproduces the reported GenRec row exactly (0.0251) and the
level-1 top-1 rate below reproduces the teacher-forced 9.8%, from an independent code path.

The first code is expensive: handing it over multiplies HR@10 by **12.4×**. But it is not the
binding constraint, because handing it over does not rescue the model — **with the right region and
a median of 51 candidates left, 69% of targets still miss the top 10.** Retrieval only becomes
reliable at depth 2, where the median candidate set is 2.

Splitting users by how well the model placed level 1 on its own, and then measuring **unaided**
HR@10, separates "cannot find the region" from "finds it and cannot rank inside it":

| the model's level-1 code | users | unaided HR@10 |
|---|---|---|
| top-1 correct | 2,182 | 0.0907 |
| in its top-10 | 5,291 | 0.0558 |
| in its top-64 | 8,154 | 0.0083 |
| outside top-64 | 6,736 | 0.0001 |

**Even when the model's own first choice of level-1 code is correct, unaided HR@10 is 0.0907** —
so the residual loss sits inside the region, not in reaching it. And the model is not lost:
the true level-1 code has a median rank of **26 of 256** against 128 for chance, and is in the
model's top-25 for 49.1% of users. It localizes the neighbourhood well and rarely commits to it.

Both halves are weak, which is consistent with the diversity result above and against the tidier
story: a better first-code predictor — a wider level-1 codebook, a two-stage decoder, more beam at
level 1 — would move 0.0251 toward 0.3117 at best, and 0.3117 is still a model that misses two
targets in three with fifty candidates in front of it.

Reproduce: `uv run python -m scripts.first_code_ceiling` (~55 min; one exhaustive pass serves every
depth). Full report: [`results/tables/first_code_ceiling.md`](results/tables/first_code_ceiling.md).

#### 3.5 Beam search as a ranker

**Constrained beam search is not a faithful ranker, and every generative number on this page is
therefore exhaustively scored rather than beam-ranked.**

The case for beam-ranking was that widening the beam 20 → 200 changed nothing, so the approximation
"can only cost the generative side." That reasoning does not hold. On the same 1,500 users,
beam-20 reports HR@10 0.0407 where exhaustive scoring gives 0.0240, and
the **mean true rank of a beam-reported top-10 hit is 167**. Beam pruning discards 236 of 256 first
codes, so the high-scoring items that should have outranked the target never enter the returned
list. The width sweep missed this because widening the beam finds more targets *and* more
competitors simultaneously; the two cancel, and a flat curve reads as convergence. The sweep tested
whether the beam finds the target, never whether its ranking is faithful.

Superseded beam figures: −44.6% overall (vs −57.7% exhaustive), 839 distinct items / 84.7% head
(vs 1,749 / 74.1%). The collapse is real either way, and about half as severe as the beam made it
look.

#### 3.6 The dense regime — ML-1M

Beauty is the regime semantic IDs should suit: 12,101 items, sparse, 11.78% of them sharing a code
prefix with a catalog neighbour. ML-1M is the opposite — 3,416 items, dense, 1.46% collisions, and
98.9% of unconstrained greedy decodes already legal against Beauty's 81.8%. Running the identical
comparison here is the check on whether Beauty's result belongs to semantic IDs or to Beauty.

| | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 | parameters |
|---|---|---|---|---|---|
| SASRec (atomic) | **0.8190** | **0.5948** | **0.2475** | **0.1322** | 212,000 |
| GenRec (semantic) | 0.6260 | 0.3997 | 0.1164 | 0.0607 | **109,800** |
| relative | −23.6% | −32.8% | −53.0% | −54.1% | **51.8%** |

**The accuracy verdict repeats, and it does not depend on sparsity.** −53.0% full HR@10 here
against −57.7% on Beauty: the generative model loses by roughly the same margin in the dense
regime it was supposed to struggle in as in the sparse one it was supposed to suit. That margin was
stated against measured spreads on both sides rather than against a borrowed one:

| ML-1M GenRec, 3 seeds | mean | rel. std | min | max |
|---|---|---|---|---|
| full HR@10, exhaustive | 0.1164 | 3.48% | 0.1124 | 0.1205 |
| full NDCG@10, exhaustive | 0.0601 | 4.87% | 0.0569 | 0.0627 |

The floor that produces is 10.03% against the borrowed 3.37%, and −53.0% still clears it 5.3×. Note
what does *not* transfer: the generative spread here is a third of Beauty's 13.57%, so even GenRec's
own noise cannot be carried between the two datasets. Every generative margin on this page is now
measured on both sides; nothing in the second deliverable is borrowed.

**What does not carry over is the compression.** Beauty's headline is 13.7% of the parameters;
here it is 51.8%, and the reason is worth stating precisely. The token table does shrink as
advertised — 38,650 parameters against SASRec's 170,850, a 4.4× cut. But an item is four tokens,
so the sequence the model attends over is four times longer, and the positional table grows from
10,050 to 40,050 to cover it. That single term eats most of the saving and is over a third of the
generative model's parameters. **Semantic IDs compress a catalog, not a model.** On 3,416 items
there was not much catalog to compress, and the argument that carries the whole method on Beauty
mostly evaporates.

Both full-ranking columns score all 3,416 items for all 6,040 test users
([`results/tables/atomic_vs_semantic_ml-1m.md`](results/tables/atomic_vs_semantic_ml-1m.md)). The
`full_*` columns for `genrec_ml1m` in [`results/tables/master.md`](results/tables/master.md) are
beam-ranked and read 0.1086 / 0.0579 — here beam *costs* 6.7% rather than the 32% it gained on
Beauty, because at 1.46% collisions almost nothing can be credited by prefix and only the pruning
is left. Both numbers superseded an earlier pair produced by an evaluator that scored NaN as a
rank-0 hit; the account is in [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md).

The cold-start buckets carry nothing on this dataset. After 5-core filtering ML-1M's test split has
no unseen items and two tail users, so 5,990 of 6,040 targets are head. The bucketed table is
generated for symmetry and reports what little there is; the cold-start claim rests on Beauty
alone.

### 4. Supporting measurements

#### 4.1 Hyperparameter ablations

*ML-1M, test, k=10, all rows at 100 epochs.*

Deltas against the 100-epoch baseline, marked against the BCE baseline's floor (sampled 0.96%,
full 3.37%). `~` means inside seed noise. Both sides of every delta are that configuration, so the
floor is the right family — but it was measured at 200 epochs, not the 100 these rows share, and no
ablation arm has a second seed of its own.

| Ablation | sampled HR@10 | Δ | full HR@10 | Δ | avg s/epoch |
|---|---|---|---|---|---|
| **Baseline (learnable pos emb, maxlen 200)** | **0.8152** | — | **0.2349** | — | 5.85 |
| positional embedding = none | 0.8066 | −1.06% | 0.2291 | ~ −2.47% | 7.47 |
| positional embedding = sinusoidal | 0.8147 | ~ −0.06% | 0.2182 | −7.12% | 6.91 |
| maxlen = 50 | 0.7858 | −3.61% | 0.2033 | −13.46% | 1.53 |
| maxlen = 100 | 0.8058 | −1.16% | 0.2346 | ~ −0.14% | 2.85 |
| negative sampling = popularity-weighted | 0.7540 | −7.51% | 0.1871 | −20.37% | 6.93 |

Two notes on reading this table. First, it is baselined at a **matched** budget, which matters more
than it sounds: charging each ablation against a baseline trained 100 epochs longer shifts sampled
HR@10 by only +0.47% but full HR@10 by +5.36%, enough to turn pos-emb-none and maxlen-100 into
apparent full-ranking regressions when both are inside noise. Read literally, **maxlen 100 and 200
are indistinguishable on full ranking at this budget**, at less than half the per-epoch cost.

Second, positional embeddings are the interesting row: dropping them costs ~1% sampled and nothing
detectable on full ranking, while *sinusoidal* is the one variant that clearly hurts full ranking
(−7.12%). Learnable-vs-none is nearly a wash; learnable-vs-sinusoidal is not.

#### 4.2 Seed-noise floors

*ML-1M SASRec, 5 seeds (42, 1–4).*

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

**That floor describes this configuration and no other.** Treating it as a constant of the repo
errs in both directions, so every configuration with three or more seeds carries its own:

| configuration | seeds | sampled | full |
|---|---|---|---|
| SASRec, BCE, 200ep (the floor above) | 5 | 0.34% | 1.19% |
| SASRec, CE control | 3 | 0.22% | **0.58%** |
| CE + hidden_dim 64 | 3 | 0.23% | 0.92% |
| CE + batch_size 19 | 3 | 0.33% | 0.71% |
| CE + 2 heads | 3 | 0.50% | **1.32%** |
| RecBole SASRec, dropout 0.2, rescored here | 3 | 0.23% | **1.83%** |
| RecBole SASRec, dropout 0.2, RecBole's own uni100 | 3 | 0.24% | — |
| SASRec on Beauty, BCE | 3 | 0.78% | 3.73% |
| GenRec on Beauty, semantic IDs | 3 | 0.98% | **13.57%** |
| GenRec on ML-1M, semantic IDs | 3 | 2.40% | 4.87% |

Per-run relative standard deviation, worst metric per protocol. Read the full-ranking column: it
spans **0.58% to 13.57%**, a factor of twenty-three across configurations that differ by one field,
one framework, one dataset, or one item representation.

Borrowing the BCE baseline's 1.19% therefore errs both ways. It overstates the CE family's noise by
2×, which is enough to hide `width64`'s real effect, and *understates* RecBole's by 1.5×, which
would wave through a full-ranking margin RecBole's own seeds cannot support. There is no corrected
constant to substitute — only per-configuration spreads.

Two consequences for how margins are judged here:

- **No single floor.** `scripts/seed_variance.py` checks each claimed margin against
  2·√(σ_a² + σ_b²) for the two configurations it actually compares, which collapses to the familiar
  2·√2·σ only when both sides have the same spread. Rows where one side has no seeds of its own
  print as `borrowed`, so it stays visible which verdicts are still proxies.
- **The GenRec spreads are not read from MLflow.** `train_genrec` logs a beam-20 `test_full_*`,
  while every full-ranking margin on this page is exhaustive, so the two GenRec rows take their
  full-ranking spread from `scripts/genrec_seed_spread.py` — the same exhaustive pass that produces
  the tables — and only their sampled pair from MLflow. Using the logged spread would be a proxy of
  exactly the kind this section exists to remove.

`uv run python -m scripts.seed_variance` prints all of it. Three to five seeds estimate σ loosely —
these are sanity floors, not significance tests, and the seeded arms carry Welch tests as well.

#### 4.3 Semantic ID quality

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

All commands assume `uv sync` has been run. Timings are for an Apple M-series GPU (MPS).

### Setup and data

```bash
uv sync
uv run python -m src.data.download --dest data/raw --dataset all
uv run python -m src.data.preprocess --dataset ml-1m  --out-dir data/processed/ml-1m
uv run python -m src.data.preprocess --dataset beauty --out-dir data/processed/beauty
```

### Baselines and SASRec

```bash
uv run python -m src.baselines --data-dir data/processed/ml-1m
uv run python -m src.train --config configs/sasrec_ml1m.yaml
uv run python -m src.train --config configs/sasrec_beauty.yaml
```

### Ablations

```bash
# Loss only: full-catalog softmax instead of BCE-with-one-negative (~56 min)
uv run python -m src.train --config configs/ablation/sasrec_ml1m_loss_ce.yaml

# Architecture arms on top of CE, one field each (~30-55 min apiece)
uv run python -m src.train --config configs/ablation/sasrec_ml1m_ce_batch19.yaml
uv run python -m src.train --config configs/ablation/sasrec_ml1m_ce_width64.yaml
uv run python -m src.train --config configs/ablation/sasrec_ml1m_ce_heads2.yaml
```

### Semantic IDs and the generative model

```bash
# Beauty metadata is an extra ~99MB
uv run python -m src.data.download --dest data/raw --dataset beauty --with-meta
uv run python -m src.semantic_ids.embed     --dataset ml-1m
uv run python -m src.semantic_ids.rq_kmeans --dataset ml-1m
uv run python -m scripts.inspect_semantic_ids --dataset ml-1m   # quality report

uv run python -m src.train_genrec --config configs/genrec_beauty.yaml
uv run python -m src.train_genrec --config configs/genrec_ml1m.yaml    # ~3h20m
```

### Analysis

Each script below scores all 12,101 items for all 22,363 Beauty test users — budget ~55 min apiece.
Add `--beam` to the first for the superseded beam-ranked numbers.

```bash
uv run python -m scripts.compare_atomic_vs_semantic  # bucket table + cold-start figure
uv run python -m scripts.diagnose_genrec             # diversity + per-level accuracy
uv run python -m scripts.debias_decoding             # popularity-debiasing α sweep
uv run python -m scripts.first_code_ceiling          # oracle-prefix ladder + level-1 localization
uv run python -m scripts.seed_variance               # noise floor + margin re-check
uv run python -m scripts.seed_variance --prefix sasrec_beauty   # the same, for Beauty

# The same comparison on ML-1M (~8 min: 3,416 items × 6,040 test users)
uv run python -m scripts.compare_atomic_vs_semantic \
    --sasrec-config configs/sasrec_ml1m.yaml --genrec-config configs/genrec_ml1m.yaml

uv run python -m src.export_results   # rebuild results/tables/master.md + figures
uv run pytest tests/
```

### Optional: seed replication

The per-configuration noise floors in [§4.2](#42-seed-noise-floors) come from these runs. They are
the expensive part of the page and are not needed to reproduce any single result.

```bash
# Beauty SASRec, ~22 min each. Run one at a time so they do not contend for the GPU.
for s in 1 2; do uv run python -m src.train --config configs/sasrec_beauty.yaml \
    --seed $s --run-name sasrec_beauty_seed$s; done

# GenRec, ~1h per Beauty run and ~3h20m per ML-1M run.
for s in 1 2; do uv run python -m src.train_genrec --config configs/genrec_beauty.yaml \
    --seed $s --run-name genrec_beauty_seed$s; done

# The exhaustive spread those seeds feed: ~55 min per Beauty seed, ~8 min per ML-1M seed.
uv run python -m scripts.genrec_seed_spread   # Beauty; --limit N to sanity-check on a subset
uv run python -m scripts.genrec_seed_spread --genrec-config configs/genrec_ml1m.yaml \
    --run-names genrec_ml1m genrec_ml1m_seed1 genrec_ml1m_seed2
```

The run name must be overridden alongside the seed, or each seed overwrites the previous checkpoint.
Start seed runs on an idle machine: under contention this repo has measured training epochs of
3,000-4,000s against a 16s median, on an identical config and seed.

### How the numbers stay honest

`results/tables/master.md` is generated from MLflow rather than transcribed, and every table on
this page is asserted against the thing that produced it by
`tests/test_readme_matches_results.py` — measured cells against
`results/tables/` and `mlflow.db`, derived cells recomputed from those at full precision — so a
figure that drifts from its source fails CI instead of waiting to be noticed by a reader. Long GPU
sweeps run detached on remote sandboxes via `scripts/daytona_*.py`, with result recovery for
interrupted runs.

---

## Limitations and open questions

Kept current, on the principle that a mixed result reported is worth more than a clean result
implied.

- **Roughly a third of the full-ranking divergence between the two SASRecs is still unattributed.**
  The loss ablation accounts for 56% of the HR@10 gap and 60% of the NDCG@10 gap. Width, measured at
  three seeds per arm, accounts for a further 18.7% of the HR@10 residual and 16.4% of the NDCG@10
  residual. What remains has no measured explanation. FFN inner size (RecBole 256, here tied to
  hidden_dim) and RecBole's per-position augmentation (981,491 targets per epoch against 647,430) are
  the remaining uncontrolled differences; neither is a config field to flip, since both would change
  the model or the data pipeline rather than a setting.
- **heads2 is undecided, not null, and will stay that way.** Its full-ranking deltas are positive
  (+1.11% HR@10, +0.99% NDCG@10) but not significant (p=0.29, 0.30), because the arm's own seed spread
  is the widest measured here — 1.32% on full HR@10 against the control's 0.19%. From the measured
  pooled sd, 80% power at that effect size needs ~12 seeds per arm (15 for NDCG@10), ~20 GPU-hours to
  resolve an effect smaller than width's. Not spent, so the arm is recorded as unresolved.
- **The seeded comparison is three seeds, and the arms are not budget-matched.** Welch df are 2-4 and
  full HR@10's 95% CI runs [+0.48%, +4.96%], so width's sign is established and its magnitude is not.
  Early stopping also gave the arms 82-200 epochs against the control's 200.
- **The blanket noise floor is retired, and what replaced it is still partial.** The 0.96% / 3.37%
  figures come from five seeds of one configuration, and applying them elsewhere was wrong in both
  directions: too wide for the CE family (hiding width64's real effect) and too narrow for RecBole's
  full-ranking numbers (1.83% per run against the borrowed 1.19%). Ten configurations now carry
  their own measured spread and every margin is judged against the two it actually compares. What is
  still unmeasured: RecBole's dropout-0.5 and BERT4Rec runs, and the ablation arms at their own
  100-epoch budget. Beauty's SASRec and GenRec on both datasets have three seeds each. Those
  remaining rows print as `borrowed` in `seed_variance` rather than being quietly proxied, but
  borrowed is what they remain.
- **The loss ablation is matched-budget, not converged.** Both arms are still improving at epoch
  200. CE's advantage is measured where BCE has not finished training, which understates BCE — the
  conservative direction for the claim being made, but not a converged comparison.
- **The loss ablation is one seed, and CE ran at a learning rate tuned for BCE.** Both arms share
  seed 42 and lr 0.001. CE winning at a rate it was never tuned for makes the result a lower bound
  rather than an artifact, but no lr sweep or second seed was run.
- **No training-budget curve, and there deliberately will not be one.** Only the 1× (200-epoch)
  point exists. Measured epoch times put a 4× point for the model pair at ~42 GPU-hours and the full
  1×/4×/10× trajectory at ~105; running it to this repo's own seeding standard — three seeds of a
  configuration whose spread has never been measured — is ~315 GPU-hours, and at RecBole's defaults
  it would measure training budget crossed with the dropout default shown above to be the dominant
  term. It was cut because it would reproduce an existing study on someone else's question and
  would not change a claim on this page. The scaling claim at the heart of the BERT4Rec
  controversy is therefore untested here, and stays untested. The one piece of free evidence points
  the interesting way: the dropout-0.2 SASRec was still improving at epoch 189 while the
  default-dropout run had plateaued — the configuration that had not finished training is the one
  that wins. Full reasoning in [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md) §6.4.
- **BERT4Rec comparisons rest on the sampled protocol.** Only one RecBole run has full-ranking
  numbers, and for the one pair where both exist they disagree by 40–53%. Nothing here establishes
  that the SASRec-vs-BERT4Rec ordering survives full-catalog evaluation. The cross-framework
  comparison also varies loss, batch size and architecture at once. Full accounting in
  [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md) §4.
- **RecBole's dropout-0.2 run now has three seeds; every other RecBole number is still one.** The
  dropout-0.2 SASRec was re-run at seeds 1 and 2, which is what put a measured floor under the
  headline margin it carries (+3.71% / +6.33% on uni100, against that configuration's own floors —
  0.67% on HR@10 and 0.38% on NDCG@10 — 5.5× and 17× clear). The dropout-0.5 SASRec and both
  BERT4Rec runs remain single-seed, so any margin involving them is still judged against a partly
  borrowed floor. RecBole's full-ranking
  spread was the widest measured here until Beauty's SASRec was seeded (3.73%), and that in turn
  until Beauty's GenRec was (13.57%).
- **Ablations are a statement about 100-epoch training.** Deliberate compute saving; both sides of
  every delta share the budget, but the popularity-negatives result in particular may be a
  convergence-speed effect rather than a final ranking.
- **Beauty SASRec sits above the accepted reproduction band on all three seeds** (0.4654–0.5054;
  seeds give 0.5097 / 0.5098 / 0.5154, i.e. +0.43pp, +0.44pp and +1.00pp, mean +0.62pp). Reported
  as-is rather than tuned until it lands inside. Seeding moved this the unflattering way: seed 42
  was the *lowest* of the three, so the single-seed disclosure understated the overshoot — the same
  thing the RecBole seeds did to the residual.
- **The atomic-vs-semantic comparison is not parameter-matched** and cannot be, by construction.
- **The compression claim is a Beauty result, not a method result.** 13.7% of the parameters on a
  12,101-item catalog becomes 51.8% on ML-1M's 3,416, because the positional table grows with the
  token sequence while the item table shrinks with the catalog. Two catalog sizes is not a curve;
  where the trade turns favourable is unmeasured.
- **GenRec's reported numbers are seed 42 of three, and the seed changes more than the metrics
  suggest.** On Beauty the exhaustive full-ranking spread is 13.57% per run, fourteen times the
  sampled spread of the same three checkpoints, and the two protocols disagree about which seed is
  best. Early stopping ran 62, 100 and 184 epochs on Beauty; on ML-1M two of the three seeds hit
  the 200-epoch budget without converging and the third stopped at 136. The reported row is one draw,
  the floors in `seed_variance` are three-seed measurements, and the cold-start count is 7–10 rather
  than the 10 the headline quotes.
- **The serving demo's GenRec list is beam-ranked** and therefore more optimistic than every table
  on this page.

---

## References

- Kang & McAuley, *Self-Attentive Sequential Recommendation* (ICDM 2018) — SASRec
- Sun et al., *BERT4Rec* (CIKM 2019); Petrov & Macdonald, *A Systematic Review and Replicability
  Study of BERT4Rec* (RecSys 2022)
- Rajput et al., *Recommender Systems with Generative Retrieval* (NeurIPS 2023) — TIGER
- Krichene & Rendle, *On Sampled Metrics for Item Recommendation* (KDD 2020)

### Supporting documents

| Document | What it carries |
|---|---|
| [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md) | The dated debugging and decision trail, including every result later corrected and why |
| [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md) | Claim-by-claim analysis of what this repo's data does and does not establish |
| [`docs/original-plan.md`](docs/original-plan.md), [`docs/execution-plan.md`](docs/execution-plan.md) | The pre-registered plan and acceptance criteria, written before any experiment ran and kept unedited — several predictions did not survive the data |
| [`results/tables/master.md`](results/tables/master.md) | Script-generated master table, never hand-edited |

---

## License and dataset terms

**The code in this repository is MIT-licensed** ([`LICENSE`](LICENSE)). That covers everything under
`src/`, `scripts/`, `serving/`, `tests/`, and `configs/`, plus the generated reports in
`results/tables/`.

**It does not cover the datasets, which are not redistributed here.** `data/raw/` and
`data/processed/` are gitignored; `src.data.download` fetches each dataset from its original host at
setup time, and each arrives under its own terms.

| Dataset | Source | Terms that bind a user of this repo |
|---|---|---|
| MovieLens-1M | [GroupLens](https://files.grouplens.org/datasets/movielens/ml-1m.zip), University of Minnesota | Research use permitted. **No redistribution without separate permission**, and **no commercial or revenue-bearing use** without permission from a GroupLens faculty member. No implied endorsement by UMN or GroupLens. Publications must cite the dataset paper. |
| Amazon Reviews (Beauty) | [SNAP / UCSD](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/), Julian McAuley | Provided for research use; the dataset page asks that publications cite the two papers below. |

Dataset citations:

- Harper & Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015.
  [doi:10.1145/2827872](https://doi.org/10.1145/2827872)
- McAuley, Targett, Shi & van den Hengel, *Image-based Recommendations on Styles and Substitutes*,
  SIGIR 2015.
- He & McAuley, *Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class
  Collaborative Filtering*, WWW 2016.

The MovieLens non-commercial condition is the one most likely to matter in practice: the MIT licence
on this code does **not** grant any right to use ML-1M commercially, and running these scripts
against it does not create one.
