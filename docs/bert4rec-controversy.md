# The BERT4Rec Reproducibility Controversy — What This Repo's Data Does and Doesn't Show

**Status: partial.** One training-budget point (1x = 200 epochs) landed; the 4x/10x points
and the same-framework RecBole-SASRec cross-check did not. This document reports what the
single point supports, and is explicit about which claims it cannot reach. It is written to
be read alongside [`REPRODUCTION_LOG.md`](../REPRODUCTION_LOG.md), which carries the raw
debugging trail.

---

## 1. The controversy, and the claims under test

Sun et al. (2019) introduced BERT4Rec and reported it beating SASRec (Kang & McAuley, 2018)
by a wide margin. Petrov & Macdonald (2022), *A Systematic Review and Replicability Study
of BERT4Rec for Sequential Recommendation* (RecSys '22), examined that result and the many
papers that cite it as a baseline, and argued the picture is considerably messier than the
headline suggests.

The claims this project set out to test with its own data:

| # | Claim | Testable here? |
|---|---|---|
| **C1** | BERT4Rec's *default* training configuration is severely undertrained — the originally reported numbers are not reachable with default settings in a reasonable budget. | ❌ Not testable as run (see §4) |
| **C2** | Given a sufficiently large training budget, BERT4Rec becomes competitive with — and can exceed — SASRec. | ⚠️ One point, consistent with |
| **C3** | BERT4Rec baseline numbers reported across the downstream literature are inconsistent and frequently not reproducible. | ❌ Out of scope (systematic-review claim, not an experiment) |
| **C4** | The SASRec-vs-BERT4Rec ranking is driven by training budget and training objective, not by architecture alone. | ❌ Confounded by design (see §4) |

C1–C3 are my reading of Petrov & Macdonald's argument; C4 is this project's own framing,
prompted by the same authors' later work on SASRec training objectives. **Before publishing
anything derived from this document, the four claims should be checked line-by-line against
the paper** — they are paraphrased from the argument, not quoted, and no page or table
references have been verified.

---

## 2. Experimental design

The single thing that makes a SASRec-vs-BERT4Rec comparison worth anything is protocol
identity — most published comparisons of these two models differ in split, negative
sampling, or metric definition, which makes the numbers incomparable before the models even
start training. The design here holds the protocol fixed and varies the model:

| Dimension | Setting | Applies to |
|---|---|---|
| Dataset | MovieLens-1M, 5-core filtered | both |
| Split | Leave-one-out, grouped by user, ordered by timestamp | both |
| Max sequence length | 200 | both |
| Eval protocol | Sampled: 1 positive + 100 uniform negatives | both |
| Metrics | HR@10, NDCG@10, test set | both |
| Epoch budget | 200 | both |

SASRec is this repo's own from-scratch PyTorch implementation
([`src/models/sasrec.py`](../src/models/sasrec.py), `configs/sasrec_ml1m.yaml`). BERT4Rec is
RecBole 1.2.1 ([`configs/recbole/ml1m_base.yaml`](../configs/recbole/ml1m_base.yaml),
[`src/recbole_run.py`](../src/recbole_run.py)) — reimplementing BERT4Rec from scratch was
judged low-ROI relative to buying a credible third-party implementation, per the project
plan.

Both results are logged to the same MLflow experiment and exported into
[`results/tables/master.md`](../results/tables/master.md) by script, never hand-copied.

---

## 3. Result

**ML-1M, sampled protocol (1 positive + 100 uniform negatives), test set, k=10, 200 epochs:**

| Model | Implementation | HR@10 | NDCG@10 | s/epoch | Total train time |
|---|---|---|---|---|---|
| SASRec | this repo (PyTorch, MPS) | **0.8190** | 0.5948 | ~7.0 | ~25 min |
| BERT4Rec | RecBole 1.2.1 (CUDA) | 0.8031 | **0.6036** | 104.1 | 5.8 h |

MLflow runs `sasrec_ml1m` and `bert4rec_recbole_1x`.

**It is a wash.** SASRec takes HR@10 by 1.6pp; BERT4Rec takes NDCG@10 by 0.9pp. Neither
model dominates, and the margins are small enough that a different seed could plausibly
reorder them — no seed-variance study was run, so that possibility is not excluded.

This is squarely at odds with the original BERT4Rec paper's reported margin over SASRec,
and squarely in line with the broad direction of the reproducibility literature: **a
properly trained SASRec is not the weak baseline that BERT4Rec's introduction made it look
like.** That is the one substantive conclusion this repo's data supports on its own.

The wall-clock column is the other half of the story, with the caveat that it compares
different hardware (Apple M-series MPS vs. a CUDA GPU), different batch sizes, and different
loss functions — so it is not a controlled efficiency measurement. It is nonetheless a
~14x gap in seconds-per-epoch to reach a statistical tie, and directionally consistent with
BERT4Rec being the more expensive way to buy this level of accuracy.

---

## 4. What this does **not** show

These are the reasons the result above is a single honest data point rather than a
resolution of the controversy.

**The negative sets are not literally identical.** The repo's central methodological
commitment is one frozen `negatives.json` (seed=42) shared by every model. RecBole's
evaluator does not consume it — `eval_args.mode: uni100` makes RecBole draw its own 1+100
uniform negatives. The protocol is therefore identical in *shape* (same split, same k, same
sampling distribution, same exclusion rule) but not the same *draw*. The plan's fallback for
this — export RecBole's raw prediction scores and rescore them through this repo's own
evaluator — was not implemented. Given that ~1.6pp separates the models, this is not a
negligible caveat.

**Loss function is confounded with framework.** RecBole runs both its models with
`loss_type: CE` (full softmax over the catalog, `train_neg_sample_args: ~`), while this
repo's SASRec trains with BCE against one sampled negative per position, per the original
SASRec paper. So the comparison is not SASRec-vs-BERT4Rec; it is
*SASRec+BCE+own-implementation* vs *BERT4Rec+CE+RecBole*. Three things vary at once. This
matters especially for C4, which is precisely a claim about the loss — the experiment as run
cannot separate the effect it is meant to measure.

**No same-framework control.** The intended fix for the confound above was the RecBole
SASRec run: same framework, same loss, same protocol, differing only in architecture, and
simultaneously serving as third-party cross-validation of this repo's implementation
(EXECUTION_PLAN.md's M4 criterion: <2% relative difference). **It was never launched.** The
only RecBole-SASRec sandbox run was a 20-epoch pipeline smoke test, which itself came back
with 0 MLflow runs and was discarded as a throwaway (commit `75d5ca5`); the real sweep was
then cut along with the 4x/10x budgets. Until it runs, M4 is not met and the framework
confound stays open.

**Only one budget point exists, so the budget claim is untested.** The sweep was built to
train once to 2000 epochs and recover 1x/4x/10x milestones from that single trajectory
(`src/recbole_run.py`, `scripts/daytona_week4.py`), which is why `BUDGETS` in the launcher
still lists all three. What was actually launched and completed was the 1x point alone:
`trained_to_epochs=200` in the run's params. At 104 s/epoch, the full 2000-epoch trajectory
would have cost ~58 GPU-hours per model, and the sweep was cut to the 1x point on cost
grounds. Consequently there is **no training-budget curve**, `results/figures/training_budget.png`
does not exist, and C2 rests on a single point that is *consistent with* the claim rather
than evidence *for* it.

**"1x" here is not the original paper's default configuration.** This is the most important
limitation for C1. The undertraining claim is about BERT4Rec's *original released
configuration* (the reference implementation's default training-step count and masked-LM
setup). What was run is 200 epochs of RecBole's BERT4Rec with full cross-entropy and a 2048
batch — a different implementation, a different objective, and a budget chosen to match this
repo's SASRec rather than to match the original release. That BERT4Rec is already
competitive at this repo's "1x" therefore says nothing about whether the *original* default
was undertrained. C1 is not weakly supported or contradicted here; it is simply not tested.

**No full-ranking number for BERT4Rec.** The RecBole evaluation was uni100-only, so
`full_HR@10` / `full_NDCG@10` are blank for `bert4rec_recbole_1x` in the master table. Since
this project's own methodology section argues at length that sampled metrics inflate results
(Krichene & Rendle, 2020), the headline comparison resting entirely on the sampled protocol
is a real weakness — the sampled protocol is exactly the one this repo elsewhere warns
against trusting alone.

---

## 5. Verdict, claim by claim

| # | Claim | Verdict from this repo's data |
|---|---|---|
| C1 | Default BERT4Rec config is severely undertrained | **Not tested.** The "1x" run is RecBole+CE at 200 epochs, not the original release configuration. |
| C2 | Adequately trained, BERT4Rec is competitive with SASRec | **Consistent with, not demonstrated.** At a matched 200-epoch budget and matched protocol the two tie (SASRec +1.6pp HR@10, BERT4Rec +0.9pp NDCG@10). No budget curve, so "adequately trained" is untested as a *variable*. |
| C3 | Downstream BERT4Rec baseline numbers are unreliable | **Out of scope.** Literature-survey claim; no experiment here addresses it. |
| C4 | Ranking is driven by budget and objective, not architecture | **Confounded, cannot answer.** Loss, framework, and architecture all vary between the two runs. |

**The one thing the data does say clearly:** under an identical evaluation protocol and an
identical epoch budget, BERT4Rec does *not* reproduce a decisive win over SASRec on ML-1M.
Whatever else remains open, the original paper's margin does not survive contact with a
protocol-controlled comparison here — which is the reproducibility community's central
complaint, arrived at independently.

---

## 6. What would close the gaps

In cost order, cheapest first:

1. **RecBole SASRec at 200 epochs** (~5 GPU-hours, ~$12 on an RTX 4090 sandbox). SASRec is
   the far smaller model, but that buys less than it looks like: the measured bottleneck is
   CPU-side — RecBole rebuilds the maxlen-200 sequence augmentation and feeds the GPU from
   one dataloader — and that cost is essentially model-independent, so this lands in the same
   order as BERT4Rec's 5.8 h rather than an order below it. Still the best value in this
   list: it closes the framework confound, gives the same-framework architecture comparison,
   and supplies the M4 cross-validation number in one run.
2. **Rescore RecBole predictions through this repo's evaluator** using the frozen
   `negatives.json`. Pure CPU work; removes the negative-set caveat entirely and would also
   yield the missing full-ranking numbers.
3. **One 4x point (800 epochs) per model** (~23 GPU-hours for BERT4Rec). Turns C2's single
   point into an actual budget curve and produces `training_budget.png`. The full 10x point
   is likely not worth its ~58 GPU-hours unless the 4x point shows the curve still climbing.

---

## References

- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation.* ICDM.
- Sun et al. (2019). *BERT4Rec: Sequential Recommendation with Bidirectional Encoder
  Representations from Transformer.* CIKM.
- Krichene & Rendle (2020). *On Sampled Metrics for Item Recommendation.* KDD.
- Petrov & Macdonald (2022). *A Systematic Review and Replicability Study of BERT4Rec for
  Sequential Recommendation.* RecSys.
