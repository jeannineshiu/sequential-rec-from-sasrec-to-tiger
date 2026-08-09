# The BERT4Rec Reproducibility Controversy — What This Repo's Data Does and Doesn't Show

**Status: partial.** Both models ran at one training-budget point (1x = 200 epochs) under a
matched protocol; the 4x/10x points did not run. The headline finding is that **the answer
flips depending on which SASRec you compare against**, and both SASRecs are defensible. This
document reports what the two runs support and is explicit about which claims they cannot
reach. Read alongside [`REPRODUCTION_LOG.md`](../REPRODUCTION_LOG.md), which carries the raw
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
| Dataset | MovieLens-1M, 5-core filtered | all three runs |
| Split | Leave-one-out, grouped by user, ordered by timestamp | all three runs |
| Max sequence length | 200 | all three runs |
| Eval protocol | Sampled: 1 positive + 100 uniform negatives | all three runs |
| Metrics | HR@10, NDCG@10, test set | all three runs |
| Epoch budget | 200 | all three runs |

Three runs: this repo's own from-scratch PyTorch SASRec
([`src/models/sasrec.py`](../src/models/sasrec.py), `configs/sasrec_ml1m.yaml`), plus SASRec
and BERT4Rec from RecBole 1.2.1
([`configs/recbole/ml1m_base.yaml`](../configs/recbole/ml1m_base.yaml),
[`src/recbole_run.py`](../src/recbole_run.py)). Reimplementing BERT4Rec from scratch was
judged low-ROI relative to buying a credible third-party implementation, per the project
plan.

Note what this table does and does not cover: it fixes everything about the *evaluation*
and nothing about the *models*. Model hyperparameters were left at each implementation's
own defaults — which turns out to be where the interesting result comes from (§3).

All three results are logged to the same MLflow experiment and exported into
[`results/tables/master.md`](../results/tables/master.md) by script, never hand-copied.

---

## 3. Result

**ML-1M, sampled protocol (1 positive + 100 uniform negatives), test set, k=10, 200 epochs:**

| Model | Implementation | HR@10 | NDCG@10 | s/epoch | Total train time |
|---|---|---|---|---|---|
| SASRec | this repo (PyTorch, MPS) | **0.8190** | 0.5948 | ~7.0 | ~25 min |
| SASRec | RecBole 1.2.1 (CUDA) | 0.7768 | 0.5702 | 84.2 | 4.7 h |
| BERT4Rec | RecBole 1.2.1 (CUDA) | 0.8031 | **0.6036** | 104.1 | 5.8 h |

MLflow runs `sasrec_ml1m`, `sasrec_recbole_1x`, `bert4rec_recbole_1x`.

### The result depends on which SASRec you ask

Those three rows contain two different answers to "does BERT4Rec beat SASRec?", and the
only thing that changes between them is which SASRec is used as the baseline:

| Comparison | HR@10 | NDCG@10 | Winner |
|---|---|---|---|
| RecBole BERT4Rec vs. **RecBole** SASRec | +3.39% | +5.86% | **BERT4Rec, on both** |
| RecBole BERT4Rec vs. **this repo's** SASRec | −1.95% | +1.47% | **a tie** (one metric each) |

Against RecBole's own SASRec — same framework, same loss, same protocol, same budget, the
most "controlled" comparison available here — BERT4Rec wins cleanly on both metrics, which
would reproduce the original BERT4Rec paper's direction. Against this repo's from-scratch
SASRec, the same BERT4Rec run is merely tied. **Same BERT4Rec number, opposite conclusion.**

### Why the two SASRecs differ, and why it matters

The two SASRecs are not the same model in two frameworks. RecBole ships per-model default
hyperparameters, and they are not matched across its own models:

| | this repo's SASRec | RecBole SASRec | RecBole BERT4Rec |
|---|---|---|---|
| hidden size | 50 | 64 | 64 |
| attention heads | 1 | 2 | 2 |
| **dropout** | **0.2** | **0.5** | **0.2** |
| loss | BCE + 1 sampled neg | CE (full softmax) | CE (full softmax) |
| batch size | 128 | 2048 | 2048 |

RecBole gives SASRec dropout 0.5 and BERT4Rec dropout 0.2 by default. On ML-1M — dense, 165
actions/user — 0.5 is a lot of regularization; this repo's own ML-1M config uses 0.2 (0.5 is
reserved for sparse Beauty), and it scores 4–5% higher than RecBole's SASRec.

**Hypothesis, not a demonstrated cause:** the dropout asymmetry is the single most likely
driver of the gap, and therefore of the flipped conclusion. It is *not* tested here — no
ablation isolating dropout was run. The experiment that would settle it is cheap and
specific: rerun RecBole SASRec with `hidden_dropout_prob: 0.2` / `attn_dropout_prob: 0.2`
and see whether the same-framework comparison collapses back to a tie.

If it does, this is a small, concrete instance of exactly the failure mode the
reproducibility literature describes — an apparent architectural win that is partly a
baseline-configuration artifact, arrived at accidentally by taking a framework's defaults at
face value. **This repo did the same thing**: `configs/recbole/ml1m_base.yaml` deliberately
matched the *protocol* across models (split, negatives, maxlen, budget) and never checked
that the *model* hyperparameters were comparable.

### Cost

The wall-clock column compares different hardware (Apple M-series MPS vs. an RTX 4090),
batch sizes, and losses, so it is not a controlled efficiency measurement. Within RecBole,
where hardware and batch size *are* matched, BERT4Rec costs 104.1 s/epoch against SASRec's
84.2 — about 24% more per epoch for its win.

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

**The cross-framework comparison confounds loss with framework.** RecBole runs both its
models with `loss_type: CE` (full softmax, `train_neg_sample_args: ~`), while this repo's
SASRec trains with BCE against one sampled negative per position, per the original SASRec
paper. So "this repo's SASRec vs. RecBole BERT4Rec" varies architecture, framework, loss,
and batch size at once. The same-framework row fixes framework, loss, and batch size — but
introduces the dropout asymmetry described in §3 instead. **Neither of the two comparisons
is clean**; they are confounded in different directions, which is precisely why they
disagree.

**The M4 cross-validation failed, and that failure is itself uninterpretable as stated.**
EXECUTION_PLAN.md's M4 criterion was "RecBole SASRec within 2% of this repo's SASRec" as
third-party evidence of implementation correctness. Measured: **−5.16% HR@10, −4.14%
NDCG@10** — comfortably outside the band. But the criterion assumed the two runs would
differ only by implementation, and they do not: they differ in dropout (0.2 vs 0.5), hidden
size (50 vs 64), heads (1 vs 2), loss (BCE+1neg vs CE), and batch size (128 vs 2048). A gap
of that size between two *differently configured* models is unremarkable and is not evidence
of a bug in this repo's SASRec.

It is also not evidence of *correctness*. The check that would actually test the
implementation is to run RecBole's SASRec with **this repo's** hyperparameters (d=50, 1
head, dropout 0.2) and compare like with like. That has not been run, so the third-party
verification of this repo's SASRec remains outstanding — the milestone is unmet, not passed
by reinterpretation.

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

**No full-ranking numbers for either RecBole run.** The RecBole evaluation was uni100-only,
so `full_HR@10` / `full_NDCG@10` are blank for both `sasrec_recbole_1x` and
`bert4rec_recbole_1x` in the master table. Since
this project's own methodology section argues at length that sampled metrics inflate results
(Krichene & Rendle, 2020), the headline comparison resting entirely on the sampled protocol
is a real weakness — the sampled protocol is exactly the one this repo elsewhere warns
against trusting alone.

---

## 5. Verdict, claim by claim

| # | Claim | Verdict from this repo's data |
|---|---|---|
| C1 | Default BERT4Rec config is severely undertrained | **Not tested.** The "1x" run is RecBole+CE at 200 epochs, not the original release configuration. |
| C2 | Adequately trained, BERT4Rec is competitive with SASRec | **Supported at one budget point.** BERT4Rec is at minimum competitive (tie vs. this repo's SASRec) and beats RecBole's own SASRec on both metrics. But "adequately trained" is untested as a *variable* — there is no budget curve. |
| C3 | Downstream BERT4Rec baseline numbers are unreliable | **Out of scope as a survey claim — but see below.** This project accidentally produced one instance of the underlying mechanism. |
| C4 | Ranking is driven by budget and objective, not architecture | **Not answered, but sharpened.** Budget is untested. What the data *does* show is that the ranking flips with baseline *configuration* — the same BERT4Rec run wins or ties depending only on which SASRec it is measured against. |

**The clearest thing in this data is not about either architecture.** It is that a
protocol-controlled comparison is not the same as a *fair* comparison. Every knob this
project set out to control — split, negatives, maxlen, epoch budget, metric — was matched
across the two models, and the conclusion still inverts depending on a baseline
hyperparameter nobody chose deliberately, inherited from a framework default file.

That is C3's mechanism observed first-hand rather than cited: published BERT4Rec-beats-
SASRec results are only as trustworthy as the SASRec they were measured against, and a
"reasonable default" is not a neutral choice. This repo fell into it too, and the honest
version of the headline is:

> Under a matched protocol on ML-1M at 200 epochs, BERT4Rec beats RecBole's default-
> configured SASRec on both metrics, and ties a paper-configured SASRec. The gap is
> plausibly a dropout artifact, and that hypothesis is untested.

---

## 6. What would close the gaps

In cost order, cheapest first:

0. **RecBole SASRec at 200 epochs with dropout 0.2** (~4.7 GPU-hours, ~$10). Now the single
   highest-value run available: it tests §3's dropout hypothesis, and it is simultaneously
   the like-for-like M4 cross-validation that the completed run was not. If the
   same-framework gap collapses, the flipped conclusion is explained; if it survives, the
   architecture claim gets much stronger. Either outcome is publishable.
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
