# The BERT4Rec Reproducibility Controversy — What This Repo's Data Does and Doesn't Show

**Status: partial, and closed at partial by decision.** All models ran at one training-budget
point (1x = 200 epochs) under a matched protocol; the 4x/10x points did not run and, as of
2026-08-25, will not — see §6.4 for the cost and the reasoning. The headline finding is that **the answer
flips depending on which SASRec you compare against** — and, as of the fourth run, that the
flip is *caused* by a single framework default rather than merely correlated with it. This
document reports what the runs support and is explicit about which claims they cannot reach.
Read alongside [`REPRODUCTION_LOG.md`](../REPRODUCTION_LOG.md), which carries the raw
debugging trail.

> **Update (2026-08-09).** The dropout hypothesis in §3 was previously flagged as untested.
> It has now been tested with a single-variable rerun and **confirmed**: changing only
> RecBole SASRec's dropout from its default 0.5 to 0.2 moves it by +3.71% HR@10 / +6.33%
> NDCG@10 — *more* than the entire margin (+3.39% / +5.86%) by which BERT4Rec was said to
> beat it. With dropout matched, SASRec edges ahead of BERT4Rec on both metrics. The same
> run meets M4's cross-validation criterion on HR@10 (+0.61%) but not NDCG@10 (+7.41%), and
> its full-ranking numbers — the first available for any RecBole run here — diverge from this
> repo's SASRec far more than the sampled ones do. §3, §4, §5 and §6 are updated below.

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
| **C4** | The SASRec-vs-BERT4Rec ranking is driven by training budget and training objective, not by architecture alone. | ⚠️ Partially — configuration shown to flip it (§3); budget still untested |

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
| Dataset | MovieLens-1M, 5-core filtered | all four runs |
| Split | Leave-one-out, grouped by user, ordered by timestamp | all four runs |
| Max sequence length | 200 | all four runs |
| Eval protocol | Sampled: 1 positive + 100 uniform negatives | all four runs |
| Metrics | HR@10, NDCG@10, test set | all four runs |
| Epoch budget | 200 | all four runs |

Four runs: this repo's own from-scratch PyTorch SASRec
([`src/models/sasrec.py`](../src/models/sasrec.py), `configs/sasrec_ml1m.yaml`), plus SASRec
and BERT4Rec from RecBole 1.2.1
([`configs/recbole/ml1m_base.yaml`](../configs/recbole/ml1m_base.yaml),
[`src/recbole_run.py`](../src/recbole_run.py)). Reimplementing BERT4Rec from scratch was
judged low-ROI relative to buying a credible third-party implementation, per the project
plan.

Note what this table does and does not cover: it fixes everything about the *evaluation*
and nothing about the *models*. Model hyperparameters were left at each implementation's
own defaults — which turns out to be where the interesting result comes from (§3).

All four results are logged to the same MLflow experiment and exported into
[`results/tables/master.md`](../results/tables/master.md) by script, never hand-copied.

---

## 3. Result

**ML-1M, sampled protocol (1 positive + 100 uniform negatives), test set, k=10, 200 epochs:**

| Model | Implementation | HR@10 | NDCG@10 | s/epoch | Total train time |
|---|---|---|---|---|---|
| SASRec | this repo (PyTorch, MPS) | **0.8190** | 0.5948 | ~7.0 | ~25 min |
| SASRec | RecBole 1.2.1, dropout 0.5 (default) | 0.7768 | 0.5702 | 84.2 | 4.7 h |
| SASRec | RecBole 1.2.1, **dropout 0.2** | 0.8056 | **0.6063** | 84.2 | 4.7 h |
| BERT4Rec | RecBole 1.2.1 (CUDA, dropout 0.2) | 0.8031 | 0.6036 | 104.1 | 5.8 h |

MLflow runs `sasrec_ml1m`, `sasrec_recbole_1x`, `sasrec_recbole_1x_dropout02`,
`bert4rec_recbole_1x`.

### The result depends on which SASRec you ask

The first three rows contain two different answers to "does BERT4Rec beat SASRec?", and the
only thing that changes between them is which SASRec is used as the baseline:

| Comparison | HR@10 | NDCG@10 | Winner |
|---|---|---|---|
| RecBole BERT4Rec vs. RecBole SASRec (**dropout 0.5**) | +3.39% | +5.86% | **BERT4Rec, on both** |
| RecBole BERT4Rec vs. **this repo's** SASRec | −1.94% | +1.48% | **a tie** (one metric each) |
| RecBole BERT4Rec vs. RecBole SASRec (**dropout 0.2**) | −0.31% | −0.45% | **SASRec, on both** |

Against RecBole's default-configured SASRec — same framework, same loss, same protocol, same
budget, the most "controlled" comparison available here — BERT4Rec wins cleanly on both
metrics, which would reproduce the original BERT4Rec paper's direction. Against this repo's
from-scratch SASRec, the same BERT4Rec run is merely tied. Against the *same* RecBole SASRec
with one hyperparameter changed, it loses on both. **Same BERT4Rec number, three different
conclusions.**

### The dropout asymmetry, tested

RecBole's SASRec and BERT4Rec property files
(`recbole/properties/model/{SASRec,BERT4Rec}.yaml`) are identical on every architectural
default — `n_layers: 2`, `n_heads: 2`, `hidden_size: 64`, `inner_size: 256`,
`loss_type: CE`. They differ on exactly one thing:

```
SASRec.yaml:   hidden_dropout_prob: 0.5    attn_dropout_prob: 0.5
BERT4Rec.yaml: hidden_dropout_prob: 0.2    attn_dropout_prob: 0.2
```

That makes a dropout-only rerun a genuine single-variable experiment rather than an
approximation of one, and it was run: `configs/recbole/ml1m_sasrec_dropout02.yaml`, an
overlay on the shared base config, changing dropout and nothing else. Seed, split,
negatives, budget, batch size, loss and hardware all held.

| RecBole SASRec | HR@10 | NDCG@10 |
|---|---|---|
| dropout 0.5 (default) | 0.7768 | 0.5702 |
| dropout 0.2 | 0.8056 | 0.6063 |
| **effect of dropout alone** | **+3.71%** | **+6.33%** |

Compare that against the margin it was supposed to explain — BERT4Rec's +3.39% / +5.86% win
over the default-configured SASRec. **The dropout default alone more than accounts for the
entire BERT4Rec advantage.** The hypothesis is confirmed, and with the asymmetry removed the
same-framework comparison does not merely collapse to a tie, it reverses: SASRec is ahead by
+0.31% HR@10 / +0.45% NDCG@10, which is itself small enough to read as a tie.

An incidental observation from the training curve: at dropout 0.2 the valid NDCG@10 was
still climbing at epoch 189 (0.6394, a new best), whereas the run was cut at 200. The
default-dropout run had plateaued. Whatever the budget curve in §6 eventually shows, the
lower-dropout configuration is the one that had not finished improving.

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
reserved for sparse Beauty), and it scores 4–5% higher than RecBole's default SASRec.

**Demonstrated cause, as of the dropout-0.2 rerun above.** This is a small, concrete
instance of exactly the failure mode the reproducibility literature describes — an apparent
architectural win that is entirely a baseline-configuration artifact, arrived at
accidentally by taking a framework's defaults at face value. **This repo did the same
thing**: `configs/recbole/ml1m_base.yaml` deliberately matched the *protocol* across models
(split, negatives, maxlen, budget) and never checked that the *model* hyperparameters were
comparable. The defaults were not adversarial or unusual — they are what anyone running
RecBole out of the box gets, which is the point.

One process change came out of this: `src/recbole_run.py` now logs the hyperparameters that
actually vary (dropout, width, heads, loss, batch size) to MLflow. Two runs that differ only
by an *unlogged* framework default are indistinguishable in the master table, which is how
the asymmetry survived a week of analysis.

### Cost

The wall-clock column compares different hardware (Apple M-series MPS vs. an RTX 4090),
batch sizes, and losses, so it is not a controlled efficiency measurement. Within RecBole,
where hardware and batch size *are* matched, BERT4Rec costs 104.1 s/epoch against SASRec's
84.2 — about 24% more per epoch for its win.

---

## 4. What this does **not** show

These are the reasons the result above is a single honest data point rather than a
resolution of the controversy.

**The negative sets are not literally identical, and it demonstrably matters.** The repo's
central methodological commitment is one frozen `negatives.json` (seed=42) shared by every
model. RecBole's evaluator does not consume it — `eval_args.mode: uni100` makes RecBole draw
its own 1+100 uniform negatives. The protocol is therefore identical in *shape* (same split,
same k, same sampling distribution, same exclusion rule) but not the same *draw*.

The fallback is now implemented — `src/recbole_run.export_scores` dumps the test-set score
matrix and `scripts/rescore_recbole.py` rescores it offline against the frozen negatives —
and it puts a number on the caveat. On **identical predictions** from the dropout-0.2 run:

| Scoring of the same predictions | HR@10 | NDCG@10 |
|---|---|---|
| RecBole's own uni100 draw | 0.8056 | 0.6063 |
| This repo's frozen `negatives.json` | 0.8240 | 0.6389 |
| difference from the negative draw alone | +2.28% | +5.38% |

That is the same order of magnitude as every margin discussed in this document. Cross-model
comparisons here are therefore made on uni100 numbers throughout, because all four runs have
them; the frozen-negative numbers are reported separately (master table row
`sasrec_recbole_1x_dropout02_ourprotocol`) and are **not** differenced against another run's
uni100 figure.

**The other two RecBole runs cannot be rescored.** `export_scores` did not exist when
`sasrec_recbole_1x` and `bert4rec_recbole_1x` ran, their sandboxes were ephemeral and are
deleted, and no checkpoint was ever pushed. Their predictions are gone permanently; only the
scalar uni100 metrics survive. Closing the negative-draw and full-ranking gaps for those two
means retraining them (~10.5 GPU-hours combined).

**The cross-framework comparison confounds loss with framework.** RecBole runs both its
models with `loss_type: CE` (full softmax, `train_neg_sample_args: ~`), while this repo's
SASRec trains with BCE against one sampled negative per position, per the original SASRec
paper. So "this repo's SASRec vs. RecBole BERT4Rec" varies architecture, framework, loss,
and batch size at once. The same-framework row fixes framework, loss, and batch size — but
introduces the dropout asymmetry described in §3 instead. **Neither of the two comparisons
is clean**; they are confounded in different directions, which is precisely why they
disagree.

**The M4 cross-validation now passes — on the dropout-matched run.** EXECUTION_PLAN.md's M4
criterion was "RecBole SASRec within 2% of this repo's SASRec" as third-party evidence of
implementation correctness. Against RecBole's *default* SASRec it measured −5.15% HR@10 /
−4.14% NDCG@10, comfortably outside the band, but that comparison was uninterpretable: the
two runs differ in dropout (0.2 vs 0.5), hidden size (50 vs 64), heads (1 vs 2), loss
(BCE+1neg vs CE) and batch size (128 vs 2048), so the number measured configuration rather
than correctness.

With dropout matched, the criterion must be evaluated on the *same* negative draw — the
frozen `negatives.json`, for both sides. Anything else mixes the two draws quantified above
and measures the sampling, not the implementations:

| RecBole SASRec (dropout 0.2) vs. this repo's SASRec | HR@10 | NDCG@10 |
|---|---|---|
| both on frozen `negatives.json` (**the valid comparison**) | **+0.61%** ✅ | **+7.41%** ❌ |
| *mixed draws (RecBole uni100 vs. frozen) — do not use* | *−1.64%* | *+1.93%* |

**M4's criterion is met on HR@10 and missed on NDCG@10.** The mixed-draw row is recorded
only because it is the comparison one gets by reading the two runs' headline numbers
straight out of the master table, and it happens to fall inside ±2% on both metrics — a
false pass. It is the exact error this section warns about one paragraph earlier, and it is
easy to make.

The split verdict is more informative than either number alone. HR@10 only asks whether the
true item landed in the top 10; NDCG@10 asks *where*. The two implementations put the true
item in the top 10 at essentially the same rate (+0.61%), while RecBole's ranks it
distinctly higher within that window (+7.41%). That is the same direction, and the same
suspected cause, as the full-ranking divergence documented below: cross-entropy over the
full catalog produces sharper ordering than BCE against one sampled negative, and the effect
grows as the metric becomes more sensitive to position — invisible to HR, visible in NDCG,
large in full ranking.

So the third-party evidence is partial: consistent with a correct implementation on the
coarsest metric, and showing a real, systematic, and plausibly loss-driven difference on the
finer ones. Note also that this is not the fully-matched run originally described as
necessary — hidden size, heads, loss and batch size still differ.

The same result relocates the earlier failure. The −5.15% / −4.14% figure was never about
this repo's implementation; it was a dropout measurement wearing a correctness label.

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

**Full-ranking numbers exist for one RecBole run only — and they disagree with the sampled
verdict.** `full_HR@10` / `full_NDCG@10` are still blank for `sasrec_recbole_1x` and
`bert4rec_recbole_1x` (their predictions are gone, see above). The dropout-0.2 run has them,
via offline rescoring, and they are worth pausing on:

| SASRec, ML-1M, test, k=10 | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 |
|---|---|---|---|---|
| this repo (BCE + 1 sampled negative) | 0.8190 | 0.5948 | 0.2475 | 0.1322 |
| RecBole, dropout 0.2 (CE over full catalog) | 0.8240 | 0.6389 | **0.3467** | **0.2029** |
| relative difference | +0.61% | +7.41% | **+40.1%** | **+53.5%** |

All four sampled figures are on the frozen `negatives.json`, so the row is directly
comparable throughout.

The divergence grows monotonically with how much the metric cares about position: +0.61% on
sampled HR, +7.41% on sampled NDCG, +40% on full HR, +53% on full NDCG. The most likely
explanation is the training
objective, which the sampled protocol is largely blind to: RecBole trains with cross-entropy
over the entire catalog, which is close to a direct optimization of full-ranking, while this
repo follows the original SASRec recipe of BCE against one sampled negative per position.
Ranking against 100 random negatives barely distinguishes the two; ranking against 3,416
items does.

This is unverified — no ablation swapping only the loss was run — but it is a concrete
warning about the M4 criterion above. **"Within 2%" is a statement about one metric under
one protocol, not about the models.** Two implementations can agree almost exactly on the
protocol this repo elsewhere warns against trusting (Krichene & Rendle, 2020) while
differing by 40–50% on the one it recommends. The cross-model BERT4Rec comparisons in §3 all
rest on the sampled protocol, and this row is a reason to hold them loosely: nothing here
establishes that the §3 ordering would survive full-ranking evaluation, and the one case
where both protocols are available shows them disagreeing by a wide margin.

---

## 5. Verdict, claim by claim

| # | Claim | Verdict from this repo's data |
|---|---|---|
| C1 | Default BERT4Rec config is severely undertrained | **Not tested.** The "1x" run is RecBole+CE at 200 epochs, not the original release configuration. |
| C2 | Adequately trained, BERT4Rec is competitive with SASRec | **Supported at one budget point, weakened by the dropout result.** BERT4Rec is competitive — it ties this repo's SASRec and loses narrowly to a dropout-matched RecBole SASRec. Its apparent *win* over RecBole's SASRec was a configuration artifact. "Adequately trained" remains untested as a *variable*: there is no budget curve. |
| C3 | Downstream BERT4Rec baseline numbers are unreliable | **Out of scope as a survey claim — but demonstrated first-hand.** This project produced, and then diagnosed, one instance of the underlying mechanism. |
| C4 | Ranking is driven by budget and objective, not architecture | **Partially answered, in the claim's favour.** Budget is still untested. But the ranking is now shown to flip on a *non-architectural* variable: one dropout default, worth more than the entire architectural margin. The suspected loss effect on full ranking (§4) points the same way. Architecture alone does not determine the ordering. |

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
> configured SASRec by +3.39% HR@10 / +5.86% NDCG@10. Changing one line of that SASRec's
> configuration — dropout 0.5 → 0.2, the value RecBole already uses for BERT4Rec — is worth
> +3.71% / +6.33%, and the ordering reverses. BERT4Rec's advantage on this dataset, at this
> budget, under this protocol, is a baseline-configuration artifact.

Two limits on that sentence, both from §4. It rests on the sampled protocol, and the one run
with both protocols shows them diverging by 40–50%. And every RecBole number is a single
seed — though the measured noise floor (§6) puts the dropout effect at 4–6x it, so that
concern is now quantified rather than open. The residual SASRec-over-BERT4Rec margin
(+0.31% / +0.45%) sits *inside* the floor: that is a tie, measured.

---

## 6. What would close the gaps

~~RecBole SASRec at dropout 0.2~~ — **done** (2026-08-09), §3. ~~Rescore RecBole predictions
through this repo's evaluator~~ — **done** for that run, §4; impossible for the other two,
whose predictions no longer exist.

Remaining, in cost order, cheapest first:

1. ~~A seed-variance study~~ — **done cheaply, on the wrong model on purpose.** Five seeds of
   *this repo's* SASRec cost nothing (laptop, ~20 min each) and establish a noise floor of
   **0.96% sampled / 3.37% full** (2·√2·σ; see README and `scripts/seed_variance.py`). Read
   against it, every conclusion in §3 holds: the dropout effect is 4–6x the floor, and the
   residual SASRec-over-BERT4Rec margin (+0.31% / +0.45%) is inside it, confirming the tie as
   a measurement rather than a hedge. **The floor is borrowed, not measured, for the RecBole
   runs** — different model, framework and loss. **Superseded on 2026-08-24**: RecBole's
   dropout-0.2 config was re-run at seeds 1 and 2, so that family now carries its own measured
   spread and the blanket floor quoted above has been retired (see the README's noise-floor
   section). Its full-ranking spread is 1.83%, *wider* than the borrowed 1.19% — the proxy was
   too narrow there. Beauty, seeded on 2026-08-25, is wider still at 3.73%.
2. **A loss ablation: RecBole SASRec at dropout 0.2 with BCE + 1 sampled negative**
   (~4.7 GPU-hours). Tests §4's explanation for the 40–53% full-ranking divergence, which is
   currently the largest unexplained effect in the project and the one that most undermines
   confidence in the sampled-protocol comparisons.
3. **Retrain the other two RecBole runs with score export** (~10.5 GPU-hours). Gives
   `bert4rec_recbole_1x` and `sasrec_recbole_1x` frozen-negative and full-ranking numbers, so
   the §3 comparison table can be repeated on the full-ranking protocol rather than resting
   on the sampled one.
4. ~~One 4x point (800 epochs) per model~~ — **not funded, decided 2026-08-25, and the
   reasoning is part of the result.** The measured epoch times are 84.2 s (SASRec) and
   104.1 s (BERT4Rec), so a 4x point for the pair is ~42 GPU-hours (~$95) and the full
   2000-epoch trajectory — which yields 1x/4x/10x from one run via the milestone snapshots in
   `src/recbole_run.py` — is ~105 GPU-hours (~$240). Neither figure is the real price. This
   repo's own standard, established by the seed work in the README, is that a new
   configuration is not reported against a borrowed noise floor; a budget trajectory is a new
   configuration whose spread nobody has measured, so meeting that standard means three seeds
   per model — ~315 GPU-hours, ~$700 — and running it at RecBole's defaults would measure
   budget crossed with the very dropout default §3 shows is the dominant term, so an honest
   version wants a matched-dropout pair as well.

   What that buys is a reproduction of a study that already exists (Petrov & Macdonald 2022),
   on someone else's question, and it would not change a single claim in this repo. Every
   other experiment that was funded here overturned or corrected something the repo itself had
   said. This one has no such lever, and the risk of leaving it undone — a reader thinking
   the project adjudicated the controversy — was closed for free by the framing paragraph in
   the README and by §4 and §5 of this document, which state outright that C2 rests on one
   point and is *consistent with* rather than *evidence for* the claim.

   The directional evidence that does exist is free and is already recorded: at dropout 0.2
   the valid NDCG@10 was still climbing at epoch 189 while the default-dropout run had
   plateaued (§3). **The configuration that had not finished training is the one that wins**,
   which is a sharper sentence than a two-point curve at defaults would have produced.
   `results/figures/training_budget.png` therefore does not exist by decision rather than by
   omission, and M4 stays partially met on the record.

   If it is ever funded, the sweep is already built and takes one command per model:
   `uv run python scripts/daytona_week4.py --model SASRec --detached` (add `--budgets` to
   pick the trajectory). A smoke run on 2026-08-25 confirmed the launcher still provisions
   and starts; the sandboxes were deleted before training, for ~$1.

---

## References

- Kang & McAuley (2018). *Self-Attentive Sequential Recommendation.* ICDM.
- Sun et al. (2019). *BERT4Rec: Sequential Recommendation with Bidirectional Encoder
  Representations from Transformer.* CIKM.
- Krichene & Rendle (2020). *On Sampled Metrics for Item Recommendation.* KDD.
- Petrov & Macdonald (2022). *A Systematic Review and Replicability Study of BERT4Rec for
  Sequential Recommendation.* RecSys.
