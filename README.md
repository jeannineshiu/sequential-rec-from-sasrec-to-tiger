# Sequential Recommendation: From SASRec to TIGER

Reproduce a field-standard sequential recommender (SASRec), independently verify the
BERT4Rec reproducibility controversy under a training-budget-controlled comparison, then
rebuild the same codebase as a TIGER-style generative recommender with RQ-quantized
semantic IDs — walking through 2018→2025 sequential recommendation in one repo.

Full plan: [`sequential-rec-project-plan.md`](sequential-rec-project-plan.md)
Execution checklist: [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)
Debugging trail: [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md)
BERT4Rec controversy analysis: [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md)

## Status: Week 3 done (M3 met) — Week 4 mostly done (M4 partially met)

Week 4 ran four protocol-matched models at 200 epochs on ML-1M. The finding is not about
either architecture: **BERT4Rec's win over SASRec on this dataset is a baseline-configuration
artifact.**

RecBole's SASRec and BERT4Rec property files are identical on every architectural default
(2 layers, 2 heads, hidden 64, inner 256, CE loss) except one — SASRec gets dropout 0.5,
BERT4Rec gets 0.2. Changing that one line and nothing else:

| Comparison (same protocol, same budget, sampled test HR@10 / NDCG@10) | HR@10 | NDCG@10 | Winner |
|---|---|---|---|
| BERT4Rec vs. RecBole's SASRec (**dropout 0.5**, its default) | +3.39% | +5.86% | BERT4Rec, on both |
| BERT4Rec vs. **this repo's** SASRec | −1.94% | +1.48% | a tie |
| BERT4Rec vs. RecBole's SASRec (**dropout 0.2**) | −0.31% | −0.45% | SASRec, on both |
| *effect of the dropout default alone* | *+3.71%* | *+6.33%* | — |

The dropout default is worth **more than the entire margin it was supposed to explain**. Same
BERT4Rec run, three different conclusions, decided by a value nobody chose deliberately. This
repo walked into it by matching the *evaluation* protocol across models with great care and
never checking that the *model* hyperparameters were comparable — which is the failure mode
the BERT4Rec reproducibility literature describes, reproduced here by accident and then
diagnosed.

**M4 is partially met.** The cross-validation criterion (<2% vs this repo's SASRec, both on
the frozen negatives) passes on HR@10 (+0.61%) and fails on NDCG@10 (+7.41%). The signature
training-budget figure was cut on GPU cost — only the 1x point exists — so it remains
outstanding. Full claim-by-claim analysis:
[`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md).

One result deserves more attention than the headline. The same two SASRec implementations
agree to +0.61% on sampled HR@10 and diverge by **+40% / +53%** on full ranking (see the
full-ranking table below). Agreement under the sampled protocol is not agreement.

## Reproduction table (updated as results land)

### ML-1M, sampled protocol (1 positive + 100 uniform random negatives), test set, k=10

| Model | HR@10 | NDCG@10 | Note |
|---|---|---|---|
| Popularity | 0.4363 | 0.2401 | floor |
| BPR-MF (implicit) | 0.5745 | 0.3357 | floor |
| SASRec (paper, Kang & McAuley 2018) | 0.8245 | 0.5905 | target |
| **SASRec (this repo)** | **0.8190** | **0.5948** | ✅ within target range (0.80–0.83 / 0.57–0.60) |
| SASRec (RecBole, 200 epochs, default hparams — dropout 0.5) | 0.7768 | 0.5702 | −5.15% / −4.14% vs this repo, but this measures dropout, not implementation |
| SASRec (RecBole, 200 epochs, **dropout 0.2**) | 0.8056 | **0.6063** | the dropout-only rerun; beats BERT4Rec on both |
| SASRec (RecBole, dropout 0.2, **rescored on this repo's frozen negatives**) | 0.8240 | 0.6389 | ✅ +0.61% HR@10 vs this repo (M4 <2% met) / ❌ +7.41% NDCG@10 (not met) |
| **BERT4Rec (RecBole, 200 epochs)** | 0.8031 | 0.6036 | beats RecBole's *default* SASRec; ties this repo's; loses to the dropout-matched one — see [controversy analysis](docs/bert4rec-controversy.md) |

The three RecBole rows above marked otherwise are on RecBole's own `uni100` negative draw;
only the explicitly-marked row uses this repo's frozen `negatives.json`. On identical
predictions the two draws differ by +2.28% HR@10 / +5.38% NDCG@10 — the same order as every
margin being discussed — so rows from different draws are never differenced against each
other.

### ML-1M, full ranking (rank against entire catalog, excl. history), test set, k=10

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0369 | 0.0180 |
| BPR-MF (implicit) | 0.0671 | 0.0333 |
| SASRec (this repo) | 0.2475 | 0.1322 |
| SASRec (RecBole, dropout 0.2, rescored) | **0.3467** | **0.2029** |

That last row is +40% / +53% over this repo's SASRec, against +0.61% on sampled HR@10 for the
same pair. The divergence grows monotonically with how much the metric cares about *where* in
the ranking the true item lands, which points at the training objective: RecBole trains with
cross-entropy over the full catalog, this repo with BCE against one sampled negative per
position, per the original SASRec paper. Untested as a hypothesis — no loss-only ablation was
run — but it is the largest unexplained effect in the project.

### Amazon Beauty, sampled protocol, test set, k=10

| Model | HR@10 | NDCG@10 | Note |
|---|---|---|---|
| SASRec (paper reference, ~0.4854 ±2pp) | 0.4654–0.5054 | — | target |
| **SASRec (this repo)** | **0.5097** | 0.3453 | ⚠️ +0.43pp over the accepted band (see REPRODUCTION_LOG.md — reporting as-is rather than tuning to fit) |

TIGER-style comparison pending Week 6.

### Ablations (ML-1M, sampled/full HR@10, test set, k=10 — 100 epochs, see REPRODUCTION_LOG.md for full analysis)

| Ablation | sampled HR@10 | full HR@10 | avg sec/epoch |
|---|---|---|---|
| Baseline (learnable pos emb, maxlen 200) | 0.8190 | 0.2475 | ~7.0 |
| A1: positional embedding = none | 0.8066 | 0.2291 | 7.47 |
| A1: positional embedding = sinusoidal | 0.8147 | 0.2182 | 6.91 |
| A2: maxlen = 50 | 0.7858 | 0.2033 | 1.53 |
| A2: maxlen = 100 | 0.8058 | 0.2346 | 2.85 |
| A4: negative sampling = popularity-weighted | 0.7540 | 0.1871 | 6.93 |

Full table (script-generated, not hand-copied): [`results/tables/master.md`](results/tables/master.md)
A3 sampled-vs-full-ranking scatter: [`results/figures/sampled_vs_full.png`](results/figures/sampled_vs_full.png) — clean
positive correlation within ML-1M, but Beauty's full-ranking score sits well below what
the ML-1M trend would predict, since full-ranking difficulty scales with catalog size
(12,101 items vs. 3,416) independent of model quality — see REPRODUCTION_LOG.md for the
full discussion.

## Methodology notes

- **Two evaluation protocols, always reported side by side.** The sampled protocol
  (1 positive + 100 uniform negatives) matches the original SASRec paper but is known to
  inflate metrics relative to ranking against the full catalog (Krichene & Rendle, 2020 —
  "On Sampled Metrics for Item Recommendation"). Note above how much higher every
  sampled number is than its full-ranking counterpart, even for the weakest baseline.
- **Fixed negatives across all models.** `data/processed/*/negatives.json` is generated
  once (seed=42) and reused by every model so comparisons are apples-to-apples. **Exception:
  the RecBole runs**, whose evaluator draws its own 1+100 uniform negatives
  (`eval_args.mode: uni100`) rather than consuming that file — same protocol shape, different
  draw. This is no longer a hand-wave: rescoring one run's raw predictions both ways puts the
  effect at +2.28% HR@10 / +5.38% NDCG@10, on the same order as the margins being discussed.
  `src/recbole_run.py` now exports the test score matrix and
  `scripts/rescore_recbole.py` rescores it offline; that path was added too late for the
  first two RecBole runs, whose ephemeral sandboxes are gone and whose predictions cannot be
  recovered.
- **Leave-one-out split**, 5-core filtered, per user: last item → test, second-to-last →
  valid, rest → train. No-leakage checked by assertion at preprocessing time.

## Reproduce

```bash
uv sync
uv run python -m src.data.download --dest data/raw --dataset all
uv run python -m src.data.preprocess --dataset ml-1m --out-dir data/processed/ml-1m
uv run python -m src.data.preprocess --dataset beauty --out-dir data/processed/beauty
uv run python -m src.baselines --data-dir data/processed/ml-1m
uv run python -m src.train --config configs/sasrec_ml1m.yaml
uv run python -m src.train --config configs/sasrec_beauty.yaml
uv run python -m src.export_results   # rebuilds results/tables/master.md + the A3 figure
uv run pytest tests/
```

## Repo structure

See [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) Appendix / original plan Appendix A.

## Limitations

Kept current as the project progresses, per EXECUTION_PLAN.md's guiding principle of
reporting negative/mixed results rather than hiding them.

- **Beauty SASRec sits 0.43pp above the accepted reproduction band** (0.5097 vs. the
  0.4654–0.5054 target). Reported as-is rather than tuned until it lands inside the band.
- **Ablations ran at 100 epochs, headline numbers at 200.** Deliberate compute saving —
  ablations compare configs against each other, not against the paper — but it means the A4
  popularity-negatives result in particular may be a convergence-speed effect rather than a
  final ranking.
- **The BERT4Rec comparisons still rest on the sampled protocol.** Only one RecBole run has
  full-ranking numbers, and for the one pair where both protocols exist they disagree by
  40–53%. Nothing here establishes that the SASRec-vs-BERT4Rec ordering would survive
  full-ranking evaluation. The cross-framework comparison also still varies loss, batch size
  and architecture at once. Full accounting in
  [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md) §4.
- **The full-ranking divergence between the two SASRecs is unexplained.** +40% HR@10 / +53%
  NDCG@10 for RecBole's over this repo's, from implementations that agree to +0.61% on
  sampled HR@10. Cross-entropy over the full catalog vs. BCE with one sampled negative is the
  obvious suspect and is consistent with the pattern, but no loss-only ablation was run.
- **No training-budget curve.** Only the 1x (200-epoch) point was run; the 4x/10x points
  were cut on cost (~58 GPU-hours per model for the full trajectory). The scaling claim at
  the heart of the BERT4Rec controversy is therefore untested here.
- **No seed-variance study.** Every headline number is a single seed (42). The residual
  SASRec-vs-BERT4Rec margins (+0.31% / +0.45%) are almost certainly inside seed noise and
  should be read as a tie. The dropout effect (+3.71% / +6.33%) is an order of magnitude
  larger and probably survives, but that is an assumption, not a measurement.
