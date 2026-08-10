# Sequential Recommendation: From SASRec to TIGER

Reproduce a field-standard sequential recommender (SASRec), independently verify the
BERT4Rec reproducibility controversy under a training-budget-controlled comparison, then
rebuild the same codebase as a TIGER-style generative recommender with RQ-quantized
semantic IDs — walking through 2018→2025 sequential recommendation in one repo.

Full plan: [`sequential-rec-project-plan.md`](sequential-rec-project-plan.md)
Execution checklist: [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)
Debugging trail: [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md)
BERT4Rec controversy analysis: [`docs/bert4rec-controversy.md`](docs/bert4rec-controversy.md)

## Status: Weeks 1–5 done — Week 6 main experiment done (the cold-start hypothesis fails, in the direction it was supposed to win)

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
| GenRec (semantic IDs, same backbone) | 0.3621 | 0.2235 | −28.96% / −35.27% — see below |

### Amazon Beauty: atomic vs. semantic IDs (Week 5 result, full comparison in Week 6)

Same backbone, same protocol, same frozen negatives; the only variable is whether an item is
one embedding or a sequence of four semantic tokens.

| test, k=10 | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 | parameters |
|---|---|---|---|---|---|
| SASRec (atomic) | **0.5097** | **0.3453** | **0.0594** | **0.0303** | 828,352 |
| GenRec (semantic) | 0.3621 | 0.2235 | 0.0329 | 0.0168 | **113,472** |
| relative | −28.96% | −35.27% | −44.6% | −44.6% | **13.7%** |

**The generative model loses, and it is not a decoding artifact** — widening the beam from 20
to 200 moves full HR@10 by 0.0002 (`uv run python -m scripts.beam_sensitivity`), and every
margin is far outside the seed-noise floor.

The parameter column is what makes this interesting rather than just negative. 12,101 item
embeddings collapse into 782 token embeddings — a 15.5x smaller table — so GenRec gives up
29% of sampled HR@10 while running on an eighth of the parameters. That capacity gap cannot be
controlled away: matching parameter counts would mean crippling SASRec's item table or
inflating GenRec's hidden dimension, and the compression *is* the method under test. It gets
stated rather than eliminated.

Unconstrained greedy decoding is legal 81.8% of the time after training (32.8% after two
epochs), so constrained decoding is doing real work — without the Trie, nearly one in five of
the model's first-choice recommendations would not be an item that exists.

### Cold-start bucketing: the hypothesis fails in the direction it was supposed to win

Semantic IDs are supposed to help where atomic IDs are weakest — a rarely-seen item has a
barely-trained embedding, but its semantic ID is built from codes thousands of other items
share. Prediction: GenRec loses on the head and closes the gap on the tail.

![cold-start buckets](results/figures/cold_start_buckets.png)

| bucket (target's train frequency) | users | SASRec HR@10 | GenRec HR@10 | relative |
|---|---|---|---|---|
| unseen (0) | 138 | 0.0000 | 0.0000 | — |
| tail (1–4) | 4,594 | 0.0185 | 0.0022 | **−88.2%** |
| torso (5–19) | 9,539 | 0.0427 | 0.0085 | **−80.1%** |
| head (20+) | 8,092 | 0.1033 | 0.0797 | −22.8% |
| overall | 22,363 | 0.0594 | 0.0329 | −44.6% |

**The gap widens monotonically as items get rarer — the opposite of the prediction.** Re-running
every bucket at beam 200 instead of 20 makes it slightly worse, not better (tail −89.4%), so it
is not a decoding artifact. The `unseen` bucket is uninformative: 138 users, both models at
zero; GenRec's zero is consistent with any true rate below ~2.2%.

### Why: recommendation diversity collapses

| model | distinct items across all top-10s | median train freq | % head | % torso | % tail |
|---|---|---|---|---|---|
| SASRec (atomic) | **9,221** (76% of catalogue) | 22 | 54.2% | 42.1% | 3.7% |
| GenRec (semantic) | **839** (7%) | 84 | 84.7% | 13.1% | 2.2% |

GenRec's entire output covers 7% of the catalogue. Generation collapsed onto a small set of
high-probability code sequences, and the tail result is a symptom of that.

The likely mechanism is not really about semantic IDs: GenRec ranks by P(item | history),
which contains the popularity prior, while SASRec's dot product is unnormalized and carries no
such prior. Swapping atomic for semantic IDs also silently swaps an unnormalized scorer for a
normalized one — and that second change, which nobody set out to make, is doing much of the
damage. It arrived with the architecture the same way Week 4's dropout arrived with the
framework.

Per-level code accuracy (teacher-forced on the true prefix) is 9.8% / 17.9% / 22.4% / 86.3%
across the four levels. Accuracy *rises* with depth as the prefix narrows the choice, and the
content-free disambiguation token is nearly free — so Beauty's 11.78% collision rate is not
the bottleneck it appeared to be. The binding constraint is the first code.

Reproduce: `uv run python -m scripts.compare_atomic_vs_semantic` and
`uv run python -m scripts.diagnose_genrec`.

### Ablations (ML-1M, test set, k=10 — **all rows at 100 epochs**, see REPRODUCTION_LOG.md)

Deltas are against the 100-epoch baseline, and marked against the measured seed-noise floor
(sampled 0.96%, full 3.37% — see below). `~` means the difference is inside seed noise.

| Ablation | sampled HR@10 | Δ | full HR@10 | Δ | avg sec/epoch |
|---|---|---|---|---|---|
| **Baseline (learnable pos emb, maxlen 200)** | **0.8152** | — | **0.2349** | — | 5.85 |
| A1: positional embedding = none | 0.8066 | −1.05% | 0.2291 | ~ −2.47% | 7.47 |
| A1: positional embedding = sinusoidal | 0.8147 | ~ −0.06% | 0.2182 | −7.11% | 6.91 |
| A2: maxlen = 50 | 0.7858 | −3.61% | 0.2033 | −13.45% | 1.53 |
| A2: maxlen = 100 | 0.8058 | −1.15% | 0.2346 | ~ −0.13% | 2.85 |
| A4: negative sampling = popularity-weighted | 0.7540 | −7.51% | 0.1871 | −20.35% | 6.93 |

**This table previously used the 200-epoch headline run as its baseline**, against which every
ablation was also being charged for 100 fewer epochs of training. That budget effect is small
on sampled HR@10 (+0.47%) but large on full HR@10 (+5.36%), and correcting it reverses two
conclusions: A1-none and A2-maxlen100 both looked like real full-ranking regressions
(−7.43% / −5.21%) and are in fact inside seed noise. Read literally, **maxlen 100 and 200 are
indistinguishable on full ranking at this budget** — at less than half the per-epoch cost.

Positional embeddings are the interesting row: dropping them entirely costs ~1% on sampled
HR@10 and nothing detectable on full ranking, while *sinusoidal* embeddings are the one
variant that clearly hurts full ranking (−7.11%). Learnable-vs-none is nearly a wash here;
learnable-vs-sinusoidal is not.

### Seed variance (ML-1M, this repo's SASRec, 5 seeds: 42, 1, 2, 3, 4)

Only the training seed varies — weight init and the training negative sampler. Evaluation
negatives stay frozen (`negatives.json`, seed 42), so this is training noise alone.

| Metric | mean | rel. std | range |
|---|---|---|---|
| sampled HR@10 | 0.8188 | 0.28% | 0.71% |
| sampled NDCG@10 | 0.5925 | 0.34% | 0.83% |
| full HR@10 | 0.2453 | 1.19% | 2.97% |
| full NDCG@10 | 0.1305 | 1.08% | 2.45% |

**Full-ranking metrics are ~4x noisier than sampled ones**, which is worth knowing before
reading any full-ranking comparison in this repo: separating 3,416 items is far more
sensitive to initialization than separating 101.

Comparisons here are between two runs each measured once, so the relevant floor is
2·√2·σ: **0.96% sampled, 3.37% full**. `uv run python -m scripts.seed_variance` prints every
margin claimed in this repo against it. Not a significance test — five seeds estimate σ
loosely — but enough to separate the claims that survive from the ones that do not.

Full table (script-generated, not hand-copied): [`results/tables/master.md`](results/tables/master.md)
A3 sampled-vs-full-ranking scatter: [`results/figures/sampled_vs_full.png`](results/figures/sampled_vs_full.png) — clean
positive correlation within ML-1M, but Beauty's full-ranking score sits well below what
the ML-1M trend would predict, since full-ranking difficulty scales with catalog size
(12,101 items vs. 3,416) independent of model quality — see REPRODUCTION_LOG.md for the
full discussion.

### Semantic IDs (Week 5, input to the TIGER-style model)

Item text → `all-MiniLM-L6-v2` (384-dim) → residual KMeans, 3 levels × 256 codes, plus a 4th
token that disambiguates items landing on an identical 3-token code. ML-1M text is title +
genres, Beauty is title + deepest category path + brand. Metadata coverage is 100% on both
datasets (3,416/3,416 and 12,101/12,101).

| | ML-1M | Beauty |
|---|---|---|
| dead codes (any level) | 0 | 0 |
| collision rate on the 3-token code | 1.46% | 11.78% |
| largest colliding group | 3 | 12 |
| embedding norm explained by 3 tokens | 55.7% | 48.7% |
| within-prefix cosine @ depth 3 (vs. random pair) | 0.753 (0.439) | 0.871 (0.288) |

No dead codes on either dataset — the collapse failure mode RQ-VAE exists to fix does not
appear here, so the RQ-VAE stretch goal stays skipped. Prefix coherence rises monotonically
with depth on both, which is the coarse-to-fine property the generative model depends on.

Two things worth knowing before reading any Week 6 result. **Beauty collides at 11.78%**: for
~1 item in 8, the only thing separating it from a catalog neighbour is a token carrying no
content signal, which caps what semantic IDs can do there. And **ML-1M's codes encode release
year at least as strongly as genre** — items sharing a 2-token prefix are 1.68 years apart on
average against a 15.85-year baseline — because MovieLens titles embed the year in the string.
Nobody chose that; it came in with the text format.

Per-dataset reports with sampled prefix groups:
[`results/tables/semantic_ids_ml-1m.md`](results/tables/semantic_ids_ml-1m.md),
[`results/tables/semantic_ids_beauty.md`](results/tables/semantic_ids_beauty.md).

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
uv run python -m scripts.seed_variance  # noise floor + every claimed margin against it

# Week 5: semantic IDs (needs the extra ~99MB Beauty metadata file)
uv run python -m src.data.download --dest data/raw --dataset beauty --with-meta
uv run python -m src.semantic_ids.embed --dataset ml-1m
uv run python -m src.semantic_ids.rq_kmeans --dataset ml-1m
uv run python -m scripts.inspect_semantic_ids --dataset ml-1m   # quality report
uv run python -m src.train_genrec --config configs/genrec_beauty.yaml

uv run pytest tests/
```

## Repo structure

See [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) Appendix / original plan Appendix A.

## Limitations

Kept current as the project progresses, per EXECUTION_PLAN.md's guiding principle of
reporting negative/mixed results rather than hiding them.

- **Beauty SASRec sits 0.43pp above the accepted reproduction band** (0.5097 vs. the
  0.4654–0.5054 target). Reported as-is rather than tuned until it lands inside the band.
- **Ablations run at 100 epochs, headline numbers at 200.** Deliberate compute saving. The
  ablation table now uses a 100-epoch baseline so both sides of every delta share a budget
  (it previously did not — see the note under that table), but the results are still a
  statement about 100-epoch training. A4's popularity-negatives result in particular may be a
  convergence-speed effect rather than a final ranking.
- **GenRec's full-ranking numbers come from beam search, SASRec's from exhaustive scoring.**
  Beam width 20 is saturated (200 recovers nothing), so the two are comparable in practice,
  but they are not the same procedure and the generative side can only lose from it.
- **The atomic-vs-semantic comparison is not parameter-matched**, and cannot be — see the
  Beauty comparison table. GenRec runs on 13.7% of SASRec's parameters by construction.
- **Seed variance is measured for this repo's SASRec only.** The 0.96% / 3.37% floors come
  from five seeds of one model on ML-1M and are applied to RecBole runs and Beauty results as
  a proxy. Those are different models, frameworks, and datasets; the floors are indicative
  there, not measured.
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
- **Every RecBole number is still a single seed.** Seed variance was measured for this repo's
  SASRec (above) and used as the floor; it confirms the residual SASRec-vs-BERT4Rec margins
  (+0.31% / +0.45%) are inside noise and should be read as a tie, and that the dropout effect
  (+3.71% / +6.33%) is 4–6x the floor. But no RecBole run was repeated across seeds, so the
  floor applied to them is borrowed rather than measured.
