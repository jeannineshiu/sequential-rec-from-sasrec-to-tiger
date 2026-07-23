# Sequential Recommendation: From SASRec to TIGER

Reproduce a field-standard sequential recommender (SASRec), independently verify the
BERT4Rec reproducibility controversy under a training-budget-controlled comparison, then
rebuild the same codebase as a TIGER-style generative recommender with RQ-quantized
semantic IDs — walking through 2018→2025 sequential recommendation in one repo.

Full plan: [`sequential-rec-project-plan.md`](sequential-rec-project-plan.md)
Execution checklist: [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)
Debugging trail: [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md)

## Status: Week 3 done (M3 milestone met) — Week 4 next

## Reproduction table (updated as results land)

### ML-1M, sampled protocol (1 positive + 100 uniform random negatives), test set, k=10

| Model | HR@10 | NDCG@10 | Note |
|---|---|---|---|
| Popularity | 0.4363 | 0.2401 | floor |
| BPR-MF (implicit) | 0.5745 | 0.3357 | floor |
| SASRec (paper, Kang & McAuley 2018) | 0.8245 | 0.5905 | target |
| **SASRec (this repo)** | **0.8190** | **0.5948** | ✅ within target range (0.80–0.83 / 0.57–0.60) |
| SASRec (RecBole, cross-val) | — | — | deferred, see REPRODUCTION_LOG.md |
| BERT4Rec (RecBole) | — | — | Week 4 |

### ML-1M, full ranking (rank against entire catalog, excl. history), test set, k=10

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0369 | 0.0180 |
| BPR-MF (implicit) | 0.0671 | 0.0333 |
| SASRec (this repo) | 0.2475 | 0.1322 |

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
  once (seed=42) and reused by every model so comparisons are apples-to-apples.
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

(To be filled in honestly as the project progresses — see EXECUTION_PLAN.md's guiding
principle of reporting negative/mixed results rather than hiding them.)
