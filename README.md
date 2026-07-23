# Sequential Recommendation: From SASRec to TIGER

Reproduce a field-standard sequential recommender (SASRec), independently verify the
BERT4Rec reproducibility controversy under a training-budget-controlled comparison, then
rebuild the same codebase as a TIGER-style generative recommender with RQ-quantized
semantic IDs — walking through 2018→2025 sequential recommendation in one repo.

Full plan: [`sequential-rec-project-plan.md`](sequential-rec-project-plan.md)
Execution checklist: [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)
Debugging trail: [`REPRODUCTION_LOG.md`](REPRODUCTION_LOG.md)

## Status: Week 1 in progress

## Reproduction table (updated as results land)

### ML-1M, sampled protocol (1 positive + 100 uniform random negatives), test set, k=10

| Model | HR@10 | NDCG@10 | Note |
|---|---|---|---|
| Popularity | 0.4363 | 0.2401 | floor |
| BPR-MF (implicit) | 0.5745 | 0.3357 | floor |
| SASRec (paper, Kang & McAuley 2018) | 0.8245 | 0.5905 | target |
| SASRec (this repo) | — | — | Week 2 |
| SASRec (RecBole, cross-val) | — | — | Week 2 |
| BERT4Rec (RecBole) | — | — | Week 4 |

### ML-1M, full ranking (rank against entire catalog, excl. history), test set, k=10

| Model | HR@10 | NDCG@10 |
|---|---|---|
| Popularity | 0.0369 | 0.0180 |
| BPR-MF (implicit) | 0.0671 | 0.0333 |
| SASRec (this repo) | — | — |

### Amazon Beauty / TIGER-style comparison

Pending Week 3 / Week 6.

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
uv run python -m src.data.download --dest data/raw
uv run python -m src.data.preprocess --raw-dir data/raw/ml-1m --out-dir data/processed/ml-1m
uv run python -m src.baselines --data-dir data/processed/ml-1m
uv run pytest tests/
```

## Repo structure

See [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) Appendix / original plan Appendix A.

## Limitations

(To be filled in honestly as the project progresses — see EXECUTION_PLAN.md's guiding
principle of reporting negative/mixed results rather than hiding them.)
