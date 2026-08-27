# Atomic vs. semantic IDs — MovieLens-1M, full ranking, test set

Buckets by the target item's training-split frequency (unseen: 0–0, tail: 1–4, torso: 5–19, head: 20+).
Both models rank exhaustively over the whole catalogue, so no beam approximation is
involved on either side. `debiased a=1` subtracts the log training-frequency prior.

| bucket | users | SASRec (atomic) HR@10 | GenRec (semantic) HR@10 | GenRec (semantic), debiased a=1 HR@10 | GenRec (semantic) vs SASRec (atomic) | GenRec (semantic), debiased a=1 vs SASRec (atomic) |
|---|---|---|---|---|---|---|
| unseen | 0 | nan | nan | nan | — | — |
| tail | 2 | 0.0000 | 0.0000 | 0.0000 | — | — |
| torso | 48 | 0.0000 | 0.0000 | 0.0625 | — | — |
| head | 5990 | 0.2496 | 0.1174 | 0.0598 | -53.0% | -76.1% |
| overall | 6040 | 0.2475 | 0.1164 | 0.0598 | -53.0% | -75.9% |

## Significance vs the atomic baseline

Fisher exact on hits/misses per bucket. `p(>)` is one-sided for the semantic model
being better (the cold-start prediction); `p(2)` is two-sided.

| bucket | users | model | hits@10 | baseline hits | p(>) | p(2) |
|---|---|---|---|---|---|---|
| tail | 2 | GenRec (semantic) | 0 | 0 | 1.0000 | 1.0000 |
| tail | 2 | GenRec (semantic), debiased a=1 | 0 | 0 | 1.0000 | 1.0000 |
| torso | 48 | GenRec (semantic) | 0 | 0 | 1.0000 | 1.0000 |
| torso | 48 | GenRec (semantic), debiased a=1 | 3 | 0 | 0.1211 | 0.2421 |
| head | 5990 | GenRec (semantic) | 703 | 1495 | 1.0000 | 0.0000 |
| head | 5990 | GenRec (semantic), debiased a=1 | 358 | 1495 | 1.0000 | 0.0000 |
| overall | 6040 | GenRec (semantic) | 703 | 1495 | 1.0000 | 0.0000 |
| overall | 6040 | GenRec (semantic), debiased a=1 | 361 | 1495 | 1.0000 | 0.0000 |

![cold start buckets](../figures/cold_start_buckets_ml-1m.png)
