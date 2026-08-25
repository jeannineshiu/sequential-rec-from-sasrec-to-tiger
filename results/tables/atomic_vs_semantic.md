# Atomic vs. semantic IDs — Amazon Beauty, full ranking, test set

Buckets by the target item's training-split frequency (unseen: 0–0, tail: 1–4, torso: 5–19, head: 20+).
Both models rank exhaustively over the whole catalogue, so no beam approximation is
involved on either side. `debiased a=1` subtracts the log training-frequency prior.

| bucket | users | SASRec (atomic) HR@10 | GenRec (semantic) HR@10 | GenRec (semantic), debiased a=1 HR@10 | GenRec (semantic) vs SASRec (atomic) | GenRec (semantic), debiased a=1 vs SASRec (atomic) |
|---|---|---|---|---|---|---|
| unseen | 138 | 0.0000 | 0.0072 | 0.0725 | — | — |
| tail | 4594 | 0.0185 | 0.0026 | 0.0144 | -85.9% | -22.4% |
| torso | 9539 | 0.0427 | 0.0060 | 0.0081 | -86.0% | -81.1% |
| head | 8092 | 0.1033 | 0.0606 | 0.0138 | -41.4% | -86.6% |
| overall | 22363 | 0.0594 | 0.0250 | 0.0118 | -57.8% | -80.0% |

## Significance vs the atomic baseline

Fisher exact on hits/misses per bucket. `p(>)` is one-sided for the semantic model
being better (the cold-start prediction); `p(2)` is two-sided.

| bucket | users | model | hits@10 | baseline hits | p(>) | p(2) |
|---|---|---|---|---|---|---|
| unseen | 138 | GenRec (semantic) | 1 | 0 | 0.5000 | 1.0000 |
| unseen | 138 | GenRec (semantic), debiased a=1 | 10 | 0 | 0.0008 | 0.0016 |
| tail | 4594 | GenRec (semantic) | 12 | 85 | 1.0000 | 0.0000 |
| tail | 4594 | GenRec (semantic), debiased a=1 | 66 | 85 | 0.9498 | 0.1394 |
| torso | 9539 | GenRec (semantic) | 57 | 407 | 1.0000 | 0.0000 |
| torso | 9539 | GenRec (semantic), debiased a=1 | 77 | 407 | 1.0000 | 0.0000 |
| head | 8092 | GenRec (semantic) | 490 | 836 | 1.0000 | 0.0000 |
| head | 8092 | GenRec (semantic), debiased a=1 | 112 | 836 | 1.0000 | 0.0000 |
| overall | 22363 | GenRec (semantic) | 560 | 1328 | 1.0000 | 0.0000 |
| overall | 22363 | GenRec (semantic), debiased a=1 | 265 | 1328 | 1.0000 | 0.0000 |

![cold start buckets](../figures/cold_start_buckets.png)
