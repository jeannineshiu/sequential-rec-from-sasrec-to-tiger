# GenRec seed spread — Amazon Beauty, exhaustive full ranking

Every row is one training run of the same configuration, differing only in
`train.seed`, scored through the exhaustive pass the headline table uses. The
evaluation negatives and the frequency buckets are frozen, so what moves here is
training noise and nothing else.

| run | GenRec (semantic) HR@10 | GenRec (semantic), debiased a=1 HR@10 | GenRec (semantic) NDCG@10 | GenRec (semantic), debiased a=1 NDCG@10 |
|---|---|---|---|---|
| `genrec_beauty` | 0.0251 | 0.0117 | 0.0131 | 0.0053 |
| `genrec_beauty_seed1` | 0.0199 | 0.0104 | 0.0108 | 0.0050 |
| `genrec_beauty_seed2` | 0.0258 | 0.0111 | 0.0133 | 0.0054 |

| metric | mean | std | rel. std | min | max |
|---|---|---|---|---|---|
| full HR@10 | 0.0236 | 0.0032 | 13.57% | 0.0199 | 0.0258 |
| full NDCG@10 | 0.0124 | 0.0014 | 11.28% | 0.0108 | 0.0133 |

## Hits per bucket, per seed

Counts, not rates, because the cold-start claim is quoted as counts and the
Fisher tests run on them.

| run | model | unseen (138) | tail (4594) | torso (9539) | head (8092) |
|---|---|---|---|---|---|
| `genrec_beauty` | GenRec (semantic) | 1 | 12 | 57 | 492 |
| `genrec_beauty` | GenRec (semantic), debiased a=1 | 10 | 65 | 76 | 111 |
| `genrec_beauty_seed1` | GenRec (semantic) | 0 | 14 | 47 | 385 |
| `genrec_beauty_seed1` | GenRec (semantic), debiased a=1 | 7 | 70 | 71 | 85 |
| `genrec_beauty_seed2` | GenRec (semantic) | 0 | 22 | 51 | 504 |
| `genrec_beauty_seed2` | GenRec (semantic), debiased a=1 | 8 | 82 | 80 | 78 |

## What gets recommended, per seed

Distinct items across every user's top-10, and the share of those
recommendations that are items the training split never contained.

| run | model | distinct items in top-10 | % unseen |
|---|---|---|---|
| `genrec_beauty` | GenRec (semantic) | 1,749 | 0.04% |
| `genrec_beauty` | GenRec (semantic), debiased a=1 | 2,084 | 1.79% |
| `genrec_beauty_seed1` | GenRec (semantic) | 2,329 | 0.05% |
| `genrec_beauty_seed1` | GenRec (semantic), debiased a=1 | 2,462 | 1.28% |
| `genrec_beauty_seed2` | GenRec (semantic) | 1,330 | 0.00% |
| `genrec_beauty_seed2` | GenRec (semantic), debiased a=1 | 1,605 | 1.84% |
