# GenRec seed spread — MovieLens-1M, exhaustive full ranking

Every row is one training run of the same configuration, differing only in
`train.seed`, scored through the exhaustive pass the headline table uses. The
evaluation negatives and the frequency buckets are frozen, so what moves here is
training noise and nothing else.

| run | GenRec (semantic) HR@10 | GenRec (semantic), debiased a=1 HR@10 | GenRec (semantic) NDCG@10 | GenRec (semantic), debiased a=1 NDCG@10 |
|---|---|---|---|---|
| `genrec_ml1m` | 0.1164 | 0.0598 | 0.0607 | 0.0286 |
| `genrec_ml1m_seed1` | 0.1124 | 0.0475 | 0.0569 | 0.0203 |
| `genrec_ml1m_seed2` | 0.1205 | 0.0598 | 0.0627 | 0.0271 |

| metric | mean | std | rel. std | min | max |
|---|---|---|---|---|---|
| full HR@10 | 0.1164 | 0.0041 | 3.48% | 0.1124 | 0.1205 |
| full NDCG@10 | 0.0601 | 0.0029 | 4.87% | 0.0569 | 0.0627 |

## Hits per bucket, per seed

Counts, not rates, because the cold-start claim is quoted as counts and the
Fisher tests run on them.

| run | model | unseen (0) | tail (2) | torso (48) | head (5990) |
|---|---|---|---|---|---|
| `genrec_ml1m` | GenRec (semantic) | 0 | 0 | 0 | 703 |
| `genrec_ml1m` | GenRec (semantic), debiased a=1 | 0 | 0 | 3 | 358 |
| `genrec_ml1m_seed1` | GenRec (semantic) | 0 | 0 | 0 | 679 |
| `genrec_ml1m_seed1` | GenRec (semantic), debiased a=1 | 0 | 0 | 4 | 283 |
| `genrec_ml1m_seed2` | GenRec (semantic) | 0 | 0 | 0 | 728 |
| `genrec_ml1m_seed2` | GenRec (semantic), debiased a=1 | 0 | 0 | 4 | 357 |

## What gets recommended, per seed

Distinct items across every user's top-10, and the share of those
recommendations that are items the training split never contained.

| run | model | distinct items in top-10 | % unseen |
|---|---|---|---|
| `genrec_ml1m` | GenRec (semantic) | 838 | 0.00% |
| `genrec_ml1m` | GenRec (semantic), debiased a=1 | 1,059 | 0.00% |
| `genrec_ml1m_seed1` | GenRec (semantic) | 729 | 0.00% |
| `genrec_ml1m_seed1` | GenRec (semantic), debiased a=1 | 919 | 0.00% |
| `genrec_ml1m_seed2` | GenRec (semantic) | 854 | 0.00% |
| `genrec_ml1m_seed2` | GenRec (semantic), debiased a=1 | 1,025 | 0.00% |
