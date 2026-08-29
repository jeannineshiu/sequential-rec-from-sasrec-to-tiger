# What the two models actually recommend — Amazon Beauty, test top-10

Both models ranked exhaustively over the whole catalogue, so no beam pruning is
constraining what can appear in a top-10 on either side.

| model | distinct items in all top-10s | median train freq | mean | % head | % torso | % tail | % unseen |
|---|---|---|---|---|---|---|---|
| SASRec (atomic) | 9,221 | 22 | 42.7 | 54.2% | 42.1% | 3.7% | 0.0% |
| GenRec (semantic) | 1,749 | 63 | 97.3 | 74.1% | 21.3% | 4.6% | 0.0% |
| GenRec (semantic), debiased a=1 | 2,084 | 5 | 10.1 | 8.1% | 50.3% | 39.8% | 1.8% |

Catalogue size: 12,101 items.

## GenRec per-level code accuracy (teacher-forced on the true prefix)

| level | argmax accuracy | codebook size |
|---|---|---|
| 1 | 0.0976 | 256 |
| 2 | 0.1788 | 256 |
| 3 | 0.2243 | 256 |
| 4 | 0.8627 | 12 |

All four correct if independent: 0.00338 (not the retrieval rate -- levels are not independent -- but it bounds how much compounding costs).

