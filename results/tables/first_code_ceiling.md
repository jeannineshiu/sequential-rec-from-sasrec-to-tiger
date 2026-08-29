# What the first code costs — amazon-beauty, full ranking, test set, k=10

Oracle depth *d* hands the model the target's true first *d* codes and ranks only the
items sharing that prefix. `d=0` is the model as it actually runs. The candidate column
is the median number of items still competing, so the size of the hint stays visible.
**Not comparable to SASRec** — SASRec gets no oracle.

| oracle depth | median candidates | HR@10 | NDCG@10 | vs d=0 |
|---|---|---|---|---|
| 0 (as trained) | 12,096 | 0.0251 | 0.0131 | — |
| 1 | 51 | 0.3117 | 0.1627 | +1140% |
| 2 | 2 | 0.9880 | 0.7767 | +3831% |
| 3 | 1 | 1.0000 | 0.9458 | +3879% |

Handing over the level-1 code alone multiplies HR@10 by **12.4x** (0.0251 -> 0.3117).

But it does not rescue the model: with the right region and a median of 51 candidates left, **69% of targets still miss the top 10**. Level 2 is where retrieval actually becomes reliable (0.9880 from a median of 2 candidates). So the first code is not a lone bottleneck -- selecting the region and ranking within it are both weak.

## Is the model near the right first code, or nowhere near it?

Rank of the true level-1 code among all 256, from the model's own logits:

| true level-1 code in top-n | share of users |
|---|---|
| 1 | 9.8% |
| 5 | 24.2% |
| 10 | 33.4% |
| 25 | 49.1% |
| 64 | 69.9% |
| 128 | 86.9% |

Median rank of the true level-1 code: **26** of 256 (random would be 128), so the model localizes the region far better than chance while rarely nailing it.

### Unaided HR@10, split by how well the model placed the level-1 code

| level-1 outcome | users | HR@10 (no oracle) |
|---|---|---|
| top-1 correct | 2,182 | 0.0907 |
| in top-10 | 5,291 | 0.0558 |
| in top-64 | 8,154 | 0.0083 |
| outside top-64 | 6,736 | 0.0001 |

Even when the model's own first choice of level-1 code is correct, unaided HR@10 is only **0.0907** -- the residual loss is inside the region, not in reaching it.

Reproduce: `uv run python -m scripts.first_code_ceiling`
