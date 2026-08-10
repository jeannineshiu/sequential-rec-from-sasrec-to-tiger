# Popularity-debiased decoding — GenRec, Amazon Beauty, exhaustive full ranking

Every one of the catalogue's items scored for every user, so there is no beam
approximation here at all. `alpha` divides out the add-one-smoothed training-frequency
prior: 0 is the model as trained, 1 is pointwise mutual information.

| alpha | HR@10 | NDCG@10 | unseen HR@10 | tail HR@10 | torso HR@10 | head HR@10 | distinct items in top-10 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0391 | 0.0233 | 0.0000 | 0.0000 | 0.0000 | 0.1136 | 332 |
| 0.25 | 0.0391 | 0.0281 | 0.0000 | 0.0000 | 0.0000 | 0.1136 | 345 |
| 0.5 | 0.0234 | 0.0234 | 0.0000 | 0.0000 | 0.0000 | 0.0682 | 375 |
| 0.75 | 0.0234 | 0.0186 | 0.0000 | 0.0000 | 0.0000 | 0.0682 | 366 |
| 1 | 0.0234 | 0.0119 | 0.0000 | 0.0000 | 0.0000 | 0.0682 | 344 |
