# Popularity-debiased decoding — GenRec, Amazon Beauty, exhaustive full ranking

Every one of the catalogue's items scored for every user, so there is no beam
approximation here at all. `alpha` divides out the add-one-smoothed training-frequency
prior: 0 is the model as trained, 1 is pointwise mutual information.

| alpha | HR@10 | NDCG@10 | unseen HR@10 | tail HR@10 | torso HR@10 | head HR@10 | distinct items in top-10 |
|---|---|---|---|---|---|---|---|
| 0 | 0.0251 | 0.0131 | 0.0072 | 0.0026 | 0.0060 | 0.0608 | 1,749 |
| 0.25 | 0.0232 | 0.0121 | 0.0072 | 0.0048 | 0.0073 | 0.0526 | 1,838 |
| 0.5 | 0.0198 | 0.0097 | 0.0145 | 0.0065 | 0.0089 | 0.0402 | 1,974 |
| 0.75 | 0.0155 | 0.0072 | 0.0362 | 0.0122 | 0.0084 | 0.0255 | 2,021 |
| 1 | 0.0117 | 0.0053 | 0.0725 | 0.0141 | 0.0080 | 0.0137 | 1,976 |
