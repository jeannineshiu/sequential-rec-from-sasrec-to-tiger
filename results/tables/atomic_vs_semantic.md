# Atomic vs. semantic IDs — Amazon Beauty, full ranking, test set

Buckets by the target item's training-split frequency (unseen: 0–0, tail: 1–4, torso: 5–19, head: 20+).
SASRec ranks exhaustively; GenRec ranks by constrained beam search (beam 20),
which can only cost the generative side — see the beam-sensitivity table in the log.

| bucket | users | SASRec (atomic) HR@10 | SASRec (atomic) NDCG@10 | GenRec (semantic) HR@10 | GenRec (semantic) NDCG@10 | HR@10 rel. |
|---|---|---|---|---|---|---|
| unseen | 138 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | — |
| tail | 4594 | 0.0185 | 0.0108 | 0.0022 | 0.0008 | -88.2% |
| torso | 9539 | 0.0427 | 0.0224 | 0.0085 | 0.0035 | -80.1% |
| head | 8092 | 0.1033 | 0.0512 | 0.0797 | 0.0418 | -22.8% |
| overall | 22363 | 0.0594 | 0.0303 | 0.0329 | 0.0168 | -44.6% |

![cold start buckets](../figures/cold_start_buckets.png)
