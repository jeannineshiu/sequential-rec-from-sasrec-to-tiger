# Master results table

Generated from MLflow (`sqlite:///mlflow.db`, experiment `sequential-rec`) via `uv run python -m src.export_results` -- do not hand-edit.

| run                                     | dataset       |   maxlen | pos_emb_type   | neg_sampling   |   avg_epoch_time_sec |   epochs_trained |   sampled_HR@10 |   sampled_NDCG@10 |   full_HR@10 |   full_NDCG@10 |
|:----------------------------------------|:--------------|---------:|:---------------|:---------------|---------------------:|-----------------:|----------------:|------------------:|-------------:|---------------:|
| ablation_ml1m_baseline_100ep            | ml-1m         |      200 |                |                |               5.8544 |         100.0000 |          0.8152 |            0.5859 |       0.2349 |         0.1252 |
| ablation_ml1m_ce_batch19                | ml-1m         |      200 |                |                |              13.9646 |         116.0000 |          0.8166 |            0.6114 |       0.3023 |         0.1741 |
| ablation_ml1m_loss_ce                   | ml-1m         |      200 |                |                |              16.8590 |         200.0000 |          0.8159 |            0.6133 |       0.3033 |         0.1749 |
| ablation_ml1m_maxlen100                 | ml-1m         |      100 |                |                |               2.8457 |         100.0000 |          0.8058 |            0.5762 |       0.2346 |         0.1228 |
| ablation_ml1m_maxlen50                  | ml-1m         |       50 |                |                |               1.5268 |         100.0000 |          0.7858 |            0.5539 |       0.2033 |         0.1080 |
| ablation_ml1m_negsampling_popularity    | ml-1m         |      200 |                | popularity     |               6.9264 |         100.0000 |          0.7540 |            0.5225 |       0.1871 |         0.0995 |
| ablation_ml1m_posemb_none               | ml-1m         |      200 | none           |                |               7.4679 |         100.0000 |          0.8066 |            0.5707 |       0.2291 |         0.1222 |
| ablation_ml1m_posemb_sinusoidal         | ml-1m         |      200 | sinusoidal     |                |               6.9149 |         100.0000 |          0.8147 |            0.5763 |       0.2182 |         0.1134 |
| bert4rec_recbole_1x                     | ml-1m         |          |                |                |             104.1169 |         200.0000 |          0.8031 |            0.6036 |     nan      |       nan      |
| bpr_mf_ml1m                             | ml-1m         |          |                |                |             nan      |         nan      |          0.5745 |            0.3357 |       0.0671 |         0.0333 |
| genrec_beauty                           | amazon-beauty |       50 |                |                |              37.7224 |         100.0000 |          0.3621 |            0.2235 |       0.0329 |         0.0168 |
| popularity_ml1m                         | ml-1m         |          |                |                |             nan      |         nan      |          0.4363 |            0.2401 |       0.0369 |         0.0180 |
| sasrec_beauty                           | amazon-beauty |       50 |                |                |             nan      |         nan      |          0.5097 |            0.3453 |       0.0594 |         0.0303 |
| sasrec_ml1m                             | ml-1m         |      200 |                |                |             nan      |         nan      |          0.8190 |            0.5948 |       0.2475 |         0.1322 |
| sasrec_ml1m_seed1                       | ml-1m         |      200 |                |                |               6.8701 |         200.0000 |          0.8202 |            0.5932 |       0.2455 |         0.1291 |
| sasrec_ml1m_seed2                       | ml-1m         |      200 |                |                |               6.8272 |         140.0000 |          0.8192 |            0.5899 |       0.2465 |         0.1313 |
| sasrec_ml1m_seed3                       | ml-1m         |      200 |                |                |               6.9172 |         181.0000 |          0.8149 |            0.5936 |       0.2467 |         0.1310 |
| sasrec_ml1m_seed4                       | ml-1m         |      200 |                |                |               6.1889 |         179.0000 |          0.8207 |            0.5910 |       0.2402 |         0.1290 |
| sasrec_recbole_1x                       | ml-1m         |          |                |                |              84.2353 |         200.0000 |          0.7768 |            0.5702 |     nan      |       nan      |
| sasrec_recbole_1x_dropout02             | ml-1m         |          |                |                |              84.1664 |         200.0000 |          0.8056 |            0.6063 |     nan      |       nan      |
| sasrec_recbole_1x_dropout02_ourprotocol | ml-1m         |          |                |                |             nan      |         nan      |          0.8240 |            0.6389 |       0.3467 |         0.2029 |
