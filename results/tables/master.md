# Master results table

Generated from MLflow (`sqlite:///mlflow.db`, experiment `sequential-rec`) via `uv run python -m src.export_results` -- do not hand-edit.

| run                                  | dataset       |   maxlen | pos_emb_type   | neg_sampling   |   avg_epoch_time_sec |   epochs_trained |   sampled_HR@10 |   sampled_NDCG@10 |   full_HR@10 |   full_NDCG@10 |
|:-------------------------------------|:--------------|---------:|:---------------|:---------------|---------------------:|-----------------:|----------------:|------------------:|-------------:|---------------:|
| ablation_ml1m_maxlen100              | ml-1m         |      100 |                |                |               2.8457 |         100.0000 |          0.8058 |            0.5762 |       0.2346 |         0.1228 |
| ablation_ml1m_maxlen50               | ml-1m         |       50 |                |                |               1.5268 |         100.0000 |          0.7858 |            0.5539 |       0.2033 |         0.1080 |
| ablation_ml1m_negsampling_popularity | ml-1m         |      200 |                | popularity     |               6.9264 |         100.0000 |          0.7540 |            0.5225 |       0.1871 |         0.0995 |
| ablation_ml1m_posemb_none            | ml-1m         |      200 | none           |                |               7.4679 |         100.0000 |          0.8066 |            0.5707 |       0.2291 |         0.1222 |
| ablation_ml1m_posemb_sinusoidal      | ml-1m         |      200 | sinusoidal     |                |               6.9149 |         100.0000 |          0.8147 |            0.5763 |       0.2182 |         0.1134 |
| bert4rec_recbole_1x                  | ml-1m         |          |                |                |             104.1169 |         200.0000 |          0.8031 |            0.6036 |     nan      |       nan      |
| bpr_mf_ml1m                          | ml-1m         |          |                |                |             nan      |         nan      |          0.5745 |            0.3357 |       0.0671 |         0.0333 |
| popularity_ml1m                      | ml-1m         |          |                |                |             nan      |         nan      |          0.4363 |            0.2401 |       0.0369 |         0.0180 |
| sasrec_beauty                        | amazon-beauty |       50 |                |                |             nan      |         nan      |          0.5097 |            0.3453 |       0.0594 |         0.0303 |
| sasrec_ml1m                          | ml-1m         |      200 |                |                |             nan      |         nan      |          0.8190 |            0.5948 |       0.2475 |         0.1322 |
