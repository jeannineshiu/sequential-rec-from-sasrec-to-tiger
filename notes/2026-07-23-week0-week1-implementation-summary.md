Week 0 與 Week 1 的大部分工作已完成並驗證。摘要如下:

**已完成:**
- 用 `uv` 建好專案骨架(`pyproject.toml`,依賴:torch/numpy/pandas/scipy/mlflow/implicit + 開發工具),已確認 MPS(Apple GPU)可用
- 完整目錄骨架(`src/{data,models,semantic_ids,eval}`、`configs/`、`tests/`、`results/`、`docs/`、`serving/`)
- `src/data/download.py` + `preprocess.py`:MD5 校驗的 ML-1M 下載、5-core 過濾、leave-one-out split、無洩漏 assertion — 實際跑過,統計數字幾乎完全對齊 Kang & McAuley (2018) 論文 Table 1(6040 users / 3416 items / 平均序列長 165.5)
- `src/data/dataset.py`:PyTorch 訓練用 Dataset,含左側 padding 與 shifted-target 建構
- `src/eval/{metrics,sampled,full_ranking}.py`:雙軌評估器(sampled 1正+100負協定,以及 full-ranking),加上固定 seed 的負樣本產生,供之後所有模型共用
- 16 個單元測試(在 toy 資料上手算 HR@10/NDCG@10 驗證)— 全數通過;`ruff` 與 `black` 皆乾淨
- Popularity + BPR-MF baseline(`src/baselines.py`),已在真實 ML-1M 資料上實際跑過,結果記錄進 MLflow(sqlite backend):

| Model      | sampled HR@10 | sampled NDCG@10 | full HR@10 | full NDCG@10 |
| ---------- | ------------- | --------------- | ---------- | ------------ |
| Popularity | 0.4363        | 0.2401          | 0.0369     | 0.0180       |
| BPR-MF     | 0.5745        | 0.3357          | 0.0671     | 0.0333       |

- `REPRODUCTION_LOG.md` 與 `README.md` 已用真實數字起頭,而非佔位符

**有一點值得注意:** 這個資料夾本身不是獨立的 git repo — `git rev-parse --show-toplevel` 顯示 repo 根目錄是你整個 home 目錄(`~/.git`),而且 `git status` 目前列出一堆與舊 Android 專案相關、不相干的已刪除檔案。我沒有動到那些東西,只有在這個資料夾範圍內檢查狀態,但你之後照計畫要推上 GitHub 前,可能會想幫這個專案獨立建一個 repo。

**Week 1 尚未完成的部分:** 用 RecBole 或官方 repo 跑一次 SASRec,取得第三方參考數字,供 Week 2 三角驗證使用 — 這是我留下來的唯一未做項目,因為它需要額外安裝工具,不屬於這個 repo 內的程式碼。請告訴我要不要現在處理這項,或直接進入 Week 2(從零實作 SASRec)。
