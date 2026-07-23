# Sequential Recommendation 復現與現代化專案 — 完整執行計畫

**專案名稱(建議 repo 名):** `sequential-rec-from-sasrec-to-tiger`
**時程:** 6 週(每週約 12–15 小時)
**核心敘事:** 從 field-standard baseline(SASRec)→ 復現爭議驗證(BERT4Rec)→ 現代生成式範式(TIGER-style Semantic ID),完整走過 2018–2025 推薦系統演進。

---

## 0. 專案目標與成功標準

### 三個交付物
1. **GitHub repo** — 可復現、有對照表、有 ablation 的完整實作
2. **Medium 文章** — 標題方向:「I Reproduced SASRec, Verified the BERT4Rec Controversy, Then Rebuilt It as a Generative Recommender」
3. **面試話術包** — 一句話定位 + 三層深度的技術問答準備

### 量化成功標準
| 里程碑 | 指標 | 目標 |
|---|---|---|
| SASRec 復現 (ML-1M, sampled) | HR@10 | 0.80–0.83(論文 0.8245) |
| SASRec 復現 (ML-1M, sampled) | NDCG@10 | 0.57–0.60(論文 0.5905) |
| RecBole 交叉驗證 | 與自實作差距 | < 2% 相對差 |
| TIGER-style (Amazon Beauty) | Recall@10 | 論文 ~0.0648 的 ±15% 內即可 |
| Cold-start 實驗 | Semantic ID vs Atomic ID 差距 | 有明確量化結論(方向不預設) |

---

## Phase 1:基礎建設與 SASRec 復現(Week 1–3)

### Week 1 — Data Pipeline 與評估框架(先寫 eval,再寫模型)

**原則:評估協定是復現成敗的最大變因,必須第一週就凍結。**

**Day 1–2:環境與資料**
- 建 repo 骨架(結構見附錄 A)、poetry/uv 管理依賴、pre-commit(black + ruff)
- 下載 MovieLens-1M;寫 data pipeline:
  - 過濾:保留 rating 作為 implicit feedback(全部視為正樣本,標準做法)
  - user grouping → 按 timestamp 排序 → 5-core filtering(user/item 至少 5 互動)
  - Leave-one-out split:每 user 最後一個 item 為 test、倒數第二為 valid、其餘為 train
  - 序列處理:maxlen=200(ML-1M 標準),左側 padding

**Day 3–4:雙軌評估器**
- **Sampled 版(對齊原論文):** 每 user 取 1 個正樣本 + 100 個 uniform random negatives,計算 HR@10 / NDCG@10
- **Full ranking 版(現代標準):** 對全 item catalog 排序(排除 user 已互動 item),同指標
- 兩版都寫,README 明確討論 Krichene & Rendle (2020) 的 sampled metrics 偏差問題 — 這是你的差異化重點之一
- 單元測試:手工構造 3-user 小資料集,人工驗算指標正確性

**Day 5–7:Baselines**
- Popularity baseline(全域最熱門 top-K)
- BPR-MF(用 `implicit` 套件,不用自己寫)
- SASRec 官方 repo 或 RecBole 先跑一次,記下參考數字 — 之後驗證自己的實作用
- MLflow 設定:沿用你 electricity-price-forecasting 的 tracking setup,每個實驗記 config + metrics + git hash

**Week 1 產出:** 資料 pipeline + 經測試的評估器 + 2 個 baseline 數字 + MLflow 就緒

---

### Week 2 — SASRec from scratch(PyTorch)

**Day 1–3:模型實作**

架構元件(對照原論文 Kang & McAuley 2018):
- Item embedding(d=50 for ML-1M)+ **learnable** positional embedding(非 sinusoidal)
- N=2 個 self-attention blocks,每個:causal masked multi-head attention(1 head)→ point-wise FFN → residual + LayerNorm + dropout
- Prediction layer:**shared item embedding**(輸出層與輸入 embedding 共享權重)
- 訓練目標:每個位置預測下一個 item,binary cross-entropy,每個正樣本配 1 個 random negative

關鍵超參數(ML-1M):
| 參數 | 值 |
|---|---|
| maxlen | 200 |
| hidden dim | 50 |
| num blocks | 2 |
| num heads | 1 |
| dropout | 0.2 |
| lr | 0.001 (Adam, β2=0.98) |
| batch size | 128 |
| epochs | ~200(早停:valid NDCG@10, patience 20) |

**Day 4–5:訓練與 debug**

常見坑清單(先知道,省 debug 時間):
- Causal mask 方向錯誤 → 模型「偷看未來」,valid 指標異常高
- Padding position 沒有 mask 掉 → attention 對 padding 分配權重
- 評估時 negative sampling 不小心包含 user 的訓練 item
- Positional embedding 超過 maxlen 的索引錯誤
- BCE loss 中 padding 位置沒有排除

**Day 6–7:對齊論文數字**
- 若差距 > 2pp:依序檢查 (1) 評估協定 (2) negative sampling (3) 早停時機 (4) dropout
- 用 Week 1 記下的官方 repo 數字三角驗證:官方 repo vs 你的實作 vs 論文
- 記錄一份 `REPRODUCTION_LOG.md`:每次嘗試、假設、結果 — 這份 log 本身就是 Medium 素材

**Week 2 產出:** 自實作 SASRec 在 ML-1M 達標;REPRODUCTION_LOG.md

---

### Week 3 — 第二資料集 + Ablation

**Day 1–2:Amazon Beauty**
- 下載 Amazon Review dataset (Beauty 子集,5-core)
- 跑通 pipeline(maxlen 改 50,hidden dim 可試 64)
- 論文參考:HR@10 ≈ 0.4854(sampled)— sparse dataset 上模型行為與 ML-1M 明顯不同

**Day 3–5:Ablation studies(每個都是面試素材)**
1. Positional embedding:有 vs 無 vs sinusoidal — 驗證論文結論(dense data 上 learnable PE 重要)
2. maxlen:50 / 100 / 200 — 序列長度 vs 效果 vs 訓練時間曲線
3. Sampled vs full ranking:量化兩種協定的排名差距與相關性(做一張 scatter plot:每個 checkpoint 的兩種指標)
4. Negative sampling:uniform vs popularity-based(訓練時)

**Day 6–7:整理**
- 所有實驗結果進一張 master table
- 畫圖:訓練曲線、ablation 對照(matplotlib,存 `results/figures/`)

**Week 3 產出:** 兩個 dataset 上的完整結果 + 4 組 ablation

---

## Phase 2:BERT4Rec 對照與復現爭議驗證(Week 4)

**策略調整:不從零實作 BERT4Rec(投資報酬率低),改用 RecBole 做嚴謹對照實驗,聚焦「復現爭議」這個高價值敘事。**

**Day 1–2:文獻與設定**
- 精讀 Petrov & Macdonald (2022) "A Systematic Review and Replicability Study of BERT4Rec" — 核心論點:原論文的 BERT4Rec 訓練嚴重不足,充分訓練後與 SASRec 差距縮小甚至反轉
- RecBole 設定 BERT4Rec:mask ratio 0.2、與你的 SASRec 相同的評估協定(這點最關鍵 — 協定不同的比較毫無意義)

**Day 3–5:核心實驗 — 訓練時長對照**
- BERT4Rec 訓練:原論文預設 epochs vs 4x vs 10x
- SASRec(你的實作)同樣延長訓練對照
- 產出本專案的招牌圖:**x 軸 = 訓練時間,y 軸 = NDCG@10,兩模型多條曲線**
- 同時用 RecBole 跑 SASRec,與你的自實作交叉驗證(< 2% 差距 = 實作正確的第三方證據)

**Day 6–7:分析寫作**
- 寫 `docs/bert4rec-controversy.md`:你的數據支持/不支持 Petrov & Macdonald 的哪些結論
- 誠實原則:如果你的結果與文獻不一致,照實寫並分析可能原因 — 這比「完美復現」更有說服力

**Week 4 產出:** 招牌對照圖 + 交叉驗證通過 + 爭議分析文件

---

## Phase 3:TIGER-style Semantic ID 生成式推薦(Week 5–6)

**這是專案從 2019 跳到 2025 的關鍵,對應 Google TIGER (NeurIPS 2023) / Meta HSTU (2024) 的 generative recommendation 範式。**

### Week 5 — Semantic ID 構建 + 模型改造

**Day 1–2:Content embedding**
- 用 sentence-transformers(如 `all-MiniLM-L6-v2`)對 item 文字(ML-1M:title + genres;Beauty:title + category + brand)編碼
- 存成 item_id → 384-dim embedding 的 lookup

**Day 3–4:量化(兩條路,先易後難)**
- **v1(先做):RQ-KMeans** — Spotify RecSys 2025 的做法,residual quantization 每層跑 KMeans。3 levels × 256 codes/level。簡單、穩定、無 codebook collapse 問題
- **v2(時間允許):RQ-VAE** — TIGER 原版。已知坑:codebook collapse(緩解:codebook 用 KMeans init + EMA update)。你有 CVAE/DDPM 背景,理論不是障礙,工程調參是
- 碰撞處理:相同 3-level code 的 items 加第 4 個 disambiguation token(TIGER 原文做法)
- 品質檢查:隨機抽 code prefix,看同 prefix 的電影是否語意相近(手動 spot check 10 組)

**Day 5–7:生成式模型**
- 改造你的 SASRec:输入序列從 atomic item ID 換成 semantic ID token 序列(每 item 3–4 tokens)
- Decoder 端 autoregressive 生成 next item 的 3 個 code
- **Constrained beam search:** 用 Trie 存所有合法 item 的 code 序列,decoding 時只允許合法前綴(否則會生成不存在的 item)— 這是 TIGER 復現的已知難點,預留 debug 時間
- 序列長度膨脹 3–4 倍 → maxlen 相應調整,注意記憶體

### Week 6 — 對比實驗 + 包裝發布

**Day 1–3:核心對比實驗**
1. **主實驗:** SASRec (atomic ID) vs TIGER-style (semantic ID),Amazon Beauty,full ranking Recall@10 / NDCG@10
2. **Cold-start 實驗(最有話題性):** 把 test set 按 item 訓練頻次分桶(head / torso / tail / unseen-in-train),分桶報指標 — Semantic ID 的理論優勢在 tail/cold items,量化驗證它
3. 誠實預期:overall 指標 TIGER-style 可能持平或略輸(學術界已知 TIGER 在小資料集上未必贏 SASRec),但 cold-start 分桶應該有故事 — 無論結果如何,量化結論本身就是貢獻

**Day 4–5:Serving demo + README**
- FastAPI endpoint:輸入觀影/購買序列 → top-10 推薦(附 semantic ID 解碼展示)— 半天,你的舒適區
- README 重寫:復現對照表置頂、方法論討論、爭議分析連結、限制誠實列出
- (Optional)Railway 部署 demo

**Day 6–7:Medium 文章**
- 結構:為什麼復現 → SASRec 對齊過程與坑 → BERT4Rec 爭議圖 → Semantic ID 改造 → cold-start 發現 → 給想做同樣事的人的 checklist
- LinkedIn 短文版同步,tag 相關性:generative recommendation 是 2025–26 最熱的 RecSys 話題

---

## 附錄 A:Repo 結構

```
sequential-rec-from-sasrec-to-tiger/
├── README.md                  # 復現對照表置頂
├── REPRODUCTION_LOG.md        # 對齊過程記錄
├── pyproject.toml
├── configs/                   # 每實驗一個 yaml
│   ├── sasrec_ml1m.yaml
│   ├── sasrec_beauty.yaml
│   └── tiger_beauty.yaml
├── src/
│   ├── data/                  # 下載、5-core、leave-one-out、Dataset 類
│   ├── models/
│   │   ├── sasrec.py
│   │   └── genrec.py          # semantic ID 生成式模型
│   ├── semantic_ids/
│   │   ├── embed.py           # sentence-transformer
│   │   ├── rq_kmeans.py
│   │   └── rq_vae.py
│   ├── eval/
│   │   ├── sampled.py
│   │   ├── full_ranking.py
│   │   └── cold_start.py      # 分桶評估
│   ├── train.py
│   └── beam_search.py         # Trie-constrained decoding
├── notebooks/                 # 分析與畫圖
├── serving/                   # FastAPI demo
├── docs/
│   └── bert4rec-controversy.md
├── results/
│   ├── figures/
│   └── tables/
└── tests/                     # 評估器單元測試
```

## 附錄 B:必讀論文清單(按閱讀順序)

1. Kang & McAuley (2018) — SASRec 原文(精讀,實作依據)
2. Sun et al. (2019) — BERT4Rec 原文(讀方法即可)
3. Krichene & Rendle (2020) — On Sampled Metrics(評估協定依據)
4. Petrov & Macdonald (2022) — BERT4Rec Replicability Study(Phase 2 核心)
5. Rajput et al. (2023) — TIGER(Phase 3 依據,精讀)
6. Zhai et al. (2024) — HSTU / Actions Speak Louder than Words(讀 intro + scaling 部分,面試談 industry frontier 用)
7. (選讀)Spotify RecSys 2025 — Semantic IDs for Joint Search & Rec(RQ-KMeans 出處)

## 附錄 C:風險與備案

| 風險 | 機率 | 備案 |
|---|---|---|
| SASRec 對不齊論文數字 | 中 | 三角驗證(官方 repo);差 2pp 內接受並在 log 中分析 |
| RQ-VAE codebook collapse | 高 | v1 用 RQ-KMeans 保底,RQ-VAE 作為 stretch goal |
| Constrained beam search 難 debug | 中 | 先做 greedy + 合法性後過濾的簡化版,再上 Trie |
| 6 週不夠 | 中 | 砍序:先砍 RQ-VAE → 再砍 Railway 部署 → Phase 1+2 本身已是完整專案 |
| TIGER-style overall 指標輸 SASRec | 高(且正常) | 敘事轉向 cold-start 分桶發現;誠實報告 = 你的一貫風格與賣點 |

## 附錄 D:面試一句話定位

> "I reproduced SASRec from scratch within 1 point of the paper's HR@10, independently verified the BERT4Rec reproducibility controversy with a training-budget-controlled comparison, then extended the same codebase into a TIGER-style generative recommender with RQ-quantized semantic IDs — and quantified exactly when semantic IDs beat atomic IDs (cold-start) and when they don't."

三層深度問答準備方向:
- L1(概念):為什麼 causal attention?為什麼 shared embedding?Semantic ID 解決什麼問題?
- L2(實作):sampled metrics 偏差、codebook collapse 緩解、constrained decoding 實作
- L3(判斷):什麼規模的公司該用哪個範式?HSTU scaling law 對中型公司的意義?— 這層直接呼應你 Speed Demo talk 的 "right tool for the problem" 論點
