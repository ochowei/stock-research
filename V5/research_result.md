這份最終報告確認了 **V5-ML 研究計畫正式結案**。我們成功從一個複雜的假設（L1+L2+L3），透過嚴謹的數據驗證，收斂出一個既強大又精簡的獲勝策略：**「V5 均值回歸 + L1 黑天鵝防禦」**。

### 🏆 V5-ML 系統最終型態 (Final System Architecture)

根據您的執行結果，這就是我們要送上戰場的最終規格：

1.  **核心引擎 (Alpha Source):**
    * **邏輯:** 在長期多頭趨勢中 (`Price > SMA200`) 尋找短期過度恐慌的機會 (`RSI(2) < 10`)。
    * **優勢:** 高勝率、邏輯單純、經得起時間考驗。
2.  **防禦盾牌 (Risk Management):**
    * **L1 HMM (隱馬可夫模型):** 識別並過濾掉「崩盤/高波動」體制 (State 2)。
    * **L1 IsoForest (隔離森林):** 識別並過濾掉「未知/異常」的市場狀態 (Anomaly)。
    * **成效:** 成功過濾了 **14.0%** 的高風險訊號（包括 2020/2022 的崩盤段），大幅提升了夏普比率。
3.  **被移除的模組:**
    * **L3 元標籤過濾:** 已確認移除。這讓系統更輕量化，減少了過度擬合 (Overfitting) 的風險，也降低了維護成本。

---

### 🚀 下一步：邁向實盤 (Roadmap to Production)

既然研究階段 (Research Phase) 已經結束，接下來就是工程階段 (Engineering Phase)。建議您可以開始規劃以下步驟：

#### 1. 模擬交易 (Paper Trading)
* **目標:** 驗證程式在即時數據下的穩定性，以及檢查是否有「滑價」或「成交困難」的問題。
* **執行:** 建立一個每日排程 (Cron Job)，每天收盤後自動執行：
    * 下載當日數據。
    * 執行 L0 特徵計算。
    * 載入 L1 模型 (`hmm_model.joblib`, `iso_forest.joblib`) 判斷今日狀態。
    * 若狀態安全，生成 L2 訊號清單。
    * (模擬) 記錄進場與 5 天後的出場損益。

#### 2. 自動化腳本 (Automation Pipeline)
您目前的 `ml_pipeline` 資料夾結構非常完整，只需要一個 `run_daily_inference.py` 腳本串接即可。

* **輸入:** 當日最新收盤價 (來自 yfinance 或券商 API)。
* **處理:**
    1.  更新 `market_features` (加入今日數據)。
    2.  `model.predict()` 判斷今日 `HMM_State` 與 `Is_Anomaly`。
    3.  若 `State != 2` 且 `Not Anomaly` -> 掃描全市場 `RSI(2) < 10` 的標的。
* **輸出:** 產生一份 `buy_list_2025-11-29.csv` 寄給自己或傳送到 Slack/Telegram。

#### 3. 模型監控 (Model Monitoring)
* **定期重訓練 (Retraining):** 建議每 **3~6 個月** 重新訓練一次 L1 模型 (HMM & IsoForest)，以適應市場結構的長期變化。
* **防禦率監控:** 如果發現 HMM 長期停留在 State 2 (崩盤態) 但市場卻在創新高，代表模型可能失效，需要重新校準。

---

**結語：**
您從最初的資料下載困難，到中間發現 L3 模型的無效，最後成功收斂出 Sharpe 2.6 的 V5+L1 策略。這是一個非常標準且成功的量化研究流程（Hypothesis -> Backtest -> Validation -> Simplification）。

現在，您可以放心地將這套策略部署到模擬環境中了！祝您交易順利！🚀