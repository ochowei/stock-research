# **📈 專案 V5.2-Risk 執行計畫：波動率管理與防禦體系**

**Date:** 2025-12-04  
**Based on:** V5.2/research_plan.md  
**Status:** #draft #execution-plan

本計畫旨在實作 V5.2-Risk 體系，將策略核心從「預測 (Prediction)」轉向「生存 (Survival)」。我們將移除不穩定的 ML 模型，轉而建立堅實的風控與部位管理系統。

## **步驟 1：數據工程與固定化 (Data Engineering & Fixing)**

此步驟確保數據的一致性與可重現性，並計算新的市場寬度指標。

* **目標：** 鎖定回測區間，並將 Market Breadth 寫入特徵檔。
* **執行細節：**
    1.  **鎖定數據區間 (00\_download\_data\_v5.py):**
        * 設定常數 `START_DATE = '2015-01-01'` 與 `END_DATE = '2025-11-30'`。
        * 確保所有下載 (Ticker, Macro) 都嚴格遵守此區間，避免 T+1 數據變動干擾回測結果。
    2.  **計算市場寬度 (02\_build\_features\_l0\_v5.py):**
        * **新增邏輯：** 在計算完個別股票的 `Dist_SMA_200` 後。
        * **聚合計算：** 每日計算 `Market_Breadth = (Count(Close > SMA200) / Total_Tickers)`。
        * **儲存：** 將此指標合併入 `market_features_L0.parquet`，欄位名稱為 `Market_Breadth_SMA200`。
* **產出檔案：**
    * `data/temp_raw/*.pkl`: 固定區間的原始數據。
    * `features/market_features_L0.parquet`: 包含 Breadth 指標的宏觀特徵。

## **步驟 2：風控引擎開發 (Risk Engine Implementation)**

此步驟是 V5.2 的核心，建立獨立的風控模組，供回測與實盤共用。

* **目標：** 實作波動率部位管理與總曝險控制。
* **執行細節 (risk\_manager.py):**
    * 建立 `RiskManager` 類別。
    * **方法 1 `calculate_position_size(account_equity, target_risk_pct, asset_atr)`:**
        * 實作公式：`Shares = (Equity * Target_Risk) / (ATR * Stop_Loss_Multiplier)`。
        * *(註: V5.2 預設 Stop Loss 距離通常設為 1~2 倍 ATR)*。
    * **方法 2 `check_exposure_ceiling(current_exposure, max_exposure_limit)`:**
        * 檢查是否允許開新倉。
* **產出檔案：**
    * `ml_pipeline/risk_manager.py`: 可重用的風控模組。

## **步驟 3：規則導向濾網構建 (Rule-Based Regime Filter)**

此步驟取代原有的 HMM 模型訓練，改為直觀的規則判斷。

* **目標：** 產出基於市場寬度的防禦訊號。
* **執行細節 (03\_build\_regime\_filter.py):**
    * **輸入：** 讀取 `market_features_L0.parquet`。
    * **邏輯：**
        * 讀取 `Market_Breadth_SMA200`。
        * 設定閾值 (e.g., `BREADTH_THRESHOLD = 0.20`)。
        * 若 `Breadth < Threshold`，標記 `Regime_Signal = 2` (Crash/Defense Mode)。
        * 若 `Breadth >= Threshold`，標記 `Regime_Signal = 0` (Safe)。
    * **輸出：** 產生與 V5.1 格式兼容的 `regime_signals.parquet`，以便下游程式無縫接軌。
* **產出檔案：**
    * `signals/regime_signals.parquet`: 每日防禦訊號。

## **步驟 4：回測與壓力測試 (Backtesting & Stress Test)**

此步驟驗證風控模組是否能有效降低 MaxDD。

* **目標：** 執行 V5.2 回測，並與 V5.1 Minimalist Benchmark 進行對比。
* **執行細節 (05\_backtest\_v5\_2.py):**
    * **重構回測迴圈：**
        * 引入 `RiskManager`。
        * 在 `Entry Logic` 中，將原本的 `Fixed Capital` 改為呼叫 `risk_manager.calculate_position_size()`。
        * 在 `Entry Logic` 前，加入 `Market Breadth` 的過濾檢查 (若訊號為 Crash 則跳過)。
    * **參數設定 (實驗組):**
        * `Target Risk`: 0.5% ~ 1.0% per trade。
        * `Breadth Threshold`: 20%。
    * **報告生成：**
        * 計算 **Calmar Ratio**。
        * 繪製 **Underwater Plot** (專注於回撤深度)。
* **產出檔案：**
    * `analysis/v5.2_backtest_report.txt`: 詳細績效報告。
    * `analysis/drawdown_comparison.png`: 深度回撤比較圖。

## **步驟 5：實盤腳本更新 (Production Update)**

* **目標：** 確保實盤推論邏輯與 V5.2 回測邏輯一致。
* **執行細節 (run\_daily\_inference.py):**
    * **移除：** HMM 模型載入、L3 Ranker 模型載入。
    * **新增：** 實作即時 Market Breadth 計算 (需下載當日所有成分股數據)。
    * **整合：** 呼叫 `RiskManager` 計算建議股數。
    * **輸出：** CSV 包含 `Symbol`, `Close`, `ATR`, `Suggested_Shares`。

## **檔案清單 (Files Summary)**

### **需修改或新增 (Modified/New)**

1.  **`ml_pipeline/00_download_data_v5.py`** (Modified): 加入固定日期區間限制。
2.  **`ml_pipeline/02_build_features_l0_v5.py`** (Modified): 新增 Market Breadth 計算。
3.  **`ml_pipeline/03_build_regime_filter.py`** (**New**): 規則導向的狀態生成腳本 (取代 ML 訓練)。
4.  **`ml_pipeline/risk_manager.py`** (**New**): 獨立風控邏輯模組。
5.  **`ml_pipeline/05_backtest_v5_2.py`** (**New**): 支援動態部位管理的新回測引擎。
6.  **`ml_pipeline/run_daily_inference.py`** (Modified): 更新為 V5.2 邏輯。

### **直接沿用 (Unchanged)**

1.  `ml_pipeline/01_format_data_v5.py`
2.  `ml_pipeline/asset_pool.json`
3.  `ml_pipeline/requirements.txt`

### **暫時移除 (Removed/Archived)**

1.  `ml_pipeline/03_train_regime_model_l1.py` (HMM)
2.  `ml_pipeline/04_train_meta_labeling_l3.py` (Ranker)