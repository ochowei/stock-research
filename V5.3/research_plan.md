# **研究計畫：V5.3-Dynamic (Track A) 數據檢核與動態攻防系統 (Final Revised v3)**

**Date:** 2025-12-04
**Focus:** 🟢 **Free Data Only (yfinance)**
**Goal:** 在免費數據限制下，確保數據品質，並驗證動態風控是否優於「固定持有」與「死抱」。

## **階段一：數據審計與基準重現 (Data Audit & Baselining)**

**目標：** 在開發新功能前，先確保數據品質，並建立清洗後的標準測試集。

### **1.1 工作項目：數據覆蓋率分析與清洗 (Data Sufficiency Analysis & Cleaning)**
* **問題：** 繼承自 V5.2 的 `asset_pool.json` 與 `toxic_asset_pool.json` 是靜態清單，可能包含大量在早期年份（如 2015-2018）尚未上市或流動性不足的標的。
* **執行動作 (撰寫 `00_audit_data.py`):**
    1.  **備份原始清單：** 將 V5.2 的原始檔案重新命名為 `origin_asset_pool.json` 與 `origin_toxic_asset_pool.json`。
    2.  **數據審計：** 計算每檔標的在 2015-2025 間的「有效交易日」比例與成交量密度。
    3.  **清洗與生成：** 生成經過清洗的 **新版** `asset_pool.json` 與 `toxic_asset_pool.json`。
    4.  **產出報告：** 生成 `data_selection_report.md`，描述篩選邏輯與被剔除的標的。
    5.  **覆蓋率視覺化：** 產出 `data_coverage_over_time.png`，確認各年份有效標的數量（目標 > 30 檔）。

### **1.2 工作項目：基準線重現 (Benchmark Re-implementation)**
* **目標：** 使用 **清洗後的新版清單** 跑出 V5.1 與 V5.2 的成績單作為對照。
* **執行動作:**
    * **Benchmark A (V5.1 Aggressive):** 邏輯: `Fixed 5-Day` + `No Filter` + `Equal Weight`。
    * **Benchmark B (V5.2 Risk-Aware):** 邏輯: `Fixed 5-Day` + `Breadth Filter` + `ATR Sizing` (即 V5.2 Full System)。
    * **產出:** `analysis/baseline_performance.csv`。

---

## **階段二：核心系統實作 (Core Implementation - Track A)**

### **2.1 L4 出場層：動態追蹤止盈 (ATR Trailing Stop)**
* **機制:** 取代 V5.2 的固定 5 天出場。
* **邏輯:**
    * $Stop = Entry - (3.0 \times ATR)$
    * $New\_Stop = Highest\_High - (K \times ATR)$
    * **動態 K:** 當 `RSI > 70` 或 `L1 Risk = High` 時，緊縮至 $K=1.5$。

### **2.2 交易成本模型 (Transaction Cost Model - Updated)**
* **券商假設:** **Firstrade / IBKR Lite (Zero Commission)**。
* **參數設定:**
    * **Commission (佣金):** **0 bps** (因 Firstrade 免佣)。
    * **Regulatory Fees (規費):** **1 bps** (預留給 SEC/TAF 賣出規費)。
    * **Slippage (滑價):**
        * **Entry (Limit Order):** **5 bps** (假設掛限價單，但考慮未能成交的機會成本或微幅滑動)。
        * **Trailing Exit (Stop Market):** **10 bps** (模擬觸發止損時，市價單造成的較大滑價)。

### **2.3 L1 防禦層：混合式崩盤預警 (Hybrid Defense)**
* **A. 硬性熔斷:** `Breadth < 15%` -> 強制清倉。
* **B. 預測模組 (Price Action XGBoost):** 預測短期波動風險，僅調整 L4 參數，不強制清倉。

---

## **階段三：驗證與壓力測試 (Verification)**

### **3.1 剝離研究 (Ablation Study)**
在同一份數據池 (清洗後的 Normal + Toxic) 上，比較以下場景：

1.  **V5.3 (Full):** L1 混合防禦 + L4 動態出場。
2.  **V5.3 (No L1):** 僅 L4 動態出場 (測試 L4 自身的獲利能力)。
3.  **V5.3 (No Stop / 死抱):** **(新增)** 關閉 L4 動態止盈與時間止損，持有直到觸發 L1 熔斷或 RSI > 90 極端訊號。驗證 L4 的 Trailing Stop 是否比「死抱」更優秀。
4.  **V5.2 (Benchmark):** 固定 5 天 + 硬性熔斷 (測試 V5.3 是否超越前代)。

### **3.2 毒性生存測試 (Survivorship Stress Test)**
* 專注觀察 **清洗後新版** `toxic_asset_pool.json` 中的標的。
* **成功標準:** V5.3 在毒性池中的 MaxDD 需優於 V5.2，且總回報不應歸零。

---

## **執行順序 (Execution Steps)**

1.  **[Step 0] 數據清洗:** 執行 `00_download_custom.py` -> `00_audit_data.py` (產出新清單)。
2.  **[Step 1] 建立基準:** 執行 `05_backtest_benchmarks.py` (V5.1/V5.2)。
3.  **[Step 2] 開發 V5.3:** 修改 `risk_manager.py` (L4) 與 `backtesting_utils.py` (整合新的成本模型)。
4.  **[Step 3] 最終回測:** 執行 `06_backtest_v5.3.py` (含所有剝離場景)。

---

## **未來展望 (Future Work): V5.4 Track B 進階研究**

**目標:** 引入付費與高解析度數據，解決「未知倖存者偏差」並優化執行細節。

### **1. 更細顆粒度的 OHLCV (High-Frequency Data)**
* **數據來源:** yfinance (近期) 或付費源。
* **規格:** 5m 或 15m K線數據。
* **目的:**
    * **更精確的進出場:** 模擬盤中觸發 L4 止損的真實價格，而非僅依賴日線 Low/Close。
    * **微結構特徵:** 計算日內 VPIN (Volume-Synchronized Probability of Informed Trading) 或買賣壓失衡，作為 L1 防禦的新因子。

### **2. Track B (進階執行 - Paid / Sharadar or Polygon)**
* **數據:** **Point-in-Time (PIT)** 歷史價格與基本面資料庫。
    * **關鍵特性:** 包含**所有已下市股票 (Delisted Stocks)**，且數據對應到「當時」的發布狀態（無前視偏差）。
* **偏差處理:** **全真法 (Reality Check)**。
    * **動態股票池 (Dynamic Universe):** 建立 `get_historical_universe(date)` 函數，重建每一天歷史當下真實存在的股票清單，而非使用 2025 年的後見之明清單。
* **目的:**
    * **捕捉「未知的地雷股」:** 找出那些我們沒聽過、但在歷史上曾造成虧損的股票。
    * **真實生存率驗證:** 驗證策略在真實歷史洪流（包含數千檔倒閉股）中的存活能力，確認 V5.3 的風控是否具有普適性。
