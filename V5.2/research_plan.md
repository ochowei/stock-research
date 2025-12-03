# **研究計畫：V5.2-Risk 波動率管理與防禦體系 (Final)**

**Date:** 2025-12-04
**Topics:** \#quant-trading \#risk-management \#system-design \#backtesting \#dual-track
**Status:** \#research \#execution-plan

## **1. 研究背景與核心目標 (Context & Objectives)**

### **1.1 V5.1 實相檢驗總結**
V5.1 的實驗證明了簡單規則 (`RSI(2) < 10` & `Price > SMA(200)`) 具有強大的獲利能力 (9年 8.5倍)，但其 **-48.24% 的最大回撤 (MaxDD)** 意味著該策略在 2020 疫情熔斷與 2022 升息熊市中幾乎面臨破產風險。

### **1.2 V5.2 核心目標：生存優先**
本階段目標是將 V5 Baseline 的最大回撤控制在 **-20% ~ -25%** 以內，同時保留至少 **70%** 的總回報。我們將重心從預測「哪隻股票會漲」轉向「該買多少」（Position Sizing）以及「何時完全空手」（Market Filter）。

## **2. 策略方法論 (Methodology)**

為了全面評估策略的穩健性，本研究採用 **雙軌驗證 (Dual-Track Validation)** 架構：

### **2.1 Track A：自選清單驗證 (Custom Portfolio)**
* **標的:** `asset_pool.json` (包含科技巨頭與高波動妖股)。
* **目的:** 驗證策略在實際感興趣的標的上，能否透過風控模組大幅降低 V5.1 的回撤，同時保留大部分獲利。
* **基準 (Benchmark):** V5.1 Minimalist Backtest (Track A 同標的)。

### **2.2 Track B：全市場壓力測試 (Index Stress Test)**
* **標的:** **S&P 100** 與 **Nasdaq 100** 全成分股。
* **目的:** 驗證策略是否具有普適性 (Generalization)，確保績效不是源自於對特定妖股的過度擬合 (Overfitting)。
* **基準 (Benchmark):** Index Equal Weight Buy & Hold。

### **2.3 核心風控機制 (Risk Mechanics)**
1.  **波動率目標倉位 (Volatility Scaled Sizing):**
    * 公式: $$Position Size = \frac{Total Capital \times Target Risk \%}{Stock ATR}$$
2.  **市場寬度熔斷 (Breadth Thrust Filter):**
    * 指標: `Market Breadth` = (S&P 100 成分股中股價 > SMA200 的比例)。
    * 規則: 若 `Breadth < 20%`，視為系統性崩盤，強制停止開新倉。

## **3. 數據與重現性協議 (Data & Reproducibility)**

### **3.1 固定日期區間**
* **Start Date:** `2015-01-01`
* **End Date:** `2025-11-30`
* **目的:** 鎖定數據快照，確保 Track A 與 Track B 的比較基準一致。

### **3.2 數據源**
* **Source:** yfinance (Daily OHLCV)。
* **清單來源:**
    * Track A: 本地 `asset_pool.json`。
    * Track B: Wikipedia 最新成分股清單 (需注意倖存者偏差，但在壓力測試情境下可接受)。

## **4. 執行階段規劃 (Execution Phases)**

### **階段一：雙軌數據工程 (Data Engineering)**
1.  開發 `00_download_custom.py` 與 `00_download_index.py`，分別建立獨立的數據庫。
2.  計算並儲存 **Market Breadth** 指標 (基於 Index 成分股計算，應用於兩者)。

### **階段二：風控引擎開發 (Risk Engine)**
1.  建立 `risk_manager.py` 模組，實作 ATR Sizing 與 Exposure Ceiling。
2.  建立 `regime_filter`，產出明確的開關訊號。

### **階段三：雙軌回測 (Backtesting)**
1.  **Run Track A:** 對比 V5.2 (Risk-Aware) 與 V5.1 (Fixed) 在自選池的表現。目標：Calmar Ratio 提升。
2.  **Run Track B:** 觀察 V5.2 在全市場冷門股與傳產股中的表現。目標：正期望值與低回撤。

### **階段四：參數最佳化 (Optimization)**
1.  針對 `Target Risk %` (0.5% - 2.0%) 與 `Breadth Threshold` (15% - 30%) 進行 Grid Search。
2.  尋找在 Track A 與 Track B 均表現穩健的參數組 (Parameter Robustness)。

## **5. 未來展望與積壓工作 (Future Work / Backlog)**

以下項目暫不納入 V5.2，保留為後續迭代基礎：

### **5.1 ML 模組重啟 (ML Revival)**
* **L1 HMM:** 待解決滯後問題後重新引入。
* **L3 Ranker:** 待引入正交特徵後重新訓練。
* **L4 Dynamic Exit:** 待優化目標函數後重啟。

### **5.2 數據維度擴展 (Data Expansion)**
* **標的:** 擴展至 IWO (Russell 2000) 中小盤股。
* **頻率:** 引入 5m/15m K線與盤前盤後 (ETH) 數據，捕捉日內微結構。

### **5.3 進階驗證**
* **合成數據:** 使用 TimeGAN 生成極端市場路徑進行壓力測試。
* **基本面濾網:** 加入財報日迴避 (Earnings Avoidance) 邏輯。

---
**結語:** 更新後的 V5.2 計畫通過「雙軌驗證」，不僅解決了生存問題，更進一步回答了「策略是否只對特定股票有效」的關鍵質疑。這將是實盤前最重要的一次體檢。