# **研究計畫：V5.2-Risk 波動率管理與防禦體系 (Risk & Sizing)**

**Date:** 2025-12-04  
**Topics:** \#quant-trading \#risk-management \#system-design \#backtesting \#reproducibility  
**Status:** \#research \#execution-plan

## **1. 研究背景與核心目標 (Context & Objectives)**

### **1.1 V5.1 實相檢驗總結**
V5.1 的實驗證明了簡單規則 (`RSI(2) < 10` & `Price > SMA(200)`) 具有強大的獲利能力 (9年 8.5倍)，但其 **-48.24% 的最大回撤 (MaxDD)** 意味著該策略在 2020 疫情熔斷與 2022 升息熊市中幾乎面臨破產風險。

### **1.2 V5.2 核心目標：生存優先**
本階段目標是將 V5 Baseline 的最大回撤控制在 **-20% ~ -25%** 以內，同時保留至少 **70%** 的總回報。我們將重心從預測「哪隻股票會漲」轉向「該買多少」（Position Sizing）以及「何時完全空手」（Market Filter）。

### **1.3 擱置與待修復項目 (Dormant Modules)**
以下模組在 V5.1 表現不如預期或過於復雜，將在 V5.2 中暫時移除，歸類為「待修復 (On Hold)」項目，未來視需要重新啟動研究：
* **L1 HMM (隱馬可夫模型):** 因反應滯後 (Lag) 嚴重，暫時以 Rule-Based 的市場寬度取代。
* **L3 Ranker (LGBM 排序):** 因樣本外表現不穩定，暫時以純 RSI 數值排序取代。
* **L4 Dynamic Exit (ML 預測):** 因容易在反彈初期被洗出場，暫時以 ATR 動態止盈規則取代。

## **2. 策略方法論 (Methodology)**

### **2.1 基準策略 (Benchmark: V5.1 Minimalist)**
為了客觀評估風控模組的效用，我們將嚴格對比 **V5.1 Minimalist Backtest** 的結果：
* **邏輯:** `RSI(2) < 10` & `Price > SMA(200)`。
* **倉位:** 固定金額 (Equal Weight, e.g., 20% per trade)。
* **出場:** 固定持有 5 天。
* **現狀:** Sharpe 0.94, MaxDD -48.24%, Return 847%。

### **2.2 V5.2 實驗組：波動率目標倉位 (Volatility Scaled Sizing)**
V5.2 將不再使用固定金額開倉，而是根據各別標的的波動率動態調整部位大小。
* **公式:** $$Position Size = \frac{Total Capital \times Target Risk \%}{Stock ATR}$$
* **預期效果:** 在高波動 (High Vol) 時期自動縮小部位，實現「分批試單」；在低波動 (Low Vol) 時期放大部位，提升資金效率。

### **2.3 L1 重構：市場寬度熔斷 (Breadth Thrust Filter)**
* **新指標:** `Market Breadth` = (資產池中股價 > SMA200 的成分股數量) / (資產池總數)。
* **規則:** 若 `Breadth < Threshold` (e.g., 20%)，強制停止開倉或大幅降低曝險上限。這是一個硬性開關，用於防禦系統性崩盤。

## **3. 數據與重現性協議 (Data & Reproducibility)**

為確保研究期間（可能跨越數天或數週）的數據一致性，並方便未來的回測比對，我們將嚴格鎖定數據區間。

### **3.1 固定日期區間 (Fixed Date Range)**
* **Start Date:** `2015-01-01` (確保有足夠歷史數據計算 SMA200)。
* **End Date:** `2025-11-30` (鎖定於開發開始前的最後一個完整月份)。
* **目的:** 避免因為每天重新下載最新數據（T+1），導致回測結果產生微小變異，干擾對策略邏輯的判斷。

### **3.2 數據源**
* **Universe:** S&P 100 / Nasdaq 100 高流動性成分股。
* **Source:** yfinance (Daily OHLCV)。

## **4. 執行階段規劃 (Execution Phases)**

### **階段一：數據工程與固定化 (Data Engineering)**
1.  修改 `00_download_data_v5.py`，加入 `END_DATE = '2025-11-30'` 的硬限制。
2.  修改 `02_build_features_l0_v5.py`，計算每日的 **Market Breadth** 指標並存入特徵檔。

### **階段二：風控引擎開發 (Risk Engine)**
1.  建立 `ml_pipeline/risk_manager.py` 模組。
2.  實作 `ATR Position Sizing` 邏輯。
3.  實作 `Portfolio Heat` (總曝險上限) 控制邏輯。

### **階段三：回測與壓力測試 (Backtesting)**
1.  開發 `05_backtest_v5_2.py`。
2.  **實驗 A (單一變數):** 僅啟用 Vol-Sizing，對比 Benchmark。目標：降低個股暴雷風險。
3.  **實驗 B (單一變數):** 僅啟用 Market Breadth Filter，對比 Benchmark。目標：降低 2020/2022 系統性回撤。
4.  **實驗 C (整合):** 同時啟用 Vol-Sizing + Breadth Filter。

### **階段四：參數最佳化 (Optimization)**
1.  Grid Search 尋找最佳參數組合：
    * `Target Risk %` (e.g., 0.5%, 1.0%, 2.0%)
    * `Breadth Threshold` (e.g., 15%, 20%, 30%)
2.  **目標函數:** 最大化 **Calmar Ratio** (Annual Return / MaxDD)。

## **5. 預期產出 (Deliverables)**

* **Code:**
    * `risk_manager.py`: 獨立風控模組。
    * `05_backtest_v5_2.py`: 支援動態倉位的新回測腳本。
* **Report:**
    * `analysis/v5.2_benchmark_comparison.csv`: 詳細列出 V5.1 (Fixed) vs V5.2 (Risk-Aware) 的各項指標對比。
    * `analysis/drawdown_comparison.png`: 回撤深度比較圖 (Underwater Plot)。