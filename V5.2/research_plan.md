# **研究計畫：V5.2-Risk 波動率管理與防禦體系 (Risk & Sizing)**

**Date:** 2025-12-04  
**Topics:** \#quant-trading \#risk-management \#system-design \#backtesting \#reproducibility  
**Status:** \#research \#execution-plan

## **1. 研究背景與核心目標 (Context & Objectives)**

### **1.1 V5.1 實相檢驗總結**
V5.1 的實驗證明了簡單規則 (`RSI(2) < 10` & `Price > SMA(200)`) 具有強大的獲利能力 (9年 8.5倍)，但其 **-48.24% 的最大回撤 (MaxDD)** 意味著該策略在 2020 疫情熔斷與 2022 升息熊市中幾乎面臨破產風險。複雜的 ML 模型 (L3/L4) 因過度擬合與訊號雜訊，反而劣化了績效。

### **1.2 V5.2 核心目標：生存優先**
本階段目標是將 V5 Baseline 的最大回撤控制在 **-20% ~ -25%** 以內，同時保留至少 **70%** 的總回報。我們將重心從預測「哪隻股票會漲」轉向「該買多少」（Position Sizing）以及「何時完全空手」（Market Filter）。

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

## **5. 未來展望與積壓工作 (Future Work / Backlog)**

以下項目為潛在的高價值研究方向，暫不納入 V5.2 的核心開發，但保留作為 V6 或後續版本的迭代基礎。

### **5.1 ML 模組重啟與修復 (Revival of ML Modules)**
* **L1 HMM (Regime Detection):**
    * *問題:* 在 V5.1 中對崩盤反應有顯著滯後 (Lag)。
    * *計畫:* 未來可嘗試縮短窗口、加入更靈敏的特徵 (如 VIX Term Structure)，或改用 HMM-GARCH 模型來提升反應速度。
* **L3 Ranker (Signal Filtering):**
    * *問題:* 樣本外 (OOS) 表現不穩定，與簡單 RSI 排序相比無顯著優勢。
    * *計畫:* 待風控穩定後，可重新引入 L3，但需尋找與技術指標低相關的「正交特徵」(如新聞情緒、籌碼面) 作為輸入，避免資訊重疊。
* **L4 Dynamic Exit (Alpha Prediction):**
    * *問題:* 容易在反彈初期被洗出場。
    * *計畫:* 重新訓練模型以預測「持倉剩餘利潤期望值」，而非單純的漲跌機率。

### **5.2 ML 數據粒度與時段優化 (Data Granularity & Session Timing)**
針對未來重啟的 ML 模型 (L3/L4)，需考慮引入更精細的數據以捕捉微觀結構：
* **更細的時間粒度 (5m / 15m Duration):**
    * 目前模型僅使用日線 (Daily)。未來可引入 5 分鐘或 15 分鐘 K 線數據，訓練模型識別日內的反轉形態 (Intraday Reversal Patterns)，優化進場與出場的精確時機。
* **盤前盤後數據 (Extended Trading Hours, ETH):**
    * 除了正常交易時段 (RTH)，應納入 ETH 數據。許多重大消息與劇烈波動發生在盤前或盤後，將 ETH 的波動率與量能納入特徵，可能有助於 L1 模型更早偵測到異常風險。

### **5.3 交易標的池擴展 (Universe Expansion)**
* **中小盤成長股 (IWO / Russell 2000):**
    * 目前 V5.2 鎖定 S&P 100 以確保流動性。未來若風控模型 (V5.2) 驗證有效，可嘗試將標的擴展至 IWO 成分股，測試策略在高波動、高 Beta 資產上的獲利潛力。
* **板塊輪動 (Sector Rotation):**
    * 測試針對不同板塊 (如 QQQ 科技股 vs. XLP 必需消費股) 設定不同的參數或閾值，而非全市場一體適用。

### **5.4 進階驗證與數據增強 (Advanced Validation & Data)**
* **合成壓力測試 (Synthetic Stress Test):**
    * 實作 TimeGAN 生成未曾發生過的極端市場路徑 (如長期滯脹或閃崩)，驗證 L1 寬度濾網與 ATR 風控的極限生存能力。
* **基本面與替代數據濾網:**
    * **Earnings Date:** 加入財報日迴避邏輯。
    * **Sentiment Analysis:** 引入新聞或社群情緒分數作為輔助濾網。
    * **Microstructure:** 利用量能結構 (`Down_Vol_Prop`) 區分恐慌拋售與陰跌，優化進場時機。

---
**結語:** V5.2 是策略成熟化的關鍵一步。我們先放下 AI 的預測水晶球，拿起統計學的盾牌。唯有先確保在任何市場環境下都能生存，複利的力量才能發揮作用。