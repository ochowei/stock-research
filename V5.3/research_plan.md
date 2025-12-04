這份計畫的核心精神是 **「動態攻防 (Dynamic Offense & Defense)」**。在 V5.2 確立了生存基石後，V5.3 旨在透過更精細的預測模型與動態出場，找回被過度風控犧牲掉的超額報酬。

---

# **研究計畫：V5.3-Dynamic 智能攻防與動態出場系統**

**Date:** 2025-12-04
**Topics:** \#quant-trading \#machine-learning \#dynamic-exit \#supervised-learning
**Status:** \#research \#draft

## **1. 研究背景與核心目標 (Context & Objectives)**

### **1.1 V5.2 復盤總結**
V5.2 成功達成「生存優先」的目標，透過 ATR 控倉與市場寬度濾網，將最大回撤 (MaxDD) 控制在 -25% 左右。然而，剝離研究 (Ablation Study) 揭示了一個關鍵痛點：**「5 天時間止損 (Time Stop)」與「固定止盈」嚴重限制了獲利潛力**。在妖股反彈的主升段，過早離場導致策略錯失了肥尾 (Fat Tail) 利潤。

### **1.2 V5.3 核心目標：讓利潤奔跑 (Let Winners Run)**
本階段目標是在維持 V5.2 風控水準的前提下，顯著提升 **Calmar Ratio (報酬/回撤比)**。
我們將從「規則導向 (Rule-Based)」進化為「模型導向 (Model-Based)」與「動態導向 (Dynamic)」。

* **Primary Goal:** 移除時間止損，實作動態追蹤出場。
* **Secondary Goal:** 升級 L1 防禦層為「預測型」模型，解決訊號滯後問題。
* **Tertiary Goal:** 復活 L3 排序層，引入基本面數據以區分「錯殺」與「垃圾」。

## **2. 系統架構升級 (Architecture Upgrades)**

V5.3 將引入三個關鍵的新模組，形成完整的動態閉環：

### **2.1 L1 防禦層升級：監督式崩盤預警 (Supervised Crash Prediction)**
* **痛點:** V5.2 的 HMM/市場寬度指標屬於「確認型」指標，往往在崩盤發生後才動作。
* **解法:** 改用監督式學習 (XGBoost/LightGBM) 訓練預測模型。
* **Target:** 預測未來 5-10 天 `SPY` 下跌 > 5% 或 `VIX` 飆升的機率。
* **Features:** 宏觀特徵 (VIX Term Structure, Credit Spreads, Sector Rotation)。

### **2.2 L3 排序層復活：基本面增強 (Fundamental-Enhanced Ranking)**
* **痛點:** V5.1 的 L3 失敗是因為特徵與 L2 重疊。V5.2 證明「排序」有效，但僅依賴 RSI。
* **解法:** 構建 Learning to Rank (LTR) 模型，引入 **正交特徵**。
* **New Features:**
    * **基本面 (Fundamentals):** P/S Ratio, Revenue Growth, Free Cash Flow (避免買到即將破產的公司)。
    * **微結構 (Microstructure):** 買賣壓比率, 盤中波動率偏度。

### **2.3 L4 出場層新建：動態追蹤止盈 (ATR Trailing Stop)**
* **痛點:** 固定 5 天出場截斷了動能。
* **解法:** 實作路徑依賴型 (Path-Dependent) 出場機制。
* **邏輯:**
    * **初始止損:** Cost - 2 * ATR。
    * **移動止盈:** 當價格上漲，止盈線隨之移動至 `High_max - 3 * ATR`。
    * **狀態加權:** 若 L1 預測風險升高，自動收緊 ATR 倍數 (例如 3x -> 1.5x)。

## **3. 數據工程需求 (Data Engineering)**

為了支援上述模型，需擴充數據源：

1.  **基本面數據 (Fundamental Data):**
    * 來源: yfinance (Quarterly Financials) 或外部 API。
    * 頻率: 季資料，需進行前視偏差 (Look-ahead Bias) 處理 (例如：財報公告日後才可使用)。
2.  **盤中/高頻特徵 (Intraday Features):**
    * 來源: yfinance (60m data)。
    * 用途: 計算更精細的波動率與進場點。

## **4. 執行階段規劃 (Execution Phases)**

### **階段一：出場機制改革 (Exit Revolution)**
* **任務:** 修改 `backtesting_utils.py`，移除 `Hold 5 Days` 邏輯，實作 `TrailingStop` 類別。
* **驗證:** 使用 V5.2 的進場訊號，僅替換出場邏輯，對比 Normal/Toxic Pool 的績效差異。
* **預期:** 大幅提升 Total Return，MaxDD 可能微幅增加但可控。

### **階段二：L1 預測模型開發 (Predictive L1)**
* **任務:** 建立 `03_train_crash_predictor.py`。
* **方法:** 標註歷史崩盤區間 (Labeling)，訓練 XGBoost 分類器。
* **驗證:** 比較新模型與 V5.2 市場寬度濾網的 `F1-Score` 與 `Lead Time` (領先天數)。

### **階段三：L3 基本面因子整合 (Fundamental L3)**
* **任務:** 擴充 `00_download` 與 `02_features`，納入財報數據。
* **方法:** 訓練新的 Ranker 模型，目標是在 RSI<10 的股票中，排序出反彈機率最高的標的。
* **驗證:** 觀察 Top 5 選股的勝率與盈虧比是否顯著優於隨機 RSI 排序。

### **階段四：全系統整合回測 (Integration)**
* **任務:** 串聯 L1(預測) -> L2(篩選) -> L3(排序) -> Risk(控倉) -> L4(動態出場)。
* **雙軌驗證:** 再次執行 Custom vs. Index 雙軌回測與毒性壓力測試。

## **5. 成功指標 (Success Metrics)**

* **Calmar Ratio:** > 2.0 (年化報酬 / 最大回撤)。
* **Recovery Factor:** > 5.0 (總獲利 / 最大回撤絕對值)。
* **Win Rate:** 維持 > 55% (雖然動態出場主要提升的是盈虧比，但勝率不應大幅下降)。
* **Toxicity Resistance:** 在 Toxic Pool 中不發生本金歸零 (Ruin)。
