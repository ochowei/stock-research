### **V4-D.8 專案規格書 (Plan v7.0): 通用模型 (60m 回歸版)**

文件版本： v7.0 (2025年11月12日)  
取代版本： v6.2 (2025年11月12日)  
狀態： \#draft

### **1\. 核心目標與架構轉變 (Objective & Architecture Shift)**

#### **1.1. 目的**

本文件（v7.0）旨在定義 V4-D.8 機器學習 (ML) 計畫的重大架構轉變。我們將從 v6.2 的「5m 分類模型」轉向\*\*「60m 回歸模型」\*\*，以優先適應新的數據源（yfinance 60m 數據）需求，並強化特徵工程的嚴謹性。

#### **1.2. 核心架構決策 (Architecture Decision)**

* **(架構保留)** 基於 V4-D.8 專案 $N \\ll P$（樣本數遠少於特徵數）的數據稀缺困境，以及特徵工程已具備高度「跨資產可比較性」的特性，本專案 (v7.0) **繼續採用**「選項 B：通用/池化模型 (Pooled Panel Data Model)」架構。  
* 此架構旨在通過匯集（Pool）大量資產的數據，從根本上解決數據稀缺性問題，並利用面板數據（Panel Data）的優勢提升模型預測能力。

#### **1.3. (v7.0 新增) 核心轉變 (Core Pivot: v6.2 \-\> v7.0)**

v6.2 的「動態三重屏障法 (TBM)」依賴高頻（5m）數據來判斷停利/停損的「競速」路徑。為了優先適應 yfinance 60m 數據的便利性，此標籤法已不適用。

v7.0 的核心轉變如下：

5. **特徵工程 (Features)：** 簡單匯總 → **「完整/不完整」K 棒分離 (方案 C)**，以嚴謹處理不均等的 K 棒時長。

#### **1.4. (v7.0 新增) 交易模擬假設 (Trading Simulation Assumptions)**

* **目的：** 本規格書旨在模擬現實世界的隔夜交易操作。  
* **現實限制：** 根據現實規則，延長時段（盤前、盤後）僅能使用**限價單 (Limit Order)**，而正常時段（盤中）可使用**限價單或市價單 (Market Order)**。  
* 模型模擬 (v7.0 基礎版)：  
  3\. 出場 (T+1 日)： 2.2.2 節中的「T+1 日的開盤價」模擬在 T+1 開盤時以市價單出場。  
* **(新增) 未來可擴展的模擬方向：**  
  * v7.0 的標籤 (Y) 定義了基礎版的回測邏輯。未來可基於此架構擴展更複雜的模擬：  
  * **進場點 (Entry)：** 可測試不同的進場價格 p（例如 T-1 (High+Low)/2），或模擬 T 日開盤**市價進場**（p \= T Open）。  
  * **出場點 (Exit)：** 可將 T+1 開盤出場改為 T+N 日收盤出場，或模擬**動態停損/停利單**（例如基於 T-1 ATR 的 vol 單位）。  
  * **訂單類型 (Order Type)：** 可進一步模擬不同訂單類型（限價單、停損單）在 ETH/RTH 時段的成交機率與滑價。

### **2\. 階段一：數據標記與特徵工程 (v7.0 修訂版)**

#### **2.1. 特徵向量 (X) 規格 (v7.0 擴展版)**

v7.0 特徵向量（X）是在 T-1 日收盤後可用的所有資訊。此規格在 v6.2 基礎上大幅擴展，以支持 60m 頻率及 RTH/ETH 混合模型，並採納「方案 C」解決 K 棒時長不均的問題。

**(A-F 組 v7.0 修訂) 60m RTH/ETH 基礎特徵 (方案 C)**

原 v6.2 中的 A, B, C, E, F 組特徵（日線價格延遲、成交量、5m 提取、K 棒形狀、價格相對位置）全部轉為 60m 頻率，並**分離為 RTH 和 ETH 兩個版本**。

**關鍵工程 (方案 C)：** 為了處理 yfinance 中 ETH K 棒時長不均（例如 30 分鐘）導致的「不可比較」問題，我們將特徵計算分離為 \_Full 和 \_Partial 兩個獨立的數據源：

1. **\_Full (完整 K 棒) 特徵：**  
   * **定義：** 僅使用 T-1 日當天所有**完整的 60 分鐘** K 棒進行計算（例如 \_MAX, \_MIN, \_AVG）。  
   * **範例：**  
     * X\_T1\_RSI\_60m\_RTH\_AVG\_Full (T-1 日所有*完整* RTH K 棒的 RSI 平均值)  
     * X\_T1\_ATR\_60m\_ETH\_MAX\_Full (T-1 日所有*完整* ETH K 棒的 ATR 最大值)  
     * (其餘特徵依此類推...)  
2. **\_Partial (不完整 K 棒) 特徵：**  
   * **定義：** 僅使用 T-1 日當天**不滿 60 分鐘**的 K 棒（通常是 ETH 的最後一根）進行計算。  
   * **範例：**  
     * X\_T1\_RSI\_60m\_ETH\_Last\_Partial (T-1 日最後一根*不完整* ETH K 棒的 RSI 值)  
     * X\_T1\_Vol\_Ratio\_ETH\_Last\_Partial (T-1 日最後一根*不完整* ETH K 棒的成交量比率)  
     * (其餘特徵依此類推...)

**(G 組 v7.0 修訂) 上下文特徵 (共 4 個特徵)**

* **(修訂)** v6.2 (G) 組的短期量化因子被保留和修訂，用於實現「混合模型」，使模型能感知「同質性」群組差異。X\_36 已被替換為可計算的價格代理特徵。  
* X\_34\_Beta\_6M: 標識股票的系統性風險（6 個月（126 天） Beta 值）。
* X\_35\_Momentum\_6\_1M: 標識股票的中期動能（過去 6 個月扣除最近 1 個月的總回報）。  
* X\_36\_Z\_Score\_126\_Daily: **(v7.0 替換)** 標識股票的長期價值因子（126 日 Z-Score），作為 P/B 比例的價格代理。
* X\_37\_Liquidity\_Amihud: 標識股票的流動性（Amihud 非流動性指標）。

#### **2.1.1. (v7.0 新增) 完整特徵列表 (Total 67 Features)**

以下是 T-1 日收盤後，提供給模型的**一組**完整特徵向量。

**基礎 60m 核心指標 (Base Metrics) (共 9 個)：**

* RSI: 60m K 棒的 RSI (14 週期)。  
* ATR: 60m K 棒的 ATR (14 週期)。  
* MFI: 60m K 棒的 MFI (14 週期)。  
* Vol\_Ratio: 60m K 棒的成交量 / 過去 20 根 60m K 棒的平均成交量。  
* Body\_Pct\_ATR: (60m |Close \- Open|) / (60m ATR)。  
* Upper\_Wick\_Pct\_ATR: (60m High \- Max(Open, Close)) / (60m ATR)。  
* Lower\_Wick\_Pct\_ATR: (60m Min(Open, Close) \- Low) / (60m ATR)。  
* Z\_Score\_20\_60m: 60m K 棒的收盤價，相對於過去 20 根 60m K 棒的 Z-Score。  
* BBWidth\_20\_60m: 過去 20 根 60m K 棒的布林帶寬度。

**A. RTH 完整 K 棒特徵 (T-1 日匯總) (共 27 個特徵)**

* (9 個 Base Metrics) x (3 個匯總)：\_AVG\_Full, \_MIN\_Full, \_MAX\_Full  
* X\_T1\_RSI\_60m\_RTH\_AVG\_Full, X\_T1\_RSI\_60m\_RTH\_MIN\_Full, X\_T1\_RSI\_60m\_RTH\_MAX\_Full  
* X\_T1\_ATR\_60m\_RTH\_AVG\_Full, X\_T1\_ATR\_60m\_RTH\_MIN\_Full, X\_T1\_ATR\_60m\_RTH\_MAX\_Full  
* X\_T1\_MFI\_60m\_RTH\_AVG\_Full, X\_T1\_MFI\_60m\_RTH\_MIN\_Full, X\_T1\_MFI\_60m\_RTH\_MAX\_Full  
* X\_T1\_Vol\_Ratio\_60m\_RTH\_AVG\_Full, X\_T1\_Vol\_Ratio\_60m\_RTH\_MIN\_Full, X\_T1\_Vol\_Ratio\_60m\_RTH\_MAX\_Full  
* X\_T1\_Body\_Pct\_ATR\_60m\_RTH\_AVG\_Full, X\_T1\_Body\_Pct\_ATR\_60m\_RTH\_MIN\_Full, X\_T1\_Body\_Pct\_ATR\_60m\_RTH\_MAX\_Full  
* X\_T1\_Upper\_Wick\_Pct\_ATR\_60m\_RTH\_AVG\_Full, X\_T1\_Upper\_Wick\_Pct\_ATR\_60m\_RTH\_MIN\_Full, X\_T1\_Upper\_Wick\_Pct\_ATR\_6RTH\_MAX\_Full  
* X\_T1\_Lower\_Wick\_Pct\_ATR\_60m\_RTH\_AVG\_Full, X\_T1\_Lower\_Wick\_Pct\_ATR\_60m\_RTH\_MIN\_Full, X\_T1\_Lower\_Wick\_Pct\_ATR\_RTH\_MAX\_Full  
* X\_T1\_Z\_Score\_20\_60m\_RTH\_AVG\_Full, X\_T1\_Z\_Score\_20\_60m\_RTH\_MIN\_Full, X\_T1\_Z\_Score\_20\_60m\_RTH\_MAX\_Full  
* X\_T1\_BBWidth\_20\_60m\_RTH\_AVG\_Full, X\_T1\_BBWidth\_20\_60m\_RTH\_MIN\_Full, X\_T1\_BBWidth\_20\_60m\_RTH\_MAX\_Full

**B. ETH 完整 K 棒特徵 (T-1 日匯總) (共 27 個特徵)**

* (9 個 Base Metrics) x (3 個匯總)：\_AVG\_Full, \_MIN\_Full, \_MAX\_Full  
* X\_T1\_RSI\_60m\_ETH\_AVG\_Full, X\_T1\_RSI\_60m\_ETH\_MIN\_Full, X\_T1\_RSI\_60m\_ETH\_MAX\_Full  
* X\_T1\_ATR\_60m\_ETH\_AVG\_Full, X\_T1\_ATR\_60m\_ETH\_MIN\_Full, X\_T1\_ATR\_60m\_ETH\_MAX\_Full  
* X\_T1\_MFI\_60m\_ETH\_AVG\_Full, X\_T1\_MFI\_60m\_ETH\_MIN\_Full, X\_T1\_MFI\_60m\_ETH\_MAX\_Full  
* X\_T1\_Vol\_Ratio\_60m\_ETH\_AVG\_Full, X\_T1\_Vol\_Ratio\_60m\_ETH\_MIN\_Full, X\_T1\_Vol\_Ratio\_60m\_ETH\_MAX\_Full  
* X\_T1\_Body\_Pct\_ATR\_60m\_ETH\_AVG\_Full, X\_T1\_Body\_Pct\_ATR\_60m\_ETH\_MIN\_Full, X\_T1\_Body\_Pct\_ATR\_60m\_ETH\_MAX\_Full  
* X\_T1\_Upper\_Wick\_Pct\_ATR\_60m\_ETH\_AVG\_Full, X\_T1\_Upper\_Wick\_Pct\_ATR\_60m\_ETH\_MIN\_Full, X\_T1\_Upper\_Wick\_Pct\_ATR\_ETH\_MAX\_Full  
* X\_T1\_Lower\_Wick\_Pct\_ATR\_60m\_ETH\_AVG\_Full, X\_T1\_Lower\_Wick\_Pct\_ATR\_60m\_ETH\_MIN\_Full, X\_T1\_Lower\_Wick\_Pct\_ATR\_ETH\_MAX\_Full  
* X\_T1\_Z\_Score\_20\_60m\_ETH\_AVG\_Full, X\_T1\_Z\_Score\_20\_60m\_ETH\_MIN\_Full, X\_T1\_Z\_Score\_20\_60m\_ETH\_MAX\_Full  
* X\_T1\_BBWidth\_20\_60m\_ETH\_AVG\_Full, X\_T1\_BBWidth\_20\_60m\_ETH\_MIN\_Full, X\_T1\_BBWidth\_20\_60m\_ETH\_MAX\_Full

**C. ETH 不完整 K 棒特徵 (T-1 日) (共 9 個特徵)**

* (9 個 Base Metrics) x (1 個匯總)：\_Last\_Partial  
* X\_T1\_RSI\_60m\_ETH\_Last\_Partial  
* X\_T1\_ATR\_60m\_ETH\_Last\_Partial  
* X\_T1\_MFI\_60m\_ETH\_Last\_Partial  
* X\_T1\_Vol\_Ratio\_60m\_ETH\_Last\_Partial  
* X\_T1\_Body\_Pct\_ATR\_60m\_ETH\_Last\_Partial  
* X\_T1\_Upper\_Wick\_Pct\_ATR\_60m\_ETH\_Last\_Partial  
* X\_T1\_Lower\_Wick\_Pct\_ATR\_60m\_ETH\_Last\_Partial  
* X\_T1\_Z\_Score\_20\_60m\_ETH\_Last\_Partial  
* X\_T1\_BBWidth\_20\_60m\_ETH\_Last\_Partial

**G. 上下文特徵 (T-1 日) (共 4 個特徵)**

* X\_34\_Beta\_6M
* X\_35\_Momentum\_6\_1M  
* X\_36\_Z\_Score\_126\_Daily
* X\_37\_Liquidity\_Amihud

#### **2.2. 標籤 (Y) 規格 (v7.0 動態回歸版)**

* **(v7.0 廢除)** v6.2 的「動態三重屏障法 (TBM)」被**廢除**。  
* **(v7.0 核心)** 採用\*\*「動態風險標準化回報 (Dynamic Risk-Normalized Return)」\*\*。  
* **理由：** 此方法繼承了 v6.2 規避「波動率偏誤 (Volatility Bias)」 的核心精神。它使模型預測的不是「絕對回報 %」，而是「標準化後的風險單位 (ATR 倍數)」，確保不同資產的 Y 標籤具有「風險可比性」。

**2.2.1. 標籤輸入參數 (T-1 日計算)**

* 進場價格 (p)：p \= T-1 Close (T-1 日的日線收盤價)。  
* 波動率單位 (vol)：vol \= X\_T1\_ATR\_Daily (T-1 日的 14 天 ATR **非標準化**值)。  
* 時間障礙 (p\_VT)：(v7.0 修訂) T+1 日開盤 (Open)。

**2.2.2. Y 標籤定義 (回歸)**

* **情境 (A) \- 獲利/虧損 (T+1 開盤結算)：**  
  * T 日進場成功（T 日 60m Low 觸及 p）。  
  * 結算價格 ($p\_{exit}$)：T+1 日的**開盤價**。  
  * **Y (標籤) \=** $(p\_{exit} \- p) / vol$  
    * *(範例：若 T+1 開盤價為* $p+1.5 \\times vol$*，則 Y \= 1.5)*  
    * *(範例：若 T+1 開盤價為* $p-0.8 \\times vol$*，則 Y \= \-0.8)*  
* **情境 (B) \- 未成交 (NO\_FILL)：**  
  * T 日 60m Low 從未觸及 p。  
  * **Y (標籤) \= 0** (或 Null，見 3.1 節)。

### **3\. 階段二：訓練架構與驗證協議**

#### **3.1. 數據清理規則 (Cleaning Rules)**

* **規則一：** 刪除「財報污染」數據（T日或T+N日區間內有財報日的樣本）。  
* **規則二：** 刪除「未成交」數據 (Label Isolation)。在 v7.0 中，Y=0 的樣本（未成交）**必須被移除**，以免模型混淆「未成交」與「零回報」（即 $p\_{exit} \\approx p$）這兩種截然不同的情境。

#### **3.2. 資產池設計 (Asset Pool Design)**

* **決策：** 繼續採用「混合模型 (Hybrid Model)」策略。  
* **執行 (1) \- 多樣性 (Diversity)：** 訓練數據池應包含最大化的多樣性（跨板塊、跨市值、跨波動率），以增強模型的泛化能力並最大化樣本 $N$。  
* **執行 (2) \- 同質性 (Homogeneity)：** v7.0 架構**強化**了此策略。模型現在可利用 G 組 (量化因子) 和 A-F 組 (RTH/ETH, Full/Partial 分離) 感知多維度的群組差異。

#### **3.3. 驗證協議 (Validation Protocol)**

* **決策：** 嚴格採用\*\*「滾動窗口驗證 (Walk-Forward Validation)」\*\*。  
* **理由：** 這是唯一能真實模擬時間序列預測，並避免「數據洩漏」的驗證方法。標準 k-fold 交叉驗證在時間序列上是錯誤的。

#### **3.4. 特徵標準化標準作業程序 (SOP)**

此 SOP 是 V4-D.8 專案必須遵守的關鍵協議，以防止「前視偏差 (Look-Ahead Bias)」。

在**每一個**滾動窗口 $k$ 中，必須執行以下步驟：

1. **定義窗口 (Define Window)：**  
   * Train\_k \= 窗口 $k$ 的訓練數據（面板數據）。  
   * Test\_k \= 窗口 $k$ 的測試數據（面板數據）。  
2. **擬合標量 (Fit Scaler)：**  
   * scaler\_k \= StandardScaler()  
3. **關鍵：** **僅**在訓練集 Train\_k 上計算統計參數（均值、標準差）。  
   * scaler\_k.fit(Train\_k\[X\_features\])  
4. **轉換數據 (Transform Data)：**  
   * **關鍵：** 使用 scaler\_k（攜帶 Train\_k 的統計參數）來轉換**兩個**數據集。  
   * X\_train\_scaled \= scaler\_k.transform(Train\_k\[X\_features\])  
   * X\_test\_scaled \= scaler\_k.transform(Test\_k\[X\_features\])  
5. **訓練與預測 (Train & Predict)：**  
   * model\_k.fit(X\_train\_scaled, Train\_k)  
   * predictions \= model\_k.predict(X\_test\_scaled)  
6. **嚴禁 (Prohibited Operation)：**  
   * **絕不**允許在 Test\_k 上調用 fit() 或 fit\_transform()。此操作會將未來數據的統計分佈洩漏到測試過程中，導致回測結果虛高。  
7. **滾動 (Roll Forward)：**  
   * 移動窗口，重複步驟 1-6。

### **4\. Works Cited**

* $$研究文件$$$$ML$$  
  V4-D.8 (階段一) 特徵工程規格書 (Plan v5)  
*   
  31. Panel Data vs Time Series Analysis \- GitHub  
* Financial Machine Learning \- The University of Chicago  
* Deep Neural Network Estimation in Panel Data Models \- Federal ...  
* The "Universal Model" by Justin Sirignano and Rama Cont \- Quantitative Finance Stack Exchange  
* Are Sector-Specific Machine Learning Models Better Than ... \- Quantpedia  
* Unsupervised ML in Algorithmic Trading | by Parker Carrus \- Medium  
* How Traders Can Take Advantage of Volatile Markets \- Charles Schwab International  
* Full article: Conditional Volatility Targeting \- Taylor & Francis Online  
* The Triple Barrier Method: Labeling Financial Time Series for ML in ... \- Medium  
* MetaTrader 5 Machine Learning Blueprint (Part 2): Labeling ... \- MQL5  
* Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets \- arXiv  
* Creative target variables for supervised ML? : r/algotrader \- Reddit  
* Why does my AI keep suggesting me to use ATR as an indicator for ... \- Reddit  
* 5 ATR Stop-Loss Strategies for Risk Control \- LuxAlgo  
* Average True Range (ATR) Indicator & Strategies \- AvaTrade  
* Normalized ATR: Two Ways of Expressing ATR as Percentage \- Macroption  
* Why AI & ML Models Need Diverse Training Data | Grepsr  
* Why is Training Data Diversity Important for Machine Learning, AI \- FutureBeeAI  
* Improve your AI models with diverse data \- Paradime.io  
* Can generative AI provide better data for financial models? \- Warwick Business School  
* Putting Your Forecasting Model to the Test: A Guide to Backtesting | Towards Data Science  
* How To Backtest Machine Learning Models for Time Series ... \- MachineLearningMastery.com  
* Building a Backtesting Service to Measure Model Performance at Uber-scale | Uber Blog  
* Deep Learning Enhanced Multi-Day Turnover Quantitative Trading Algorithm for Chinese A-Share Market \- arXiv  
* python \- fit-transform on training data and transform on test data ... \- Stack Overflow  
* What's the difference between fit and fit\_transform in scikit-learn models? \- Data Science Stack Exchange  
* why it uses fit\_transform on training dataset and transform in validation dataset? \- Kaggle  
* Use "fit\_transform" on training data, but "transform" (only) on testing/new data \- YouTube
