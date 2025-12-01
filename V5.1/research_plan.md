# **研究計畫：V5.1-ML 正交資訊增強與動態出場優化 (Orthogonal Alpha & Dynamic Exit)**

Date: 2025-12-01 20:30 PM

Topics: \#quant-trading \#system-design \#ai-collaboration \#risk-management \#data-analysis

Status: \#research \#draft

## **1\. 研究背景與核心目標 (Context & Objectives)**

### **1.1 V5 復盤總結**

在 V5-ML 的研究中，我們成功驗證了 **「L1 黑天鵝防禦系統」** 的有效性，將夏普比率提升至 2.60。然而，**L3 元標籤過濾層 (Meta-Labeling)** 表現失敗 (Sharpe 降至 1.82)。

**失敗原因分析：**

1. **資訊重疊 (Information Redundancy):** L3 模型使用了與 L2 訊號生成器高度相似的特徵 (如 RSI, SMA Distance)。在 L2 已經篩選出 RSI \< 10 的樣本後，L3 無法僅憑「RSI 是 8 還是 9」來區分勝負。  
2. **目標函數偏差:** 原 L3 採用二元分類 (Profit \> 0)，但在資源有限的交易場景中，我們更需要知道「哪一檔反彈幅度最大」，而非僅僅是「哪一檔會賺錢」。  
3. **出場僵化:** 固定的「持有 5 天」邏輯導致獲利回吐或資金佔用。

### **1.2 V5.1 核心目標**

本階段 (V5.1) 旨在透過引入 **「正交資訊 (Orthogonal Information)」** 與 **「動態出場 (Dynamic Exit)」** 來解決上述瓶頸。

1. **修復 L3 (大腦):** 從「過濾毒藥」轉向 **「機會排序 (Learning to Rank)」**，並引入板塊與基本面特徵。  
2. **新增 L4 (手腳):** 實作 **「剩餘 Alpha 預測」** 或 **「波動率動態止盈」**，取代固定持有期。  
3. **數據增強 (燃料):** 引入板塊 ETF 與更嚴格的滑價模擬。

## **2\. 策略方法論優化 (Methodology Upgrades)**

### **2.1 L3 層重構：排序與正交特徵 (L3: Ranking & Orthogonality)**

**目標：** 讓 L3 模型具備「選股智慧」，能在同樣超賣的標的中，挑出反彈機率最高者。

* **模型架構變更:**  
  * **From:** LGBMClassifier (Binary: Will it rise?)  
  * **To:** LGBMRanker (Ranking: Which will rise the most?)  
  * **Loss Function:** LambdaRank 或 NDCG (Normalized Discounted Cumulative Gain)。  
* **特徵工程 (Feature Engineering) \- 引入正交資訊:**  
  * **移除:** 原始技術指標 (RSI, SMA Distance)，避免與 L2 重疊。  
  * **新增 A \- 板塊輪動特徵 (Sector Features):**  
    * 計算標的所屬板塊 (如 XLK, XLF) 的 RSI 與相對強弱。  
    * *邏輯假設:* 若個股 RSI \< 10 但板塊 RSI \> 40 (個股獨跌)，風險高；若板塊 RSI \< 20 (集體恐慌)，反彈機率高。  
  * **新增 B \- 估值與事件特徵 (Valuation & Events) (需數據支持):**  
    * **Earnings Date:** 距離上次財報發布的天數 (避免接剛暴雷的刀)。  
    * **Relative Valuation:** 當前價格在過去 252 天的百分位 (Price Rank)，作為估值的簡易代理。  
  * **新增 C \- 微結構特徵 (Microstructure):**  
    * **Volume Structure:** 下跌段的量能變化 (無量下跌 vs. 爆量下跌)。

### **2.2 L4 層新增：動態出場機制 (L4: Dynamic Exit)**

**目標：** 提升資金周轉率 (Turnover) 與資本效率。

* **方案 A \- 剩餘 Alpha 預測模型 (Residual Alpha Model):**  
  * 在持倉的每一天 ($T+1$ to $T+4$) 執行推論。  
  * **Input:** 持倉當前損益、今日 RSI、L1 市場狀態。  
  * **Output:** $P(\\text{明天繼續上漲})$。  
  * **Action:** 若機率 \< 閾值 (e.g., 0.45)，提前出場。  
* **方案 B \- 波動率動態止盈 (Volatility-Adjusted TP):**  
  * 設定動態目標價：$Target \= Entry \+ K \\times ATR\_{entry}$  
  * 其中 $K$ 係數由簡單的回歸模型預測，或根據 L1 狀態動態調整 (e.g., 牛市 $K=2.0$, 震盪市 $K=1.0$)。

## **3\. 數據與驗證協議 (Data & Verification Protocol)**

### **3.1 數據源擴充**

為了支援板塊特徵，需擴充 00\_download\_data.py：

* **Sector ETFs:** XLK (Tech), XLF (Financials), XLV (Health), XLY (Discretionary), XLP (Staples), XLE (Energy), XLI (Industrial), XLB (Materials), XLU (Utilities).

### **3.2 更嚴格的滑價模擬 (Slippage Modeling)**

V5 回測假設以收盤價/開盤價完美成交，V5.1 需加入流動性懲罰。

* **模型:** $P\_{buy} \= P\_{open} \\times (1 \+ \\text{Slippage})$  
* **參數:**  
  * 高流動性 (SPY, AAPL): 0.01%  
  * 中流動性 (一般 S\&P 500): 0.05%  
  * 低流動性 (小型股/極端恐慌時): 0.1% \~ 0.2%

### **3.3 滾動視窗優化 (Walk-Forward Optimization)**

* L1 模型 (HMM) 將從靜態訓練改為 **滾動訓練 (Rolling Window)**。  
* **週期:** 每 6 個月重新訓練一次，使用過去 2 年數據。  
* **目的:** 適應市場從低息 \-\> 高息 \-\> 降息循環的結構性改變。

## **4\. 執行階段規劃 (Execution Phases)**

### **階段一：數據工程 (Data Engineering)**

1. 修改 00\_download\_data\_v5.py，加入 Sector ETFs 下載清單。  
2. 修改 02\_build\_features\_l0\_v5.py，實作「板塊相對強弱」與「個股/板塊 RSI 乖離」特徵。  
3. (Optional) 嘗試獲取 Earnings Date 數據 (若 yfinance 不支援，則使用 Price Gap 作為財報日代理特徵)。

### **階段二：L3 模型升級 (L3 Model Upgrade)**

1. 修改 04\_train\_meta\_labeling\_l3.py。  
2. 將標籤生成邏輯從 Classification (0/1) 改為 Ranking Score (e.g., 5日漲幅 %).  
3. 替換模型為 LGBMRanker。  
4. 特徵篩選：剔除原始 RSI，僅保留正交特徵。

### **階段三：L4 出場機制實作 (L4 Implementation)**

1. 實作「波動率動態止盈」邏輯 (較易實作，優先驗證)。  
2. 回測比較：Fixed 5-Day vs Dynamic ATR Exit 的 Sharpe 與 Win Rate。

### **階段四：整合回測 (Integrated Backtest)**

1. 執行 V5.1 全流程：L1 防禦 \-\> L2 掃描 \-\> L3 排序 (取 Top K) \-\> L4 動態出場。  
2. 產出新的 equity\_curves\_comparison.png，目標 Sharpe \> 3.0 且 Max DD \< 10%。

## **5\. 預期產出 (Deliverables)**

* V5.1/ml\_pipeline/features/sector\_features.parquet  
* V5.1/ml\_pipeline/models/l3\_ranker.joblib  
* V5.1/research\_result\_v5.1.md (包含滑價壓力測試報告)
