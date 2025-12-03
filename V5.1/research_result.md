# **研究結案報告：V5.1-ML 正交資訊與回測實相檢驗**

Date: 2025-12-03 21:30 PM  
Topics: \#quant-trading \#backtesting \#strategy-validation \#post-mortem  
Status: \#published \#insight

## **1\. 執行摘要 (Executive Summary)**

V5.1 專案原定目標是透過引入正交資訊 (L3 Ranking) 與動態出場 (L4 Dynamic Exit) 來優化 V5 策略。然而，在實施嚴格的 **滾動視窗驗證 (Rolling Window)** 與 **除權息數據修正 (Split Adjustment)** 後，我們獲得了極具價值的「實相檢驗 (Reality Check)」。

**結論：** 複雜的 ML 模型 (L3/L4) 在樣本外表現不如預期，但我們意外挖掘出了一個極具韌性的 **極簡基石 (Minimalist Baseline)**。

### **績效對比 (Benchmark Comparison)**

| 策略版本 | 總回報 (Return) | 夏普比率 (Sharpe) | 最大回撤 (MaxDD) | 狀態 |
| :---- | :---- | :---- | :---- | :---- |
| **V5 (In-Sample)** | \~1600% (幻覺) | 2.60 (幻覺) | \-10% | *過度擬合 / 未來數據* |
| **V5.1 (Rolling ML)** | 26.25% | 0.23 | \-40.13% | *模型雜訊干擾* |
| **V5 Baseline (Fixed)** | **847.95%** | **0.94** | **\-48.24%** | **真實物理法則** |

## **2\. 關鍵發現 (Key Findings)**

### **2.1 奧卡姆剃刀的勝利 (Victory of Occam's Razor)**

實驗證明，簡單的物理法則勝過複雜的預測模型：

* **有效邏輯:** Price \> SMA200 (趨勢) \+ RSI(2) \< 10 (均值回歸) \+ Hold 5 Days (時間止損)。  
* **無效優化:** L3 的排序模型與 L4 的 ATR 動態出場，在樣本外測試中反而截斷了獲利並引入了雜訊。

### **2.2 工程與數據的陷阱 (Engineering Pitfalls)**

本階段最重要的成就是修復了兩個足以毀滅實盤的工程漏洞：

1. **數據源污染:** 修正了 yfinance 未開啟 auto\_adjust=True 導致的拆股虧損誤判 (如 NVDA, AMZN)。  
2. **邏輯黑洞:** 修正了倉位管理邏輯中「覆蓋持倉 (Overwrite)」而非「累加/檢查」的 Bug，找回了憑空消失的資金。

### **2.3 風險的真實面貌 (The True Risk)**

V5 Baseline 雖然獲利強勁 (9年 8.5倍)，但 **\-48.24% 的最大回撤** 是不可接受的。這發生在 2020 與 2022 年的系統性崩盤期間。這確立了下一階段的唯一目標：**風控**。

## **3\. 策略決策 (Strategic Decisions)**

1. **保留 (Keep):**  
   * 核心進場邏輯 (RSI \< 10 & Stock \> SMA200)。  
   * 核心出場邏輯 (固定持有 5 天)。  
   * Top 5 輪動機制。  
2. **移除 (Remove):**  
   * **L3 Ranker:** 暫時退役，回歸最簡單的 RSI 數值排序 (Deepest Value)。  
   * **L4 Dynamic Exit:** 廢除，避免在反彈初期被洗出場。  
   * **Rolling HMM (L1):** 因反應滯後 (Lag) 嚴重，暫時移除，需尋找更敏銳的替代方案。  
3. **待解決 (To-Do):**  
   * 如何將 \-48% 的回撤壓制到 \-20% 以內，而不腰斬總回報？

## **4\. 結語**

V5.1 雖然在模型表現上「失敗」了，但在科學驗證上取得了巨大的「成功」。我們剝除了所有過度包裝，找到了一個會賺錢的引擎。現在，我們只需要為它裝上煞車。