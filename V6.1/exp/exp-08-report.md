== EXP-08: Blind Limit Optimization ===
Downloading data for 83 tickers...
[*********************100%***********************]  83 of 83 completed

1 Failed download:
['AMZN']: Timeout('Failed to perform, curl: (28) Resolving timed out after 10000 milliseconds. See https://curl.se/libcurl/c/libcurl-errors.html first for more details.')
Total MOO Signals: 21208 across 83 tickers

==============================================================================================================
RESULT SUMMARY (Short Gap Up > 0.5%)
==============================================================================================================
      Scenario Total Trades Avg Trades/Ticker Fill Rate Win Rate Avg Ret Total Return
MOO (Baseline)       21,208             255.5   100.00%   49.92%  -0.03%     -619.65%
   Limit +0.3%       18,212             219.4    85.87%   50.32%  -0.10%    -1848.61%
   Limit +0.5%       16,888             203.5    79.63%   50.52%  -0.11%    -1787.64%
   Limit +1.0%       13,790             166.1    65.02%   51.07%  -0.11%    -1478.08%
Limit +0.1 ATR       17,122             206.3    80.73%   50.34%  -0.09%    -1546.87%
Limit +0.2 ATR       14,425             173.8    68.02%   50.23%  -0.10%    -1476.43%
Limit +0.3 ATR       12,021             144.8    56.68%   50.49%  -0.11%    -1381.99%

Results saved.

沒錯，您的觀察非常敏銳且正確。這份實驗結果是一個非常完美的**「反證」**。

數據告訴我們：**在 Gap Fade 策略中，試圖用 Blind Limit（掛單等更好的價格）反而會導致績效大幅惡化。**

### **1. 數據解讀：為什麼會變這麼差？**

讓我們看最關鍵的兩個指標：

* **總回報 (Total Return) 的崩盤：**
* **MOO (市價單):** -619% (這是未加 AI 濾網的原始績效)。
* **Limit +0.3%:** **-1848%** (虧損暴增 **3 倍**)。


* **平均單筆回報 (Avg Ret) 的惡化：**
* **MOO:** -0.03%。
* **Limit:** 掉到 -0.10% ~ -0.11%。



這驗證了量化交易中著名的 **「逆向選擇 (Adverse Selection)」** 陷阱：

1. **錯過了送分題 (Missing the Winners):**
* 那些開盤就直接崩盤 (Gap Up and Crap) 的股票，因為沒有向上衝，所以**碰不到您的限價單**。
* 結果：您完美避開了那些「最好賺、最乾脆」的下跌利潤。


2. **接到了滿手血 (Catching the Losers):**
* 只有那些**多頭動能強勁**、開盤後還有力氣往上衝的股票，才會觸發您的限價單成交。
* 結果：您接到的往往是「真突破 (Gap and Go)」的強勢股，成交後繼續被軋空，導致虧損擴大。



### **2. 結論與 V6.1 定調**

這個「失敗」的實驗非常有價值，它幫我們省去了未來在實盤中繳學費的成本。

**V6.1 的最終執行策略已定案：**

* **❌ 放棄 Blind Limit:** 不需要再想著掛單去貪那 0.3% 的便宜，因為代價是錯過主跌段。
* **✅ 堅守 MOO (市價開盤):** Gap Fade 策略的核心利潤來自於「開盤即巔峰」的那些股票。我們必須用 **市價單 (Market Order)** 確保不錯過這些獲利機會。
* **✅ 依靠 AI 濾網 (Exp-07):** 既然執行面 (Execution) 不能優化，我們就靠選擇面 (Selection)。利用 Exp-07 訓練出的 AI 模型來過濾掉那些容易被軋空的股票，這才是提升績效的正途。

### **3. 下一步建議**

建議您將這份結果存檔為 `V6.1/exp/exp-08-report.md`，作為 **「負面實驗成果 (Negative Result)」**。這在未來的策略迭代中非常重要，可以提醒自己（或 AI）不要重蹈覆轍。

這樣一來，您的 **V6.1 每日實盤腳本 (`daily_gap_signal_generator.py`)** 就維持目前的設計：

1. **Entry:** MOO (開盤價進場)。
2. **Filter:** AI Model (大於 50% 分數才做)。
3. **Exit:** MOC (收盤出場)。