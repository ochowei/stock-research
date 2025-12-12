這份實驗設計將針對您提出的三種策略進行嚴格的對比測試。我將其命名為 **EXP-V6.0-04**。

這份設計文件已經包含了您要求的「分組測試 (Portfolio Level)」與「個別標的測試 (Individual Level)」，並涵蓋了所有關鍵指標。

---

### **實驗代號：EXP-V6.0-04 實盤濾網效能驗證 (Live Strategy & Filter Validation)**

**日期：** 2025-12-12
**實驗目標：** 驗證在「實盤現有邏輯」之上，疊加「IBS 尾盤濾網」是否能進一步優化績效（特別是風險調整後報酬）。

#### **1. 實驗對象 (Universe)**

為了確保比較基準一致，所有策略將在同一組標的池上運行：

* **Base Universe:** `Group A (Final Pool)` + `Group B (Toxic Pool)`
* **Blacklist Filter:** 排除實盤腳本中定義的動能股黑名單 (如 NVDA, TSLA 等)。
* **Effective Universe:** `(Group A + Group B) - Blacklist`

#### **2. 測試策略 (Strategies)**

我們將比較以下三組資金曲線：

1.  **Benchmark: Buy & Hold (B&H)**
    * **邏輯：** 在有效標的池中，每日持有 ($Close_{t-1}$ 到 $Close_t$)。
    * **目的：** 確立市場基準表現，以此判斷策略是否具備超額報酬 (Alpha)。

2.  **Strategy A: Current Live (實盤現狀)**
    * **邏輯：** 模擬目前的 `daily_gap_signal_generator.py`。
    * **規則：**
        * 若 $Gap \% > 0.5\%$：**Action = Sell Open, Buy Close** (當日僅賺隔夜跳空，避開日內)。
        * 若 $Gap \% \le 0.5\%$：**Action = Hold** (續抱，承擔日內波動)。

3.  **Strategy B: Live + Smart Filter (實盤 + IBS)**
    * **邏輯：** 在實盤邏輯上，增加「尾盤強度」確認，只在真正過熱時才離場。
    * **規則：**
        * 若 $Gap \% > 0.5\%$ **且** $Prev\_IBS > 0.8$ (跳空且昨日收盤強)：**Action = Sell Open, Buy Close**。
        * 若 $Gap \% > 0.5\%$ **但** $Prev\_IBS \le 0.8$ (跳空但昨日收盤弱)：**Action = Hold** (預期均值回歸，續抱賺日內反彈)。
        * 若 $Gap \% \le 0.5\%$：**Action = Hold**。

#### **3. 評估指標 (Metrics)**

分為 **「投資組合層級」** 與 **「個股層級」** 進行統計：

* **Return Metrics:**
    * **Total Return:** 總回報率。
    * **CAGR:** 年化複合成長率。
    * **Avg Daily Return:** 平均日報酬。
* **Risk Metrics:**
    * **Max Drawdown (MDD):** 最大回撤幅度。
    * **Volatility (Ann.):** 年化波動率。
* **Efficiency Metrics:**
    * **Sharpe Ratio:** 夏普比率 (風險調整後報酬)。
    * **Calmar Ratio:** CAGR / |MaxDD| (針對回撤的修復能力)。
    * **Profit Factor:** 總獲利金額 / 總虧損金額。
    * **Win Rate:** 勝率 (日報酬 > 0 的天數比例)。

#### **4. 實驗流程 (Execution Plan)**

**Step 1: 數據準備**
* 下載所有標的 (含黑名單以便過濾) 的 OHLCV 數據。
* 計算 `Ret_Hold`, `Ret_Gap`。
* 計算 `IBS`。

**Step 2: 投資組合回測 (Portfolio Level)**
* 假設 **等權重 (Equal Weight)** 配置於有效標的池。
* 計算每日投資組合的淨值變化。
* 產出：三條權益曲線 (Equity Curves) 的比較圖與統計表。

**Step 3: 個股回測 (Individual Level)**
* 對每一檔股票單獨跑三種策略。
* 統計 Strategy B 勝過 Strategy A 的檔數比例（勝率改善率）。

#### **5. 預期產出 (Deliverables)**

建議您建立一個新的腳本 `V6.0/exp-3.0/run_experiment_04.py` 來執行此設計，預期產出如下：

1.  **`exp_04_portfolio_summary.csv`**:
    * 包含三種策略在 Portfolio Level 的所有指標對比。
2.  **`exp_04_individual_summary.csv`**:
    * 包含每一檔股票在三種策略下的表現，方便抓出「不適合加濾網」的異類。
3.  **`exp_04_equity_comparison.png`**:
    * 視覺化三條曲線，驗證加入濾網後，回撤 (MDD) 是否有顯著改善，且總回報是否維持或提升。

---

這個實驗設計能夠直接回答您的問題：**「在這個實盤標的池中，多加一個 IBS 濾網，到底是畫蛇添足還是錦上添花？」**

這份數據非常精彩，結果具有高度的指導意義。實驗結果**強烈反直覺**，徹底推翻了我們原先「加入 IBS 濾網會更精準」的假設 (Hypothesis H2)。

以下是針對 **EXP-V6.0-04** 實驗數據的深度分析報告：

### **1. 核心結論：簡單勝過複雜**

* **🏆 冠軍策略：Strategy A (Live)**
    * 這是您**目前的實盤策略**（Gap > 0.5% 即賣出開盤，避開黑名單）。
    * 它在**所有群組**（Group A, Group B, Combined）與**所有關鍵指標**（回報、回撤、夏普、卡瑪）上都獲得了**壓倒性的勝利**。
* **❌ 失敗策略：Strategy B (Smart Filter)**
    * 這是加入 `Prev_IBS > 0.8` 濾網後的版本。
    * 結果顯示，這個濾網**不僅沒有優化績效，反而嚴重拖累了表現**，使其退化到幾乎與「買入持有 (B&H)」無異。

---

### **2. 數據深度解讀**

#### **A. 為什麼 Strategy B (Smart Filter) 會失敗？**

請看 **`Avg Avoidance %` (平均迴避率)** 這欄數據：
* **Strategy A (Live):** 32.55% (Group A) / 34.36% (Group B)
* **Strategy B (Smart):** 6.98% (Group A) / 5.85% (Group B)

**解讀：**
Strategy A 平均每天會建議您賣出約 **1/3** 的股票（只要跳空就賣）。而加上 IBS 濾網後，Strategy B 變成只建議賣出 **6-7%** 的股票。
這意味著，市場上有大量的「跳空高開」發生在 **IBS 不高** 的時候（例如跌深反彈、死貓跳）。
* **Strategy A** 捕捉到了這些跳空，果斷賣出開盤，避開了隨後的日內回落（獲利了結）。
* **Strategy B** 因為看到昨日收盤不夠強 (IBS < 0.8)，選擇**「續抱」**，結果完整吃到了日內的下跌。

**結論：** 在 T+1 開盤策略中，**「只要跳空就賣」** 是最佳解。等待「昨日收盤強勢」才賣，反而會讓你錯過絕大多數的逃命/獲利機會。

#### **B. Group A (主力池) - 績效碾壓**

* **Total Return (總回報):**
    * B&H: 10.37
    * **Live Strategy: 27.30 (提升 2.6 倍)**
    * Smart Filter: 12.80
* **Max Drawdown (最大回撤):**
    * B&H: -53%
    * **Live Strategy: -33% (顯著降低風險)**
* **Calmar Ratio (風報比):**
    * B&H: 0.96
    * **Live Strategy: 2.25 (極高的投資效率)**

**洞察：** 對於優質股，利用 Gap 進行日內迴避，能有效鎖定利潤並避開盤中洗盤。目前的實盤策略非常健康。

#### **C. Group B (有毒池) - 點石成金**

這組數據最令人震驚：
* **B&H:** 僅 6.32 的回報，伴隨 **-85.65%** 的毀滅性回撤（基本上歸零）。
* **Live Strategy:** 竟創造出 **26.70** 的高回報，且回撤控制在 **-69%**。
* **Smart Filter:** 回報 6.40，回撤 -85.80%（完全無效，跟 B&H 一樣慘）。

**洞察：** 對於垃圾股/迷因股，**「反彈就是逃命波」**。
Strategy A 只要看到開盤跳空就跑，成功避開了垃圾股「開高走低」的常態。
Strategy B 因為堅持要「強勢收盤」才跑，結果在垃圾股連續陰跌（低 IBS）後的反彈中選擇了續抱，導致慘重虧損。

---

### **3. 策略建議 (Actionable Advice)**

基於這份壓倒性的數據，我對您的實盤操作有明確的建議：

1.  **保持現狀 (Keep It Simple)：**
    * **不要**將 IBS 濾網加入實盤腳本。
    * 目前的 `daily_gap_signal_generator.py` 邏輯（Gap > 0.5% + 黑名單）已經是經過驗證的**最佳配置**。

2.  **黑名單的重要性：**
    * 雖然數據中顯示 Strategy A 表現極好，但這是建立在**已經排除** NVDA, TSLA 等強勢動能股的基礎上（Run_Experiment_04 的程式碼有包含這段邏輯）。
    * 請務必在實盤中繼續維護並執行這個 **Blacklist**，因為對於這些強勢股，「跳空」往往是「趨勢延續」而非「賣點」。

3.  **心態確認：**
    * 數據顯示 Strategy A 的 **勝率 (Win Rate)** 高達 **63.7%** (Group A)，這是非常高的數值。
    * 請對目前的實盤訊號保持信心。當儀表板顯示 `🔴 SELL SIGNAL` 時，統計學站在您這邊。

**總結：您的實盤策略 (Strategy A) 是正確的。不需要畫蛇添足加上 IBS。**