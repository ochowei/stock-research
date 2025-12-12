### **實驗代號：EXP-V6.0-03 智能持有策略：基於尾盤與缺口特徵的日內迴避 (Smart Hold & Intraday Avoidance)**

**日期：** 2025-12-12  
**實驗目標：** 在以「買入持有 (Buy & Hold)」為核心基礎下，利用 **IBS (尾盤強度)** 與 **Gap (隔夜跳空)** 作為濾網，動態迴避「高回撤風險」的日內時段，旨在**維持總報酬 (Total Return) 的同時，顯著降低最大回撤 (Max Drawdown)**。

#### **1. 實驗背景與假設 (Background & Hypothesis)**

* **既有發現 (EXP-01 & 02)：**
    1.  **隔夜紅利：** 大部分的風險溢價來自隔夜 (Night)，而日內 (Day) 往往伴隨高波動與均值回歸。
    2.  **尾盤反轉：** $IBS < 0.2$ (弱勢收盤) 傾向於導致隔夜反彈；反之，$IBS > 0.8$ (強勢收盤) 往往意味著短線過熱。
* **核心假設 (H1 - 獲利了結假說)：** 當一檔股票在 $T-1$ 日收盤極強 ($High \ IBS$)，且在 $T$ 日開盤進一步跳空高開 ($Gap \ Up$) 時，短線動能達到極致，隨後 $T$ 日的日內時段 (Open to Close) 極易發生獲利了結或動能耗盡的下跌。
* **策略意義：** 如果我們在這種時刻選擇「賣出開盤 (Sell Open)，買回收盤 (Buy Close)」，即**跳過當天的日內持有**，理論上可以避開回撤，並鎖定隔夜的獲利。

#### **2. 實驗對象 (Asset Universe)**

1.  **Group A: Final Asset Pool (主力池)**
2.  **Group B: Toxic Asset Pool (有毒池)** – *預期此策略在有毒池效果最顯著，因為垃圾股常出現「拉高出貨」的日內走勢。*
3.  **Group C: SPY/VOO (市場基準)**

#### **3. 策略邏輯與變數 (Strategy Logic)**

我們將比較三種策略的資金曲線：

**1. Benchmark: Buy and Hold (B&H)**
* **邏輯：** 每日持有 ($Close_{t-1}$ 到 $Close_t$)。不進行任何擇時。

**2. Strategy A: Gap Filter (純缺口過濾)**
* **邏輯：** 認為「大幅跳空高開」必有回落。
* **規則：**
    * 若 $Gap \% > 0.5\%$ (開盤比前收盤高 0.5% 以上)：
        * **Action:** Sell Open, Buy Close (當日日內空手，僅持有隔夜)。
    * 否則：
        * **Action:** Hold (當日繼續持有)。

**3. Strategy B: IBS + Gap Smart Filter (智能過濾)**
* **邏輯：** 結合尾盤型態。只有在「收盤強勢」且「開盤跳空」的雙重確認下，才認定為過熱並離場。
* **規則：**
    * 若 $IBS_{t-1} > 0.8$ **AND** $Gap \% > 0\%$ (前日收盤強 + 今日開盤漲)：
        * **Action:** Sell Open, Buy Close (鎖定利潤，避開日內回調)。
    * 否則：
        * **Action:** Hold (包含 $IBS$ 低時的日內反彈機會)。

#### **4. 評估指標 (Metrics)**

本實驗不再只看報酬率，重點在於 **「風險調整後的報酬」**：

1.  **Max Drawdown (MDD):** 策略能否有效減少資產回落的幅度？
2.  **Calmar Ratio:** $CAGR / |Max \ Drawdown|$。這是衡量「為了賺這筆錢，我承受了多大痛苦」的最佳指標。
3.  **Intraday Avoided Return:** 我們「主動避開」的那幾天日內報酬總和。如果是負值，代表策略成功（避開了虧損）。

#### **5. 實驗流程 (Execution Plan)**

**Step 1: 數據準備**
* 計算每日 $IBS_{t-1}$。
* 計算每日 $Gap \% = (Open_t - Close_{t-1}) / Close_{t-1}$。
* 計算每日 $Day \ Return = (Close_t - Open_t) / Open_t$。

**Step 2: 模擬交易**
* 對每一天 $t$：
    * 計算 **B&H Return**: $(Close_t - Close_{t-1}) / Close_{t-1}$
    * 計算 **Strategy Return**:
        * 若觸發「迴避條件」：Return = $Gap \%$ (僅賺取隔夜跳空，日內報酬設為 0)。
        * 若未觸發：Return = B&H Return。

**Step 3: 績效歸因**
* 統計各策略觸發了多少次「迴避」。
* 分析被迴避掉的日內行情主要是上漲還是下跌。

#### **6. 預期產出報表 (Output Deliverables)**

1.  **exp_03_smart_hold_summary.csv:**
    * Columns: Pool, Strategy, Total Return, **Max Drawdown**, **Calmar Ratio**, Avoided_Days_Count, Avoided_Day_Return_Sum.
2.  **exp_03_equity_comparison.png:**
    * 繪製 B&H vs. Smart Filter 的權益曲線。
    * *預期視覺效果：* Smart Filter 的曲線在市場大跌段（如 2022 年）應呈現較平緩的走勢（因為避開了許多開盤後的殺盤）。
3.  **exp_03_drawdown_depth.png:**
    * 水下曲線圖 (Underwater Plot)。直觀展示回撤恢復的速度。

---

**實驗意義：**
如果 **Strategy B (IBS + Gap)** 能在不犧牲太多總報酬的情況下，將 MDD 降低 10%~20%，這將是極具實戰價值的「高勝率持有策略」。它不需要頻繁進出，只需要在特定的「過熱早晨」按一下賣出，收盤再買回即可。