### **實驗代號：EXP-V6.0-02 尾盤強弱與隔夜反轉效應 (Tail-End Reversion)**

**日期：** 2025-12-11  
**實驗目標：** 驗證 Survey 第 4.2 節提出的「IBS 尾盤效應」假說，即「收盤疲弱（Weak Close）的資產傾向於在隔夜產生正向反彈」，從而確認在 T+1 開盤執行的潛在優勢來源。

#### **1. 實驗背景與假設 (Background & Hypothesis)**

* **Survey 依據：** 報告指出 `IBS < 0.2`（收盤接近當日最低）代表日內出現過度拋售，次日開盤往往有均值回歸（Mean Reversion）的動力。
* **核心問題：** 隔夜的獲利（Overnight Alpha）是普遍存在的，還是集中在特定的「尾盤型態」之後？
* **假設 (H1)：** **低 IBS (Weak Tail)** 的日子，其 $R_{night}$ 顯著高於 **高 IBS (Strong Tail)** 的日子（驗證均值回歸）。
* **假設 (H2)：** 在有毒資產池 (Toxic Pool) 中，這種反轉效應可能更劇烈或失效（因基本面惡化導致的連續下跌）。

#### **2. 實驗對象 (Asset Universe)**

沿用 EXP-01 的分組設定：

1.  **Group A: Final Asset Pool (主力池)**
2.  **Group B: Toxic Asset Pool (有毒池)**
3.  **Group C: VOO/SPY (市場基準)**

#### **3. 關鍵指標與變數 (Variables & Metrics)**

**自變數 (Independent Variable): 尾盤強度 (Tail Strength)**

* **指標：** Internal Bar Strength (IBS)
* **公式：** $IBS_t = \frac{Close_t - Low_t}{High_t - Low_t}$
* **分組 (Buckets)：**
    * **Tier 1 (Bearish Tail):** $IBS \le 0.2$ (收盤極弱，殺尾盤)
    * **Tier 2 (Neutral):** $0.2 < IBS < 0.8$ (收盤中性)
    * **Tier 3 (Bullish Tail):** $IBS \ge 0.8$ (收盤極強，拉尾盤)

**應變數 (Dependent Variable): 隔夜表現**

* **Overnight Return:** $R_{night} = \frac{Open_{t+1} - Close_t}{Close_t}$
* **Next Day Gap:** 觀察開盤跳空的方向與幅度。

#### **4. 實驗流程 (Execution Plan)**

**Step 1: 數據計算**

* 對每一檔標的，計算每日的 $IBS_t$。
* 計算對應的次日隔夜報酬 $R_{night, t+1}$。

**Step 2: 條件過濾與聚合 (Conditioning & Aggregation)**

* 將所有交易日按 $IBS$ 分組 (Tier 1, 2, 3)。
* 分別計算各組的平均隔夜報酬 (Mean Overnight Return)、勝率 (Win Rate) 與 波動率。

**Step 3: 累積績效模擬 (Conditional Equity Curve)**

* **Strategy_Weak_Tail:** 僅在 $IBS_t \le 0.2$ 時持有隔夜（收盤買，隔天開盤賣）。
* **Strategy_Strong_Tail:** 僅在 $IBS_t \ge 0.8$ 時持有隔夜。
* **Benchmark:** 持有所有隔夜 (All Nights)。

#### **5. 預期產出報表 (Output Deliverables)**

1.  **exp_02_ibs_summary.csv (統計分析表):**
    * **Columns:** Pool, IBS_Bucket (Low/Mid/High), Avg_Night_Return, Win_Rate, Count (樣本數)
    * **預期發現：** Group A 的 Low IBS 組別應有最高的 Avg_Night_Return。

2.  **exp_02_reversion_equity.png (策略走勢圖):**
    * 比較「專門承接殺尾盤 (Weak Tail Strategy)」與「追逐拉尾盤 (Strong Tail Strategy)」的累計隔夜損益曲線。
    * 這將直觀展示「到底該買強勢收盤還是弱勢收盤」。

3.  **exp_02_toxic_divergence.png (有毒資產特徵):**
    * 對比 Group A 與 Group B 在 Low IBS 下的表現。
    * *風險預警：* 有毒資產的低 IBS 可能代表真實崩盤而非錯殺，因此其隔夜反彈可能不如優質資產明顯。

---

**解讀重點：**
如果實驗結果顯示 **Low IBS** 的隔夜回報顯著優於 **High IBS**，則證實了 Survey 中關於「使用限價單在低位承接」或「T+1 開盤買入前日殺尾盤股票」的策略邏輯是正確的。這意味著我們不應該盲目持有所有隔夜，而是可以根據尾盤的強弱進行擇時。