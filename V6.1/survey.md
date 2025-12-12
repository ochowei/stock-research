這份文件將 V6.0 實驗後的優化方向整理為 **V6.1 策略優化路線圖**。這份文件以數據為導向，旨在進一步提升實盤策略（Strategy A: Gap > 0.5% Sell Open）的風險調整後報酬（Risk-Adjusted Return）。

---

# **V6.1 策略優化與開發路線圖 (Strategy Optimization Roadmap)**

**版本:** V6.1-Draft  
**基石:** 基於 V6.0 EXP-03/04 驗證成功的實盤策略 (Strategy A)  
**目標:** 透過更精細的「資產篩選」與「微觀結構判斷」，進一步提升勝率並降低賣飛風險。

---

## **方向一：資產適性分析與動態選股 (Asset Suitability & Dynamic Selection) [NEW]**

**核心問題：**
目前的策略假設「Gap > 0.5% 賣出」對所有非黑名單股票都有效。但實際上，某些股票具有強烈的「趨勢慣性」（Gap 是突破而非過熱），而某些股票則具有強烈的「均值回歸」特性。我們需要一個機制來識別**「哪些股票適合這個策略」**。

**實作方法：**

1.  **建立標籤 (Labeling) - 基於 EXP-03 數據：**
    * 利用 `V6.0/exp-3.0/output/exp_03_individual_stock_report.csv` 中的回測結果。
    * **Target (Y):** `Calmar Delta` (策略 A 的卡瑪比率 - B&H 的卡瑪比率) 或 `Win Rate (Strat)`。
    * **定義：**
        * **適合 (Positive):** Calmar Delta > 0.5 且 Win Rate > 55%。
        * **不適合 (Negative):** Calmar Delta < 0 或 Win Rate < 50%。

2.  **特徵工程 (Feature Engineering) - 引入基本面與微結構：**
    * 參考 `V5.3/ml_pipeline/02_build_features.py` 中的特徵建構邏輯。
    * **基本面特徵 (Fundamental):**
        * **Sector:** 板塊分類 (Tech vs Utilities)。公用事業股可能不適合此策略（波動太小）。
        * **Market Cap:** 市值大小。
    * **微結構特徵 (Microstructure):**
        * **Amihud Illiquidity:** 衡量流動性風險。流動性差的股票 Gap 往往是真實的重新定價，不宜逆勢操作。
        * **Down_Vol_Prop:** 下跌量能佔比。
    * **技術特徵 (Technical):**
        * **Volatility (ATR/Close):** 歷史波動率。
        * **Gap Frequency:** 過去一年發生 Gap > 0.5% 的頻率。

3.  **模型訓練 (Model Training):**
    * 使用隨機森林 (Random Forest) 或 XGBoost 訓練一個分類器。
    * **輸入:** 上述特徵。
    * **輸出:** 該股票是否適合執行 Gap Fade 策略的機率。

4.  **實盤應用:**
    * 在 `daily_gap_signal_generator.py` 中引入此模型或規則。
    * **白名單機制：** 每日僅針對「適性分數」高的股票監控 Gap 訊號，自動過濾掉那些「跳空即噴出」的趨勢股。

---

## **方向二：引入 ATR 動態門檻 (Dynamic Threshold with ATR)**

**核心問題：**
固定 0.5% 的 Gap 門檻對於不同波動率的股票缺乏適應性。對於低波動股（如 KO），0.5% 已經是顯著偏離；對於高波動股（如 GME），0.5% 僅是雜訊。

**實作方法：**

1.  **公式調整：**
    將 `daily_gap_signal_generator.py` 中的固定閾值改為：
    $$Gap\_Threshold_i = k \times \frac{ATR(14)_i}{Close_i}$$
2.  **參數設定：**
    * 建議初始設定 $k = 0.2$ 或 $0.3$（即要求跳空幅度達到日均波動的 20%~30%）。
3.  **預期效果：**
    * **高波動股 (Toxic/Meme)：** 門檻自動提高（例如需 Gap > 2% 才觸發），避免在無效的雜訊中頻繁進出。
    * **低波動股 (Defensive)：** 門檻自動降低，增加交易機會。

---

## **方向三：盤前微結構優化 (Pre-market Microstructure)**

**核心問題：**
目前僅使用「盤前最新價」計算 Gap，忽略了盤前時段（4:00 AM - 9:30 AM）的價格路徑（Path）隱含的資訊。

**實作方法：**

1.  **數據源：** 利用 yfinance 下載 5m 數據 (`interval="5m", prepost=True`)。
2.  **新指標 - 盤前衰竭 (Pre-market Fade):**
    * 計算盤前最高價 ($High_{pre}$) 與盤前現價 ($Price_{curr}$) 的距離。
    * **規則：** 若 `(High_pre - Price_curr) / High_pre > 0.5%`，代表盤前多頭已經力竭，開盤後賣壓湧現機率高 -> **加強賣出訊號**。
3.  **新指標 - T-1 尾盤動能 (Tail Momentum):**
    * 參考 `V5.3/ml_pipeline/spy.py` 的邏輯。
    * 計算 T-1 日 15:30 至 16:00 的漲跌幅。
    * **規則：** 若 T-1 尾盤爆量大漲 (Panic Buying) 且 T 日開盤跳空 -> **極度過熱，強力賣出**。

---

## **方向四：跨資產濾網 (Cross-Asset Filters)**

**核心問題：**
Group B (Toxic/Meme) 資產與加密貨幣市場高度連動，週一開盤的表現常受週末幣圈情緒影響。

**實作方法：**

1.  **數據源：** 參考 `V5.3/ml_pipeline/eth.py`。
2.  **週一專用濾網 (Monday Only):**
    * 在週一執行腳本時，自動抓取 ETH 或 BTC 過去 48 小時 (週末) 的漲跌幅。
    * **規則：**
        * 若策略發出賣訊 (Gap > 0.5%) **且** ETH 週末大漲 (>5%) -> **暫緩賣出** (預期情緒外溢，開盤後可能續攻)。
        * 若策略發出賣訊 **且** ETH 週末持平或下跌 -> **果斷賣出**。

---

## **方向五：日曆效應 (Calendar Effects)**

**核心問題：**
特定日期存在結構性的「日內買盤」，此時賣出開盤容易「賣飛」。

**實作方法：**

1.  **月初效應 (Turn of the Month, TOTM):**
    * **定義：** 每月的最後 1 個交易日 與 下個月的前 3 個交易日。
    * **動作：** 在此窗口期，將 Gap 觸發門檻提高（例如從 0.5% 提高至 1.0%），或直接暫停賣出策略，以享受機構資金的日內推升。
2.  **節假日前夕 (Pre-Holiday):**
    * 同上，因假期前交易量縮且情緒偏多，建議暫停 Gap Fade 策略。

---

### **建議執行優先順序**

1.  **Priority 1 (高 CP 值):** **方向二 (ATR 動態門檻)**。改動極小，但能顯著提升策略對不同波動率資產的適應性。
2.  **Priority 2 (核心優化):** **方向一 (資產適性分析)**。這是解決「哪些股票該做、哪些不該做」的根本方法，建議優先進行離線研究 (Offline Research)。
3.  **Priority 3 (精細化):** **方向三 (盤前微結構)** 與 **方向四 (Crypto 濾網)**。可作為輔助指標加入儀表板，供人工決策參考。