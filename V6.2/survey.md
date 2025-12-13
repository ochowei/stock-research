這份研究計劃是基於我們剛才探討的「日夜效應 (Day/Night Effect)」與「拔河理論 (Tug of War)」，專門針對 **V6.2 (或 V6.1 進階版)** 所設計的實驗路線圖。

這份計劃的核心目標是回答：**「如何利用『現在』的時段（如日內），去預測『下一個』時段（如隔夜）的漲跌？」**

---

# **V6.2 專案研究計劃：日夜結構預測模型 (Intraday-Overnight Prediction Model)**

**核心假設：** 股市的利潤在「日內 (Intraday)」與「隔夜 (Overnight)」之間存在結構性的輪動與博弈。
**目標：** 找出能預測 $R_{next\_night}$ (今晚收盤至明早開盤) 與 $R_{next\_day}$ 的因子。

---

### **方向一：散戶情緒反轉 (Direction 1: Retail Sentiment Reversal)**

* **核心問題：** 如果一支股票在白天（日內）漲了很多，晚上會繼續漲還是會跌？
* **理論依據：** 根據「拔河理論」，如果日內的上漲是由 **散戶 (Retail)** 推動的（無重大新聞、緩步推升、成交量普通），這通常代表情緒過熱。機構往往會在隔夜或隔日開盤進行修正。
* **預測邏輯 (Signal to Prediction)：**
    * **因子 (X):** $R_{day}$ (今日開盤至收盤漲幅) > 2% 且 無重大新聞公告。
    * **預測 (Y):** $R_{next\_night}$ (隔夜跳空) 為 **負 (Gap Down)**。
* **執行策略：**
    * 若符合特徵，於收盤前 (MOC) **做空留倉**，賭明早開低。

### **方向二：機構尾盤動能 (Direction 2: Institutional Closing Momentum)**

* **核心問題：** 收盤前 30 分鐘的急拉，是否暗示著明天會有行情？
* **理論依據：** 機構投資者 (Smart Money) 因為部位龐大，往往在流動性最好的 **收盤前半小時 (15:30 - 16:00)** 進行建倉或再平衡。尾盤的買壓通常代表「聰明錢」看好後市。
* **預測邏輯 (Signal to Prediction)：**
    * **因子 (X):** $R_{last30m}$ (尾盤30分漲幅) 顯著大於 $R_{mid}$ (盤中漲幅)，且伴隨 $Vol_{last30m}$ (尾盤量) 放大。
    * **預測 (Y):** $R_{next\_night}$ 為 **正 (Gap Up)**。
* **執行策略：**
    * 若偵測到「尾盤搶籌」訊號，於 MOC **做多留倉**，賭明早開高。

### **方向三：隔夜動能的持續性 (Direction 3: Overnight Momentum Persistence)**

* **核心問題：** 對於垃圾股 (Toxic Pool)，「弱者恆弱」是否發生在隔夜？
* **理論依據：** V6.1 報告提到垃圾股可能存在「動能崩潰」。如果一檔垃圾股今天日內被殺爛 (Intraday Crash)，恐慌情緒會在夜間發酵，導致隔天開盤繼續大跌。
* **預測邏輯 (Signal to Prediction)：**
    * **因子 (X):** 標的屬於 `Toxic Pool` 且 $R_{day}$ < -3% (日內大跌)。
    * **預測 (Y):** $R_{next\_night}$ 為 **負 (Gap Down)**。
* **執行策略：**
    * 針對垃圾股的空單，若日內利潤豐厚，**不平倉 (Hold Overnight)**，博取隔夜的恐慌溢酬。

### **方向四：日夜相關性濾網 (Direction 4: Day-Night Correlation Regime)**

* **核心問題：** 市場現在是「趨勢盤」還是「震盪盤」？
* **理論依據：**
    * **負相關 (Negative Correlation):** 日內漲、隔夜跌（或反之）。代表市場處於「震盪/拔河」狀態 -> **適合 V6.1 反轉策略 (Fade Gap)**。
    * **正相關 (Positive Correlation):** 日內漲、隔夜也漲。代表市場處於「強趨勢/動能」狀態 -> **反轉策略會失效，應切換為動能策略**。
* **預測邏輯 (Signal to Prediction)：**
    * **因子 (X):** 計算過去 20 天 $(R_{day}, R_{night})$ 的相關係數 (Correlation Coefficient)。
    * **應用:**
        * 若 Corr < -0.2 (常態): 啟動 V6.1 策略。
        * 若 Corr > 0.3 (罕見): **暫停 V6.1**，甚至考慮順勢交易。

### **方向五：缺口型態學 (Direction 5: Gap Profile Classification)**

* **核心問題：** 所有的 Gap 都能做空嗎？如何區分「逃逸缺口 (Breakaway Gap)」與「竭盡缺口 (Exhaustion Gap)」？
* **理論依據：** * **竭盡缺口:** 發生在趨勢末端，通常伴隨日內反轉 -> **V6.1 的主要利潤來源**。
    * **逃逸/突破缺口:** 發生在盤整區間突破，通常日內會續漲 (Gap and Go) -> **V6.1 的主要虧損來源**。
* **預測邏輯 (Signal to Prediction)：**
    * **因子 (X):** 結合 $Gap$ 大小與 $P_{pre}$ (盤前價格) 相對於 $MA_{20}$ (20日均線) 的位置。
    * **預測 (Y):** $R_{day}$ 的方向。
* **執行策略：**
    * 若 Gap 發生在突破長期盤整區間（Breakout），則 **SKIP** (不做空) 或 **反手做多**。

---

### **總結與建議優先順序**

這五個方向是從「簡單統計」走向「行為金融」的過程：

1.  **優先執行 [方向三] (Toxic Overnight):** 這是最容易驗證的，直接看垃圾股收盤不平倉會不會賺更多。
2.  **其次執行 [方向四] (Regime Filter):** 這能作為 V6.1 的總開關，避免在強趨勢中逆勢被軋。
3.  **進階研究 [方向一 & 二] (Intraday Prediction):** 這需要更細的數據（如分鐘線），屬於 V6.2 開發新策略的核心。

---

Q_6.1.1

能否僅透過 OHLCV 來判斷屬於哪一個 Pool ?

---

這是一個非常務實且具備高度可行性的方向。若受限於數據源（例如只使用免付費的 `yfinance` 且不抓取 `info` 基本面欄位），我們**絕對可以僅透過 OHLCV 數據**來建立一個高準確度的「代理分類模型 (Proxy Classifier)」。

根據您在 **V6.1 報告**與 **V5.3 研究計畫**中的定義，`Final Pool`（績優/權值）與 `Toxic Pool`（垃圾/Meme）在價格行為上有著天壤之別。

以下是僅利用 **OHLCV** 就能計算出的四大特徵維度，用來區分這兩個池：

### **1. 價格水平 (Nominal Price Level)**

這是最簡單暴力但最有效的篩選器。

  * **邏輯：** 機構偏好高價股（流動性好、滑價低），散戶偏好低價股（便宜、好炒作）。`Toxic Pool` 中的標的（如 `BZFD`, `RENT`, `PSNY`）大多屬於「水餃股」。
  * **OHLCV 指標：** `Close` (收盤價)
  * **判斷標準：**
      * **Final Pool:** $Price > 50$ (通常 \> 100)
      * **Toxic Pool:** $Price < 10$ (俗稱 Penny Stocks)

### **2. 波動率特徵 (Volatility Profile)**

垃圾股的波動通常是劇烈且不穩定的。

  * **邏輯：** 績優股走勢通常較平滑，垃圾股則容易暴漲暴跌。
  * **OHLCV 指標：** `ATR%` (真實波幅百分比) 或 `HV` (歷史波動率)
      * $$ATR\% = \frac{ATR(14)}{Close}$$
  * **判斷標準：**
      * **Final Pool:** $ATR\% < 3\%$ (每日波動約 1-2%)
      * **Toxic Pool:** $ATR\% > 5\%$ (每日波動常超過 5-10%)

### **3. 流動性健康度 (Liquidity Health / Amihud)**

這是您在 **V5.3 計畫**中提到的核心概念。

  * **邏輯：** 垃圾股通常「虛胖」，少量的成交金額就能把價格拉得很高（或殺得很低）。
  * **OHLCV 指標：** `Amihud Illiquidity Ratio` (阿米胡德非流動性指標)
      * $$Amihud = \frac{|Return|}{Close \times Volume}$$
      * (註：分母為「成交金額 Dollar Volume」)
  * **判斷標準：**
      * **Final Pool:** 數值極小（成交金額巨大，價格難撼動）。
      * **Toxic Pool:** 數值較大（成交金額小，一點錢就能砸盤）。

### **4. 缺口行為 (Gap Behavior)**

這與 **V6.1** 的核心策略直接相關。

  * **邏輯：** 垃圾股因為缺乏做市商 (Market Maker) 深度，隔夜更容易出現巨大的跳空，且往往伴隨著「動能崩潰」。
  * **OHLCV 指標：** `Gap_Size_Avg` (平均跳空幅度)
      * $$Gap\% = \frac{|Open_t - Close_{t-1}|}{Close_{t-1}}$$
  * **判斷標準：**
      * **Final Pool:** 跳空通常在 0.5% - 1.0% 之間。
      * **Toxic Pool:** 經常出現 \> 5% 甚至 \> 10% 的極端跳空。

-----

### **實作：自動分類器腳本 (Python 範例)**

您可以直接將此邏輯整合進 `daily_gap_signal_generator.py`，完全不需額外 API 請求。

```python
import pandas as pd
import numpy as np

def classify_pool_by_ohlcv(df):
    """
    輸入: 單一股票的 OHLCV DataFrame (至少包含最近 20 天數據)
    輸出: 'Final', 'Toxic', 或 'Neutral'
    """
    # 1. 取得最近一天的數據
    latest = df.iloc[-1]
    avg_price = df['Close'].tail(20).mean()
    
    # 2. 計算特徵
    # (A) 價格水平
    price = latest['Close']
    
    # (B) 波動率 (ATR% - 簡化版)
    # True Range / Close
    tr = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
    atr_pct = (tr.rolling(14).mean() / df['Close']).iloc[-1] * 100
    
    # (C) 成交金額 (Dollar Volume)
    dollar_vol = (df['Close'] * df['Volume']).rolling(20).mean().iloc[-1]
    
    # 3. 分類邏輯 (閾值可根據回測調整)
    
    # --- TOXIC 判定 ---
    # 股價 < $10 或 波動率 > 5% 或 日均成交額 < 500萬鎂
    if (price < 10) or (atr_pct > 5.0) or (dollar_vol < 5_000_000):
        return "Toxic"
        
    # --- FINAL 判定 ---
    # 股價 > $20 且 波動率 < 4% 且 日均成交額 > 5000萬鎂
    elif (price > 20) and (atr_pct < 4.0) and (dollar_vol > 50_000_000):
        return "Final"
        
    else:
        return "Neutral" # 中性/觀察名單

# 模擬使用
# category = classify_pool_by_ohlcv(stock_data)
# if category == 'Final': ...
```

### **總結**

使用 OHLCV 分類不僅可行，而且可能**更即時**。

  * **基本面數據 (EPS/機構持股)** 通常一季才更新一次，有滯後性。
  * **OHLCV 數據** 每天都在變。如果一檔權值股突然發生流動性危機（崩盤、波動率飆升），OHLCV 分類器會立刻捕捉到它變成了 `Toxic` 特徵，進而觸發 **V6.1** 的避險或特定策略（如 Toxic 的 TOTM 加碼）。

**建議：** V6.2 可以採用這種「基於行為的動態分類 (Behavior-based Dynamic Classification)」，這比靜態清單更具備反脆弱性。

---

Q_6.1.2

那麼能不能透過 OHLCV 判斷動能股呢？

---

這是一個非常好的問題。若我們能用 OHLCV 區分出 **「動能股 (Momentum Stocks)」**，就能將原本寫死的黑名單 (`MOMENTUM_BLACKLIST`) 替換為動態邏輯，進一步強化 V6.2 的自動化程度。

答案是：**絕對可以。** 事實上，經典的動能策略（如 Jegadeesh & Titman, 1993）本來就是僅基於價格（Price）與成交量（Volume）建構的。

在您的 **V6.1/V6.2 策略脈絡**中，識別「動能股」的主要目的是 **「避開做空 (Avoid Shorting)」**，因為這些股票在跳空後往往會「開高走高 (Gap and Go)」，是反轉策略 (Gap Fade) 的天敵。

以下是僅使用 OHLCV 數據來識別動能股的邏輯與實作代碼：

### **1. 動能股的 OHLCV 特徵 (Profile)**

與「垃圾股 (Toxic)」不同，真正的「動能股 (Momentum)」通常具有**機構支撐**，走勢較為連續且穩健，而非無厘頭的暴漲暴跌。

| 特徵維度 | 垃圾股 (Toxic) | 動能股 (Momentum) | 一般資產 (Asset/Mean Reversion) |
| :--- | :--- | :--- | :--- |
| **價格趨勢** | 雜亂、空頭排列或單日暴漲 | **多頭排列 ($P > MA_{20} > MA_{50} > MA_{200}$)** | 區間震盪或緩慢趨勢 |
| **相對強度 (RSI)** | 極端 (常 $>80$ 或 $<20$) | **強勢區間 (50 - 70 且不輕易跌破 40)** | 回歸中軸 (30 - 70) |
| **價格水平** | 通常 $< \$10$ (Penny Stocks) | **通常 $> \$20$ (機構股)** | 各種價位 |
| **歷史新高** | 遠低於歷史高點 (深淵反彈) | **接近 52 週新高 (52-wk High)** | 距離新高有段距離 |
| **策略動作** | **Short / Fade** | **AVOID (避開)** | **Fade (做空)** |

### **2. 判斷邏輯 (The Algorithm)**

我們可以結合以下三個條件來判定一檔股票是否進入「強動能狀態」：

1.  **趨勢濾網 (Trend Alignment):**
      * 收盤價 $>$ 20日均線 $>$ 50日均線 $>$ 200日均線。
      * (這是最經典的多頭排列，代表短、中、長期趨勢一致向上)。
2.  **新高接近度 (Proximity to Highs):**
      * 目前價格在 52 週最高價的 15% 範圍內。
      * 公式：$Close > 0.85 \times High_{252}$。
3.  **動能強度 (ROC - Rate of Change):**
      * 過去 3 個月或 6 個月的漲幅顯著 (例如 $> 20\%$)。

### **3. Python 實作代碼**

我們將這個邏輯整合進原本的分類器中，讓系統輸出三種狀態：`Toxic` (垃圾)、`Momentum` (強勢/避開)、`Asset` (一般/可做)。

```python
def classify_ticker_behavior_v2(df):
    """
    輸入: 單一股票的 OHLCV DataFrame (需包含至少 200 天數據以計算 MA200)
    輸出: 'Toxic', 'Momentum', 'Asset', 或 'Neutral'
    """
    if len(df) < 200:
        return "Unknown" # 數據不足

    # --- 1. 準備數據 ---
    latest = df.iloc[-1]
    close = latest['Close']
    
    # 計算均線
    ma20 = df['Close'].rolling(20).mean().iloc[-1]
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    ma200 = df['Close'].rolling(200).mean().iloc[-1]
    
    # 計算 ATR% (波動率)
    tr = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)))
    atr_pct = (tr.rolling(14).mean() / df['Close']).iloc[-1] * 100
    
    # 計算成交額 (Dollar Volume)
    avg_dvol = (df['Close'] * df['Volume']).rolling(20).mean().iloc[-1]
    
    # 計算 52 週新高位置
    high_252 = df['High'].rolling(252).max().iloc[-1]
    
    # --- 2. 判斷邏輯 (優先級: Toxic > Momentum > Asset) ---
    
    # [A] Toxic (垃圾股/高風險)
    # 特徵: 股價低、波動極大、流動性差
    if (close < 10) or (atr_pct > 5.0) or (avg_dvol < 5_000_000):
        return "Toxic"

    # [B] Momentum (強動能股) - V6.1 應避開做空
    # 特徵: 多頭排列 + 接近新高 + 股價正常
    # 邏輯: Price > MA20 > MA50 > MA200 (最強趨勢)
    # 放寬版邏輯: Price > MA50 > MA200 (中長期多頭) 且 接近年高
    is_trend_up = (close > ma50) and (ma50 > ma200)
    is_near_high = (close >= 0.85 * high_252)
    
    if is_trend_up and is_near_high:
        return "Momentum"

    # [C] Asset (一般資產) - 適合反轉策略
    # 特徵: 非垃圾，但也非強動能 (可能在區間震盪)
    return "Asset"
```

### **4. 整合進儀表板後的策略動作**

這樣您的 `daily_gap_signal_generator.py` 決策邏輯會變得更細緻：

  * **Category = Toxic:**

      * 訊號：Gap \> 0.5%。
      * 動作：**GO (Short)** (甚至在 TOTM 加碼)。
      * *理由：垃圾股的動能通常是不可持續的，容易崩盤。*

  * **Category = Momentum (New\!):**

      * 訊號：Gap \> 0.5%。
      * 動作：**SKIP (Avoid)** (或轉為 Long 策略)。
      * *理由：強勢股跳空通常代表新一波行情的開始 (Gap and Go)，做空會被軋死。*
      * *這取代了原本人工維護的 `MOMENTUM_BLACKLIST`。*

  * **Category = Asset:**

      * 訊號：Gap \> 0.5%。
      * 動作：**GO (Short)** (但在 TOTM/Holiday 暫停)。
      * *理由：一般股票跳空後容易均值回歸 (Mean Reversion)。*

### **5. 實際案例驗證 (模擬)**

  * **NVDA (2024年初):** 股價 \> $500，均線多頭排列，天天創新高。
      * 判定：**Momentum** -\> **SKIP** (成功避開做空被軋)。
  * **AMC (Meme潮):** 波動率 ATR% 極高 (\>10%)，股價低。
      * 判定：**Toxic** -\> **GO** (捕捉崩盤利潤)。
  * **KO (可口可樂):** 股價穩定，均線糾結或緩漲，ATR% 低。
      * 判定：**Asset** -\> **GO** (賺取日內回調的小利潤)。

### **總結**

加入 **「OHLCV 動能判斷」** 後，您的系統將具備 **「三重過濾網」**：

1.  **濾掉垃圾 (Toxic Check):** 用於鎖定高波動做空機會。
2.  **濾掉強者 (Momentum Check):** 用於保護自己不被強勢股軋空。
3.  **留下凡人 (Asset Check):** 用於執行穩健的均值回歸策略。

這完全不需要外部數據 (如基本面)，僅靠 `yfinance` 的歷史 K 線即可完成。