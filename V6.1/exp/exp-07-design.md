# EXP-07: Next-Day Suitability Classifier (ML Filter)

## 1. 核心假設
* **前提：** Gap 策略的獲利分佈極度不均勻。少數的大行情貢獻了大部分利潤，而大量的雞肋行情只貢獻了雜訊與手續費。
* **假說：** 在「已知跳空 (Gap > 0.5%)」的前提下，我們可以利用「前日量價結構」與「市場情緒」來預測當日交易的期望值是否大於 0。
* **關鍵機制：** 使用 **Sample Weighting (樣本權重)** 強迫模型關注「大賺與大賠」的案例，忽略小波動噪音。

## 2. 實驗設定
* **Universe:** V6.1 Final Asset Pool (2025_final_asset_pool.json)
* **Data Period:**
    * **Training:** 2020-01-01 ~ 2023-12-31 (包含疫情、升息循環)
    * **Testing (OOS):** 2024-01-01 ~ 2025-12-31 (AI 牛市)
* **Filter Condition:** `Gap % > 0.5%` (只針對有訊號的日子進行訓練與預測)

## 3. 特徵工程 (Features)
所有特徵皆使用 **T-1 (昨日)** 或 **T-0 Open (今日開盤)** 可得之資訊，無未來視。
1.  **Market Context:** VIX Level, VIX Trend (RSI).
2.  **Asset Status:** RSI(14), ATR%, Volume Ratio (量能異常), Sector.
3.  **Gap Info:** Gap Size (跳空幅度).

## 4. 標籤與權重 (Label & Weight)
* **Target:** `(Open - Close) / Open` (做空日內報酬)
* **Label (Y):**
    * `1` (Suitable): Target > 0.2% (扣除滑價成本後仍獲利)
    * `0` (Unsuitable): Target <= 0.2%
* **Sample Weight (W):** `log(1 + abs(Target))` 或 `abs(Target)`。讓模型對大行情的判斷權重加倍。

## 5. 評估指標
比較 **"All Gap Trades" (Baseline)** 與 **"Model Selected Trades" (Exp-07)** 的：
* Win Rate (勝率)
* Profit Factor (獲利因子)
* Sharpe Ratio (夏普比率)
* Avg Return per Trade (單筆平均回報)