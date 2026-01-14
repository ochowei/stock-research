# EXP-11: Non-Tech Feature Augmentation (SPY Context)

## 1. Context
* **Success of EXP-07:** The "Tech" model gained +3.55% Win Rate by adding `QQQ` features, proving that sector context is critical.
* **Current State:** The "Non-Tech" model (V6.3) still relies on the minimal 5-feature Base set.
* **Hypothesis:** Adding `SPY` (S&P 500) features to the **Non-Tech Model** will provide necessary market context and improve predictive power. `SPY` is the most representative proxy for the "Non-Tech" (general market) universe.

## 2. Plan
1.  **Data Scope:**
    *   **Universe:** Non-Tech tickers from `2025_final_asset_pool.json`.
    *   **Benchmark:** `SPY` (S&P 500 ETF).
    *   **Period:** Train (2020-2023), Test (2024-2025).

2.  **Feature Engineering:**
    *   **SPY_Gap_Pct:** `(SPY_Open - SPY_Prev_Close) / SPY_Prev_Close`
    *   **SPY_RSI_14:** RSI of SPY Close (Shifted T-1)
    *   **SPY_Dist_MA20:** Distance of SPY Open to SPY 20-day MA (Shifted T-1 data).
    *   **Market_Corr:** Rolling 20-day correlation between Stock Close and SPY Close (Shifted T-1).

3.  **Models to Train:**
    *   **Baseline:** `[Gap_Pct, RSI_14, ATR_Pct, Vol_Ratio, Dist_MA20]`
    *   **Experiment:** Baseline + `[SPY_Gap_Pct, SPY_RSI_14, SPY_Dist_MA20, Market_Corr]`

4.  **Evaluation:**
    *   **Primary:** Win Rate (Precision).
    *   **Secondary:** Average Return per trade.
    *   **Check:** Feature Importance (is SPY usage dominant or balanced?).

## 3. Success Metrics
*   **Win Rate:** Increase > 1.0% over Baseline.
*   **Avg Return:** Positive delta.
*   **Feature Importance:** SPY features should appear in the top 50% of importance but not completely drown out stock specifics (unlike the failed Crypto experiment).
