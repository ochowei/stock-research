# EXP-10: Crypto-Specific Ensemble (The "Crypto Branch")

## 1. Context
*   **Previous Findings:** EXP-01 showed that Crypto features (`Crypto_Corr`, `BTC_Trend`) are powerful predictors but reduce signal count drastically when applied globally. EXP-03 showed they caused overfitting on non-crypto stocks.
*   **Current State:** The V6.3 Production system routes tickers to "Tech" or "Non-Tech" models. Crypto-sensitive stocks (e.g., COIN, MSTR) are forced into these buckets, likely missing their primary driver: Bitcoin's price action.
*   **Goal:** Determine if a dedicated pipeline for crypto-sensitive tickers using specific crypto features outperforms the generic model.

## 2. Hypothesis
Creating a dedicated model pipeline for **Crypto-Sensitive Tickers** (using `Base` + `Crypto` features) will outperform the generic "Base Features" model (Baseline) for this specific subset of stocks.

## 3. Plan
1.  **Data Loading:**
    *   Target Pool: `V6.2/resource/2025_final_crypto_sensitive_pool.json`
    *   Market Data: OHLCV for targets + BTC-USD (Bitcoin).
2.  **Feature Engineering:**
    *   **Base Features:** `Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20`.
    *   **Crypto Features:** `BTC_Gap` (Open/PrevClose - 1), `BTC_RSI`, `BTC_Change` (PrevClose/PrevPrevClose - 1), `Crypto_Corr` (Rolling correlation between Stock Close and BTC Close).
    *   *Constraint:* Ensure 1-day shift for all closing-based indicators to prevent look-ahead bias.
3.  **Model Training:**
    *   **Baseline Model:** LightGBM trained on the Crypto Pool using **only Base Features**.
    *   **Crypto Model:** LightGBM trained on the Crypto Pool using **Base + Crypto Features**.
    *   *Split:* Chronological Split (Train: 2020-2023, Test: 2024).
4.  **Evaluation:**
    *   Strategy: Hold-to-Close (Open to Close return).
    *   Metrics: Win Rate, Average Return per Trade, Total Return.

## 4. Metrics
*   **Win Rate:** Percentage of trades with Return > 0.
*   **Avg Return:** Mean of (Close - Open) / Open.
*   **Total Return:** Sum of returns.
