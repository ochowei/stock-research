# EXP-15: Crypto-Specific Ensemble (Clean Data Redux)

## 1. Context & Hypothesis
*   **Context:** EXP-10 failed because the "Crypto Sensitive Pool" contained non-crypto stocks (e.g., Dutch Bros), causing feature confusion. However, it hinted that crypto features could mitigate losses.
*   **Hypothesis:** By restricting the training and evaluation to a **Pure Crypto Pool** (COIN, MSTR, RIOT, MARA) and adding Bitcoin (`BTC-USD`) context features (`Trend`, `RSI`, `Correlation`), we can achieve a Win Rate > 53% and positive alpha, validating a dedicated "Crypto Model" route.

## 2. Strategy
*   **Model Architecture:** LightGBM (Gradient Boosting).
*   **Features:**
    *   **Baseline:** `Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20` (Standard 5 Base Features).
    *   **Experiment:** Baseline + `BTC_Trend`, `BTC_RSI`, `BTC_Gap`, `Crypto_Corr`.
*   **Target Label:** `(Open - Close) / Open > 0.002` (0.2% Profit).
*   **Execution:** Market On Close (Hold to Close).

## 3. Data Scope
*   **Asset Pool:** Pure Crypto Tickers: `['COIN', 'MSTR', 'RIOT', 'MARA']`.
*   **Training Period:** 2023-01-01 to 2024-06-30.
*   **Testing Period:** 2024-07-01 to Present (Out-of-Sample).

## 4. Evaluation Metrics
*   **Primary:** Win Rate (Target: > 53%).
*   **Secondary:** Average Return per Trade (Target: > 0.10%).
*   **Comparison:**
    *   **Control Group:** Base Features on Pure Crypto Pool.
    *   **Test Group:** Base + BTC Features on Pure Crypto Pool.

## 5. Implementation Plan
1.  **Data Acquisition:** Fetch OHLCV for the 4 tickers and `BTC-USD`.
2.  **Feature Engineering:**
    *   Calculate Base Features for stocks.
    *   Calculate BTC Features (Shifted T-1 to avoid look-ahead).
    *   Merge.
3.  **Modeling:**
    *   Train Control Model (Base features).
    *   Train Test Model (Base + BTC features).
4.  **Evaluation:** Compare performance on the Test set.
