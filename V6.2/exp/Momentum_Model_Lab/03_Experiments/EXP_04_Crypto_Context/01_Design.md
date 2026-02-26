# EXP-04: Crypto Context Integration (Risk-On Regime)

## 1. Hypothesis
Momentum strategies (Buy Gap Up) perform significantly better in "Risk-On" environments.
Cryptocurrency markets (BTC/ETH) are the ultimate "Risk-On" asset class and trade 24/7.
Therefore, the trend and momentum of Crypto assets leading up to the Equity Market Open (9:30 AM ET) should serve as a powerful leading indicator for high-beta equity risk appetite.

We hypothesize that adding Crypto Context features (e.g., BTC Trend, BTC Momentum) will:
1.  Improve the Win Rate of the Momentum Model by filtering out "false breakouts" during Risk-Off crypto dumps.
2.  Increase the Average Return per trade.
3.  Rank as high-importance features in the model.

## 2. Plan

### Step 1: Data Acquisition
*   **Equity:** Load tickers from `2025_final_asset_pool.json`.
*   **Crypto:** Download `BTC-USD` and `ETH-USD` daily data from `yfinance`.
*   **Alignment:** Ensure strict avoidance of Look-Ahead Bias.
    *   Equity Trade Date: T
    *   Equity Open: T 9:30 AM
    *   Crypto Feature: T-1 Daily Close (available at 00:00 UTC on day T, well before 9:30 AM ET).

### Step 2: Feature Engineering
*   **Baseline Features:** `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Gap_Pct`, `VIX` (T-1).
*   **Experiment Features (Crypto Context):**
    *   `BTC_Ret_1d`: (BTC_Close(T-1) - BTC_Close(T-2)) / BTC_Close(T-2)
    *   `BTC_RSI_14`: RSI(14) of BTC Close (T-1)
    *   `BTC_Trend`: Boolean (BTC_Close(T-1) > BTC_MA50(T-1))
    *   `ETH_Ret_1d`: (ETH_Close(T-1) - ETH_Close(T-2)) / ETH_Close(T-2)

### Step 3: Modeling & Evaluation
*   **Model:** XGBoost Classifier.
*   **Training Period:** 2020-01-01 to 2023-12-31.
*   **Testing Period:** 2024-01-01 to 2025-12-31 (Out-of-Sample).
*   **Comparison:**
    *   **Baseline Model:** Trained on Baseline Features only.
    *   **Crypto Model:** Trained on Baseline + Crypto Context Features.
*   **Metrics:** Win Rate, Average Return, Equity Curve, Feature Importance.

## 3. Success Metrics
*   **Win Rate:** Improvement > 0.5% over Baseline.
*   **Avg Return:** Improvement > 0.05% per trade.
*   **Feature Importance:** Crypto features appear in the Top 5 most important features.
