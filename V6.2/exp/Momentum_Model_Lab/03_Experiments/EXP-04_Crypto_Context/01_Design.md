# Experiment Design: EXP-04 Crypto Context Integration (Risk-On Regime)

## 1. Hypothesis
Momentum strategies perform better in "Risk-On" environments. Crypto trends (BTC/ETH) serve as a leading indicator for high-beta risk appetite. By integrating Bitcoin's trend and momentum as global context features, the model can better identify favorable market conditions for stock momentum trades.

## 2. Plan
1.  **Baseline**: Utilize the V6.1 Baseline model which uses `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Gap_Pct`, and `VIX`.
2.  **Implementation**:
    *   Fetch `BTC-USD` data.
    *   Add `BTC_Change`: The daily percentage change of Bitcoin.
    *   Add `BTC_Trend_Score`: A binary or score indicating if BTC is in an uptrend (e.g., Close > MA20).
    *   Add `BTC_RSI`: The 14-day RSI of Bitcoin.
3.  **Model Training**:
    *   Train an XGBoost model with the augmented feature set.
    *   Compare performance (Win Rate, Avg Return) against the Baseline on the Out-of-Sample period (2024-2025).

## 3. Metrics
*   **Win Rate**: Percentage of trades with `Strategy_Ret > 0`. Target: > 58%.
*   **Avg Return**: Mean return per trade. Target: > 0.25%.
*   **Feature Importance**: Verify if `BTC_Change` or `BTC_Trend_Score` appear in the top features.

## 4. Success Criteria
*   Improvement in Win Rate or Avg Return compared to the Baseline.
*   Confirmation that Bitcoin price action acts as a valid "Risk-On" filter for equity momentum.
