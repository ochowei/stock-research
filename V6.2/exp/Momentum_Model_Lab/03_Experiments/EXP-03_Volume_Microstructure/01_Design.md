# Experiment Design: EXP-03 Volume Microstructure (False Breakout Filter)

## 1. Hypothesis
High price momentum accompanied by low or declining volume ("Fake Breakout") has low persistence. Specifically, gaps that occur when volume has been trending downwards or is significantly below average are more likely to revert.

## 2. Plan
1.  **Baseline**: Utilize the V6.1 Baseline model which uses `RSI_14`, `ATR_Pct`, `Vol_Ratio` (Day/Avg), `Gap_Pct`, and `VIX`.
2.  **Implementation**:
    *   Add `Vol_MA5` (5-day rolling average of volume).
    *   Add `Vol_MA5_Slope`: The rate of change of the 5-day volume moving average. Formula: `(Vol_MA5 - Vol_MA5.shift(1)) / Vol_MA5.shift(1)`.
    *   Retain `Vol_Ratio`: `Prev_Vol / Vol_MA20.shift(1)`.
3.  **Model Training**:
    *   Train an XGBoost model with the augmented feature set.
    *   Compare performance (Win Rate, Avg Return) against the Baseline.

## 3. Metrics
*   **Win Rate**: Percentage of trades with `Strategy_Ret > 0`. Target: > 58%.
*   **Avg Return**: Mean return per trade. Target: > 0.25%.
*   **Sharpe Ratio**: Risk-adjusted return.
*   **Feature Importance**: Verify if `Vol_MA5_Slope` or `Vol_Ratio` appear in the top features.

## 4. Success Criteria
*   Improvement in Win Rate or Avg Return compared to the Baseline (Win Rate ~57.6%, Avg Return ~0.98%).
*   Identification of "Fake Breakouts" (high gap, low/declining volume) leading to lower returns.
