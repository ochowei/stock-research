# Experiment Review: EXP-03 Volume Microstructure

## 1. Results Summary
| Metric | Baseline (V6.1) | EXP-03 (Vol Trend) | Diff |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 57.96% | 57.84% | -0.12% |
| **Avg Return** | 0.967% | 0.936% | -0.031% |
| **Total Return** | 5879% | 5802% | -77% |

## 2. Analysis
*   **Hypothesis Rejection**: The hypothesis that declining volume trends (`Vol_MA5_Slope`) indicate "Fake Breakouts" was **rejected**. The addition of this feature slightly degraded both Win Rate and Average Return.
*   **Feature Importance**:
    1.  `RSI_14` (0.41) - Dominant
    2.  `VIX` (0.17)
    3.  `ATR_Pct` (0.11)
    4.  `Vol_Ratio` (0.11)
    5.  `Gap_Pct` (0.11)
    6.  `Vol_MA5_Slope` (0.09) - Lowest
*   **Interpretation**: The model assigned the lowest importance to the new volume trend feature. This suggests that the volume trend leading up to the gap is not a strong predictor of the gap's resolution. The existing `Vol_Ratio` (Gap Day Volume / Avg) captures the immediate liquidity shock better than the pre-gap trend.

## 3. Conclusion & Recommendations
*   **Conclusion**: Adding pre-gap volume trend complexity adds noise. The market regime (VIX) and momentum state (RSI) are far more critical.
*   **Next Steps**:
    *   Abandon `Vol_MA5_Slope`.
    *   Proceed to **EXP-04 (Crypto Context)** or **EXP-05 (Lookback Tuning)** as per backlog priority.
    *   Consider if "Volume" needs to be measured differently (e.g., Intraday Volume Profile), but for daily data, this avenue is likely exhausted.
