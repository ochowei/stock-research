# EXP-02: LightGBM Migration - Review

## 1. Results Summary

The goal of this experiment was to migrate the Sell Model from XGBoost to LightGBM, hypothesizing that LightGBM would offer better handling of the feature set (especially with new Crypto features) and potentially reduce overfitting.

During implementation, a critical data leakage issue was identified in the VIX feature (using today's Close instead of T-1). This was corrected. We also identified and fixed potential look-ahead bias in `Dist_MA20` calculation and Crypto indicator shifts.

### Performance Metrics (OOS 2024-2025)

| Metric | Baseline (XGBoost) | Model (LightGBM) | Difference |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 52.25% | 52.23% | -0.02% |
| **Avg Return** | 0.124% | 0.159% | **+0.036%** |
| **Total Return** | 1288.2% | 1150.4% | -137.8% |
| **Count (Trades)** | 10422 | 7214 | -3208 |

*Note: The baseline metrics here are re-calculated on the same dataset/subset for fair comparison, as the exact dataset size slightly fluctuates due to data availability/cleaning.*

## 2. Findings

1.  **Selectivity Improvement:** LightGBM was significantly more selective (7214 trades vs 10422), filtering out ~30% of the baseline trades.
2.  **Profitability:** Despite a similar Win Rate (-0.02%), the **Average Return per trade increased by ~29%** (0.124% -> 0.159%). This suggests LightGBM is better at identifying higher-quality setups or avoiding large losers.
3.  **Feature Importance:**
    *   **VIX** remains the dominant feature (#1).
    *   **Crypto Features (BTC_Trend, BTC_RSI, Crypto_Corr)** are extremely significant, occupying 3 of the top 7 spots. This confirms the hypothesis from EXP-01 that crypto market state is a strong predictor for equity sell-offs/gaps.
    *   **Vol_Ratio** had 0 importance, suggesting it might be noise in this tree configuration.

## 3. Conclusion

**SUCCESS.** Although the Win Rate is flat, the significant improvement in Average Return and the increased selectivity align with the lab's goal of "Quality over Quantity". The model takes fewer trades but they are more profitable on average.

## 4. Recommendations

1.  **Adopt LightGBM** as the new standard model for future experiments.
2.  **Investigate Vol_Ratio:** It has 0 importance. Consider removing it or re-engineering it (e.g., Vol_Shock).
3.  **VIX Sensitivity:** Since VIX is the #1 feature, future experiments should explore "Regime Switching" (EXP-04) based on VIX levels to see if we can further boost performance in high/low vol environments.
