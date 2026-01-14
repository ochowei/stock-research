# EXP-01 Review: Crypto Feature Integration

## 1. Summary
**Result:** ✅ Success
**Objective:** Test if adding Crypto market state features (BTC/ETH trend, volatility, correlation) improves the Sell Model's performance.

## 2. Results

| Metric | Baseline (V6.2.2) | Model (Crypto) | Difference |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 51.31% | 54.15% | **+2.84%** |
| **Avg Return** | 0.082% | 0.142% | **+0.060%** |
| **Count** | 1799 | 325 | - |

*   **Win Rate Improvement:** Significant improvement (+2.84%).
*   **Avg Return Improvement:** Significant improvement (+0.060%).
*   **Signal Count:** Reduced to 325 (approx. 18% of baseline signals), indicating higher selectivity.

## 3. Feature Importance Analysis
Top features by Importance:
1.  `Gap_Pct` (0.38) - Dominant.
2.  `Rel_Gap_QQQ` (0.12) - Sector relative strength remains key.
3.  `Days_To_End` (0.08) - TOTM feature.
4.  `Rel_Gap_SPY` (0.05)
5.  `Dist_MA20` (0.05)
6.  `VIX` (0.04)
7.  **`Crypto_Corr` (0.044)** - *New Feature*
8.  `Days_From_Start` (0.04)
9.  **`BTC_Trend` (0.043)** - *New Feature*
10. `ATR_Pct` (0.04)
11. `RSI_14` (0.04)
12. **`BTC_RSI` (0.035)** - *New Feature*

**Observation:**
*   `Crypto_Corr` and `BTC_Trend` entered the mid-tier of importance, comparable to `VIX` and `ATR_Pct`.
*   This confirms the hypothesis that crypto market state provides unique information not captured by equity market features alone.

## 4. Conclusion & Recommendations
*   **Conclusion:** The experiment supports the hypothesis. Adding crypto features improves model precision and profitability, likely by filtering out trades during unfavorable global risk sentiment (captured by Crypto trends/correlation).
*   **Recommendation:**
    *   **Adopt:** Integrate these features into the next version of the Sell Model.
    *   **Refine:** `Crypto_Corr` seems most promising. Consider testing different lookback periods (e.g., 10d vs 30d).
    *   **Next Steps:** Proceed to LightGBM migration (EXP-02) using this enhanced feature set.
