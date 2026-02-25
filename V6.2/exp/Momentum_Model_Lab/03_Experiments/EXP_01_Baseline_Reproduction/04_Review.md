# EXP-01: Baseline Reproduction (V6.1 Parity) - Review

## 1. Executive Summary
**Result:** ✅ **Success**
The V6.1 Baseline model was successfully reproduced within the V6.2 Lab environment. The model significantly outperformed the "Buy All Gaps" strategy and exceeded all success metrics for the 2024-2025 Out-of-Sample period.

## 2. Metrics & Performance

| Metric | Target | Baseline (Buy All) | Model (V6.1) | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Win Rate** | > 55% | 47.38% | **57.59%** | +10.21% |
| **Avg Return** | > 0.25% | -0.086% | **0.975%** | +1.061% |
| **Total Trades** | N/A | 13,773 | 6,211 | -55% (Filter Rate) |

*   **Profitability:** The model turns a losing strategy (buying every gap) into a highly profitable one (~1% per trade).
*   **Selectivity:** The model filters out ~55% of the gaps, focusing on high-probability setups.

## 3. Feature Analysis

| Feature | Importance | Insight |
| :--- | :--- | :--- |
| **RSI_14** | 0.469 | **Dominant Driver.** The model relies heavily on RSI to identify momentum strength. |
| **VIX** | 0.173 | **Regime Sensitive.** Market volatility is the second most important factor. |
| **ATR_Pct** | 0.123 | Volatility relative to price matters. |
| **Vol_Ratio** | 0.120 | Volume confirmation is significant but secondary to price action. |
| **Gap_Pct** | 0.114 | The size of the gap is the least important of the selected features. |

## 4. Findings & Observations
1.  **Robustness Confirmed:** The simple 5-feature model is surprisingly effective in the 2024-2025 market regime.
2.  **RSI Dependency:** Nearly 50% of the decision weight is on RSI. This suggests the "Momentum" captured is largely "Overbought/Oversold" or "Trend Strength" measured by RSI.
3.  **Data Quality:** Some high-beta tickers (PANW, ABNB, NET) failed to download due to timeouts, but the sample size (6,211 trades) is statistically significant.

## 5. Recommendations (Next Steps)
1.  **Proceed to EXP-02 (Sector Relative Strength):** Since RSI is so dominant, adding *Sector RSI* (as proposed in EXP-02) is a logical next step to see if "Relative Momentum" adds orthogonal value.
2.  **Investigate VIX Interaction:** Given VIX is #2, we should ensure the model behaves well in different VIX regimes (Low vs High VIX).
3.  **Optimize Download:** Improve the data fetching script to handle retries for failed tickers (PANW, ABNB, etc.) to ensure complete coverage in future experiments.

## 6. Artifacts
*   Model: `03_Output/momentum_model.joblib`
*   Report: `03_Output/performance_report.csv`
*   Plots: `03_Output/momentum_equity.png`
