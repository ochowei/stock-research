# EXP-11 Review: Non-Tech Feature Augmentation (SPY Context)

## 1. Summary
*   **Status:** ✅ Success
*   **Hypothesis:** Validated. Adding SPY features significantly improved performance for the Non-Tech sector, mirroring the success of QQQ features for the Tech sector.
*   **Outcome:** Adopt the "Base + SPY" model for Non-Tech stocks.

## 2. Results Comparison (Test Set: 2024-2025)

| Model | Win Rate | Avg Return | Trades |
| :--- | :--- | :--- | :--- |
| **Non-Tech Baseline** | 52.01% | +0.13% | 4774 |
| **Non-Tech + SPY Features** | **52.84%** | **+0.19%** | 4351 |
| **Delta** | **+0.83%** | **+0.06%** | -423 |

## 3. Analysis

### Performance
*   **Win Rate:** The experiment achieved a **+0.83%** increase in Win Rate (from 52.01% to 52.84%). This is a substantial improvement, nearly reaching the 53% target.
*   **Profitability:** Average Return increased by **46%** (from 0.13% to 0.19% per trade).
*   **Selectivity:** The model became slightly more selective (discarding ~8.8% of trades), which aligns with the higher quality of remaining signals.

### Feature Importance
SPY features completely dominated the model, occupying the top 3 spots. This confirms that Non-Tech stocks are heavily driven by broad market sentiment.

| Feature | Importance | Rank |
| :--- | :--- | :--- |
| **SPY_Gap_Pct** | 433 | 1 |
| **SPY_RSI_14** | 351 | 2 |
| **SPY_Dist_MA20** | 346 | 3 |
| Vol_Ratio | 157 | 4 |
| Gap_Pct | 154 | 5 |

*   **Interpretation:** The "market context" (where SPY opens and its trend) is more predictive of a Non-Tech stock's intraday reversal than the stock's own technicals. This is a critical finding.

## 4. Conclusion & Recommendations
1.  **Adopt:** Update the V6.3 Production System to use the `NonTech_SPY_Feats` model for all Non-Tech tickers.
2.  **Next Step:** Since we now have optimized Tech (QQQ) and Non-Tech (SPY) models, we should verify the combined system performance.
3.  **Observation:** The dominance of SPY features suggests we might even be able to prune some stock-specific features later, but for now, the combined set works well.
