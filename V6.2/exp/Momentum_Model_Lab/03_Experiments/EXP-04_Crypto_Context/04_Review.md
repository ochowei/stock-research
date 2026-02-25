# Experiment Review: EXP-04 Crypto Context Integration

## 1. Outcome Summary
*   **Status**: **Failure** (Win Rate -1.74%, Avg Return -0.057% vs Corrected Baseline)
*   **Hypothesis**: **Rejected**. Adding Bitcoin price action (`BTC_Change`, `BTC_Trend_Score`) as a context feature degraded performance when properly implemented without lookahead bias.
*   **Critical Finding**: During this experiment, a severe **Lookahead Bias** was discovered in the "Baseline" model (V6.1 Parity). The original Baseline used `RSI_14` calculated on `Close[T]` to predict the Intraday Return of Day `T`. This artificially inflated the Win Rate to ~58%.
*   **Corrected Reality**: When `RSI_14` and other features are correctly shifted to use `T-1` data (known at Open), the Baseline Win Rate drops to **48.60%**.

## 2. Performance Metrics (OOS 2024-2025)

| Metric | Corrected Baseline | EXP-04 (Crypto) | Difference |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 48.60% | **46.86%** | **-1.74%** |
| **Avg Return** | -0.032% | **-0.089%** | **-0.057%** |
| **Total Return** | -134% | **-414%** | **-280%** |

*Note: The negative returns indicate that a simple "Buy Gap" strategy with these features is not viable without the lookahead advantage.*

## 3. Feature Importance Analysis (Corrected)
Even though the model performed poorly, it still found BTC features "important", likely overfitting to noise or inverse correlations.

1.  **BTC_Trend_Score**: 0.26 (Dominant feature, but led to worse results)
2.  **BTC_RSI**: 0.17
3.  **BTC_Change**: 0.15
4.  **VIX**: 0.12
5.  **Gap_Pct**: 0.08

**Insight**: The high importance score for BTC features combined with poor performance confirms that the model is finding spurious correlations (noise) rather than predictive signal.

## 4. Conclusion & Next Steps
*   **Action**: **Reject** `BTC_Change`, `BTC_Trend_Score`, and `BTC_RSI`. Do not deploy to production.
*   **Urgent Recommendation**:
    *   **Halt Production Deployment** of V6.1/V6.2 models based on the flawed "58% Win Rate" baseline.
    *   **Refactor Baseline**: The entire feature engineering pipeline must be audited to ensure all features are strictly `T-1` (or earlier) relative to the trade execution time.
    *   **Re-evaluate Strategy**: With a true win rate of ~48%, the "Gap and Go" strategy is likely losing money. We need to investigate "Gap and Fade" (Shorting) or find truly predictive features (e.g., Level 2, Intraday price action in first 5 mins).
