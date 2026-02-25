# EXP-02: Sector Relative Strength (Orthogonal Alpha) - Review

## 1. Results
### Performance Metrics (OOS 2024-2025)
| Metric | Baseline (EXP-01) | Model (EXP-02) | Diff |
| :--- | :--- | :--- | :--- |
| Win Rate | 57.59% | 57.51% | -0.08% |
| Avg Return | 0.975% | 0.944% | -0.031% |
| Count (Signals) | 6211 | 7060 | +849 (13.6%) |
| Total Return | 60.57 | 66.65 | +6.08 (10%) |

### Feature Importance
1.  **RSI_14:** 0.337
2.  **Rel_Strength_RSI:** 0.138 (New)
3.  **VIX:** 0.132
4.  **Sector_RSI:** 0.117 (New)
5.  **ATR_Pct:** 0.100

## 2. Analysis
*   **Hypothesis Check:** **Rejected.** The addition of Sector features did **not** improve Win Rate or Average Return.
*   **Observations:**
    *   While `Rel_Strength_RSI` and `Sector_RSI` were ranked highly in feature importance (2nd and 4th), they did not translate to better predictive power.
    *   The model took significantly more trades (+13.6%), likely due to the larger dataset or slightly looser criteria (since feature importance is spread out).
    *   However, the Win Rate dropped slightly (-0.08%), and Avg Return dropped slightly (-0.03%).
    *   This suggests that `Sector_RSI` is largely redundant with `Stock_RSI` (if a stock is up, its sector is usually up) or adds noise for smaller components that don't track the ETF perfectly.
    *   The added complexity of fetching external ETF data (which caused timeouts initially) introduces a failure point without clear benefit.

## 3. Conclusion
*   **Verdict:** **Fail** (or at least, not worth the complexity).
*   **Next Steps:**
    *   Do **not** deploy Sector Relative Strength features to production yet.
    *   Stick to the robust V6.1 Baseline (RSI, ATR, Vol_Ratio).
    *   Proceed to **EXP-03: Volume Microstructure** to see if volume quality filters can improve Win Rate where sector context failed.
