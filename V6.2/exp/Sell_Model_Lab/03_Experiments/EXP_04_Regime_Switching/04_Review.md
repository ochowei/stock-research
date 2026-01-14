# EXP-04: Regime-Switching Model Review

## 1. Experiment Overview
*   **Objective:** Determine if splitting the model into High VIX (>20) and Low VIX (<=20) regimes improves performance.
*   **Hypothesis:** Market behavior changes with volatility; specialized models should outperform a generalist global model.
*   **Model:** LightGBM with "Base" feature set (Gap_Pct, RSI_14, ATR_Pct, Vol_Ratio, Dist_MA20).
*   **Data Period:** Training (2020-2023), Testing (2024-2025).

## 2. Results (2024-2025 Out-of-Sample)

| Metric | Global Model | Regime System | Diff |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 52.11% | **52.33%** | +0.22% |
| **Avg Return** | 0.0013 | 0.0013 | -0.0000 |
| **Total Return** | 10.63 | **10.74** | +0.11 |
| **Signals** | 8159 | 8335 | +176 |

## 3. Analysis
*   **Marginal Improvement:** The regime-switching system showed a very slight improvement in Win Rate (+0.22%) and Total Return (+1.08% relative increase).
*   **Signal Volume:** The regime system generated slightly more signals (+2.1%), suggesting that specialized models might be slightly more confident in their respective domains, or finding edge cases that the global model smooths over.
*   **Complexity vs. Reward:** The performance gain is minimal. Maintaining two separate models adds infrastructure complexity (routing logic, double training) for <0.25% win rate gain.
*   **Hypothesis Verdict:** **Weak Support**. While specialized models didn't hurt, they didn't provide the "step-change" improvement expected. The "Global" model seems robust enough to handle varying VIX levels, likely because `ATR_Pct` and `Vol_Ratio` already encode volatility information effectively.

## 4. Conclusion & Recommendation
*   **Conclusion:** Splitting by VIX=20 provides a negligible benefit. The added complexity is not justified by the marginal performance boost.
*   **Recommendation:**
    1.  **Discard** the Regime-Switching architecture for now. Stick to the single Global Model.
    2.  **Pivot** to Sector-Specific models (EXP-05). It is possible that "Sector" is a more meaningful regime than just "VIX". Tech stocks might behave fundamentally differently than Utilities, regardless of VIX.
    3.  **Investigate** if VIX should be explicitly added as a feature to the Base model (Control), rather than just a splitter. (Note: EXP-03 suggested it caused overfitting, but maybe as an interaction term it helps).

## 5. Next Steps
*   Proceed to **EXP-05: Sector-Specific Ensembles**.
