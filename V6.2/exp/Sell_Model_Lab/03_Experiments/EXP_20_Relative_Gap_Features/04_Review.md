# EXP-20 Review: Relative Gap Features

## 1. Results Summary

| Sector | Model | Win Rate | Avg Return | Count |
| :--- | :--- | :--- | :--- | :--- |
| **Tech** | **Baseline (Control)** | **52.91%** | **0.1971%** | 2731 |
| Tech | Test (+RelGap) | 52.42% | 0.1686% | 2772 |
| **Non-Tech** | **Baseline (Control)** | **53.17%** | **0.1924%** | 4243 |
| Non-Tech | Test (+RelGap) | 52.53% | 0.1889% | 4356 |

## 2. Analysis
*   **Hypothesis Rejected:** Explicitly calculating `Stock_Gap - Index_Gap` **degraded** performance in both sectors.
*   **Tech Sector:** Win Rate dropped by **-0.49%**. The model produced slightly more signals (2772 vs 2731) but of lower quality.
*   **Non-Tech Sector:** Win Rate dropped by **-0.64%**.
*   **Feature Redundancy:** The baseline models already include `Gap_Pct` and `Index_Gap_Pct`. LightGBM (and tree models in general) are capable of learning non-linear interactions between these two features without needing explicit engineering.
*   **Noise Injection:** Adding the linear difference likely added noise or diluted the importance of the raw features, leading to suboptimal splits.
*   **Feature Importance:** In both sectors, the raw Index features (`Index_Gap_Pct`, `Index_RSI`) remained the top predictors. The new `Rel_Gap` feature did not displace them in the top 5 (based on the implementation output).

## 3. Conclusion & Recommendation
*   **Do Not Adopt:** We will **not** include `Rel_Gap` features in the production model.
*   **Stick to Baseline:** The current V6.2.4.RC architecture (Base + Raw Index Features) remains the champion.
*   **Insight:** "Don't fix what isn't broken." The model already understands the context of the gap relative to the market via the existing feature set. Explicitly enforcing a linear relationship (Subtraction) might be too rigid compared to the decision boundaries the tree learns naturally.

## 4. Next Steps
*   Mark EXP-20 as **Failed** in the backlog.
*   Proceed to the next experiment (EXP-21: Limit Order Entry) which addresses execution rather than feature engineering.
