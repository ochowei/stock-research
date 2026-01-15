# EXP-15 Review: Crypto-Specific Ensemble (Clean Data Redux)

## 1. Executive Summary
*   **Result:** ❌ **Failed (Hypothesis Rejected)**.
*   **Performance:**
    *   **Control (Base Features):** 52.84% Win Rate, +0.33% Avg Return.
    *   **Test (Base + BTC):** 48.80% Win Rate, +0.29% Avg Return.
*   **Key Insight:** Adding Bitcoin context features (`BTC_Ret`, `BTC_Trend`, `BTC_RSI`) **degraded** performance significantly (-4.04% Win Rate) even on a pure crypto stock pool. The model over-weighted these external features, leading to overfitting or noise injection.

## 2. Detailed Metrics (Test Set: 2024-07-01 to Present)

| Model | Win Rate | Avg Return | Total Return | Trades |
| :--- | :--- | :--- | :--- | :--- |
| **Control (Base)** | **52.84%** | **0.3276%** | **57.66%** | 176 |
| Test (Base + BTC) | 48.80% | 0.2906% | 48.24% | 166 |

## 3. Analysis
*   **Feature Dominance:** `BTC_Ret`, `BTC_Trend`, and `BTC_RSI_14` were the Top 3 most important features in the Test model. This confirms the model *did* learn from them, but this learning was detrimental in the Out-of-Sample period.
*   **Distraction Effect:** The heavy reliance on macro/sector context (BTC) seemingly distracted the model from the asset-specific price action (Gap, RSI, Vol) which proved to be more robust (as seen in the Control model's performance).
*   **Comparison to Tech:** Unlike the Tech sector (where QQQ features *improved* performance), the Crypto sector seems to follow its own idiosyncratic gap resolution logic better than it follows a T-1 Bitcoin trend signal.
*   **Control Performance:** The Base model on Crypto stocks (52.84%) is respectable and aligns with the Non-Tech model performance. It suggests that a dedicated "Crypto Model" with extra features is unnecessary; the standard Base features work best.

## 4. Conclusion & Recommendations
*   **Reject** the Crypto-Specific Ensemble.
*   **Do not deploy** a separate Crypto model pipeline with BTC features.
*   **Action:** Route Crypto stocks to the standard **Non-Tech (Base + SPY)** or just **Base** model. Given the high volatility, treating them as high-beta Tech or just using the Base model is safer.
*   **Next Steps:** Focus optimization efforts elsewhere (e.g., Stop-Loss optimization, EXP-16).
