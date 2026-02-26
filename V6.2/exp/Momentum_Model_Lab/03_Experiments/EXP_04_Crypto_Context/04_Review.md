# EXP-04 Review: Crypto Context Integration

## 1. Executive Summary
*   **Status:** **Failed** (Hypothesis Rejected)
*   **Experiment:** Tested whether adding `BTC` and `ETH` trend/momentum features improves Equity Momentum strategies.
*   **Result:** Adding Crypto Context features **significantly degraded** performance in the 2024-2025 Out-of-Sample period.
    *   **Win Rate:** Decreased by **2.42%** (56.24% -> 53.82%).
    *   **Avg Return:** Decreased by **0.184%** (1.213% -> 1.029%).

## 2. Detailed Metrics
| Metric | Baseline (Equity Only) | Experiment (Crypto Context) | Difference |
| :--- | :--- | :--- | :--- |
| **Win Rate** | **56.24%** | 53.82% | -2.42% |
| **Avg Return** | **1.213%** | 1.029% | -0.184% |
| **Signal Count** | 3846 | 4110 | +264 |

## 3. Feature Importance Analysis
Despite the poor performance, the model assigned significant importance to Crypto features, indicating they were "noisy predictors" that the model overfitted to during training (2020-2023), but which failed to generalize to 2024-2025.

1.  **RSI_14 (Equity):** 0.32 (Dominant)
2.  **BTC_RSI (Context):** 0.13
3.  **ETH_Ret (Context):** 0.11
4.  **BTC_Trend (Context):** 0.11
5.  **BTC_Ret (Context):** 0.10
6.  **VIX (Context):** 0.09

The fact that `BTC_RSI` was the 2nd most important feature yet performance dropped confirms that **Crypto momentum is not a stable predictor of Equity momentum** for this asset pool.

## 4. Key Learnings & Conclusion
1.  **Crypto is Idiosyncratic:** Bitcoin and Ethereum price action does not consistently lead broad equity momentum for the selected asset pool (Tech/Growth).
2.  **Overfitting Risk:** The high feature importance of Crypto variables combined with poor OOS performance is a classic sign of overfitting. The relationship likely existed in the 2020-2021 "Everything Bubble" but broke down or inverted in 2024-2025.
3.  **Simplicity Wins:** The Baseline model relying on internal momentum (RSI) and Volatility (VIX) remains robust. Adding external asset class context introduced noise.

## 5. Next Steps
*   **Action:** **Discard** Crypto Context features for the general Production Model.
*   **Recommendation:** Focus on internal market breadth (e.g., % of stocks > MA50) or sector-specific context rather than cross-asset correlations which are unstable.
*   **Update:** Mark EXP-04 as Failed in the backlog.
