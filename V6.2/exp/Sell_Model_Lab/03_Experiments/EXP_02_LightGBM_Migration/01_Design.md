# EXP-02: LightGBM Migration - Design Document

## 1. Hypothesis
LightGBM will provide competitive or superior performance to XGBoost (used in EXP-01) while potentially being faster and less prone to overfitting due to its leaf-wise growth strategy. We expect a win rate similar to or higher than EXP-01 (approx 53%) with similar selectivity.

## 2. Plan
*   **Base:** EXP-01 (Crypto Feature Integration).
*   **Change:**
    *   Replace `XGBClassifier` with `lightgbm.LGBMClassifier`.
    *   Use the same feature set as EXP-01 (including `BTC_RSI`, `BTC_Trend`, `Crypto_Corr`).
    *   Hyperparameters to test (initial guess): `num_leaves=31`, `learning_rate=0.05`, `n_estimators=200`.
*   **Execution:**
    *   Load data using the same pipeline as EXP-01.
    *   Train LightGBM.
    *   Evaluate on Out-Of-Sample (OOS) data (2024-2025).
    *   Compare with EXP-01 results.

## 3. Metrics
*   **Win Rate:** % of trades with Profit > 0.
*   **Avg Return:** Mean return per trade.
*   **Precision:** (Implied by Win Rate).
*   **Count:** Number of signals generated (Selectivity).

## 4. Success Criteria
*   Win Rate >= EXP-01 (53.86%).
*   Avg Return >= EXP-01 (0.245%).
*   Execution speed improvement is a bonus but secondary to performance.
