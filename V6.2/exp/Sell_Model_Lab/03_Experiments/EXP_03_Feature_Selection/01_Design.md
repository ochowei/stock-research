# EXP-03: Feature Selection (Ablation Study)

## 1. Hypothesis
*   **Hypothesis**: Some features in the current "All" set (V6.2.2 + Crypto) may be redundant or noisy. Specifically, we suspect that `Dist_MA20` or `Vol_Ratio` might be less predictive when powerful market features like `VIX` and `Crypto_Corr` are present.
*   **Goal**: Identify the minimal optimal feature set that maintains or improves performance (Win Rate / Avg Return) while reducing model complexity.

## 2. Plan
*   **Base Model**: LightGBM (from EXP-02 success).
*   **Technique**:
    1.  **Permutation Importance**: Analyze feature importance on the full model to rank features by their impact on validation accuracy.
    2.  **Subset Testing (Ablation)**: Train and evaluate models on specific subsets:
        *   **Subset A (Base 5)**: `['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']` (The classic technicals).
        *   **Subset B (Base + TOTM)**: Subset A + `['Days_From_Start', 'Days_To_End']`.
        *   **Subset C (Base + Crypto)**: Subset A + `['BTC_RSI', 'BTC_Trend', 'Crypto_Corr']`.
        *   **Subset D (All)**: All available features (including VIX, Rel_Gap).
*   **Metrics**: Win Rate, Average Return, Number of Signals (Selectivity).

## 3. Success Criteria
*   Identify a subset that performs within 1% of the "All" model's Win Rate but with fewer features.
*   OR identify a subset that significantly outperforms the "All" model by removing noise.
