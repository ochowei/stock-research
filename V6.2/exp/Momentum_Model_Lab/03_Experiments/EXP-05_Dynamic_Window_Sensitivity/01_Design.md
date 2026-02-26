# EXP-05: Dynamic Window Sensitivity (Lookback Tuning)

## 1. Hypothesis
The standard 14-day lookback window (RSI_14, ATR_14) may be too slow for the current high-volatility regime (2024-2025). Shorter windows (e.g., 5 or 10 days) might capture momentum shifts earlier, improving entry timing and win rates, while longer windows (e.g., 20 or 50 days) might filter out noise but react too slowly.

**Core Question:** Does reducing the lookback window improve the predictive power of momentum indicators for "Gap Up -> Continuation" setups?

## 2. Experiment Plan
*   **Target Variable:** `Strategy_Ret` = (Close - Open) / Open
    *   Label = 1 if `Strategy_Ret` > 0.2% (PROFIT_THRESHOLD)
*   **Model:** XGBoost Classifier (Fixed Hyperparameters)
*   **Training Period:** 2020-01-01 to 2023-12-31
*   **Testing Period (OOS):** 2024-01-01 to Present

### Variables
*   **Independent Variable (Window Size `W`):** `[5, 10, 14, 20, 50]`
*   **Features per Window:**
    *   `RSI_W` (Relative Strength Index)
    *   `ATR_W_Pct` (ATR / Prev_Close)
    *   `ROC_W` (Rate of Change / Momentum)
*   **Control Features:**
    *   `Gap_Pct`
    *   `Vol_Ratio` (Volume / MA20_Vol)
    *   `VIX`

### Look-Ahead Bias Correction
**CRITICAL:** All Close-based indicators (RSI, ATR, ROC, VIX) must be calculated using `Close` but then **shifted by 1** to represent the value available at `T_Open`.
*   Example: `Feature_T` = `Indicator(Close_T-1)`

## 3. Success Metrics
*   **Win Rate:** Comparison against Baseline (Window=14).
*   **Avg Return:** Must not degrade significantly.
*   **Signal Count:** Ensure shorter windows don't drastically reduce sample size due to volatility noise.

## 4. Output Artifacts
*   `comparison_results.csv`: Summary table of performance by Window size.
*   `window_comparison.png`: Equity curves for each window size.
*   `momentum_model_W{best}.joblib`: The best performing model.
