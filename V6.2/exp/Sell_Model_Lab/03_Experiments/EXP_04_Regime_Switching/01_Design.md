# EXP-04: Regime-Switching Model (High/Low VIX)

## 1. Hypothesis
Market behavior changes significantly under different volatility regimes. A model trained specifically on High VIX data might learn different patterns than one trained on Low VIX data.
**Hypothesis:** A "Regime-Switching System" that routes trades to specialized models based on VIX will outperform a single "Global" model.

## 2. Experiment Plan
*   **Feature Set:** "Base" features only (Gap_Pct, RSI_14, ATR_Pct, Vol_Ratio, Dist_MA20).
*   **Regime Definition:**
    *   **High Volatility:** VIX (T-1) > 20
    *   **Low Volatility:** VIX (T-1) <= 20
*   **Models:**
    1.  `Model_HighVIX`: Trained only on samples where VIX > 20.
    2.  `Model_LowVIX`: Trained only on samples where VIX <= 20.
    3.  `Model_Global` (Control): Trained on all samples.
*   **Evaluation:**
    *   Test on Out-of-Sample data (2024-2025).
    *   For each test sample, pick the model based on VIX.
    *   Compare "Regime System" metrics (Win Rate, Avg Return) vs "Global Model".

## 3. Metrics
*   Win Rate (%)
*   Average Return per Trade (%)
*   Total Return (Sum of Returns)
*   Number of Signals
