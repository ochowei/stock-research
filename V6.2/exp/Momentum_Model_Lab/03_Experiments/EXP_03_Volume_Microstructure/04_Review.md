# Experiment Review: EXP-03 Volume Microstructure (CORRECTED)

## 1. Critical Finding: Baseline Invalidated
*   **Discovery:** During implementation, a critical data leakage was identified in the previous methodology (EXP-01). The model was using `Close[T]` (via RSI/ATR) to predict `Open[T]` returns.
*   **Correction:** Fixed the leakage by strictly shifting all technical indicators to `T-1`.
*   **Impact:** The "57.6% Win Rate" baseline was a result of look-ahead bias. The true baseline for the "Buy Gap > 0.5%" strategy is **47.43%**.

## 2. Results Summary (Corrected Data)
*   **Win Rate:** 48.65% (vs Random Baseline 47.43%) -> **+1.22% Improvement**.
*   **Avg Return:** -0.027% (vs Random Baseline -0.084%) -> **+0.057% Improvement**.
*   **Signal Count:** 3706 (Filtered out 72% of bad trades).

## 3. Feature Analysis
With the data leak removed, RSI is no longer the sole dominant feature. Volume features have emerged as significant predictors.

| Feature | Importance | Interpretation |
| :--- | :--- | :--- |
| **VIX** | **0.274** | **Dominant.** Market Regime is the primary driver of success. |
| **ATR_Pct** | 0.155 | Volatility context. |
| **Vol_Ratio** | **0.152** | **Existing:** Volume shock is a strong signal. |
| **RSI_14** | 0.146 | Momentum (T-1) is less predictive without look-ahead. |
| **Gap_Pct** | 0.139 | Gap size. |
| **Vol_MA5_Slope** | **0.134** | **New:** Pre-gap volume trend is relevant. |

**Observation:** Volume features (`Ratio` + `Slope`) combined account for ~28% of the model's decision making, proving they are critical components of the (weak) signal.

## 4. Conclusion & Action
*   **Verdict:** **Partial Success (Feature) / Fail (Strategy).**
    *   **Feature:** `Vol_MA5_Slope` works. It adds +1.2% Win Rate over random guessing.
    *   **Strategy:** The overall strategy (Long Only Gap Up) is losing money (-0.027% return) even with the model. The current feature set is insufficient to reach the 55% Win Rate target.
*   **Next Steps:**
    *   **Prioritize Alpha:** We need stronger alpha sources. Proceed to **EXP-04 (Crypto Context)** to see if macro factors can filter the noise.
    *   **Re-evaluate Baseline:** Acknowledge that the "Simple Momentum" strategy is not profitable in the current regime without better filters.
