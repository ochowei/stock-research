# EXP-03: Volume Microstructure (False Breakout Filter)

## 1. Hypothesis (假設)
**Context:** Momentum strategies often fail when price breaks out on low volume ("Fake Breakout"), indicating a lack of institutional participation.
**Hypothesis:** High price momentum accompanied by declining or low relative volume has low persistence. Filtering these setups will improve Win Rate.
**Logic:**
*   **Strong Breakout:** Price Up + Volume Up (High Participation).
*   **Weak Breakout:** Price Up + Volume Down (Divergence).

## 2. Implementation Plan (實作計畫)

### **Features to Add:**
1.  **`Vol_MA5_Slope`**: The slope of the 5-day Volume Moving Average leading up to the gap.
    *   Formula: `(Vol_MA5_T-1 - Vol_MA5_T-6) / Vol_MA5_T-6`
    *   Interpretation: Positive slope = Increasing volume trend into the breakout.
2.  **`Vol_Ratio`**: Existing feature (Volume T-1 / Vol_MA20 T-1).
    *   Already in baseline but critical to re-validate with Slope.

### **Model Configuration:**
*   **Algorithm:** XGBoost Classifier (same as Baseline).
*   **Features:** `['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Vol_MA5_Slope', 'Gap_Pct', 'VIX']`
*   **Target:** `Strategy_Ret > 0.2%` (Intraday Momentum).

## 3. Success Metrics (驗收標準)
*   **Win Rate:** > 57.6% (Baseline V6.1).
*   **Avg Return:** > 0.98% (Baseline V6.1).
*   **Feature Importance:** `Vol_MA5_Slope` should rank in top 5 features.
