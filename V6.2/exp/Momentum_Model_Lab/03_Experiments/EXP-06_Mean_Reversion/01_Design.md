# EXP-06: Mean Reversion Signal (Gap Fade)

## 1. Hypothesis
High Momentum (RSI > 70, ROC > 5%) at T-1 predicts gap fading (Reversal) rather than continuation. The hypothesis is that when a stock is already overbought (High RSI) and then gaps up, it is an exhaustion move, and the price is likely to revert intraday (Open > Close).

**Core Question:** Can we generate positive Alpha by Shorting Gap Ups on High Momentum stocks?

## 2. Experiment Plan
*   **Target Variable:** `Strategy_Ret` = (Open - Close) / Open  (Short Strategy)
*   **Filter Condition:** `Gap_Pct` > 0.5%
*   **Regime Split:**
    *   **High Momentum:** `RSI_14` > 70
    *   **Low Momentum:** `RSI_14` <= 70 (Control Group)
*   **Training Period:** 2020-01-01 to 2023-12-31
*   **Testing Period (OOS):** 2024-01-01 to Present

### Variables
*   **Independent Variable:** Momentum Regime (High vs Low)
*   **Features:**
    *   `RSI_14` (T-1)
    *   `ROC_14` (T-1)
*   **Control Features:**
    *   `Gap_Pct`
    *   `Vol_Ratio`

### Look-Ahead Bias Correction
**CRITICAL:** All Close-based indicators (RSI, ATR, ROC, VIX) must be calculated using `Close` but then **shifted by 1** to represent the value available at `T_Open`.

## 3. Success Metrics
*   **Win Rate:** > 55% for the Short Strategy in High Momentum Regime.
*   **Avg Return:** > 0.25% per trade.
*   **Comparison:** Short Strategy Return (High Mom) vs Long Strategy Return (High Mom).

## 4. Output Artifacts
*   `performance_report.csv`: Summary table of performance (Long vs Short) by Momentum Regime.
*   `equity_curve.png`: Equity curves for High Mom Short vs Baseline Long.
