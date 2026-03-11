# EXP-08: High Momentum Short Strategy - Candle Wick Analysis

## 1. Hypothesis
**Hypothesis:** True reversals in High Momentum (`RSI > 70`) Gap Up scenarios fail almost immediately. If a gap opens and pushes significantly higher (forming a large upper wick), it is likely a continuation breakout, not a reversal.
Therefore, implementing a **Tight Stop Loss (SL)** relative to the Open price should improve the net profitability of the Short Strategy by cutting losses early on continuation days, even if it slightly reduces the win rate due to volatility noise.

## 2. Plan
We will simulate a Short Strategy at the Open price for High Momentum Gaps (`Gap > 0.5%`, `RSI_14 > 70`) and compare the performance of different Stop Loss (SL) thresholds.

### Variants to Test:
1.  **Baseline (No SL):** Short at Open, Close at Close. Return = `(Open - Close) / Open`.
2.  **SL 0.2%:** If `(High - Open) / Open >= 0.002`, Return = `-0.002`. Else Baseline.
3.  **SL 0.5%:** If `(High - Open) / Open >= 0.005`, Return = `-0.005`. Else Baseline.
4.  **SL 1.0%:** If `(High - Open) / Open >= 0.01`, Return = `-0.01`. Else Baseline.

## 3. Metrics
*   **Win Rate (%):** Percentage of profitable trades.
*   **Avg Return (%):** Average return per trade.
*   **Net Profit (Sum R):** Sum of all returns (proxy for total profit).
*   **Stop-Out Rate (%):** Percentage of trades that hit the SL.
*   **Sharpe Ratio (Proxy):** Avg Return / Std Dev of Return.

## 4. Success Criteria
*   A specific SL threshold significantly improves **Avg Return** or **Net Profit** compared to the Baseline.
*   The strategy remains viable with `> 50` signals per year.
