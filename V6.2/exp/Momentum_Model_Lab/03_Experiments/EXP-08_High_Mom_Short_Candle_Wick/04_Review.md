# EXP-08: High Momentum Short Strategy - Candle Wick Analysis

## 1. Conclusion
**Result: Success**

The experiment confirms the hypothesis that implementing a tight Stop Loss (SL) relative to the Open price significantly improves the profitability of the High Momentum Short Strategy (`RSI > 70`).

While the **Win Rate** drops dramatically (from 54% to 30% with a 1.0% SL), the **Average Return** increases by nearly **50%** (from +0.20% to +0.30%). This indicates that the "No SL" baseline suffers from large drawdown days where the gap continues to rally (Breakaway Gaps). Cutting these losses early (even if it means stopping out on some winning reversals) yields a better net expectancy.

## 2. Key Findings

| Strategy | Win Rate | Avg Return | Total Return | Stop Out Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (No SL)** | **54.12%** | +0.206% | +6.00 R | 0% |
| **SL 0.2%** | 12.25% | +0.233% | +6.79 R | 87.7% |
| **SL 0.5%** | 20.62% | +0.289% | +8.43 R | 78.0% |
| **SL 1.0%** | 30.00% | **+0.306%** | **+8.91 R** | 65.0% |

1.  **Stop Loss Effectiveness:** The `SL 1.0%` variant is the superior configuration. It achieves the highest Average Return (+0.306%) and Total Return (+8.91 R), outperforming the baseline by 48%.
2.  **Trade-off:** The Win Rate drops significantly (to 30%), meaning the strategy relies on the "fat tail" of successful reversals where the price drops significantly without ever touching the +1.0% SL level.
3.  **Volatility Noise:** The `SL 0.2%` is too tight. It stops out 87.7% of trades, likely getting whipsawed by normal opening volatility, despite having a slightly higher expectancy than baseline.
4.  **Asymmetry:** The success of the SL strategy confirms that "Failed Reversals" (Continuation) are costly. When a High Momentum Gap *does* go up, it goes up a lot. Capping that loss at 1.0% allows the profitable mean-reversion days to carry the PnL.

## 3. Recommendations
1.  **Implement SL 1.0%:** Update the production Short Strategy logic to include a hard Stop Loss at `Open * 1.01`.
2.  **Regime Filter:** Further investigate if the `Stop Out Rate` correlates with specific market regimes (e.g., VIX level). In high VIX environments, a wider SL might be needed.
3.  **Next Step:** Test a "Trailing Stop" or "Move to Breakeven" logic to see if we can improve the Win Rate of the SL strategy without sacrificing the downside protection.
