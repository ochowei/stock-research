# EXP-21 Review: Limit Order Entry Optimization

## 1. Experiment Overview
*   **Objective:** Test if placing Limit Orders slightly above the Open price (Short into Strength) improves performance by capturing better entry prices and fading "Morning Fake-Outs".
*   **Hypothesis:** Limit orders (Open + X%) will increase Average Return and Sharpe Ratio, despite lower Fill Rates.
*   **Variants:** Baseline (Open), Limit +0.5%, +1.0%, +1.5%.

## 2. Results
| Scenario | Fill Rate | Signal Count | Win Rate | Avg Return | Total Return | Sharpe |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Open)** | **100.00%** | **6744** | **66.38%** | **1.49%** | **100.71** | **5.46** |
| Limit +0.5% | 76.44% | 5155 | 65.63% | 1.37% | 70.82 | 4.90 |
| Limit +1.0% | 61.45% | 4144 | 66.05% | 1.46% | 60.47 | 5.02 |
| Limit +1.5% | 49.67% | 3350 | 66.21% | 1.57% | 52.69 | 5.20 |

## 3. Analysis
*   **Hypothesis Rejected:** Every Limit Order variant underperformed the Baseline in Total Return and Sharpe Ratio.
*   **Adverse Selection:** Surprisingly, the Average Return for `Limit +0.5%` (1.37%) was *lower* than the Baseline (1.49%). This indicates that the trades we "missed" (stocks that did not rally 0.5% after open) were actually the most profitable ones.
*   **Momentum vs. Reversion:** The data suggests that the best short signals are those that drop immediately. Stocks that rally initially (filling the limit order) demonstrate buying strength that often persists or reduces the magnitude of the subsequent drop.
*   **Opportunity Cost:** waiting for a better price sacrifices 24% - 50% of the trade opportunities, and these missed opportunities are high-quality (immediate winners).

## 4. Conclusion
*   **Action:** **Reject Limit Orders.** Continue using **Market On Close** (Entry at Open) for the V6.2 Sell Model.
*   **Insight:** "Shorting into strength" is a fallacy for this specific Gap Strategy. The alpha is in the immediate gap resolution; waiting for a pullback allows the "strength" to invalidate the bearish thesis.
