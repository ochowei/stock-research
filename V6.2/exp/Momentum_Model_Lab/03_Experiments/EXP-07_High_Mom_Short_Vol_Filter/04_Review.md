# EXP-07: High Momentum Short Strategy - Volume Filter - Review

## 1. Executive Summary
*   **Result:** **Fail (Hypothesis Rejected)**
*   **Key Finding:** Applying a high volume filter (`Vol_Ratio > 2.0`) to the High Momentum Short Strategy (`RSI_14 > 70`) **did not significantly improve performance**.
    *   **Win Rate:** Improved slightly from 53.84% (Baseline) to **54.46%** (+0.62%), but failed to meet the target of 55%.
    *   **Avg Return:** Decreased slightly from +0.19% to **+0.18%**, indicating that while the filter captures slightly more winners, it misses out on some high-magnitude moves or incurs larger losses on the losers.
    *   **Signal Count:** The filter reduced the sample size from 2,901 to 516 trades (a 82% reduction), making the strategy much harder to trade for minimal gain.

## 2. Performance Analysis

| Strategy | Win Rate | Avg Return | Count | Note |
| :--- | :--- | :--- | :--- | :--- |
| **High_Mom_Short_Base (All RSI>70)** | 53.84% | +0.193% | 2,901 | Robust baseline. |
| **High_Mom_High_Vol_Short (Vol>2.0)** | **54.46%** | **+0.184%** | 516 | **Marginal Improvement.** Not worth the reduced frequency. |
| **High_Mom_Norm_Vol_Short (Vol<=2.0)** | 53.71% | +0.195% | 2,385 | Performs almost identically to baseline. |

### Interpretation
*   **Volume is Neutral in Exhaustion:** The hypothesis that "Extreme Volume = Exhaustion" is not strongly supported by this data. High volume gaps in overbought conditions behave similarly to normal volume gaps.
*   **No Strong Signal:** The difference in Win Rate (54.46% vs 53.71%) is likely noise given the sample size reduction.
*   **Avg Return Decay:** The fact that Avg Return *dropped* suggests that the "High Volume" subset might include some "Breakaway Gaps" that continue higher (blowing out the short), or that the reversals are less violent than expected.

## 3. Comparison to Targets
*   **Win Rate:** 54.46% (Target > 55%). *Missed.*
*   **Avg Return:** +0.18% (Target > 0.30%). *Missed.*

## 4. Conclusion
The experiment failed to produce a superior strategy. The "Volume Ratio > 2.0" filter is not a reliable discriminator for exhaustion gaps in the High Momentum regime.

**Actionable Insight:**
*   **Reject the Volume Filter** for the Short Strategy.
*   The Baseline Short Strategy (RSI > 70, No Vol Filter) remains the best candidate (53.8% Win Rate, +0.19% Avg Return), but it is still borderline for production (needs > 55% Win Rate to cover costs/slippage safely).
*   **Next Steps:** Investigate **Candle Shape** (e.g., Opening vs High, Wick size) or **Intraday Price Action** (e.g., Open vs Prev Close location) to refine the Short entry.
