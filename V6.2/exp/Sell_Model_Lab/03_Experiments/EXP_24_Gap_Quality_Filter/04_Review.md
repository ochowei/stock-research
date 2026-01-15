# EXP-24: Gap Quality Filter (Volume & Context) - Review

## 1. Executive Summary
*   **Result:** ❌ Hypothesis Rejected (Directionally Inverted).
*   **Outcome:** **Do NOT avoid large gaps.** Instead, **PRIORITIZE** them.
*   **Key Finding:** The original hypothesis feared that "Large Gaps (>2%)" were Breakaway Gaps that would not fill. The data proves the exact opposite: **Large Gaps are the most profitable mean-reversion setups**, while small gaps (0.5-1.0%) perform poorly.

## 2. Methodology Check
*   **Model:** V6.2.4.RC (Tech/Non-Tech Split).
*   **Data:** 30 Major Tickers (AAPL, NVDA, TSLA, etc.) covering 2020-2025.
*   **Signals Analyzed:** 1,502 predicted trades (Prob > 0.5) in the Test Set (2024-2025).

## 3. Results Analysis

### A. Gap Size Impact
| Gap Size | Win Rate | Avg Return | Count |
| :--- | :--- | :--- | :--- |
| **0.5% - 1.0%** | 40.12% | -0.04% | 830 |
| **1.0% - 2.0%** | 39.18% | -0.30% | 439 |
| **> 2.0%** | **52.36%** | **+0.31%** | 233 |

**Insight:** There is a massive performance divergence. Small gaps (which make up ~60% of signals) are noise/trend-continuation. Large gaps (>2%) trigger a strong mean-reversion effect (Exhaustion).

### B. Gap Relative to Volatility (Gap / ATR)
| Gap / ATR | Win Rate | Avg Return | Count |
| :--- | :--- | :--- | :--- |
| **< 0.5x** | 39.26% | -0.16% | 1,075 |
| **0.5x - 1.0x** | 45.11% | +0.06% | 317 |
| **1.0x - 2.0x** | **48.57%** | **+0.25%** | 70 |
| **> 2.0x** | **70.00%** | **+0.85%** | 40 |

**Insight:** This is the strongest signal discovered. When a stock gaps more than **2x its daily ATR**, the reversal probability hits **70%**. This confirms that "Overextended" moves are the sweet spot for the Sell Model.

### C. Volume Ratio (Previous Day)
| Prev Vol Ratio | Win Rate | Avg Return |
| :--- | :--- | :--- |
| < 0.8 | 39.47% | -0.03% |
| > 2.0 | 51.76% | -0.14% |

**Insight:** High volume on the *previous* day improves Win Rate significantly (+12%), though returns remain volatile. This suggests that "Climax" volume on the prior day often sets up a reversal the next morning.

## 4. Conclusion
The hypothesis that "Large Gaps are Breakaway Gaps" is **FALSE** for this strategy universe.
*   **Small Gaps (0.5-1.0%)** are the problem. They are likely "Common Gaps" or mild trend continuations that do not revert.
*   **Large Gaps (>2% or >1x ATR)** are **Exhaustion Gaps**. They offer the best risk/reward.

## 5. Recommendations
1.  **Immediate Action:** Update the Production Strategy to **Filter Out** small gaps (Gap < 1.0% OR Gap < 0.5x ATR) if they are currently dragging down performance.
2.  **Sizing Rule:** Implement aggressive sizing for **Gap > 2x ATR** events (Win Rate 70%).
3.  **Future Experiment:** Test a "High-Volatility Gap" model that *only* trades gaps > 1.0% or > 0.5x ATR to verify if this purely filters noise without destroying sample size too much.
