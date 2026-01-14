# EXP-12: Time-Decay Exit Optimization Review

## 1. Executive Summary
*   **Result:** ❌ Failed (Hypothesis Rejected).
*   **Outcome:** **Retain "Hold to Close" (Market On Close) strategy.**
*   **Key Finding:** The "Hold to Close" strategy (Total Return: +31.15%) vastly outperforms any early exit strategy (Best Early Exit: +7.81%). The data suggests that the "Sell" signal edge materializes slowly throughout the day, rather than being a quick morning flush.

## 2. Metrics Comparison

| Strategy | Win Rate | Avg Return | Total Return | Trade Count |
| :--- | :--- | :--- | :--- | :--- |
| **Ret_MOC (Baseline)** | **53.79%** | **+0.37%** | **+31.16%** | 8438 |
| Ret_4H (~1:30 PM) | 50.11% | +0.09% | +7.82% | 8421 |
| Ret_3H (~12:30 PM) | 49.95% | +0.07% | +6.13% | 8422 |
| Ret_5H (~2:30 PM) | 49.85% | +0.07% | +5.94% | 8421 |
| Ret_2H (~11:30 AM) | 50.87% | +0.07% | +5.69% | 8422 |
| Ret_1H (~10:30 AM) | 49.07% | -0.01% | -0.86% | 8423 |

## 3. Detailed Analysis
### 3.1. The "Morning Fake-Out" Phenomenon
*   **1H Performance (-0.86%):** The negative return in the first hour is striking. Since we are Shorting at the Open, a negative return means the price **rose** in the first hour (above the Open).
*   **Interpretation:** This suggests a common pattern where the stock gaps up/down (triggering the signal), and then **rallies/retraces against our direction** in the first hour ("morning chop" or "trap").
*   **Resolution:** By holding until the close, we allow the intraday trend to reverse back in our favor (fading the morning move).

### 3.2. Alpha Accumulation
*   The returns monotonically increase as time passes (with a slight dip at 5H, but MOC is king).
*   1H: -0.86%
*   2H: +5.69%
*   3H: +6.13%
*   4H: +7.82%
*   **MOC: +31.16%**
*   **Jump to MOC:** The massive jump from 4H/5H to MOC suggests that a significant portion of the alpha comes from the **Closing Auction** or the final hour of trading. This is consistent with institutional flows "unloading" positions EOD.

## 4. Conclusion & Recommendations
*   **Reject Hypothesis:** "Time-Decay" does not apply to this specific Sell Model. The signal is not a "quick scalps" setup but a "day-long distribution" setup.
*   **Strategic Implication:** Do not implement early exits. The patience to hold through morning volatility is the primary source of edge.
*   **Future Investigation:** Investigate the "Morning Fake-Out". Could we improve entry prices by waiting 1 hour?
    *   *New Hypothesis:* Entering at 10:30 AM (or fading the 1H High) might yield better entries than entering at the Open.
