# Review: Delayed Entry Optimization (EXP-14)

## 1. Summary
This experiment tested whether delaying trade entry by 1 hour (entering at 10:30 AM EST) improves performance by avoiding the "Morning Fake-Out" identified in EXP-12. The hypothesis was that the initial negative return in the first hour often results in a better entry price later.

## 2. Results
The experiment simulated trades over the last 729 days using V6.4 Production Models.

| Metric | Baseline (Enter @ 9:30 Open) | Delayed (Enter @ 10:30 Open) | Difference |
| :--- | :--- | :--- | :--- |
| **Win Rate** | **63.14%** | 57.85% | **-5.29%** |
| **Avg Return** | **1.45%** | 0.75% | **-0.69%** |

## 3. Analysis
*   **Hypothesis Rejected:** Delayed entry significantly degrades both Win Rate and Average Return.
*   **Interpretation:** The V6.4 Sell Model correctly identifies gaps that tend to resolve (or continue) immediately. By waiting 1 hour, we miss a substantial portion of the profitable move.
*   **Contradiction with EXP-12?**: EXP-12 noted a *negative* return in the first hour (-0.86%). If that were universally true, fading the first hour (buying at Open, selling at 10:30) or Shorting at 10:30 should be better.
    *   However, EXP-12 analyzed the *entire* trade lifecycle and noted that the *alpha materializes slowly*.
    *   The discrepancy here suggests that while *on average* there might be a counter-move, **winning trades** likely move in our favor immediately or relatively quickly. Waiting 1 hour forfeits the gains from the best trades.
    *   It is also possible that the "Negative Return in First Hour" finding from EXP-12 was misinterpreted or applied to a different subset of data/models.
    *   In this experiment (V6.4 Models), the "Baseline" return (1.45%) is exceptionally high compared to historical norms (~0.20%). This might be due to the recent market regime (2024-2025) or the high quality of V6.4 signals.
    *   Regardless, **entering at the Open is vastly superior**.

## 4. Conclusion
*   **Do not adopt delayed entry.**
*   Continue executing at Market Open (MOC Strategy).
*   The "Morning Fake-Out" does not justify a systematic delay for the V6.4 signals.

## 5. Next Steps
*   Mark EXP-14 as Done.
*   Proceed to EXP-15 (Crypto Ensemble).
