# EXP-25 Review: Large Gap Specialization

## 1. Experiment Overview
*   **Hypothesis:** A specialized model trained *only* on large gaps (>2%) will outperform the general V6.2.4.RC model on that specific subset by learning the unique "exhaustion" dynamics of high-volatility moves.
*   **Target Regime:** Gap > 2%.
*   **Metric Goal:** Win Rate > 55% or significantly better than Baseline.

## 2. Results
| Metric | Baseline (V6.2.4.RC) | Specialized (High-Vol) | Delta |
| :--- | :--- | :--- | :--- |
| **Win Rate** | **54.21%** | 52.29% | -1.92% |
| **Avg Return** | **+0.51%** | +0.26% | -0.25% |
| **Trade Count** | 2,579 | 3,534 | +955 |

## 3. Analysis
*   **Failure of Specialization:** The hypothesis is **rejected**. The specialized model significantly underperformed the baseline general model (-1.92% Win Rate).
*   **Data Scarcity vs. Generalization:** Even though large gaps have unique dynamics, the specialized model was trained on only ~6,000 samples (vs ~100k+ for the general model). The loss of training data outweighed the benefit of regime specificity.
*   **Baseline Robustness:** The V6.2.4.RC model (Base + Sector Context) is remarkably robust. It correctly generalizes to large gaps (achieving a solid 54.2% Win Rate and +0.51% Avg Return) without needing explicit retraining.
*   **Overfitting Risk:** The specialized model likely overfitted to the noise in the smaller high-vol dataset, whereas the general model learned more stable feature interactions (RSI, ATR) from the broader dataset.

## 4. Conclusion
*   **Outcome:** ❌ Failed (Hypothesis Rejected).
*   **Action:** Do not deploy a specialized High-Vol model. Continue using the V6.2.4.RC/V6.2.5.RC architecture for all gap sizes.
*   **Insight:** "More data beats better data" in this context. The structural signals (Reversion to Mean) are consistent across gap sizes (0.5% to 5.0%), so splitting the data fragments the signal more than it purifies it.
