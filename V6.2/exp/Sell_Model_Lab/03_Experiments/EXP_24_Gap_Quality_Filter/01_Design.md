# EXP-24: Gap Quality Filter (Volume & Context)

## 1. Objective
To distinguish between "Exhaustion Gaps" (High Probability Short) and "Breakaway Gaps" (Low Probability Short) to improve the V6.2.4.RC Sell Model Win Rate from ~53% to >55%.

## 2. Hypothesis
The current strategy shorts all gaps > 0.5% indiscriminately.
*   **Hypothesis A (Relative Volume):** Gaps accompanied by extreme volume (e.g., 5x average) are more likely to be Breakaway Gaps (Trend continuation), and should be *avoided*. Gaps on moderate volume are more likely to fill.
*   **Hypothesis B (Gap Size):** "Too big to fill?" Extremely large gaps (>2-3%) might indicate fundamental news and should be avoided.
*   **Hypothesis C (Open vs High):** If the Open is near the High of the session (so far, e.g. pre-market high implied), it might indicate buying pressure is exhausted.

## 3. Methodology
We will test filters on the existing V6.2.4.RC predictions (or a Base Model) to see if removing certain trades improves Win Rate.

### 3.1. Variables
1.  **`Vol_Ratio`**: Already in base features, but we will test hard filters (e.g., `Vol_Ratio < 3`).
2.  **`Gap_Size`**: Test bins (0.5-1%, 1-2%, >2%).
3.  **`Rel_Vol_PreMarket`**: (If available, otherwise proxy with Opening Volume).

### 3.2. Architecture
*   Use V6.2.4.RC Base Model logic.
*   Apply filters *post-prediction* or *pre-prediction*.

### 3.3. Data
*   **Train:** 2020-2023
*   **Test:** 2024-2025

## 4. Success Metrics
*   **Win Rate > 55%**
*   **Avg Return > 0.25%**

## 5. Artifacts
*   `gap_quality_analysis.csv`
