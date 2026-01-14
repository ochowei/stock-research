# Initial Diagnosis: Sell Model V6.2.2 Failure Analysis

## Executive Summary

The comparative analysis between **V6.1 Baseline** and **V6.2.2 Attempt** reveals a critical discrepancy between the user's intended experiment (Crypto features + LightGBM) and the actual code provided in `00_Legacy_Context`.

*   **V6.1 (Legacy):** Achieved ~60% Win Rate, but this was driven by a **Look-Ahead Bias** (Data Leak).
*   **V6.2.2 (Actual Code):** Fixed the data leak, leading to a realistic performance drop (Win Rate ~52%). It **did not** contain Crypto features or LightGBM as described in the prompt, but instead introduced **TOTM (Time of The Month)** and **Relative Strength (vs QQQ/SPY)** features using XGBoost.

**Conclusion:** The "Regression" in V6.2.2 is not a failure of the model, but a correction of an invalid baseline. The features introduced in V6.2.2 (TOTM, Relative Strength) actually perform well on the clean dataset. The "Missing" experiment (Crypto + LightGBM) remains to be done.

## 1. Critical Discrepancy: User Prompt vs. Codebase

| Item | User Description (Prompt) | Actual File Content (`exp_07_v2_training.py`) |
| :--- | :--- | :--- |
| **Model** | LightGBM | **XGBoostClassifier** |
| **New Features** | Crypto Related (BTC, ETH...) | **Dist_MA20**, **Rel_Gap**, **TOTM** (Days_From_Start) |
| **Status** | "Failed" / Regression | Report claims "Optimized V2 is recommended" despite lower metrics. |

**Implication:** We cannot diagnose the failure of "Crypto Features" or "LightGBM" because they are not present in the provided files. We must treat them as **Backlog Items**.

## 2. Feature Engineering & Data Integrity

### A. The V6.1 Data Leak (Root Cause of High Performance)
The V6.1 Baseline code contained a severe Look-Ahead Bias:
```python
# V6.1 Code
df['RSI_14'] = ta.rsi(df['Close'], length=14)
# 'Close' includes the price at the END of the trading day.
# This value was used to predict the outcome at the BEGINNING of the same day (Open).
```
This allowed the model to "peek" at the closing price to decide whether to trade at the Open.

### B. The V6.2.2 Fix
The V6.2.2 Attempt correctly shifted indicators:
```python
# V6.2.2 Code
df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
# Uses T-1 Close to predict T Open.
```

### C. New Features in V6.2.2
Despite the performance drop caused by fixing the leak, the new features showed strong importance:
1.  **Days_From_Start / Days_To_End (TOTM):** Ranked #2 and #3 in feature importance.
2.  **Rel_Gap_SPY / QQQ:** Ranked #4 and #5.

## 3. Performance Attribution

| Metric | V6.1 (Leaked) | V6.2.2 (Clean) | Analysis |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 59.97% | 52.18% | Drop due to removal of future data. 52% is the realistic baseline. |
| **Avg Return** | 0.978% | 0.114% | "Profitable" signals were heavily dependent on the leak. |
| **Signal Count** | 7,521 | 5,929 | Stricter filtering with valid data. |

## 4. Recommendations & Hypotheses

1.  **Accept V6.2.2 as the True Baseline:** Future experiments must be compared against V6.2.2, not the flawed V6.1.
2.  **Execute the Missing Experiment:** The user intended to test **Crypto Correlations** and **LightGBM**. These should be the immediate next steps.
3.  **Hypothesis for Crypto:** High correlation with BTC/ETH might indicate "Risk-On" sentiment, potentially filtering out bad short signals.
4.  **Hypothesis for LightGBM:** May handle the categorical nature of TOTM or non-linear interactions better than XGBoost.
