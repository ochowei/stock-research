# EXP-25: Large Gap Specialization (High Volatility Model)

## 1. Context & Hypothesis
**Context:** EXP-24 demonstrated that "Large Gaps" (>3%) are not noise or breakaway gaps to be avoided, but rather highly profitable exhaustion setups with a Win Rate of 57.3% and Avg Return of +0.68%. However, the current V6.2.4.RC models (Tech and Non-Tech) are trained on a broad dataset where the majority of samples are small gaps (0.5% - 2.0%).

**Hypothesis:** The general models may be under-fitting the specific dynamics of large gaps because the loss function is dominated by the more frequent small gaps. A "Specialized Model" trained *exclusively* on gaps > 2% will learn the specific nuances of high-volatility reversals better than the general model.

**Goal:** Determine if a dedicated `High_Vol_Model` outperforms the `General_Model` on the subset of data where `Gap_Pct > 2%`.

## 2. Experiment Design

### 2.1. Data Scope
*   **Asset Pool:** Standard 2025 Asset Pool (approx. 200 tickers).
*   **Training Period:** 2020-01-01 to 2023-12-31.
*   **Testing Period:** 2024-01-01 to 2025-12-31 (The current high-volatility regime).
*   **Target Subset:** `Gap_Pct > 0.02` (2%).

### 2.2. Models
1.  **Baseline (General Model):** The existing V6.2.4.RC architecture (Tech/Non-Tech split).
    *   Trained on ALL gaps > 0.5%.
    *   Evaluated on `Gap_Pct > 2%`.
2.  **Challenger (Specialized High-Vol Model):**
    *   **Training Data:** Only rows where `Gap_Pct > 0.02`.
    *   **Architecture:** Single Global Model (combining Tech/Non-Tech) to ensure sufficient sample size, OR Split if data permits. Given large gaps are rarer, a Global High-Vol model is the likely starting point.
    *   **Features:** Standard Base Features + QQQ Context + SPY Context (Union of features).
    *   **Parameters:** Standard LightGBM parameters, potentially with lower depth to prevent overfitting on smaller sample size.

### 2.3. Metrics
*   **Win Rate:** Target > 57% (Beating the EXP-24 baseline).
*   **Avg Return:** Target > 0.70%.
*   **Signal Count:** Must remain high enough to be tradable (e.g., > 50 trades/year).

## 3. Implementation Plan
1.  **Data Loading:** Fetch standard dataset.
2.  **Feature Engineering:** Generate Base, QQQ, and SPY features.
3.  **Baseline Execution:**
    *   Train V6.2.4.RC (Tech/Non-Tech) on full training set (Gaps > 0.5%).
    *   Predict on Test Set (Gaps > 2%).
4.  **Challenger Execution:**
    *   Filter Training Set for `Gap_Pct > 0.02`.
    *   Train `High_Vol_Model` (Global) on this subset.
    *   Predict on Test Set (Gaps > 2%).
5.  **Comparison:** Compare Win Rate and Avg Return of Baseline vs Challenger on the *same* test subset.

## 4. Success Criteria
*   The Specialized Model must beat the Baseline Win Rate by at least **+1.0%** on the high-gap subset.
*   OR The Specialized Model significantly improves Avg Return (+10%) without degrading Win Rate.
