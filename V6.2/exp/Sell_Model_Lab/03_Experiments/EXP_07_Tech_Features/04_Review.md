# EXP-07: Tech-Specific Feature Engineering - Review

## 1. Executive Summary
*   **Result:** ✅ **Major Success** (+3.55% Win Rate, High Signal Count).
*   **Outcome:** Adopt immediately for the Tech Sector model.
*   **Key Insight:** Tech stocks are overwhelmingly driven by sector-level price action (`QQQ`). Adding QQQ context transforms the Tech model from a "weak link" (49.8% WR) into a top performer (53.35% WR).

## 2. Methodology Check
*   **Target:** Tech Sector stocks only.
*   **Features Added:** `QQQ_Gap_Pct`, `QQQ_RSI_14`, `QQQ_Dist_MA20`, `Sector_Corr`.
*   **Model:** LightGBM (Tuned).
*   **Comparison:**
    *   **Baseline:** Base Features (5) only.
    *   **Experiment:** Base (5) + Tech Features (4) = 9 Features.

## 3. Detailed Results (Test Set 2024-2025)

| Model | Win Rate | Avg Return | Trade Count | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Tech Baseline** | 49.80% | -0.00% | 1,223 | Consistent with previous failures. |
| **Tech + Features** | **53.35%** | **+0.19%** | **2,549** | **Massive improvement.** |

### Feature Importance (Top 5)
1.  **QQQ_Gap_Pct** (884) - The sector's opening sentiment is the dominant factor.
2.  **QQQ_RSI_14** (839) - Sector overbought/oversold status.
3.  **QQQ_Dist_MA20** (750) - Sector trend deviation.
4.  ATR_Pct (377)
5.  Vol_Ratio (351)

*Observation:* The top 3 features are all **Sector Features**. The individual stock's technicals are secondary to the sector's movement.

## 4. Discussion
*   **Why did the signal count double?**
    *   The Baseline Tech model (Base features only) likely struggled to find *any* patterns that worked, leading to low confidence or restrictive tuning.
    *   With the Sector features, the model found clear, repeatable patterns (e.g., "If QQQ Gaps > X and Stock Gaps > Y..."), allowing it to trade much more aggressively (Higher LR, Deeper Trees) with high confidence.
*   **The "Weak Link" is fixed.**
    *   Previous experiments (EXP-05/06) showed Tech dragging down the ensemble.
    *   With this fix, the Tech model (53.35%) now rivals the Non-Tech model (53.3% in EXP-05).

## 5. Conclusion & Action Items
1.  **Production:** The V6.3 Production Model must use a **Heterogeneous Ensemble**:
    *   **Non-Tech:** Base Features (5).
    *   **Tech:** Base + QQQ Features (9).
2.  **Resources:** Ensure `QQQ` data is fetched and processed in the production pipeline.
