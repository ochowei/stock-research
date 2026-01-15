# EXP-20: Relative Gap Features (Index Interaction)

## 1. Hypothesis
Explicitly calculating the difference between Stock Gap and Index Gap (e.g., `Stock_Gap - QQQ_Gap`) will provide a stronger signal than feeding them as separate features.
*   **Logic:** A stock gapping up 1% when the market gaps up 1% is neutral. A stock gapping up 1% when the market gaps down 1% is showing extreme relative strength (or irrational exuberance), which might be a higher quality short signal (mean reversion).
*   **Prediction:** The `Rel_Gap` feature will become a top predictor and improve Win Rate/Precision.

## 2. Experimental Design

### Feature Engineering
*   **Tech Sector:**
    *   Existing: `Gap_Pct`, `QQQ_Gap_Pct`
    *   New: `Rel_Gap_QQQ = Gap_Pct - QQQ_Gap_Pct`
*   **Non-Tech Sector:**
    *   Existing: `Gap_Pct`, `SPY_Gap_Pct`
    *   New: `Rel_Gap_SPY = Gap_Pct - SPY_Gap_Pct`

### Models
*   **Control Group (Baseline):** V6.2.4.RC Configuration
    *   Tech: Base + QQQ Features (`['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20', 'QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20']`)
    *   Non-Tech: Base + SPY Features (`['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20', 'SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20']`)
    *   Hyperparameters:
        *   Tech: Depth=3, LR=0.01 (EXP-06)
        *   Non-Tech: Unlimited Depth, LR=0.02 (EXP-06)

*   **Test Group:**
    *   Tech: Baseline + `Rel_Gap_QQQ`
    *   Non-Tech: Baseline + `Rel_Gap_SPY`

### Data Split
*   **Training:** 2020-01-01 to 2023-12-31
*   **Testing (OOS):** 2024-01-01 to Present (2025)

## 3. Success Metrics
*   **Primary:** Win Rate > Baseline
*   **Secondary:** Avg Return > Baseline
*   **Check:** Feature Importance of `Rel_Gap` > `Gap_Pct` or `Index_Gap_Pct` (validating the hypothesis that the interaction is more meaningful).
