# Global Learning Log (Momentum Model Lab)

This document captures cross-experiment insights to build a cumulative knowledge base for the Momentum Model.

## 2025-05-21: EXP-03 Volume Microstructure (False Breakout Filter)

*   **Lesson:** Adding pre-gap volume trend (`Vol_MA5_Slope`) to filter "Fake Breakouts" failed to improve performance (Win Rate 57.84% vs 57.96% Baseline).
*   **Key Insight:** The `Vol_MA5_Slope` feature had the lowest importance (0.09) among all features. `RSI_14` (0.41) and `VIX` (0.17) remain dominant. This suggests that the immediate liquidity shock (measured by `Vol_Ratio`) is more relevant than the multi-day volume trend leading up to the event.
*   **Operational Risk:** Low operational risk (internal calculation), but adds unnecessary complexity for no gain.
*   **Action:** Reject `Vol_MA5_Slope`. Maintain the existing `Vol_Ratio` feature and focus on broader macro/crypto context (EXP-04) for alpha generation.

## 2025-05-18: EXP-02 Sector Relative Strength (Orthogonal Alpha)

*   **Lesson:** Adding explicit Sector context (`Sector_RSI`, `Rel_Strength`) did **not** improve Win Rate (57.51% vs 57.59% Baseline).
*   **Key Insight:** While sector features were picked up by the model (ranked 2nd/4th), they appear redundant with the stock's own momentum (`RSI_14`) or introduce noise (imperfect tracking).
*   **Operational Risk:** Relying on external Sector ETF data (XLK, XLV, etc.) introduces significant operational risk (timeouts, missing data) which can cause critical signals to be dropped.
*   **Action:** Reject Sector features for now. Prioritize robust, single-asset features (e.g., Volume Microstructure) that do not depend on external data feeds.

## 2025-05-15: EXP-01 Baseline Reproduction (V6.1 Parity)

*   **Lesson:** The V6.1 "Simple Momentum" features (`RSI_14`, `ATR_Pct`, `Vol_Ratio`) are surprisingly robust in the 2024-2025 regime, achieving a Win Rate of 57.6% and Avg Return of 0.98% without any new fancy features.
*   **Key Insight:** `RSI_14` dominates feature importance (47%), suggesting that "Momentum" in this model is primarily defined by the strength of the trend (Overbought/Oversold).
*   **Context:** Market Volatility (`VIX`) is the second most critical factor (17%), confirming that regime filters are essential.
*   **Action:** Future experiments (Sector, Volume) must beat this high bar. Adding complexity without significant gain (>1% Win Rate) should be rejected.
