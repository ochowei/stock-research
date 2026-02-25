# Global Learning Log (Momentum Model Lab)

This document captures cross-experiment insights to build a cumulative knowledge base for the Momentum Model.

## 2025-05-15: EXP-01 Baseline Reproduction (V6.1 Parity)

*   **Lesson:** The V6.1 "Simple Momentum" features (`RSI_14`, `ATR_Pct`, `Vol_Ratio`) are surprisingly robust in the 2024-2025 regime, achieving a Win Rate of 57.6% and Avg Return of 0.98% without any new fancy features.
*   **Key Insight:** `RSI_14` dominates feature importance (47%), suggesting that "Momentum" in this model is primarily defined by the strength of the trend (Overbought/Oversold).
*   **Context:** Market Volatility (`VIX`) is the second most critical factor (17%), confirming that regime filters are essential.
*   **Action:** Future experiments (Sector, Volume) must beat this high bar. Adding complexity without significant gain (>1% Win Rate) should be rejected.
