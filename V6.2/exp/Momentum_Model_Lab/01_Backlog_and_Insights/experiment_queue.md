# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which highlights the need to establish a stable baseline before introducing orthogonal features (Sector, Volume, Crypto) to improve momentum persistence.

## 🟢 Ready to Start

### **EXP-08: High Momentum Short Strategy - Candle Wick Analysis**
* **Hypothesis:** Reversals often leave "Upper Wicks" (failed breakouts). If the gap opens and immediately fails (High - Open < 0.2% * OR * Close < Open), it confirms the reversal.
* **Implementation:** Filter Gap Ups where `(High - Open) / Open < 0.002` (Small Upper Wick) or `Close < Open` (Red Candle). This requires intraday data or Open-High relationship. *Simpler:* Use `(Open - Low) / Open > 0.005` (Large Lower Wick = Buyers stepping in?) No.
* **Refined Hypothesis:** Short the "Shooting Star" or "Doji" gaps.
* **Goal:** Increase Win Rate > 55% for Short Strategy.

## ⚫ Done

### **EXP-07: High Momentum Short Strategy - Volume Filter**
* **Result:** Fail (Win Rate 54.46% vs Baseline 53.84%).
* **Key Finding:** High volume gaps (`Vol_Ratio > 2.0`) behave almost identically to normal volume gaps in overbought conditions (`RSI > 70`). The filter reduced sample size by 80% without improving Avg Return (+0.18% vs +0.19%).
* **Date:** 2025-06-02

### **EXP-06: Mean Reversion Signal (Gap Fade)**
* **Result:** Success (Win Rate 53.7%, Avg Return +0.20% for Short Strategy).
* **Key Finding:** High Momentum (RSI > 70) gaps fail to continue (Long Win Rate 45.9%) and revert instead. Shorting these gaps yields positive expectancy (+0.20%), outperforming the Baseline Long strategy (-0.11%).
* **Date:** 2025-05-30

### **EXP-05: Dynamic Window Sensitivity (Lookback Tuning)**
* **Result:** Fail (Win Rate 33.6% vs Baseline 47.4%).
* **Key Finding:** Momentum indicators (RSI, ROC) at T-1 are **counter-indicators** for Gap Continuation. High momentum predicts gap fading (reversal). Shorter windows (10D) captured this negative signal better than longer windows (50D).
* **Date:** 2025-05-27

### **EXP-04: Crypto Context Integration (Risk-On Regime)**
* **Result:** Fail (Win Rate 53.82% vs Baseline 56.24%).
* **Key Finding:** Adding Crypto Context features (`BTC_RSI`, `ETH_Ret`) significantly degraded performance (-2.42% Win Rate). Despite high feature importance, they introduced noise/overfitting and failed to generalize to the 2024-2025 OOS period.
* **Date:** 2025-05-24

### **EXP-03: Volume Microstructure (False Breakout Filter)**
* **Result:** Fail (Win Rate 57.84% vs Baseline 57.96%).
* **Key Finding:** Adding pre-gap volume trend (`Vol_MA5_Slope`) slightly degraded performance. The feature had the lowest importance (0.09). Simple `Vol_Ratio` (Gap Volume) is sufficient.
* **Date:** 2025-05-21

### **EXP-02: Sector Relative Strength (Orthogonal Alpha)**
* **Result:** Fail (Win Rate 57.51% vs Baseline 57.59%).
* **Key Finding:** Sector features (Sector RSI, Rel Strength) did not improve performance and added complexity (data fetching issues). Simple RSI remains superior.
* **Date:** 2025-05-18

### **EXP-01: Baseline Reproduction (V6.1 Parity)**
* **Result:** Success (Win Rate 57.59%, Avg Return 0.975%).
* **Key Finding:** The V6.1 model (RSI, ATR, Vol_Ratio) is highly effective in 2024-2025. RSI is the dominant feature (47% importance).
* **Date:** 2025-05-15
