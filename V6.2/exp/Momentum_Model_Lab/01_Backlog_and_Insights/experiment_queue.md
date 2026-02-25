# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which highlights the need to establish a stable baseline before introducing orthogonal features (Sector, Volume, Crypto) to improve momentum persistence.

## 🟢 Ready to Start

### **EXP-04: Crypto Context Integration (Risk-On Regime)**
* **Hypothesis:** Momentum strategies perform better in "Risk-On" environments. Crypto trends (BTC/ETH) serve as a leading indicator for high-beta risk appetite.
* **Implementation:** Add `BTC_Change` and `BTC_Trend_Score` as global context features.
* **Goal:** Optimize position sizing or entry timing based on global risk sentiment.

### **EXP-05: Dynamic Window Sensitivity (Lookback Tuning)**
* **Hypothesis:** The standard 14-day RSI/Momentum window may be too slow for the current high-volatility regime.
* **Implementation:** Test shorter (5D, 10D) vs longer (20D, 50D) lookback windows.
* **Goal:** Find the optimal sensitivity balance between noise and lag.

## ⚫ Done

### **EXP-03: Volume Microstructure (False Breakout Filter)**
* **Result:** Fail (Win Rate 48.65% vs Random 47.43%).
* **Key Finding:** Corrected a critical data leak in EXP-01 baseline (Look-ahead bias). True baseline is negative (-0.084% return). Volume Trend (`Vol_MA5_Slope`) adds +1.2% Win Rate but strategy remains unprofitable.
* **Date:** 2025-05-20

### **EXP-02: Sector Relative Strength (Orthogonal Alpha)**
* **Result:** Fail (Win Rate 57.51% vs Baseline 57.59%).
* **Key Finding:** Sector features (Sector RSI, Rel Strength) did not improve performance and added complexity (data fetching issues). Simple RSI remains superior.
* **Date:** 2025-05-18

### **EXP-01: Baseline Reproduction (V6.1 Parity)**
* **Result:** Success (Win Rate 57.59%, Avg Return 0.975%).
* **Key Finding:** The V6.1 model (RSI, ATR, Vol_Ratio) is highly effective in 2024-2025. RSI is the dominant feature (47% importance).
* **Date:** 2025-05-15
