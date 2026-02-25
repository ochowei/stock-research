# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which highlights the need to establish a stable baseline before introducing orthogonal features (Sector, Volume, Crypto) to improve momentum persistence.

## 🟢 Ready to Start

### **EXP-01: Baseline Reproduction (V6.1 Parity)**
* **Hypothesis:** Re-training the momentum model within the new `Lab` structure using exact V6.1 parameters and features will replicate historical performance.
* **Target Metrics:** Win Rate > 55%, Avg Return > 0.25%.
* **Goal:** Establish a "Control" model (The Baseline) to measure future improvements against.

### **EXP-02: Sector Relative Strength (Orthogonal Alpha)**
* **Hypothesis:** Momentum signals are more reliable when the underlying Sector (e.g., XLK, XLF) is also in a strong trend (Sector RSI > 50). "Lone Wolf" breakouts are prone to failure.
* **Implementation:** Inject `Sector_RSI` and `Stock_vs_Sector_RelStrength` features.
* **Goal:** Increase Win Rate by filtering out false positives in weak sectors.

### **EXP-03: Volume Microstructure (False Breakout Filter)**
* **Hypothesis:** High price momentum accompanied by low or declining volume ("Fake Breakout") has low persistence.
* **Implementation:** Add `Vol_MA5_Slope` and `Vol_Ratio` (Day/Avg) as features or hard filters.
* **Goal:** Improve Precision (Win Rate) by avoiding liquidity traps.

### **EXP-04: Crypto Context Integration (Risk-On Regime)**
* **Hypothesis:** Momentum strategies perform better in "Risk-On" environments. Crypto trends (BTC/ETH) serve as a leading indicator for high-beta risk appetite.
* **Implementation:** Add `BTC_Change` and `BTC_Trend_Score` as global context features.
* **Goal:** Optimize position sizing or entry timing based on global risk sentiment.

### **EXP-05: Dynamic Window Sensitivity (Lookback Tuning)**
* **Hypothesis:** The standard 14-day RSI/Momentum window may be too slow for the current high-volatility regime.
* **Implementation:** Test shorter (5D, 10D) vs longer (20D, 50D) lookback windows.
* **Goal:** Find the optimal sensitivity balance between noise and lag.

## ⚫ Done

(Empty - Awaiting First Experiment Run)