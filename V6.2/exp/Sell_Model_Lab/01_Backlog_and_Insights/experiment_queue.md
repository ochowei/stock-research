# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which identified a discrepancy between user intent (Crypto/LightGBM) and actual code state (TOTM/XGBoost).

## 🟢 Ready to Start

### **EXP-05: Sector-Specific Ensembles**
*   **Idea:** Train distinct models for Tech (QQQ components) vs Non-Tech.
*   **Rationale:** `Rel_Gap_QQQ` was significant. Tech stocks might behave differently on Gaps.

## 🟡 Backlog

### **EXP-06: Base Feature Hyperparameter Tuning**
*   **Idea:** Optimize LightGBM hyperparameters for the "Base" feature set.
*   **Rationale:** EXP-03 showed Base is best. Tuning it might squeeze out more alpha.

## ⚫ Done

### **EXP-04: Regime-Switching Model**
*   **Result:** ⏹️ Neutral (Marginal Gain).
*   **Outcome:** Discard (Too complex for +0.22% WR).
*   **Key Findings:**
    *   Regime-System Win Rate: 52.33% (+0.22% vs Global).
    *   Signals: +2.1% volume.
    *   **Lesson:** Global model with Base features is robust enough. Explicit VIX splitting adds complexity without significant alpha.

### **EXP-03: Feature Selection (Ablation Study)**
*   **Result:** ✅ Success (Simpler is Better).
*   **Winner:** **Base Model (5 Features)** (`Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20`).
*   **Key Findings:**
    *   Base model achieved 51.96% Win Rate vs 50.95% for "All" features.
    *   Adding Crypto, TOTM, or VIX features reduced performance (overfitting).
    *   We should proceed with the leaner 5-feature model.

### **EXP-02: LightGBM Migration**
*   **Result:** ✅ Success (+0.036% Avg Return, Higher Selectivity).
*   **Hypothesis:** LightGBM may offer better handling of the new feature set and faster training times, potentially reducing overfitting compared to XGBoost.
*   **Key Findings:**
    *   Win Rate remained flat (52.23%), but Avg Return increased significantly.
    *   LightGBM is more selective (filtered ~30% of trades).
    *   `Vol_Ratio` showed 0 importance.
    *   VIX and Crypto features dominate importance.

### **EXP-01: Crypto Feature Integration**
*   **Result:** ✅ Success (+2.84% Win Rate, +0.06% Avg Return).
*   **Hypothesis:** Adding Crypto market state (BTC/ETH trend, volatility) will improve the Sell Model's ability to gauge global risk sentiment.
*   **Key Findings:**
    *   `Crypto_Corr` and `BTC_Trend` are valuable features (top 10 importance).
    *   Model selectivity increased significantly (signals dropped from ~1800 to ~300).
