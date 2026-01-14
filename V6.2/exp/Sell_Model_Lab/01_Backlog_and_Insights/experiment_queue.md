# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which identified a discrepancy between user intent (Crypto/LightGBM) and actual code state (TOTM/XGBoost).

## 🟢 Ready to Start

### **EXP-03: Feature Selection (Ablation Study)**
*   **Hypothesis:** Some features in the V6.2.2 set (e.g., `Dist_MA20`) might be redundant or noise when combined with Crypto features.
*   **Action:**
    *   Run Recursive Feature Elimination (RFE) or Permutation Importance.
    *   Test subsets of features: `[Base 5]`, `[Base + TOTM]`, `[Base + Crypto]`, `[All]`.

## 🟡 Backlog

### **EXP-04: Regime-Switching Model**
*   **Idea:** Train separate models for High VIX (>20) vs Low VIX (<20) environments.
*   **Rationale:** V6.2.2 showed VIX is the #1 feature. Splitting the model might allow it to specialize.

### **EXP-05: Sector-Specific Ensembles**
*   **Idea:** Train distinct models for Tech (QQQ components) vs Non-Tech.
*   **Rationale:** `Rel_Gap_QQQ` was significant. Tech stocks might behave differently on Gaps.

## ⚫ Done

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
