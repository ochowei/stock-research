# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which identified a discrepancy between user intent (Crypto/LightGBM) and actual code state (TOTM/XGBoost).

## 🟢 Ready to Start

### **EXP-01: Crypto Feature Integration**
*   **Hypothesis:** Adding Crypto market state (BTC/ETH trend, volatility) will improve the Sell Model's ability to gauge global risk sentiment.
*   **Action:**
    *   Fetch BTC-USD and ETH-USD data.
    *   Feature Engineering: `BTC_RSI`, `BTC_Trend`, `Crypto_Correlation` (with stock).
    *   **Base:** Start from V6.2.2 (Clean Data).
    *   **Model:** XGBoost (Keep constant to isolate feature impact).

### **EXP-02: LightGBM Migration**
*   **Hypothesis:** LightGBM may offer better handling of the new feature set and faster training times, potentially reducing overfitting compared to XGBoost.
*   **Action:**
    *   Replace `XGBClassifier` with `LGBMClassifier`.
    *   Tune `num_leaves`, `learning_rate`.
    *   **Base:** V6.2.2 (or EXP-01 if successful).

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
