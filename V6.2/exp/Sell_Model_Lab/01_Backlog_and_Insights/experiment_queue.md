# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which identified a discrepancy between user intent (Crypto/LightGBM) and actual code state (TOTM/XGBoost).

## 🟢 Ready to Start

### **EXP-10: Crypto-Specific Ensemble (The "Crypto Branch")**
* **Context:** * **EXP-01** confirmed that `Crypto_Corr` and `BTC_Trend` are top-tier predictors but drastically reduce signal count (high selectivity).
    * **EXP-03** showed that adding these features globally hurt the general model (overfitting/noise for non-crypto stocks).
    * **EXP-08** established the "Heterogeneous Ensemble" architecture.
    * *Current Gap:* Crypto-sensitive stocks (e.g., `COIN`, `MSTR`, `RIOT`) are currently forced into the generic "Tech" or "Non-Tech" pipelines, likely ignoring their primary driver (Bitcoin price action).
* **Hypothesis:** Creating a dedicated model pipeline for **Crypto-Sensitive Tickers** (using `Base` + `Crypto` features) will outperform the current V6.3 routing for this specific subset.
* **Plan:**
    1.  Load tickers from `V6.2/resource/2025_final_crypto_sensitive_pool.json`.
    2.  Train a dedicated LightGBM model for this pool using the extended feature set (Base + BTC/ETH metrics).
    3.  Compare performance against the current V6.3 baseline (where they are treated as generic Tech/Non-Tech) using "Hold to Close" execution.

### **EXP-11: Non-Tech Feature Augmentation (SPY Context)**
* **Context:** * **EXP-07** transformed the "Tech" model from a weak link to a top performer (+3.55% WR) simply by adding `QQQ` (Sector ETF) features.
    * The current **"Non-Tech"** model (V6.3) still relies on the minimal 5-feature Base set. While robust, it lacks broader market context.
* **Hypothesis:** Adding `SPY` (S&P 500) features (`Gap`, `RSI`, `MA_Dist`) to the **Non-Tech Model** will provide necessary market context and improve predictive power, similar to the QQQ effect on Tech stocks.
* **Plan:**
    1.  Focus on the Non-Tech ticker universe.
    2.  Engineer `SPY`-based features (aligning with how `QQQ` features were built in EXP-07).
    3.  Train "Base + SPY" models and compare Win Rate/Avg Return against the current "Base Only" Non-Tech model.

## 🟡 Backlog

### **EXP-12: Time-Decay Exit Optimization**
* **Context:** EXP-09 proved "Hold to Close" beats fixed targets. However, alpha often decays as the day progresses.
* **Hypothesis:** A dynamic exit based on "Time in Trade" (e.g., exit after 3 hours) or a technical trigger (e.g., cross VWAP) might capture the bulk of the move while reducing exposure to late-day chop.

## ⚫ Done

### **EXP-09: Sell Strategy Logic Refinement**
*   **Result:** ✅ Major Success (+33.7% Return).
*   **Outcome:** Adopt "Hold to Close" (Exit MOC).
*   **Key Findings:**
    *   **Hold to Close** achieved **33.7% Total Return** (vs -2.3% for Baseline).
    *   Baseline strategy (PT 0.2%) "picks pennies in front of steamrollers" (93% WR but negative expectation).
    *   Intraday Stops (0.5% - 2.0%) degrade performance by exiting trades prematurely during normal volatility.

### **EXP-08: Production Integration (V6.3 Release)**
*   **Result:** ✅ Success (System Operational).
*   **Outcome:** V6.3 Production System deployed.
*   **Key Findings:**
    *   Successfully implemented **Heterogeneous Ensemble** (Non-Tech: Base Features, Tech: Base+QQQ features).
    *   Production script `production_daily_plan_v6_3.py` generates valid signals using specific sector models.
    *   Verification confirmed sector-specific routing works as intended.

### **EXP-07: Tech-Specific Feature Engineering**
*   **Result:** ✅ Major Success (+3.55% Win Rate).
*   **Outcome:** Adopt (Tech Model uses QQQ features).
*   **Key Findings:**
    *   Tech Sector moves as a herd. `QQQ` features are the top 3 most important predictors.
    *   Tech Win Rate improved from 49.8% to **53.35%**.
    *   Signal count doubled.

### **EXP-06: Base Feature Hyperparameter Tuning**
*   **Result:** ✅ Success (+0.54% Win Rate vs Baseline Ensemble).
*   **Outcome:** Adopt (Optimized Sector Ensemble).
*   **Key Findings:**
    *   **Ensemble Win Rate: 52.23%**.
    *   **Tech Sector:** Needs extreme regularization (Depth 3, LR 0.01).
    *   **Non-Tech Sector:** Supports higher complexity (Unlimited Depth, LR 0.02).
    *   Global optimization actually hurt performance (overfitting), proving the need for sector-specific tuning.

### **EXP-05: Sector-Specific Ensembles**
*   **Result:** ✅ Success (+0.82% Win Rate, +0.07% Avg Return).
*   **Outcome:** Adopt (Sector Split > Global).
*   **Key Findings:**
    *   Ensemble Win Rate: 52.19% vs Global 51.37%.
    *   Non-Tech stocks perform exceptionally well (53.3% WR).
    *   Tech stocks are the weak link (50.2% WR).

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
