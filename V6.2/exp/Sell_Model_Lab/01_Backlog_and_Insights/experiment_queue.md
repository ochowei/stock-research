# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which identified a discrepancy between user intent (Crypto/LightGBM) and actual code state (TOTM/XGBoost).

## 🟢 Ready to Start

### **EXP-14: Delayed Entry Optimization**
* **Context:** EXP-12 showed that trades have a negative return (-0.86%) in the first hour, meaning prices often move against the "Sell" signal initially.
* **Hypothesis:** Entering the trade 1 hour after Open (or fading the 1H High) might significantly improve the entry price and Win Rate by avoiding the "morning fake-out".

## 🟡 Backlog

### **EXP-15: Crypto-Specific Ensemble (Clean Data Redux)**
* **Context:** EXP-10 (Crypto Branch) 失敗，主要原因是資產池污染（混入了非加密相關股票如 Dutch Bros）。然而，該實驗也發現 Crypto 模型能有效降低虧損幅度。
* **Hypothesis:** 既然 QQQ 特徵能顯著優化 Tech 板塊 (EXP-07)，那麼在**純淨的加密貨幣相關股池** (COIN, MSTR, RIOT, MARA) 上，加入 `BTC_Trend` 與 `Crypto_Corr` 特徵，應該能顯著提升預測準確度，複製 Sector Ensemble 的成功模式。
* **Goal:** 建立第三個專用模型路由：Tech / Non-Tech / Crypto。

### **EXP-16: Catastrophe Stop-Loss Optimization**
* **Context:** EXP-09 和 EXP-12 證明了 "Hold to Close" 是期望值最高的策略，但也意味著必須承受無限的潛在虧損風險。目前策略在極端行情下缺乏保護。
* **Hypothesis:** 雖然緊迫的止損 (0.5% - 2.0%) 會損害績效，但設置一個**寬幅的「災難止損」(Catastrophe Stop)**（例如 3倍 ATR 或 固定 5%），可能可以在不觸發「早盤假動作」的前提下，切斷極端左尾風險 (Left Tail Risk)，從而提升夏普比率 (Sharpe Ratio)。

## ⚫ Done

### **EXP-13: Production Deployment (V6.4)**
*   **Result:** ✅ Success (System Operational).
*   **Outcome:** V6.4 Production System deployed.
*   **Key Findings:**
    *   Successfully implemented **Enhanced Heterogeneous Ensemble**.
    *   **Tech Model:** Uses Base + QQQ features.
    *   **Non-Tech Model:** Uses Base + SPY features.
    *   Production script `production_daily_plan_v6_4.py` generates valid signals using correct routing.

### **EXP-12: Time-Decay Exit Optimization**
* **Result:** ❌ Failed (Hypothesis Rejected).
*   **Outcome:** Retain "Hold to Close" (Market On Close).
*   **Key Findings:**
    *   **Hold to Close** (MOC) overwhelmingly outperforms early exits (+31.1% vs +7.8% Total Return).
    *   **Morning Fake-Out:** The 1H return is negative (-0.86%), meaning the price initially moves UP against the short signal before fading.
    *   The alpha materializes slowly throughout the day.

### **EXP-11: Non-Tech Feature Augmentation (SPY Context)**
*   **Result:** ✅ Success (+0.83% Win Rate).
*   **Outcome:** Adopt (Non-Tech Model uses SPY features).
*   **Key Findings:**
    *   Adding `SPY` features (Gap, RSI, Dist_MA) increased Non-Tech Win Rate from 52.01% to **52.84%**.
    *   SPY features are the top 3 most important predictors, proving Non-Tech stocks are driven by broad market sentiment.
    *   Avg Return increased by 46% (+0.06%).

### **EXP-10: Crypto-Specific Ensemble (The "Crypto Branch")**
*   **Result:** ❌ Failed (Hypothesis Rejected, but insight gained).
*   **Outcome:** Do not adopt current iteration. Revisit with corrected data.
*   **Key Findings:**
    *   **Result:** Crypto Model (Base+BTC) had lower Win Rate (52.66% vs 53.98%) but better loss mitigation (-0.06% vs -0.14% Avg Return) compared to Baseline.
    *   **Root Cause:** The provided "Crypto Sensitive Pool" was contaminated with non-crypto stocks (e.g., Dutch Bros, Hims, Trade Desk). Forcing BTC features on these introduced noise.
    *   **Feature Importance:** BTC features dominated the model (Top 3), causing it to ignore stock-specific technicals.
    *   **Action Item:** The Crypto Pool must be rebuilt with pure-play crypto stocks (COIN, MSTR, RIOT) before re-testing.

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
