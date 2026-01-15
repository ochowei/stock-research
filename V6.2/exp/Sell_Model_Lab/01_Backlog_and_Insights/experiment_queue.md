# Experiment Queue

This queue is prioritized based on the `initial_diagnosis.md` which identified a discrepancy between user intent (Crypto/LightGBM) and actual code state (TOTM/XGBoost).

## 🟢 Ready to Start

(Empty)

## 🟡 Backlog

### **EXP-21: Limit Order Entry Optimization (Short into Strength)**
* **Hypothesis:** Based on EXP-12/16 findings of "Morning Fake-Outs" (price moves against signal initially), using a Limit Order slightly above Open (e.g., Open * 1.005) may improve entry price and Sharpe, despite lower fill rates.
* **Goal:** Extract execution alpha by fading the initial morning volatility.

### **EXP-20: Relative Gap Features (Index Interaction)**
* **Hypothesis:** Explicitly calculating the difference between Stock Gap and Index Gap (e.g., `Stock_Gap - QQQ_Gap`) will provide a stronger signal than feeding them as separate features. "Gapping up against a red market" might be a higher conviction short.
* **Goal:** Enhance Feature Engineering for the Heterogeneous Ensemble.

## ⚫ Done

### **EXP-19: Crypto Sector Specific Model (Pure Play Data)**
*   **Result:** ✅ Success (Incremental Improvement).
*   **Outcome:** Adopt Crypto-Enhanced Model for Pure Play Crypto Sector.
*   **Key Findings:**
    *   **Win Rate:** Base (52.45%) vs Test (52.95%).
    *   **Feature Importance:** `BTC_Change` and `BTC_Gap` are top predictors, confirming these stocks move as "Bitcoin Derivatives".
    *   Pure Play restriction solved the noise issue from EXP-15.
    *   **Trade-off:** Win Rate improved, but Avg Return diluted slightly due to higher signal count.

### **EXP-18: Production Script Update (Position Sizing)**
* **Result:** ✅ Success (Script Updated to V6.2.5 RC).
* **Outcome:** `production_daily_plan_v6_2_5_rc.py` deployed.
* **Key Findings:**
    *   Successfully implemented Tiered Position Sizing logic (1.5x/1.0x/0.5x).
    *   Verified signal generation and sizing output using `daily_plan_2026-01-14.csv`.
    *   System now fully automates the high-Sharpe strategy discovered in EXP-17.

### **EXP-17: Confidence-Based Position Sizing (信心權重配置)**
*   **Result:** ✅ Major Success (Highest Sharpe Ratio).
*   **Outcome:** Adopt Tiered Position Sizing (1.5x/1.0x/0.5x).
*   **Key Findings:**
    *   **Tiered Sizing** achieved Sharpe Ratio of **6.24** (vs 5.96 for Baseline) and reduced Max Drawdown by 17%.
    *   LightGBM probability is a valid proxy for trade quality.
    *   Allocating more capital to high-confidence signals (>60% Prob) significantly improves risk-adjusted returns.

### **EXP-16: Catastrophe Stop-Loss Optimization**
*   **Result:** ❌ Failed (Hypothesis Rejected).
*   **Outcome:** Reject Catastrophe Stops. Maintain "Hold to Close".
*   **Key Findings:**
    *   **Baseline (No Stop):** 84.49% Total Return, 5.22 Sharpe, -0.66 Max DD.
    *   **Catastrophe Stop (3x ATR):** 83.80% Total Return, 5.14 Sharpe, -0.70 Max DD.
    *   Stops *increased* Max Drawdown, confirming that extreme intraday moves ("morning fake-outs") are often followed by mean reversion.
    *   Exiting at the high (stop trigger) locks in losses that would otherwise recover by close.

### **EXP-15: Crypto-Specific Ensemble (Clean Data Redux)**
*   **Result:** ❌ Failed (Hypothesis Rejected).
*   **Outcome:** Do not adopt.
*   **Key Findings:**
    *   **Control (Base):** 52.84% Win Rate. **Test (Base+BTC):** 48.80% Win Rate.
    *   Adding BTC features to a pure crypto pool (COIN, MSTR, RIOT, MARA) **degraded** performance significantly (-4%).
    *   BTC features dominated feature importance (Top 3), suggesting the model overfitted to macro context at the expense of local price action.
    *   **Decision:** Crypto stocks should be treated with the standard Base model.

### **EXP-14: Delayed Entry Optimization**
* **Result:** ❌ Failed (Hypothesis Rejected).
*   **Outcome:** Reject Delayed Entry. Maintain "Market On Close" execution at Open.
*   **Key Findings:**
    *   **Baseline (Open Entry):** 63.14% Win Rate, +1.45% Avg Return.
    *   **Delayed (10:30 Entry):** 57.85% Win Rate, +0.75% Avg Return.
    *   Waiting 1 hour forfeits significant alpha; the V6.4 model signals tend to work immediately.

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
