# EXP-13: Production Deployment (V6.2.4)

## 1. Objective
Deploy the **V6.2.4 Production System** by synthesizing the validated improvements from recent experiments:
- **Tech Sector:** Use **Base + QQQ Features** (validated in EXP-07) with strict regularization (Depth 3, LR 0.01).
- **Non-Tech Sector:** Use **Base + SPY Features** (validated in EXP-11) with higher complexity (Unlimited Depth, LR 0.02).
- **Execution:** Retain "Hold to Close" (MOC) strategy (validated in EXP-09/EXP-12).

## 2. Hypothesis
Combining the specialized **Tech-QQQ** model and the **Non-Tech-SPY** model into a single heterogeneous production system will maximize overall Win Rate and Expected Return compared to V6.2.3 (which used a pure Base model for Non-Tech).

## 3. Plan
1.  **Train `v6.2.4_tech_model.joblib`**:
    *   **Data:** Tech tickers from `2025_final_asset_pool.json`.
    *   **Features:** Base (Gap, RSI, ATR, Vol_Ratio, Dist_MA) + QQQ (Gap, RSI, Dist_MA, Sector_Corr).
    *   **Algorithm:** LightGBM (Depth 3, LR 0.01, Estimators 500).
2.  **Train `v6.2.4_non_tech_model.joblib`**:
    *   **Data:** Non-Tech tickers from `2025_final_asset_pool.json`.
    *   **Features:** Base + SPY (Gap, RSI, Dist_MA, Sector_Corr).
    *   **Algorithm:** LightGBM (Num Leaves 31 (Unlimited), LR 0.02, Estimators 500).
3.  **Generate `production_daily_plan_v6_2_4.py`**:
    *   Update the V6.2.3 script to load V6.2.4 models.
    *   Implement dual benchmark fetching (QQQ for Tech, SPY for Non-Tech).
    *   Implement conditional feature engineering (Tech gets QQQ features, Non-Tech gets SPY features).
    *   Save `sector_map.json` for routing.

## 4. Success Metrics
*   **Operational Success:** The `production_daily_plan_v6_2_4.py` script runs without error and generates a signal CSV.
*   **Model Validation:** (Implicit) The models are trained using the exact configurations that achieved >52% Win Rate in prior experiments.
