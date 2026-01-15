# EXP-08: Production Integration (V6.2.3 Release)

## 1. Objective
Consolidate findings from EXP-05 (Sector Ensembles), EXP-06 (Hyperparameter Tuning), and EXP-07 (Tech-Specific Features) into a unified production generation script (`production_daily_plan_v6_2_3.py`).

## 2. Rationale
We have established three critical components for V6.2.3:
1.  **Heterogeneous Ensemble:** Separating Tech vs. Non-Tech sectors yields better results.
2.  **Tech-Specific Feature Engineering:** Tech stocks require `QQQ` context (`Gap`, `RSI`, `Dist_MA20`, `Sector_Corr`) to perform well (WR > 53%).
3.  **Divergent Regularization:**
    *   **Tech:** Needs strict regularization (Depth 3, LR 0.01) and specific features.
    *   **Non-Tech:** Performs best with Base features and higher complexity (Unlimited Depth, LR 0.02).

## 3. Plan

### A. Model Architecture
| Model | Sector Scope | Features | Hyperparameters |
| :--- | :--- | :--- | :--- |
| **Non-Tech Model** | All except Technology | Base (5): `Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20` | `LGBMClassifier(max_depth=-1, num_leaves=31, learning_rate=0.02, n_estimators=200)` |
| **Tech Model** | Technology Only | Base (5) + Tech (4): `QQQ_Gap_Pct`, `QQQ_RSI_14`, `QQQ_Dist_MA20`, `Sector_Corr` | `LGBMClassifier(max_depth=3, learning_rate=0.01, n_estimators=300)` |

### B. Implementation Steps
1.  **Train Models:**
    *   Load full dataset (2020-2023 Train, 2024-Present Test/Verify).
    *   Train Non-Tech Model on Non-Tech stocks.
    *   Train Tech Model on Tech stocks using the merged QQQ dataset.
    *   Save models as `v6.2.3_non_tech_model.joblib` and `v6.2.3_tech_model.joblib`.

2.  **Generate Production Script (`production_daily_plan_v6_2_3.py`):**
    *   Script must handle downloading Tickers + QQQ.
    *   Must dynamically split Tickers by Sector.
    *   Must apply `prepare_benchmark_features` for QQQ and merge for Tech stocks.
    *   Must apply specific models based on sector.
    *   Must output a unified JSON/CSV plan.

3.  **Verification:**
    *   Run the training to generate artifacts.
    *   Simulate a "Daily Run" for the most recent trading day to ensure the script executes without error.

## 4. Metrics
*   **Success Criteria:**
    *   Models successfully trained and saved.
    *   Production script generated.
    *   Production script successfully produces a `daily_plan.json` for a sample date.
