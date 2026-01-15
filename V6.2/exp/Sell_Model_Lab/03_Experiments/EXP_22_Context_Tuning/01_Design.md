# EXP-22: Context-Aware Hyperparameter Optimization (Re-Tune)

## 1. Context
The current hyperparameters (tuned in EXP-06) were optimized for the 5-feature Base set. V6.2.4.RC architecture introduces Sector Context (QQQ/SPY features), increasing the feature count to ~10. The strict regularization (e.g., Tech Depth=3) is likely causing underfitting by preventing the model from learning the interaction between Stock Technicals and Market Context.

## 2. Hypothesis
Re-tuning hyperparameters specifically for the V6.2.4.RC architectures (Tech=Base+QQQ, NonTech=Base+SPY) will unlock additional alpha hidden in the context features.

## 3. Plan
1.  **Data Preparation**:
    *   Load Asset Pool (`2025_final_asset_pool.json`).
    *   Fetch Data (2020-2025).
    *   Generate Features:
        *   **Tech Sector**: Base (5) + QQQ Context (4) = 9 Features.
        *   **Non-Tech Sector**: Base (5) + SPY Context (4) = 9 Features.
    *   Split Train/Test:
        *   Train: 2020-01-01 to 2023-12-31
        *   Test: 2024-01-01 to 2025-12-31

2.  **Hyperparameter Optimization (Optuna)**:
    *   **Objective**: Maximize Sharpe Ratio (with Win Rate > 50% constraint).
    *   **Tech Optimization**: Run study on Tech Signals.
    *   **Non-Tech Optimization**: Run study on Non-Tech Signals.
    *   **Search Space**:
        *   `learning_rate`: [0.005, 0.01, 0.02, 0.05, 0.1]
        *   `num_leaves`: [15, 31, 63, 127]
        *   `max_depth`: [-1, 3, 5, 7, 10, 15]
        *   `min_child_samples`: [20, 50, 100]
        *   `reg_alpha`, `reg_lambda`: [0, 0.1, 0.5, 1.0, 5.0]

3.  **Validation**:
    *   Train "Best Context Models" on full Train set.
    *   Compare against "Baseline Context Models" (using EXP-06 params) on Test set.
    *   Metrics: Win Rate, Avg Return, Sharpe Ratio.

4.  **Success Criteria**:
    *   Test Set Win Rate > 55%.
    *   Test Set Avg Return > 0.20%.
    *   Improvement over Baseline (EXP-06 Params applied to Context Features).
