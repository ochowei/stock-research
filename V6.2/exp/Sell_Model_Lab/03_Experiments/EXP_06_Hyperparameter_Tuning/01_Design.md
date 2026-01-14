# EXP-06: Base Feature Hyperparameter Tuning

## 1. Hypothesis
Optimizing LightGBM hyperparameters for the specific characteristics of "Tech" vs "Non-Tech" sectors will yield higher Win Rates and Average Returns compared to using default or global hyperparameters. We hypothesize that Tech stocks, being more volatile, might require stronger regularization (e.g., lower depth, higher L1/L2) compared to Non-Tech stocks.

## 2. Experiment Design

### Architecture
*   **Base Model:** LightGBM Classifier (proven in EXP-02).
*   **Feature Set:** "Base" (5 Features: Gap_Pct, RSI_14, ATR_Pct, Vol_Ratio, Dist_MA20) (Winner of EXP-03).
*   **Ensemble Strategy:** Sector-Specific (Tech vs Non-Tech) (Winner of EXP-05).

### Tuning Strategy
*   **Method:** RandomizedSearchCV (50 iterations).
*   **Cross-Validation:** TimeSeriesSplit (5 splits) to respect temporal order.
*   **Scoring Metric:** Precision (Class 1) - directly correlates to Win Rate.
*   **Parameter Grid:**
    *   `n_estimators`: [100, 200, 300, 500]
    *   `learning_rate`: [0.01, 0.02, 0.05, 0.1]
    *   `num_leaves`: [15, 20, 31, 50, 63]
    *   `max_depth`: [-1, 3, 5, 7, 10]
    *   `min_child_samples`: [20, 50, 100]
    *   `reg_alpha` (L1): [0, 0.1, 0.5, 1.0]
    *   `reg_lambda` (L2): [0, 0.1, 0.5, 1.0]
    *   `subsample`: [0.6, 0.8, 1.0]
    *   `colsample_bytree`: [0.6, 0.8, 1.0]

### Comparison Groups
1.  **Baseline (EXP-05):** Sector Ensemble with fixed params (`n_est=200, lr=0.05, leaves=31`).
2.  **Optimized Global:** Global Model with tuned params.
3.  **Optimized Ensemble:** Tech Model (Tuned) + Non-Tech Model (Tuned).

## 3. Metrics
*   **Win Rate:** % of trades with Return > 0.
*   **Average Return:** Mean return per trade.
*   **Trade Count:** Number of signals generated.

## 4. Success Criteria
*   **Primary:** Optimized Ensemble Win Rate > Baseline Ensemble Win Rate (Target: >52.2%).
*   **Secondary:** Insight into whether Tech and Non-Tech require significantly different hyperparameters.
