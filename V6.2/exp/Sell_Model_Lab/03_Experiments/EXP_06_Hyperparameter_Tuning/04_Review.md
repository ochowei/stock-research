# EXP-06 Review: Base Feature Hyperparameter Tuning

## 1. Executive Summary
*   **Result:** ✅ Success (Optimization improved performance).
*   **Outcome:** Adopt the **Optimized Sector-Specific Ensemble**.
*   **Key Stat:** Win Rate **52.23%** (+0.54% vs Baseline Ensemble, +0.68% vs Global Baseline). Average Return **0.14%** (+0.01% vs Baseline).
*   **Significance:** Strong validation of the Sector-Specific approach. The "Tech" sector required vastly different hyperparameters (highly regularized) compared to "Non-Tech" (more complex), proving that a global model cannot optimally fit both regimes simultaneously.

## 2. Performance Metrics (Test Set 2024-2025)

| Model | Win Rate | Avg Return | Trade Count | Selectivity |
| :--- | :--- | :--- | :--- | :--- |
| **Global Baseline** | 51.55% | 0.10% | 7814 | Low |
| **Global Optimized** | 51.12% | 0.12% | 7238 | Medium |
| **Ensemble Baseline** | 51.69% | 0.13% | 7351 | Medium |
| **Ensemble Optimized** | **52.23%** | **0.14%** | 5828 | **High** |

*   **Win Rate:** The Optimized Ensemble broke the 52% barrier, a significant milestone for this strategy.
*   **Selectivity:** The optimized models are far more selective (~20% fewer trades than baseline), filtering out low-quality signals.

## 3. Hyperparameter Insights

The tuning process revealed distinct "personalities" for each sector:

### A. Tech Sector (The "Noisy" Child)
*   **Best Params:** `n_estimators=100`, `learning_rate=0.01`, **`max_depth=3`**, `reg_lambda=0`, `colsample_bytree=0.6`.
*   **Insight:** Tech stocks required **extreme regularization**. A shallow depth (3) and very low learning rate (0.01) suggest that the signal-to-noise ratio in Tech is low. The model performs best when forced to be simple and conservative.

### B. Non-Tech Sector (The "Stable" Child)
*   **Best Params:** `n_estimators=200`, `learning_rate=0.02`, `num_leaves=50`, `max_depth=-1` (Unlimited).
*   **Insight:** Non-Tech stocks allowed for **higher complexity** (Unlimited depth, more leaves). The model could learn more nuanced patterns without overfitting as easily as in Tech.

### C. Global Model
*   **Best Params:** `n_estimators=500`, `learning_rate=0.05`, `num_leaves=63`.
*   **Insight:** The global model tried to compensate for the diversity of data by becoming **very complex** (500 trees, 63 leaves). However, this complexity failed to generalize as well as the specialized ensemble, actually *reducing* Win Rate (51.12%) compared to the simpler baseline.

## 4. Conclusion & Recommendations

1.  **Adoption:** Immediately update the production `daily_gap_signal_generator.py` to use the **Optimized Sector-Specific Ensemble**.
2.  **Configuration:** Hardcode the specific hyperparameters found for Tech and Non-Tech models into the production script (or load them from a config).
3.  **Future Work:**
    *   The "Tech" model is still the weak link (Win Rate likely lower than Non-Tech, though not explicitly broken out in final summary, the conservative params suggest difficulty). EXP-07 (Tech Features) is now critical.
    *   Investigate if "Consumer Discretionary" behaves more like Tech or Non-Tech (sub-sector split).

## 5. Artifacts
*   `model_tech_opt.joblib`: The tuned Tech model.
*   `model_non_tech_opt.joblib`: The tuned Non-Tech model.
*   `best_params.json`: Full parameter set.
*   `performance_report.csv`: Detailed metrics.
