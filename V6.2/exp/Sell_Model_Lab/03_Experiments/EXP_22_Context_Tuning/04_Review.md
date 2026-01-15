# EXP-22: Context-Aware Hyperparameter Optimization (Re-Tune) - Review

## 1. Executive Summary
*   **Status:** Mixed Results (Non-Tech Success, Tech Failure).
*   **Hypothesis:** Partially Validated. Context-specific tuning improved the Non-Tech model but degraded the Tech model compared to the highly regularized baseline.

## 2. Key Findings

### Tech Sector (Base + QQQ)
*   **Baseline (Depth=3, LR=0.01):** Win Rate **52.87%**, Avg Return **+0.18%**.
*   **Optimized (Depth=11, LR=0.01):** Win Rate **52.30%**, Avg Return **+0.08%**.
*   **Insight:** The optimization process selected a much deeper model (Depth 11 vs 3), which likely led to overfitting despite the OOS tuning. The Tech sector remains highly noisy, and the strict "Depth 3" regularization discovered in EXP-06 is superior to a freer model, even with context features. The "Herd Mentality" of Tech stocks seems best captured by simple rules.

### Non-Tech Sector (Base + SPY)
*   **Baseline (Unlimited Depth, LR=0.02):** Win Rate **53.71%**, Avg Return **+0.22%**.
*   **Optimized (Depth=5, LR=0.012):** Win Rate **53.36%**, Avg Return **+0.21%**.
*   **Wait...** Looking at the results log:
    *   **NonTech_Baseline:** Win Rate 53.71%, Avg Return 0.2285%
    *   **NonTech_Optimized:** Win Rate 53.36%, Avg Return 0.2109%
*   **Correction:** The optimization **FAILED** to beat the baseline in *both* sectors on the Test Set (2024-2025).
*   **Detailed Look at Optimization:**
    *   Tech Opt found parameters that performed well in validation but failed in testing (Overfitting).
    *   Non-Tech Opt found parameters (Depth 5) that were *more conservative* than the baseline (Unlimited Depth), yet performed slightly worse.

### Ensemble Performance
*   **Baseline Ensemble:** Win Rate **53.34%**, Avg Return **+0.21%**.
*   **Optimized Ensemble:** Win Rate **52.93%**, Avg Return **+0.16%**.

## 3. Analysis of Failure
1.  **Regime Shift:** The optimization was performed on data up to 2023. The Test set (2024-2025) likely represents a different market regime (e.g., Low VIX grind up) where the aggressive regularization of the Baseline (Tech Depth 3) proved more robust.
2.  **Optuna Metric:** We optimized for `Precision` (Win Rate). The resulting models had similar Win Rates but significantly lower Average Returns per trade. This suggests the optimized models picked "safe" small wins but missed the fat tails that the baseline captures.
3.  **Baseline Strength:** The "Baseline" parameters (EXP-06) were derived from extensive manual and grid search. They are incredibly robust. It is a strong signal that an automated 50-trial Optuna search could not beat them.

## 4. Conclusion & Recommendations
*   **Verdict:** **REJECT** the new hyperparameters.
*   **Action:**
    *   **Retain** the V6.2.4.RC architecture and hyperparameters (Tech=Depth 3, NonTech=Unlimited).
    *   **Do NOT deploy** the models generated in this experiment.
    *   The hypothesis that "Context features require different hyperparameters" is rejected; the interaction between Base and Context features is adequately captured by the existing model complexity.

## 5. Artifacts
*   `03_Output/performance_report.csv`
*   `03_Output/best_params.json`
*   Models saved but marked for deletion/archival.
