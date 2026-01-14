# EXP-05 Review: Sector-Specific Ensembles

## 1. Results Summary

| Metric | Global Model | Ensemble (Tech + Non-Tech) | Diff |
| :--- | :--- | :--- | :--- |
| **Win Rate** | 51.37% | **52.19%** | **+0.82%** |
| **Avg Return** | 0.080% | **0.148%** | **+0.068%** |
| **Trades** | 7199 | 7051 | -148 |

## 2. Analysis
*   **Success**: The experiment is a clear success. The Sector-Specific Ensemble approach outperformed the Global Model in both Win Rate (+0.82%) and Average Return (almost doubled, from 0.08% to 0.15%).
*   **Sector Behavior**:
    *   **Tech**: Win Rate ~50.22%, Avg Return ~0.02%. Tech stocks are harder to predict or have tighter margins in this regime.
    *   **Non-Tech**: Win Rate ~53.28%, Avg Return ~0.22%. The Non-Tech model is performing exceptionally well.
*   **Interpretation**:
    *   Tech stocks likely have different volatility profiles and mean-reversion speeds compared to the broader market.
    *   By splitting the models, the Non-Tech model was able to capitalize on more consistent behaviors without being "confused" by Tech stock patterns (or vice versa).
    *   The slight reduction in trade count (-148) suggests the specialized models are slightly more selective, filtering out lower-quality setups.

## 3. Conclusion & Recommendation
*   **Adopt Sector-Specific Ensembles**: This architecture should be adopted for the V6.2 production system.
*   **Next Steps**:
    *   Verify if further splitting (e.g., Energy/Financials) adds value, though sticking to a simple Tech/Non-Tech split is robust and manageable.
    *   Investigate why Tech performance is lower. Are we missing specific features for Tech (e.g., Nasdaq volatility vs VIX)?
    *   **Action**: Mark EXP-05 as Success. Add a follow-up to investigate Tech-specific features or just proceed to hyperparameter tuning for the Ensemble.

## 4. Artifacts
*   `model_tech.joblib`: Model trained on Tech stocks.
*   `model_non_tech.joblib`: Model trained on Non-Tech stocks.
*   `sector_map.json`: Mapping of tickers to sectors.
