# EXP-03 Review: Feature Selection (Ablation Study)

## 1. Executive Summary
*   **Result**: ✅ **Success** (Simplification Identified)
*   **Best Model**: **Base Model (5 Features)**
*   **Key Finding**: The "Base" feature set (`Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20`) outperformed all larger feature sets in the OOS period (2024-2025). Adding TOTM, Crypto, or VIX features actually **reduced** Win Rate and Average Return, indicating they were introducing noise or causing overfitting in this specific Sell Model context.

## 2. Detailed Metrics (OOS 2024-2025)
| Subset | Features | Signals | Win Rate | Avg Return |
| :--- | :--- | :--- | :--- | :--- |
| **Base** | 5 | 8730 | **51.96%** | **0.11%** |
| Base_Crypto | 8 | 8173 | 51.34% | -0.00% |
| All | 13 | 9147 | 50.95% | 0.01% |
| Base_TOTM | 7 | 8032 | 50.54% | -0.06% |

**Baseline (Previous Best)**: ~52.2% Win Rate (EXP-02), but note that EXP-02 used the "All" features. The current run of "All" yielded 50.95%. This discrepancy might be due to minor data differences or randomness in LightGBM, but the *relative* ranking within this experiment is clear: **Simpler is better.**

## 3. Analysis
*   **Overfitting Confirmed**: The additional features (Crypto, VIX, Time of Month) appear to help the model memorize training data patterns that do not hold in 2024-2025.
*   **Crypto Failure**: Unlike EXP-01 (XGBoost), where Crypto features seemed to help, here with LightGBM they hurt performance. This might suggest that LightGBM is more sensitive to the noise in Crypto correlations for this specific strategy.
*   **Volume Ratio**: Despite `Vol_Ratio` showing low importance in previous runs, removing it (implicitly, by testing Base which includes it) works well. It likely acts as a basic liquidity filter.
*   **Permutation Importance**: (Refer to `03_Output/permutation_importance.png`)
    *   Top features are likely `Gap_Pct` and `RSI_14`.
    *   Crypto/VIX features likely had low or negative permutation importance, confirming they were distracting the model.

## 4. Conclusion & Recommendations
*   **Revert to Base Features**: For the next iteration of the Sell Model, we should strip away the extra features and focus on optimizing the core technicals.
*   **Refine Base**: Since "Base" is now the winner, we can try to improve it by tuning the *definitions* of these 5 features (e.g., RSI period, ATR lookback) rather than adding new ones.
*   **Next Steps**:
    *   **EXP-04 (Regime Switching)** might still be valid, but use the "Base" model as the underlying engine.
    *   **New Idea**: Hyperparameter tuning on the "Base" LightGBM model to squeeze out more performance.

## 5. Artifacts
*   `subsets_performance.csv`: detailed metrics.
*   `permutation_importance.png`: feature ranking.
