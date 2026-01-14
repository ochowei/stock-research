# EXP-10 Review: Crypto-Specific Ensemble

## 1. Summary
*   **Hypothesis:** Rejected.
*   **Outcome:** The Crypto-Ensemble (Base + BTC Features) **underperformed** the Baseline (Base Only) in Win Rate (-1.32%) but **improved** Average Return (reduced average loss from -0.14% to -0.06%).
*   **Status:** Do Not Adopt (in current form).

## 2. Performance Metrics (Test Period: 2024)
| Model | Win Rate | Avg Return | Total Return | Signals |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Base Only)** | **53.98%** | -0.14% | -85.5% | 628 |
| **Crypto Model (Base + BTC)** | 52.66% | **-0.06%** | -46.9% | 771 |

*Note: Both models failed to generate positive alpha in 2024 for this specific basket of stocks, likely due to the strong bullish momentum of Growth/Tech stocks in 2024 (Shorting the Open was a losing strategy).*

## 3. Key Findings
1.  **Feature Dominance (Overfitting?):** In the Crypto Model, BTC features (`BTC_RSI`, `BTC_Gap`, `BTC_Change`) took the top 3 spots in feature importance, displacing `Gap_Pct`. The model effectively ignored the stock's own technicals in favor of Bitcoin's movement.
2.  **Data Quality Critical Failure:**
    *   **Root Cause:** The provided `2025_final_crypto_sensitive_pool.json` contains stocks with **no correlation to Crypto**, such as `BROS` (Dutch Bros Coffee), `HIMS` (Telehealth), and `TTD` (AdTech).
    *   **Impact:** Forcing Bitcoin features onto coffee and telehealth stocks introduced significant noise, explaining why the model's precision (Win Rate) dropped.
3.  **Loss Mitigation:** Despite the noise, the Crypto model was "less wrong" on average (Avg Return -0.06% vs -0.14%). This suggests that during extreme BTC moves, the model might have avoided some deep losses, or the added noise simply dampened the confidence in "bad" high-conviction signals.

## 4. Recommendations
1.  **Immediate Action:** **Audit and Rebuild the Crypto Asset Pool.** The current pool is contaminated with generic Growth/Retail stocks. A true test requires a pure-play list (e.g., `COIN`, `MSTR`, `MARA`, `RIOT`, `CLSK`, `BITF`).
2.  **Follow-Up Experiment:** Re-run EXP-10 with a corrected, pure-play Crypto Pool.
3.  **Strategic Insight:** For the 2024 regime, "Shorting at Open" was unprofitable for this high-beta basket. Future experiments should check if "Long at Open" (Dip Buy) is the correct strategy for this cluster in the current regime.
