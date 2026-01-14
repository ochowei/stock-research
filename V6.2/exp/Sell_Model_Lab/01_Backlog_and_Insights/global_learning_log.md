# Global Learning Log

## 2025-02-25: EXP-10 Crypto-Specific Ensemble
*   **Lesson:** **Domain Definition is Critical.** The attempt to train a "Crypto" model failed because the "Crypto Sensitive Pool" resource contained non-crypto stocks (e.g., Dutch Bros, Hims).
*   **Observation:** Forcing high-context features (Bitcoin Price Action) onto unrelated assets creates noise that degrades predictive power (Win Rate dropped from 53.98% to 52.66%). However, the model did reduce the average loss per trade, possibly by filtering out bad trades during extreme BTC volatility.
*   **Action:** Verify and audit asset pools *before* engineering domain-specific features. A feature is only as good as its relevance to the target asset class.

## 2025-02-24: EXP-09 Execution Logic Refinement
*   **Lesson:** **Patience Pays Off.** The standard execution logic of taking small profits (0.2%) and holding losses to close was detrimental, yielding a negative return (-2.33%). The optimal strategy for the V6.3 models is simply **Hold to Close** (Exit MOC).
*   **Observation:** The "Hold to Close" strategy achieved a **33.70% Total Return** and **0.37% Average Return**, drastically outperforming any Profit Take/Stop Loss combination. Intraday stops (0.5%-2.0%) reduced performance by cutting trades early during normal volatility.
*   **Action:** Update the production execution protocol to remove the 0.2% Profit Target. Execute purely on Time (Market Open -> Market Close).

## 2025-02-23: EXP-08 Production Integration (V6.3 Release)
*   **Lesson:** **Heterogeneous Execution is Viable.** Successfully deployed a production system that routes tickers to completely different model pipelines (Features + Hyperparameters) based on their sector.
*   **Observation:** The system correctly identified `DDOG` (Technology) and applied the Tech Model (Base+QQQ features), while applying the Non-Tech Model (Base features) to `ABAT` (Industrials), proving that complex, conditional logic can be reliably automated in the signal generation phase.
*   **Action:** Ensure future monitoring checks for "Sector Drift" (e.g., if a ticker changes sector in `yfinance`) and `Sector_Corr` stability, as these are now critical dependencies for the Tech model.

## 2025-02-21: EXP-07 Tech-Specific Feature Engineering
*   **Lesson:** **Context is King for Tech.** The Tech sector model, which was previously the "weak link" (49.8% Win Rate), was transformed into a top performer (53.35% Win Rate) by adding sector-specific context (`QQQ` Gap, RSI, MA_Dist).
*   **Observation:** The top 3 most important features for Tech stocks were *all* QQQ-based features. This proves that individual Tech stocks are heavily driven by the sector's overall sentiment and momentum.
*   **Action:** Implement a **Heterogeneous Ensemble** for Production: Use Base Features (5) for Non-Tech, but Base + QQQ Features (9) for Tech.

## 2025-02-20: EXP-06 Base Feature Hyperparameter Tuning
*   **Lesson:** **Sectors Have Distinct "Personalities".** The Tech sector model required extreme regularization (Depth=3, Learning Rate=0.01) to perform well, suggesting a low signal-to-noise ratio. In contrast, the Non-Tech sector model thrived with high complexity (Unlimited Depth, Learning Rate=0.02, Leaves=50).
*   **Observation:** A globally tuned model performed *worse* than the baseline, likely because it tried to find a "middle ground" that was suboptimal for both sectors.
*   **Action:** When deploying the Sector-Specific Ensemble, do not use a single set of hyperparameters. Hardcode the specific, divergent hyperparameters for Tech vs Non-Tech models to maximize the Ensemble's performance (Win Rate 52.23%).

## 2025-02-19: EXP-05 Sector-Specific Ensembles
*   **Lesson:** **Sector Specialization Improves Precision.** Training separate models for Tech (QQQ/Technology) and Non-Tech stocks yielded a +0.82% improvement in Win Rate (52.19% vs 51.37%) and nearly doubled the Average Return (0.15% vs 0.08%) compared to a single Global Model.
*   **Observation:** The "Non-Tech" model performed exceptionally well (53.3% WR), suggesting that non-tech stocks have more predictable mean-reversion properties in this framework. Tech stocks proved harder to predict (50.2% WR), likely due to different volatility drivers.
*   **Action:** Adopt the Sector-Specific Ensemble architecture for the V6.2 production system. This structural change offers significant alpha without increasing feature complexity.

## 2024-05-26: EXP-04 Regime-Switching Model (High/Low VIX)
*   **Lesson:** **Regime Splitting Adds Complexity, Not Value.** Training separate models for High VIX (>20) and Low VIX (<=20) environments yielded only a negligible improvement in Win Rate (+0.22%) and Total Return.
*   **Observation:** The Global model (trained on all data) performed nearly as well as the specialized ensemble. This suggests that the LightGBM model, even with just 5 base features (which includes `ATR_Pct` and `Vol_Ratio`), is capable of learning volatility dynamics internally without needing explicit hard-coded splits.
*   **Action:** Discard the Regime-Switching architecture. Focus on Sector-specific models next, as that might represent a more fundamental divergence in price behavior than just "volatility level".

## 2024-05-25: EXP-03 Feature Selection (Ablation Study)
*   **Lesson:** **Simpler is Better.** The "Base" feature set (5 features: Gap, RSI, ATR, Vol_Ratio, Dist_MA20) outperformed larger sets that included Crypto, VIX, and TOTM features in the OOS period (2024-2025).
*   **Observation:** Win Rate for "Base" was 51.96% vs 50.95% for "All". This contradicts EXP-01/02 findings slightly, suggesting that while Crypto/VIX features explain training data well, they may lead to overfitting or fail to generalize in recent market regimes (2024+).
*   **Action:** Revert to the 5-feature Base set for immediate future experiments (Regime Switching, etc.) to establish a robust, low-variance baseline.

## 2024-05-24: EXP-02 LightGBM Migration
*   **Lesson:** Migrating to LightGBM retained the Win Rate of XGBoost (~52.2%) but significantly improved Average Return (+29%) and selectivity.
*   **Observation:** VIX is by far the most dominant feature, followed by Crypto features. Traditional volume features (`Vol_Ratio`) appeared irrelevant in this tree configuration.
*   **Correction:** Identified and fixed a data leakage in VIX (look-ahead bias using today's close). Future models must ensure all daily indicators (VIX, Crypto) are shifted by 1 day (T-1) relative to the trade entry (T Open).
*   **Action:** Adopt LightGBM as the default engine. Investigate dropping `Vol_Ratio`.

## 2024-05-23: EXP-01 Crypto Feature Integration
*   **Lesson:** Crypto market state features (specifically `Crypto_Corr` and `BTC_Trend`) significantly improve the precision of the Gap Sell Model.
*   **Observation:** The model became much more selective (80% reduction in signals), suggesting that many gap trades fail during periods of crypto/equity disconnect or negative crypto trends.
*   **Action:** Incorporate Crypto features into the standard feature set for future experiments (Base = EXP-01).
