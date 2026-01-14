# Global Learning Log

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
