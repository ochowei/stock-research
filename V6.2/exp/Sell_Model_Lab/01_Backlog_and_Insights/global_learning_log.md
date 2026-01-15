# Global Learning Log

## 2026-01-28: EXP-22 Context-Aware Hyperparameter Optimization (Re-Tune)
*   **Lesson:** **Robustness > Optimization.** The rigorous manual regularization parameters discovered in EXP-06 (Tech: Depth 3, Non-Tech: Unlimited) proved superior to automated Optuna tuning for the V6.2.4.RC architecture.
*   **Observation:** The automated optimization process, driven by validation set metrics (2020-2023), selected deeper trees (Depth 11 for Tech) that overfitted to the training regime. When tested on 2024-2025 data, these "optimized" models underperformed the simpler, constrained baseline models.
*   **Insight:** The "Tech = Herd" insight implies that signal behavior is simple and driven by macro factors (QQQ). Allowing the model more complexity (Deeper Trees) just allows it to memorize noise rather than learning the core macro relationship.
*   **Action:** Reject the new hyperparameters. Stick to the battle-tested EXP-06 parameters for the V6.2 production system.

## 2026-01-27: EXP-21 Limit Order Entry Optimization
*   **Lesson:** **The Best Shorts Don't Look Back.** Attempting to "short into strength" by placing Limit Orders above the open (Open + 0.5%) degraded performance across all metrics (Sharpe 4.90 vs 5.46 for Baseline).
*   **Observation:** The trades that triggered the limit orders (by spiking up initially) had a *lower* average return (1.37%) than the baseline pool (1.49%). This implies an **Adverse Selection** bias: stocks that show immediate buying strength after the open are less likely to reverse significantly compared to those that drop immediately.
*   **Action:** Reject Limit Orders. Maintain Market On Close (MOC) execution at the Open price. The "Morning Fake-Out" (EXP-12) is likely a phenomenon of losing trades, not a reliable entry signal for winners.

## 2026-01-26: EXP-20 Relative Gap Features (Index Interaction)
*   **Lesson:** **Implicit Interaction > Explicit Engineering.** Explicitly calculating the difference between Stock Gap and Index Gap (e.g., `Rel_Gap = Stock_Gap - QQQ_Gap`) **degraded** performance in both Tech (-0.49%) and Non-Tech (-0.64%) sectors.
*   **Observation:** The baseline model (using raw `Gap_Pct` and `Index_Gap_Pct` as separate features) already captures the interaction effectively. Tree-based models (LightGBM) are naturally capable of learning these non-linear relationships. Forcing a linear subtraction likely added noise or diluted the signal.
*   **Action:** Reject `Rel_Gap` features. Stick to the V6.2.4.RC architecture (Base Features + Raw Index Features) which allows the model to determine the optimal relationship itself.

## 2026-01-25: EXP-19 Crypto Sector Specific Model (Pure Play Data)
*   **Lesson:** **Data Purity is Key for Domain Features.** When testing the Crypto-Specific Model on a strictly "Pure Play" pool (COIN, MSTR, RIOT, MARA), the addition of Bitcoin features (`BTC_Change`, `BTC_Gap`) successfully improved the Win Rate (+0.50%) over the Base model.
*   **Observation:** This reverses the negative result of EXP-15, proving that the failure was due to the inclusion of non-pure crypto stocks in the pool. For true proxy stocks, the underlying asset's price action (BTC) is the dominant driver of alpha (Top 2 Feature Importance).
*   **Action:** Adopt the Crypto-Enhanced Model (Base + BTC features) specifically for the Pure Play Crypto sector.

## 2026-01-23: EXP-18 Production Script Update (Position Sizing)
*   **Lesson:** **Operationalizing Alpha.** The Tiered Position Sizing strategy (verified in EXP-17) was successfully integrated into the daily production script `production_daily_plan_v6_2_5_rc.py`.
*   **Observation:** The system now automatically assigns position sizes (1.5x, 1.0x, 0.5x) based on the model's confidence probability. This removes manual intervention and ensures that high-conviction trades receive the capital allocation they deserve to maximize the Sharpe Ratio.
*   **Action:** Ensure all future executions use `v6.2.5.rc` (or later) scripts to benefit from this logic.

## 2026-01-22: EXP-17 Confidence-Based Position Sizing
*   **Lesson:** **Confidence Correlates with Alpha.** The LightGBM model's probability output (`predict_proba`) is a reliable indicator of trade quality. A tiered position sizing strategy (1.5x for High Prob, 1.0x for Medium, 0.5x for Low) significantly improved the Sharpe Ratio (6.24 vs 5.96) and reduced Max Drawdown by 17% compared to fixed sizing.
*   **Observation:** The "Linear" sizing approach was too aggressive, increasing total return but also volatility, leading to a lower Sharpe Ratio. The "Tiered" approach struck the optimal balance, effectively "betting big" only when the model was most certain.
*   **Action:** Adopt Tiered Position Sizing for the production system. Risk management should be proactive (before entry) rather than reactive (intraday stops).

## 2026-01-21: EXP-16 Catastrophe Stop-Loss Optimization
*   **Lesson:** **Intraday Stops Are Detrimental.** Even wide "Catastrophe Stops" (3x ATR, Fixed 10%) failed to improve the Sharpe Ratio or reduce Drawdown compared to the baseline "Hold to Close" strategy.
*   **Observation:** The "Max Drawdown" actually *worsened* with stops (from -0.66 to -0.70). This counter-intuitive result confirms that extreme intraday moves ("morning fake-outs") in this strategy are highly mean-reverting. Exiting at the high (the stop trigger) effectively locks in the maximum possible loss for the day, whereas holding often allows the price to fade back towards the entry.
*   **Action:** Do not implement trade-level stops. Risk must be managed via position sizing, not by trying to time intraday exits.

## 2026-01-20: EXP-15 Crypto-Specific Ensemble (Clean Data Redux)
*   **Lesson:** **Macro Overfitting in Niche Sectors.** Adding Bitcoin features (`BTC_Ret`, `BTC_Trend`) to a "Pure Crypto" stock pool (COIN, MSTR, RIOT) significantly *degraded* performance (Win Rate -4%) compared to a simple Base feature model.
*   **Observation:** The macro features dominated feature importance, distracting the model from asset-specific price action signals (Gap, RSI) which were actually more predictive. This mirrors the "contaminated pool" failure of EXP-10 but confirms it wasn't just about data purity—it's a fundamental issue where external macro context injects noise into mean-reversion signals for these high-beta assets.
*   **Action:** Do not build a specialized "Crypto Model" with BTC features. Treat crypto stocks as standard volatile assets using the Base (or Non-Tech) model.

## 2026-01-15: EXP-14 Delayed Entry Optimization
*   **Lesson:** **Timing the Market < Time in the Market (Even Intraday).** Delaying entry by 1 hour to avoid the "Morning Fake-Out" destroyed performance, reducing Win Rate by ~5% and Avg Return by ~50%.
*   **Observation:** While EXP-12 suggested the *average* 1H return is negative, EXP-14 proves that *winning trades* likely resolve immediately. By waiting 1 hour, we miss the most profitable moves completely.
*   **Insight:** The V6.4 signals are high quality and don't need "confirmation" via a 1-hour wait. The "Morning Fake-Out" phenomenon is likely driven by losing trades, not the winners.
*   **Action:** Reject Delayed Entry. Continue to execute "Sell at Market Open" (MOC Strategy).

## 2026-01-14: EXP-12 Time-Decay Exit Optimization
*   **Lesson:** **Alpha Does Not Decay Intraday.** The hypothesis that the edge from the Sell Model is a "quick burst" that dissipates after a few hours was decisively rejected.
*   **Observation:** The "Hold to Close" (Market On Close) strategy yielded a Total Return of **+31.16%**, whereas exiting after 1 hour resulted in a **-0.86%** return. Exiting at any hourly interval (2H, 3H, 4H) significantly underperformed holding to the close.
*   **Insight:** The negative 1H return suggests a "Morning Fake-Out" pattern where prices initially move against the short signal before reversing later in the day. The edge is a day-long mean reversion/distribution event, not a scalp.
*   **Action:** Retain "Hold to Close" as the production execution strategy. (Update: EXP-14 confirmed that trying to time this "Fake-Out" via delayed entry is counter-productive).

## 2025-02-26: EXP-11 Non-Tech Feature Augmentation (SPY Context)
*   **Lesson:** **Context Works for Everyone.** Just as QQQ features improved the Tech model, adding SPY features (`Gap`, `RSI`, `Dist_MA20`) to the Non-Tech model significantly improved performance (+0.83% Win Rate, +46% Avg Return).
*   **Observation:** SPY features became the top 3 most important predictors for Non-Tech stocks, confirming that broad market sentiment is a primary driver for mean reversion, even more so than individual stock technicals.
*   **Action:** Adopt the "Base + SPY" model for all Non-Tech tickers. This completes the "Contextual Ensemble" vision where every stock is traded with awareness of its relevant sector/market benchmark.

## 2025-02-25: EXP-10 Crypto-Specific Ensemble
*   **Lesson:** **Domain Definition is Critical.** The attempt to train a "Crypto" model failed because the "Crypto Sensitive Pool" resource contained non-crypto stocks (e.g., Dutch Bros, Hims, Trade Desk).
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
