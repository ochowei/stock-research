# EXP-05 Review: Dynamic Window Sensitivity

## 1. Summary
*   **Status:** FAIL (Hypothesis Rejected)
*   **Best Window:** 10 Days (Win Rate 33.6%), but still significantly underperformed the Baseline (47.4%).
*   **Key Insight:** Pre-market momentum (RSI, ROC at T-1) is a **counter-indicator** for Gap Continuation. High momentum predicts gap fading (reversal), not continuation.

## 2. Performance Analysis

| Window | Win Rate | Avg Return | Count | Baseline Win | Baseline Avg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5** | 31.6% | -1.36% | 605 | 47.4% | -0.08% |
| **10** | **33.6%** | **-0.91%** | 524 | 47.4% | -0.08% |
| **14** | 33.0% | -1.00% | 564 | 47.4% | -0.08% |
| **20** | 32.6% | -0.95% | 488 | 47.4% | -0.08% |
| **50** | 29.4% | -1.97% | 394 | 47.4% | -0.08% |

### Observations:
1.  **Negative Alpha:** The model, designed to pick "Momentum Continuation" trades (Green Candle after Gap Up), consistently picked losers. Win rates dropped from ~47% (random/baseline) to ~33%.
2.  **Sensitivity:** Shorter windows (10 days) performed marginally better than longer windows (50 days), likely because they react faster to recent price action, but the direction of prediction is wrong.
3.  **Regime Check:** The 2024-2025 regime (OOS) shows that "Buying Gap Ups" in general is a losing strategy (Baseline Avg Return -0.08%).

## 3. Conclusions
1.  **Look-ahead Bias Confirmation:** By strictly removing look-ahead bias (using T-1 Close for indicators), the "Momentum" edge completely disappears and flips. Previous successes (EXP-01) likely benefited from implicit look-ahead bias.
2.  **Exhaustion Signal:** High RSI/ROC at T-1 combined with a Gap Up at T Open is a classic **Exhaustion** setup. The market gaps up on hype but immediately sells off (Fade the Gap).
3.  **Window 10 is "Optimal":** Among the tested windows, 10 days captured the signal best (even though the signal was negative). If we were to build a Mean Reversion model (Short the Gap), Window 10 would likely be the best feature.

## 4. Recommendations
1.  **Pivot to Mean Reversion:** The next experiment should test **Shorting** (or fading) these high-momentum gap ups.
2.  **Intraday Confirmation:** Pre-market momentum is not enough. We need intraday confirmation (e.g., Opening Range Breakout) to filter out the immediate fades.
3.  **Abandon Pure Momentum:** Do not deploy this model. The hypothesis that "Trend continues after Gap" is false for high-RSI setups in this regime.
