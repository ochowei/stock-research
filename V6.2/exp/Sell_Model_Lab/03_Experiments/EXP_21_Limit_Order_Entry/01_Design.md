# EXP-21: Limit Order Entry Optimization (Short into Strength)

## 1. Hypothesis
Based on findings from EXP-12 and EXP-16, the Sell Model often faces a "Morning Fake-Out" where the price initially moves against the position (higher) before reversing.
*   **Hypothesis:** Placing a Limit Order to Sell Short slightly above the Open price (e.g., Open + 0.5%) will improve the average entry price and risk-adjusted returns (Sharpe Ratio), even if it results in a lower Fill Rate.
*   **Rationale:** Fading the initial morning volatility captures the "squeeze" before the drop.

## 2. Experiment Design
*   **Models:** Use the current production models (V6.2.4.RC) from EXP-18.
    *   `v6.2.4_rc_tech_model.joblib`
    *   `v6.2.4_rc_non_tech_model.joblib`
*   **Data:**
    *   Training Data: 2022-2023 (Implicit in models).
    *   Test Data: 2024 (Standard Test Set).
*   **Strategy Variants:**
    *   **Baseline:** Entry at Market Open (Fill Rate 100%).
    *   **Test 1:** Limit Order at `Open * 1.005` (+0.5%).
    *   **Test 2:** Limit Order at `Open * 1.010` (+1.0%).
    *   **Test 3:** Limit Order at `Open * 1.015` (+1.5%).
*   **Execution Logic:**
    *   For each signal, calculate `Limit_Price`.
    *   Check if `High_Day >= Limit_Price`.
        *   **If Yes:** Trade Filled. Entry Price = `Limit_Price`.
        *   **If No:** Trade Missed. Return = 0.
    *   Exit: Market On Close (MOC).
    *   Return Calculation: `(Entry_Price - Close) / Entry_Price` (Short Selling).

## 3. Metrics
*   **Win Rate (%):** Percentage of filled trades that are profitable.
*   **Fill Rate (%):** Percentage of signals that get filled.
*   **Average Return (%):** Average return per **filled** trade.
*   **Total Return (%):** Cumulative return of the strategy (sum of returns, assuming fixed capital per trade).
*   **Sharpe Ratio:** Risk-adjusted return of the strategy.
*   **Signal Count:** Number of signals generated vs filled.

## 4. Success Criteria
*   **Primary:** Improvement in Sharpe Ratio or Total Return compared to Baseline.
*   **Secondary:** Average Return per filled trade increases significantly to justify the reduced frequency.
