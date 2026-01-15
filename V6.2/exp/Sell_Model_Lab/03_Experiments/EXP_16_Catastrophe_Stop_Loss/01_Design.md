# EXP-16: Catastrophe Stop-Loss Optimization

## 1. Hypothesis
While tight stop losses (0.5% - 2.0%) have been proven to degrade performance by triggering on "morning fake-outs" (EXP-09, EXP-12), the current "Hold to Close" strategy exposes the portfolio to unlimited left-tail risk (e.g., meme stock squeezes).
We hypothesize that a **Wide "Catastrophe Stop"** (e.g., 3x ATR or fixed 5%) can truncate extreme losses without being triggered by normal volatility, thereby improving the Sharpe Ratio and Sortino Ratio.

## 2. Plan
1.  **Generate Trades**: Use the V6.4 Production Models (Tech & Non-Tech) to generate a full set of historical predictions (2024-2025).
2.  **Simulate Stops**: For each trade, use daily OHLC data to determine if a stop level was breached.
    *   Since we are Shorting at Open:
    *   Stop Price = Entry Price * (1 + Threshold)
    *   Trigger: High >= Stop Price
    *   Exit Price: Stop Price (assuming stop execution)
3.  **Test Thresholds**:
    *   No Stop (Baseline)
    *   Fixed 3%
    *   Fixed 5%
    *   Fixed 10%
    *   2x ATR (daily)
    *   3x ATR (daily)
4.  **Metrics**:
    *   **Sharpe Ratio** (Primary)
    *   **Max Drawdown**
    *   **Total Return**
    *   **Win Rate**
    *   **Stop Trigger Rate** (% of trades stopped out)

## 3. Success Criteria
*   **Sharpe Ratio** > Baseline (No Stop).
*   **Total Return** must not decrease by more than 10% (we accept a small insurance premium).
*   **Max Drawdown** should decrease.

## 4. Execution
*   Load V6.4 Models (`v6.4_tech_model.joblib`, `v6.4_non_tech_model.joblib`).
*   Generate/Load Dataset (2024-01-01 to Present).
*   Iterate through stops and calculate metrics.
