# EXP-12: Time-Decay Exit Optimization

## 1. Context
Previous experiments (EXP-09) demonstrated that a "Hold to Close" (Market On Close) strategy significantly outperforms fixed profit targets (e.g., 0.2%). However, market microstructure theory suggests that "alpha decays" over time. The signal edge might be strongest in the first few hours of trading, and holding until 4:00 PM might expose the trade to unnecessary late-day noise or mean reversion.

## 2. Hypothesis
**Hypothesis:** Exiting trades based on "Time in Trade" (e.g., after 1, 2, or 3 hours) will improve the **Risk-Adjusted Return** (Sharpe/Sortino) or **Total Return** compared to holding until Market Close.

**Rationale:**
- The Sell Model detects "gap exhaustion" or "reversal" setups at the Open.
- The correction often happens quickly (morning flush).
- Holding past the initial move might give back profits as the stock stabilizes or drifts with the market.

## 3. Plan
1.  **Signal Generation:** Use the production V6.3 models (Tech & Non-Tech) to generate signals on the Test Set (2024-01-01 to Present).
2.  **Data Acquisition (Granular):** For every valid signal, fetch **Hourly (1h)** OHLC data from `yfinance`.
3.  **Execution Simulation:**
    - Calculate exit prices at specific time horizons:
        - **1 Hour** (approx 10:30 AM)
        - **2 Hours** (approx 11:30 AM)
        - **3 Hours** (approx 12:30 PM)
        - **4 Hours** (approx 1:30 PM)
        - **Hold to Close** (Benchmark)
    - Note: Standard trading hours are 9:30 AM - 4:00 PM.
    - 1h candles usually start at 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 (30 mins).
    - We will use the **Close** of the Nth hourly bar as the exit price.
4.  **Metrics:** Compare Win Rate, Avg Return, and Total Return across time horizons.

## 4. Success Metrics
- **Primary:** Total Return > Baseline (Hold to Close).
- **Secondary:** Win Rate stability.
