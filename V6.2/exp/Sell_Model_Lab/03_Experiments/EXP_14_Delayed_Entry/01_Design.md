# Experiment Design: Delayed Entry Optimization (EXP-14)

## 1. Hypothesis
EXP-12 revealed that the Sell Model's trades often experience a negative return in the first hour (-0.86%), suggesting a "Morning Fake-Out" or squeeze before the trend reverses.
**Hypothesis:** Delaying the trade entry by 1 hour (entering at 10:30 AM EST) will bypass this initial adverse excursion, resulting in a better entry price, higher Win Rate, and higher Average Return.

## 2. Plan
*   **Models:** Use the V6.4 Production Models (Heterogeneous Ensemble: Tech w/ QQQ, Non-Tech w/ SPY).
*   **Data Scope:** Last 730 days (approx. 2 years) to accommodate `yfinance` 1-hour data availability limits.
*   **Procedure:**
    1.  **Signal Generation:** Re-run the V6.4 strategy on the test period (last 730 days) to identify all valid "Sell" signals using daily data (T-1).
    2.  **Intraday Data Fetching:** For every generated signal, fetch 1-hour resolution data for the specific trade date.
    3.  **Execution Simulation:**
        *   **Baseline (MOC):** Entry at Market Open (9:30), Exit at Market Close (16:00).
        *   **Delayed (1H Entry):** Entry at 10:30 Open (start of the 2nd hour candle), Exit at Market Close (16:00).
    4.  **Comparison:** Calculate and compare performance metrics.

## 3. Success Metrics
*   **Primary:** Win Rate > Baseline.
*   **Secondary:** Average Return > Baseline.
*   **Check:** Total Return (Ensure we don't lose too much volume/opportunity).

## 4. Resources
*   **Models:** `../EXP_13_Production_Deployment/03_Output/v6.4_tech_model.joblib` and `v6.4_non_tech_model.joblib`.
*   **Sector Map:** `../EXP_13_Production_Deployment/03_Output/sector_map.json`.
*   **Asset Pool:** `../../../../resource/2025_final_asset_pool.json`.
