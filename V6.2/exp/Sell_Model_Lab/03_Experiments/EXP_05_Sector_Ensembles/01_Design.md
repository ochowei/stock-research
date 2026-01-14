# EXP-05: Sector-Specific Ensembles

## 1. Hypothesis
Tech stocks (specifically QQQ components or Technology sector) exhibit different gap behaviors compared to the broader market. Training separate models for Tech and Non-Tech stocks will allow each model to specialize in the unique volatility and mean-reversion characteristics of its sector, leading to higher overall precision and win rates compared to a single Global Model.

## 2. Plan
1.  **Data Loading**: Load historical price data for the `2025_final_asset_pool.json`.
2.  **Sector Identification**:
    *   Fetch sector information for each ticker using `yfinance`.
    *   Classify tickers into `Tech` (Sector = 'Technology') and `Non-Tech`.
    *   Save the sector mapping for verification.
3.  **Feature Engineering**: Generate the "Base" feature set (identified as optimal in EXP-03):
    *   `['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']`
4.  **Model Training**:
    *   **Global Model**: Train LightGBM on all training data (2020-2023).
    *   **Tech Model**: Train LightGBM only on Tech stocks (2020-2023).
    *   **Non-Tech Model**: Train LightGBM only on Non-Tech stocks (2020-2023).
5.  **Ensemble Evaluation**:
    *   For the test set (2024-2025), route predictions based on the ticker's sector.
    *   `Ensemble Prediction` = `Tech Model(X)` if Tech, else `Non-Tech Model(X)`.
6.  **Comparison**: Compare `Ensemble` performance against `Global Model` baseline on the Test Set.

## 3. Metrics
*   **Win Rate (%)**: Percentage of trades with > 0% return.
*   **Average Return (%)**: Mean return per trade.
*   **Total Return**: Sum of returns.
*   **Trade Count**: Number of signals generated.
