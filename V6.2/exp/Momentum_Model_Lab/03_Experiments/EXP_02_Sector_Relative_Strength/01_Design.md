# EXP-02: Sector Relative Strength (Orthogonal Alpha)

## 1. Hypothesis
Momentum signals are more reliable when the underlying Sector (e.g., XLK, XLF) is also in a strong trend. Breakouts that align with sector momentum ("Rising Tide") should outperform "Lone Wolf" breakouts.

We hypothesize that injecting Sector Context will:
1.  Filter out false positives where the stock is rising but the sector is weak (Headfake).
2.  Prioritize setups where both Stock and Sector are strong.

## 2. Plan
1.  **Data Preparation:**
    *   Load `sector_map.json` to identify each stock's sector.
    *   Map sectors to their corresponding SPDR ETFs (e.g., Technology -> XLK).
    *   Download daily OHLCV for all tickers and Sector ETFs.

2.  **Feature Engineering:**
    *   Calculate `RSI_14` for both Stock and Sector ETF.
    *   Create `Sector_RSI`: The RSI of the sector ETF.
    *   Create `Rel_Strength_RSI`: `Stock_RSI - Sector_RSI`.
    *   (Optional) `Sector_Ret_1M` vs `Stock_Ret_1M`.

3.  **Model Training:**
    *   Model: XGBoost (Same parameters as EXP-01).
    *   Features: Baseline (`Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `VIX`) + New (`Sector_RSI`, `Rel_Strength_RSI`).
    *   Train Period: 2020-2023.
    *   Test Period: 2024-2025.

4.  **Evaluation:**
    *   Compare Win Rate and Avg Return against EXP-01 Baseline.
    *   Analyze Feature Importance to see if the model picks up the sector signal.
    *   Check for "Lone Wolf" penalty (does the model down-weight stocks with weak sectors?).

## 3. Metrics
*   **Primary:** Win Rate > 55% (Baseline was 57.6%).
*   **Secondary:** Avg Return > 0.25% (Baseline was 0.98% - High bar!).
*   **Validation:** Feature Importance of Sector features > 5%.

## 4. Sector Mapping (SPDR)
*   Technology -> XLK
*   Consumer Cyclical -> XLY
*   Communication Services -> XLC
*   Financial Services -> XLF
*   Healthcare -> XLV
*   Industrials -> XLI
*   Consumer Defensive -> XLP
*   Energy -> XLE
*   Utilities -> XLU
*   Real Estate -> XLRE
*   Basic Materials -> XLB
*   Unknown/Other -> SPY
