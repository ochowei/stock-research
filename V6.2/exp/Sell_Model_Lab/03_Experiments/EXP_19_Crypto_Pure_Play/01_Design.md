# EXP-19: Crypto Sector Specific Model (Pure Play Data)

## 1. Hypothesis
**Hypothesis:** Applying the Crypto-Enhanced Model (Base + BTC Features) to a strictly sanitized "Pure Play" Crypto pool (COIN, MSTR, RIOT, MARA) will significantly improve performance compared to the Base Model.

**Context:** EXP-15 failed because the "Crypto Sensitive" pool contained non-crypto stocks, causing BTC features to introduce noise. By restricting the universe to stocks directly correlated with Bitcoin price action, the macro features should provide valid signal.

## 2. Plan

### 2.1 Scope
*   **Universe:** Pure Play Crypto Stocks: `['COIN', 'MSTR', 'RIOT', 'MARA']`.
*   **Timeframe:** Daily Data (2020-01-01 to Present).
*   **Engine:** LightGBM.

### 2.2 Features
*   **Base Features (Control):**
    *   `Gap_Pct`: (Open - Prev Close) / Prev Close
    *   `RSI_14`: Relative Strength Index
    *   `ATR_Pct`: ATR / Close
    *   `Vol_Ratio`: Volume / MA_Volume
    *   `Dist_MA20`: (Close - MA20) / MA20
*   **Crypto Features (Test):**
    *   `BTC_Gap`: (BTC Open - BTC Prev Close) / BTC Prev Close
    *   `BTC_RSI_14`: BTC RSI
    *   `BTC_Change`: BTC Daily Change (Close - Open) / Open (Prev Day) -> *Wait, to avoid lookahead, this should be Prev Day Return.*
        *   Actually, let's stick to standard practice: Features available at Open.
        *   `BTC_Gap`: Available at Open.
        *   `BTC_RSI_14`: Based on Prev Close. Available.
        *   `BTC_Change`: Prev Day Return. Available.

### 2.3 Methodology
1.  **Data Acquisition:** Fetch daily data for Pure Play stocks and BTC-USD.
2.  **Feature Engineering:** Construct Base and Crypto features. **Crucial:** Ensure Crypto features are properly aligned (T-1 for Close-based, T for Open-based if available, but usually safe to use T-1 Close for everything to avoid any risk).
    *   *Correction:* Stock Gap is calculated at Open (Open - Prev Close). BTC Gap can also be calculated at Open if we assume we have live BTC data. However, for simplicity and robustness, we usually use T-1 BTC data or Open-PrevClose.
    *   Let's align with EXP-01/15: Use T-1 BTC data for RSI/Change. BTC Gap can be (BTC_Open_T - BTC_Close_T-1) / BTC_Close_T-1.
3.  **Model Training:**
    *   **Control Model:** Train on Base Features only.
    *   **Test Model:** Train on Base + Crypto Features.
    *   **Split:** Time Series Split (Train: 2022-2023, Test: 2024).
4.  **Evaluation:** Compare Win Rate and Avg Return on the Test Set (2024).

## 3. Success Metrics
*   **Primary:** Test Model Win Rate > Control Model Win Rate.
*   **Secondary:** Test Model Avg Return > Control Model Avg Return.
*   **Lab Standard:** Win Rate > 55%, Avg Return > 0.20%.
