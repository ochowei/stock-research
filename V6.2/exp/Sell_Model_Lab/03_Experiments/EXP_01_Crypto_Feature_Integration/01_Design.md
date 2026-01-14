# EXP-01: Crypto Feature Integration

## Hypothesis
Adding Crypto market state (BTC/ETH trend, volatility) will improve the Sell Model's ability to gauge global risk sentiment.
Specifically, we hypothesize that strong crypto trends or high volatility might correlate with "risk-on" or "risk-off" behaviors in the equity market, influencing the success of gap strategies.

## Plan

### 1. Data Fetching
*   Fetch `BTC-USD` and `ETH-USD` from Yahoo Finance.
*   Align dates with the stock data.

### 2. Feature Engineering
*   **BTC_RSI**: RSI(14) of BTC Close. Shifted by 1 (T-1).
*   **BTC_Trend**: BTC Close / BTC SMA(50) - 1. Shifted by 1 (T-1).
*   **Crypto_Corr**: Rolling correlation (30d) between Stock Close and BTC Close. Shifted by 1 (T-1).
    *   *Note*: Since we are predicting individual stock gaps, the correlation is between the *specific stock* and BTC.

### 3. Model Training
*   **Base**: V6.2.2 Clean Data (XGBoost).
*   **New Features**: Add `BTC_RSI`, `BTC_Trend`, `Crypto_Corr` to the existing feature set.
*   **Hyperparameters**: Keep constant (n_estimators=200, learning_rate=0.05, max_depth=4).

### 4. Metrics
*   **Win Rate**: Compare vs Baseline.
*   **Avg Return**: Compare vs Baseline.
*   **Feature Importance**: Check if crypto features appear in the top 10.

## Success Criteria
*   Improvement in Win Rate > 1% or Avg Return > 0.05%.
*   Crypto features show non-trivial importance (> 0.02).
