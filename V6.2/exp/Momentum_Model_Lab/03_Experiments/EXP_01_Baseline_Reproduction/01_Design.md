# EXP-01: Baseline Reproduction (V6.1 Parity)

## 1. Hypothesis (假設)
**Hypothesis:** Re-training the momentum model within the new `Lab` structure using exact V6.1 parameters and features will replicate historical performance.
**Rationale:** Before introducing new features (Sector, Volume, Crypto), we must establish a "Control" model (The Baseline) to measure future improvements against. Any deviation in the baseline performance suggests environment or data issues.

## 2. Plan (步驟)

1.  **Environment Setup**:
    *   Initialize the experiment folder `EXP_01_Baseline_Reproduction`.
    *   Ensure dependencies (yfinance, pandas-ta, xgboost) are consistent.

2.  **Data Ingestion**:
    *   Load tickers from `V6.2/resource/2025_final_asset_pool.json`.
    *   Fetch OHLCV data for 2020-2025 using `yfinance`.
    *   Split into Train (2020-2023) and Test (2024-2025).

3.  **Feature Engineering (V6.1 Legacy)**:
    *   Replicate exact feature set: `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Gap_Pct`, `VIX`.
    *   Target: `Strategy_Ret` = `(Close - Open) / Open`.
    *   Label: `Strategy_Ret > 0.2%` (PROFIT_THRESHOLD).

4.  **Model Training**:
    *   Algorithm: XGBoost Classifier.
    *   Hyperparameters: `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`.
    *   Weighting: Sample weights based on absolute return magnitude.

5.  **Evaluation**:
    *   Metrics: Win Rate, Average Return.
    *   Comparison: Model vs Baseline (Buy All Gaps).

## 3. Metrics (指標)

**Success Criteria:**
*   **Win Rate:** > 55%
*   **Average Return:** > 0.25% (per trade)
*   **Consistency:** Results should closely match the historical V6.1 performance (if known) or at least be profitable.
