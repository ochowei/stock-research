# EXP-07: Tech-Specific Feature Engineering

## 1. Hypothesis
The Tech sector model (from EXP-05/06) underperforms the Non-Tech model (50.2% WR vs 53.3% WR).
Hypothesis: Tech stocks are highly correlated with the broader Nasdaq-100 index. Adding Tech-specific market context features (e.g., QQQ volatility, momentum, and gap) will improve the model's ability to distinguish between noise and valid signals in the Tech sector.

## 2. Plan
1.  **Data Source**:
    *   Target: Tech stocks from `2025_final_asset_pool.json`.
    *   External: `QQQ` (Nasdaq-100 ETF) historical data.
2.  **Feature Engineering**:
    *   **Base Features**: `Gap_Pct`, `RSI_14`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20`.
    *   **New Tech Features**:
        *   `QQQ_Gap_Pct`: Market sentiment at Open.
        *   `QQQ_RSI_14` (T-1): Overbought/Oversold status of the sector.
        *   `QQQ_Dist_MA20` (T-1): Sector trend deviation.
        *   `Tech_Corr_20` (T-1): Correlation of stock vs QQQ over 20 days.
3.  **Model Architecture**:
    *   LightGBM Classifier.
    *   Training separate models for Tech sector with/without new features.
4.  **Evaluation**:
    *   Train Period: 2020-2023.
    *   Test Period: 2024-2025 (Out-of-Sample).
    *   Metrics: Win Rate, Average Return, Signal Count.
    *   Baseline: EXP-06 Optimized Tech Model (Base Features only).

## 3. Success Metrics
*   **Win Rate**: > 51.0% for Tech Sector (Baseline ~50.2%).
*   **Avg Return**: Improvement over baseline.
