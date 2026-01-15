# EXP-23: RSI Period Sensitivity (Short-Term Mean Reversion)

## 1. Objective
To determine if shorter RSI periods (e.g., 2, 4, 10) outperform the standard `RSI_14` in the V6.2.4.RC Sell Model.

## 2. Hypothesis
The current strategy targets overnight "Gap" overreactions, which are immediate, short-term mean reversion events. The standard `RSI_14` is a medium-term momentum indicator that may be too slow to capture the acute "overbought" conditions (spike high) that precede a successful gap fill. A shorter RSI should be more responsive and potentially improve precision.

## 3. Methodology
We will iterate through a list of RSI periods and evaluate performance on both Tech and Non-Tech sectors, maintaining the V6.2.4.RC architecture (Heterogeneous Ensemble).

### 3.1. Variable
*   **RSI_Period**: `[2, 3, 4, 5, 7, 10, 14]` (14 is Baseline)

### 3.2. Architecture (V6.2.4.RC)
*   **Tech Sector:**
    *   Features: Base (`Gap_Pct`, `RSI_X`, `ATR_Pct`, `Vol_Ratio`, `Dist_MA20`) + QQQ (`Gap`, `RSI`, `Dist_MA`).
    *   *Note: We will only change the Stock's RSI, not the Index RSI, to isolate the variable.*
    *   Model: LightGBM (Depth 3, LR 0.01).
*   **Non-Tech Sector:**
    *   Features: Base + SPY (`Gap`, `RSI`, `Dist_MA`).
    *   Model: LightGBM (Unlimited Depth, LR 0.02).

### 3.3. Data
*   **Train:** 2020-01-01 to 2023-12-31
*   **Test:** 2024-01-01 to Present (2025)

## 4. Success Metrics
*   **Win Rate (> 55%)**: Primary metric.
*   **Avg Return (> 0.20%)**: Secondary metric.
*   **Signal Count**: Must remain viable (not drop to zero).

## 5. Artifacts
*   `rsi_sensitivity_results.csv`: Aggregate metrics for each RSI period.
*   `rsi_sensitivity_plot.png`: Visualization of Win Rate vs RSI Period.
