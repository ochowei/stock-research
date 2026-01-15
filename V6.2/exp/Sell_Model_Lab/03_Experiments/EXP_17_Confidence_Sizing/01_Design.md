# EXP-17: Confidence-Based Position Sizing

## 1. Context & Hypothesis
*   **Context:** Previous experiments (EXP-16) confirmed that "Hold to Close" is the superior exit strategy. However, drawdowns are still a concern. Instead of trying to cut losses intraday (which fails due to morning volatility), we aim to manage risk *before* entry by sizing positions based on model confidence.
*   **Hypothesis:** The LightGBM model's output probability (`predict_proba`) is calibrated enough that higher probability signals have a higher expected value. Allocating more capital to these high-confidence signals will improve the risk-adjusted return (Sharpe Ratio) and potentially reduce Max Drawdown.

## 2. Experimental Design
*   **Models:**
    *   Use the deployed V6.4 models (Heterogeneous Ensemble).
    *   Tech: `v6.4_tech_model.joblib` (Base + QQQ features).
    *   Non-Tech: `v6.4_non_tech_model.joblib` (Base + SPY features).
*   **Data:**
    *   Period: Recent 2 years (e.g., 2023-01-01 to Present) or maximum available overlap with V6.4 training window validation.
    *   Assets: `2025_final_asset_pool.json`.
*   **Strategies:**
    1.  **Baseline (Equal Weight):** Fixed $10,000 per trade.
    2.  **Variant A (Tiered Sizing):**
        *   `Prob >= 0.60`: $15,000 (1.5x)
        *   `0.55 <= Prob < 0.60`: $10,000 (1.0x)
        *   `Prob < 0.55`: $5,000 (0.5x) - *Note: The model threshold is usually 0.5, so this covers the lower end of "buy" signals.*
    3.  **Variant B (Linear Sizing):**
        *   `Size = $10,000 * (1 + (Prob - 0.55) * 10)` (Example formula, will tune in implementation).
        *   Or simpler: `Size = $10,000 * (Prob / 0.55)`.
        *   Let's stick to the queue definition: `Size = (Prob - 0.5) * Scale`. Let's assume Scale to normalize around $10k.
        *   Actually, let's use a simpler mapping for Variant B to avoid over-engineering:
            *   Factor = `0.5 + (Prob - 0.5) * 5` -> If Prob=0.5, Factor=0.5. If Prob=0.6, Factor=1.0. If Prob=0.7, Factor=1.5.
            *   `Size = $10,000 * Factor`.

## 3. Metrics
*   **Primary:** Sharpe Ratio.
*   **Secondary:** Max Drawdown (%), Total Return ($), Win Rate (%).
*   **Success Criteria:** Variant A or B achieves a higher Sharpe Ratio than Baseline AND lower Max Drawdown.

## 4. Implementation Plan
1.  **Data Ingestion:** Download daily OHLCV for the asset pool and Sector/Index data (QQQ, SPY).
2.  **Feature Generation:** Re-create V6.4 features (Base, +QQQ for Tech, +SPY for Non-Tech).
3.  **Inference:** Run models to get `predict_proba`.
4.  **Simulation:** Calculate daily PnL for each strategy.
    *   Return = `(Open - Close) / Open` (Short selling).
    *   PnL = Return * Position_Size.
5.  **Evaluation:** Compute metrics.
