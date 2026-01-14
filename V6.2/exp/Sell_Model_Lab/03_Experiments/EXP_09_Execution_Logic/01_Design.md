# Experiment Design: EXP-09 Sell Strategy Logic Refinement

## 1. Hypothesis
The current execution logic (Profit Take at 0.2%, Stop Loss at Market Close) was established for earlier, weaker models. With the V6.3 Production System (Tech/Non-Tech Heterogeneous Ensemble) achieving >53% Win Rate, we hypothesize that:
1.  **Profit Taking:** The models may correctly predict larger moves than 0.2%. Extending PT thresholds (e.g., to 0.5% or 1.0%) could capture more alpha.
2.  **Stop Loss:** Explicit intraday stops (e.g., 1.0% or 2.0%) might prevent catastrophic outlier losses (tail risk) that "Hold to Close" logic incurs, potentially improving Sharpe Ratio even if Win Rate drops slightly.

## 2. Goals
*   Identify the optimal Profit Take (PT) and Stop Loss (SL) combination for the V6.3 models.
*   Maximize **Total Return** and **Avg Return per Trade** while keeping **Win Rate > 50%**.

## 3. Methodology

### A. Data & Models
*   **Period:** Test Set (2024-01-01 to Present).
*   **Models:** V6.3 Production Models (Non-Tech: Base Features, Tech: Base+QQQ Features) from EXP-08.
*   **Asset Pool:** `2025_final_asset_pool.json`.

### B. Execution Logic Simulation (Backtest)
Since we lack minute-level data, we will use **Daily OHLC conservative approximation**:

For a **Short** position at `Open`:
*   **Stop Loss (SL) Price**: `Open * (1 + SL_Pct)`
*   **Profit Take (PT) Price**: `Open * (1 - PT_Pct)`

**Outcome Determination:**
1.  **SL Hit:** If `High >= SL_Price`:
    *   Result: Loss of `SL_Pct`.
    *   *Constraint:* We assume SL is hit first if both SL and PT levels are breached in the same candle (Conservative/Pessimistic assumption).
2.  **PT Hit:** If `Low <= PT_Price` AND `High < SL_Price`:
    *   Result: Gain of `PT_Pct`.
3.  **Hold to Close:** If neither breached:
    *   Result: `(Open - Close) / Open`.

### C. Grid Search Parameters
*   **Profit Take (PT):** `[0.002 (Baseline), 0.003, 0.004, 0.005, 0.0075, 0.01, None (Hold to Close)]`
*   **Stop Loss (SL):** `[0.005, 0.01, 0.015, 0.02, None (Hold to Close)]`

## 4. Metrics
*   **Win Rate %**
*   **Total Return %** (Sum of returns)
*   **Avg Return %**
*   **Trade Count** (To ensure we aren't filtering too much, though PT/SL doesn't filter entries, just outcomes).

## 5. Success Criteria
*   Find a configuration that improves **Total Return** by at least **10%** relative to the Baseline (PT 0.2%, No SL) without dropping Win Rate below 51%.
