# Experiment Review: EXP-09 Execution Logic Refinement

## 1. Executive Summary
*   **Result:** ✅ **Major Success** (Discovered Optimal Execution).
*   **Outcome:** **Discard Profit Taking.** Adopt "Hold to Close" as the default execution strategy.
*   **Impact:**
    *   **Baseline (PT 0.2%):** -2.33% Total Return (Failed).
    *   **New Logic (Hold to Close):** **+33.70% Total Return**.
    *   **Avg Return:** Improved from -0.02% to **+0.37%**.

## 2. Detailed Findings

### A. The "Picking Pennies" Trap
The baseline strategy (Target 0.2% Profit, No Stop) achieved a very high Win Rate (93.7%), but generated a net loss. This confirms the hypothesis that capping profits at 0.2% leaves the strategy vulnerable to "tail risk" losses at Close that wipe out dozens of small wins.
*   **Baseline Stats:** Win Rate 93.7%, Avg Return -0.02%, Total Return -2.33%.

### B. "Hold to Close" Superiority
The V6.3 models (Tech/Non-Tech) have a predictive edge (Win Rate ~54%) over the full trading day. Allowing the trade to run until Market Close maximizes the extraction of this edge.
*   **Hold to Close Stats:** Win Rate 53.96%, Avg Return +0.37%, Total Return +33.70%.

### C. Stop Losses Reduce Performance
Introducing intraday stops (0.5% to 2.0%) consistently degraded performance compared to holding to close.
*   **SL 2.0% (No PT):** Total Return 24.2% (vs 33.7%).
*   **SL 0.5% (No PT):** Total Return 19.2%.
*   **Insight:** Intraday volatility frequently triggers stops before the mean reversion thesis materializes. The models are trained to predict `Open > Close`; they are not trained to predict "Low Volatility Path to Close".

### D. Profit Targets Leave Money on the Table
*   **PT 1.0% (No SL):** Total Return 9.14%.
*   **PT 0.2% (Baseline):** Total Return -2.33%.
*   **Insight:** The models identify setups with significant reversion potential. Cutting winners short severely handicaps the strategy's expectancy.

## 3. Recommendations
1.  **Update Production Logic:** Remove the `0.002` (0.2%) profit target and the standard Stop Loss logic.
2.  **New Execution Rules:**
    *   **Entry:** Market Open.
    *   **Exit:** Market Close (MOC).
    *   **Emergency Stop:** Retain a wide catastrophic stop (e.g., 5%+) just for risk management, though it wasn't hit in this simulation.
3.  **Future Work:** Investigate "Time-Based" exits (e.g., exit after 2 hours) or "Technical Exits" (e.g., cross moving average), but for now, "Hold to Close" is the clear winner.
