# EXP-18: Production Script Update (Position Sizing)

## 1. Hypothesis
Incorporating the Tiered Position Sizing logic verified in EXP-17 into the daily production script will align the operational output with the high-Sharpe strategy discovered in the lab.
Specifically:
*   Signals with `Probability > 0.60` should receive **1.5x** size.
*   Signals with `0.55 < Probability <= 0.60` should receive **1.0x** size.
*   Signals with `0.50 < Probability <= 0.55` should receive **0.5x** size.

## 2. Plan
1.  **Setup**: Create experiment folder and `03_Output` directory.
2.  **Asset Migration**: Copy the validated models (`v6.2.4_rc_tech_model.joblib`, `v6.2.4_rc_non_tech_model.joblib`) and `sector_map.json` from `EXP_13` to the local `03_Output` folder to simulate a self-contained production environment.
3.  **Implementation**:
    *   Create `production_daily_plan_v6_5_rc.py` by extending `production_daily_plan_v6_2_4_rc.py`.
    *   Add a function `get_position_size(prob)` that implements the tiered logic.
    *   Apply this function to the DataFrame after signal generation.
    *   Add a new column `Position_Size` to the output CSV.
4.  **Verification**:
    *   Run the script to generate a daily plan (using the latest available data).
    *   Inspect the output CSV to ensure the `Position_Size` column exists and correctly maps to the `Probability` column.

## 3. Metrics
*   **Success**: The script runs without error and produces a CSV with a `Position_Size` column where values strictly follow the defined logic (1.5, 1.0, 0.5).
*   **Failure**: Script crashes or sizing logic is incorrect.
