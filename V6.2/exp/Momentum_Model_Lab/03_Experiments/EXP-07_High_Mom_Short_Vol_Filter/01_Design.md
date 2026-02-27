# EXP-07: High Momentum Short Strategy - Volume Filter

## 1. Hypothesis
**Core Idea:** High volume gaps on already extended stocks (RSI > 70) represent "Exhaustion Gaps" and reverse more reliably than low volume gaps.
**Rationale:** When a stock is already Overbought (RSI > 70) and gaps up on massive volume (> 2x Average), it often indicates a climax or "blow-off top" where the last buyers are rushing in, leaving no one left to buy during the session.
**Prediction:** Applying a `Vol_Ratio > 2.0` filter to the High Momentum Short Strategy (from EXP-06) will filter out "strong continuation" moves and isolate "exhaustion" moves, improving the Short Win Rate.

## 2. Plan
1.  **Data Ingestion:** Load the standard asset pool (2020-2025).
2.  **Feature Engineering:**
    *   `RSI_14` (T-1): Relative Strength Index.
    *   `Vol_Ratio` (T-1): Volume(T-1) / MA20_Volume(T-2). *Note: We look at the volume LEADING UP TO the gap (T-1) or the Gap Day Volume?*
    *   *Correction based on Hypothesis:* The hypothesis says "High volume gaps". This usually means the volume *on the gap day* (Projected or Opening) or the volume *preceding* it. However, since we are trading at the Open (Gap Strategy), we only know T-1 Volume and T Open.
    *   *Refinement:* Let's stick to **T-1 Volume Ratio** (Volume on the day *before* the gap) to imply "Climax before the Gap".
    *   *Alternative Interpretation:* "High volume gaps" might imply `Gap Volume`. But we don't know the full day volume at the Open. We can use `Pre-Market Volume` (not available) or `Opening Volume`.
    *   *Decision:* To avoid look-ahead bias and stick to T-1 data availability, we will test **T-1 Volume Ratio > 2.0**. (Did the stock rally on huge volume *yesterday* and then gap up today? -> Exhaustion).
    *   *Wait*, looking at EXP-03, `Vol_Ratio` was defined as `Prev_Vol / Vol_MA20.shift(1)`. This is T-1 volume relative to T-2 baseline. This matches "Volume leading up to the gap".
3.  **Strategy Logic:**
    *   **Universe:** All Gap Ups (> 0.5%).
    *   **Regime:** High Momentum (`RSI_14` > 70).
    *   **Filter (Experiment):** `Vol_Ratio` > 2.0 (Extreme Volume on T-1).
    *   **Action:** Short at Open, Cover at Close (Bet on Reversal).
4.  **Control Group:**
    *   `RSI_14` > 70 AND `Vol_Ratio` <= 2.0 (Low/Normal Volume).
5.  **Baseline:**
    *   EXP-06 Result (`RSI_14` > 70, No Volume Filter).

## 3. Success Metrics
*   **Win Rate:** > 55% (Baseline was 53.7%).
*   **Avg Return:** > 0.30% (Baseline was +0.20%).
*   **Signal Count:** > 500 trades (Must be statistically significant).
