# EXP-16 Review: Catastrophe Stop-Loss Optimization

## 1. Executive Summary
*   **Result:** ❌ Failed (Hypothesis Rejected).
*   **Decision:** Maintain **"Hold to Close" (No Stop)** protocol. Do not implement catastrophe stops.
*   **Performance:**
    *   **Baseline (No Stop):** Sharpe 5.22, Total Return 84.5%.
    *   **3x ATR Stop:** Sharpe 5.14, Total Return 83.8%.
    *   **Fixed 10% Stop:** Sharpe 4.93, Total Return 79.3%.
    *   **Fixed 5% Stop:** Sharpe 4.62, Total Return 69.2%.

## 2. Analysis
### A. The "Wick" Cost
The hypothesis assumed that a wide stop would only trigger on "true disasters". However, the data shows that **Catastrophe Stops actually increased Max Drawdown** (from -0.66 to -0.70 for 3x ATR).
This implies that in many "loss" trades, the price spikes up significantly (triggering the stop at the high) but then reverts effectively by the close.
*   **Example Mechanism:**
    *   Short at \$100.
    *   Intraday spike to \$106 (Stop triggered at \$105). Loss locked at -5%.
    *   Price fades to close at \$102. "Hold" Loss would be -2%.
    *   The Stop Loss forced an exit near the high of the day.

### B. Mean Reversion Confirmation
The Sell Model exploits mean reversion. By definition, these stocks are volatile. Extreme intraday volatility is often noise that resolves itself by the close. Attempting to cut losses intraday contradicts the core thesis of the strategy (fading the gap).

### C. Frequency of Catastrophes
With a "Stop Trigger Rate" of only 0.3% for the 3x ATR stop, one would expect it to save the portfolio from rare "Black Swan" events. The fact that it *still* underperformed suggests that:
1.  The "Black Swans" (Close >> High >> Entry) are extremely rare.
2.  The "Fake Swans" (High >> Entry but Close ~ Entry) are more common, and the cost of paying the "insurance premium" on these fake swans outweighs the savings on the true black swans.

## 3. Conclusion & Recommendations
1.  **Rejection of Stops:** Intraday stop losses, even wide ones, are mathematically detrimental to this specific mean-reversion strategy.
2.  **Risk Management:** Risk must be managed via **Position Sizing** (allocating small % per trade) rather than trade-level stops. Diversification across the basket is the only effective safety net.
3.  **Future Work:** Investigate if "Time-Based Stops" (e.g., if trade is losing > 3% at 2 PM, exit?) work better, though EXP-12 suggests holding is generally best.
