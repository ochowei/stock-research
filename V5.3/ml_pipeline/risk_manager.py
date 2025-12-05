import numpy as np

class RiskManager:
    """
    Manages position sizing and exposure based on volatility and market regime.
    """
    def __init__(self, target_risk=0.01, max_position_pct=0.2):
        """
        Initializes the RiskManager.
        Args:
            target_risk (float): The target risk percentage per trade (e.g., 0.01 for 1%).
            max_position_pct (float): Maximum percentage of total capital allocated to a single position (e.g., 0.2 for 20%).
        """
        self.target_risk = target_risk
        self.max_position_pct = max_position_pct

    def calculate_position_size(self, total_capital, stock_price, atr):
        """
        Calculates the position size based on volatility-scaled sizing with a hard cap.
        Formula:
            1. Volatility Size = (Total Capital * Target Risk %) / Stock ATR
            2. Max Size = (Total Capital * Max Position %) / Stock Price
            3. Final Size = Min(Volatility Size, Max Size)
        """
        if atr == 0 or stock_price == 0:
            return 0

        # 1. Volatility-based sizing
        dollar_risk = total_capital * self.target_risk
        vol_shares = dollar_risk / atr

        # 2. Hard Cap sizing (Position Limit)
        max_dollar_alloc = total_capital * self.max_position_pct
        cap_shares = max_dollar_alloc / stock_price

        # 3. Take the smaller of the two (Conservative approach)
        final_shares = min(vol_shares, cap_shares)

        return int(final_shares) # Return integer shares

    def apply_regime_filter(self, regime_signal):
        """
        Determines if new entries are allowed.
        Regime 2 (Crash) -> Block Entry.
        """
        if regime_signal == 2:
            return False
        return True
