import numpy as np

class RiskManager:
    """
    Manages position sizing and exposure based on volatility and market regime.
    """
    def __init__(self, target_risk=0.01):
        """
        Initializes the RiskManager.
        Args:
            target_risk (float): The target risk percentage per trade (e.g., 0.01 for 1%).
        """
        self.target_risk = target_risk

    def calculate_position_size(self, total_capital, stock_price, atr):
        """
        Calculates the position size based on volatility-scaled sizing.
        Formula: Position Size = (Total Capital * Target Risk %) / Stock ATR
        Args:
            total_capital (float): The total account value.
            stock_price (float): The current price of the stock.
            atr (float): The Average True Range (ATR) of the stock.
        Returns:
            float: The number of shares to purchase. Returns 0 if ATR is zero.
        """
        if atr == 0:
            return 0

        dollar_amount = total_capital * self.target_risk
        position_size_shares = dollar_amount / atr
        return position_size_shares

    def apply_regime_filter(self, regime_signal):
        """
        Determines if new entries are allowed based on the market regime signal.
        If Regime Signal == 2 (Crash/Panic), no new entries are allowed.
        Args:
            regime_signal (int): The current regime signal (e.g., from regime_signals.parquet).
        Returns:
            bool: True if trading is allowed, False otherwise.
        """
        if regime_signal == 2:
            return False  # Do not allow new entries
        return True  # Allow new entries
