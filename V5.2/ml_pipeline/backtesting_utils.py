import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from risk_manager import RiskManager

def run_backtest(
    all_data,
    initial_capital=100000.0,
    target_risk=0.01,
    slippage_bps=5,
    transaction_cost_bps=5
):
    """
    Runs a backtest with corrected T-1 signal generation and hold-until-exit logic.
    """
    rm = RiskManager(target_risk=target_risk)

    # --- Signal Generation (Shifted to prevent lookahead bias) ---
    # Create signals based on T-1 data. A signal on day T is based on data from T-1.
    all_data = all_data.sort_index()
    all_data['entry_signal'] = (all_data['RSI_2'].shift(1) < 10) & (all_data['Close'].shift(1) > all_data['SMA_200'].shift(1))
    all_data['exit_signal'] = all_data['RSI_2'].shift(1) > 50

    cash = initial_capital
    positions = {}  # {symbol: {shares: float, entry_price: float}}
    equity = pd.Series(index=all_data.index.unique().sort_values())

    # --- Main Backtest Loop ---
    for date, daily_data in all_data.groupby(level=0):
        # 1. Value portfolio at start of day
        portfolio_value = cash
        for symbol, pos_data in positions.items():
            if symbol in daily_data['symbol'].values:
                price_data = daily_data[daily_data['symbol'] == symbol]['Open']
                current_price = price_data.iloc[0] if not price_data.empty else pos_data['entry_price']
                portfolio_value += pos_data['shares'] * current_price
        equity[date] = portfolio_value

        # 2. Process Exits
        symbols_to_exit = []
        for symbol, pos_data in positions.items():
            if symbol in daily_data['symbol'].values:
                exit_signal_row = daily_data[daily_data['symbol'] == symbol]
                if not exit_signal_row.empty and exit_signal_row['exit_signal'].iloc[0]:
                    symbols_to_exit.append(symbol)

        for symbol in symbols_to_exit:
            exit_price_data = daily_data[daily_data['symbol'] == symbol]['Open']
            if not exit_price_data.empty:
                exit_price = exit_price_data.iloc[0]
                shares = positions.pop(symbol)['shares']

                exit_price_adj = exit_price * (1 - slippage_bps / 10000)
                proceeds = shares * exit_price_adj
                costs = proceeds * (transaction_cost_bps / 10000)
                cash += proceeds - costs

        # 3. Process Entries
        entry_signals = daily_data[daily_data['entry_signal']]

        if not entry_signals.empty:
            current_regime = entry_signals['regime_signal'].iloc[0]
            if rm.apply_regime_filter(current_regime):
                for _, row in entry_signals.iterrows():
                    symbol = row['symbol']
                    if symbol in positions:
                        continue

                    entry_price = row['Open']
                    atr = row['ATR_14']

                    if atr > 0 and entry_price > 0:
                        shares = rm.calculate_position_size(portfolio_value, entry_price, atr)

                        entry_price_adj = entry_price * (1 + slippage_bps / 10000)
                        cost_of_trade = shares * entry_price_adj
                        transaction_fees = cost_of_trade * (transaction_cost_bps / 10000)
                        total_cost = cost_of_trade + transaction_fees

                        if cash >= total_cost:
                            cash -= total_cost
                            positions[symbol] = {'shares': shares, 'entry_price': entry_price_adj}

    return equity.dropna()

def analyze_performance(equity_curve, output_dir, filename_prefix, title, benchmark_curve=None, benchmark_label='Benchmark'):
    """Calculates and saves performance metrics and plots."""
    def calculate_metrics(curve):
        if curve is None or curve.empty or curve.iloc[0] == 0:
            return 0, 0, 0, -1
        returns = curve.pct_change().fillna(0)
        total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
        cagr = (curve.iloc[-1] / curve.iloc[0]) ** (252 / len(curve)) - 1 if len(curve) > 252 else total_return
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        rolling_max = curve.cummax()
        drawdown = (curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return total_return, cagr, sharpe, max_drawdown

    strat_tr, strat_cagr, strat_sharpe, strat_mdd = calculate_metrics(equity_curve)

    metrics = {
        'Metric': ['Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown'],
        'Strategy': [f"{strat_tr:.2%}", f"{strat_cagr:.2%}", f"{strat_sharpe:.2f}", f"{strat_mdd:.2%}"]
    }

    if benchmark_curve is not None:
        bench_tr, bench_cagr, bench_sharpe, bench_mdd = calculate_metrics(benchmark_curve)
        metrics['Benchmark'] = [f"{bench_tr:.2%}", f"{bench_cagr:.2%}", f"{bench_sharpe:.2f}", f"{bench_mdd:.2%}"]

    performance_df = pd.DataFrame(metrics).set_index('Metric')

    os.makedirs(output_dir, exist_ok=True)
    performance_df.to_csv(os.path.join(output_dir, f'{filename_prefix}_performance.csv'))

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    (equity_curve / equity_curve.iloc[0]).plot(ax=ax, label='Strategy', color='royalblue')
    if benchmark_curve is not None and not benchmark_curve.empty:
        (benchmark_curve / benchmark_curve.iloc[0]).plot(ax=ax, label=benchmark_label, color='grey', linestyle='--')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized Equity')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_equity.png'))
    plt.close()

    print(f"Analysis complete for {filename_prefix}. Results saved to {output_dir}")
    print(performance_df)
