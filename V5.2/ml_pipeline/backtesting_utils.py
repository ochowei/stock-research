import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from risk_manager import RiskManager

def run_backtest(
    all_data,
    initial_capital=100000.0,
    target_risk=0.01,
    max_position_pct=0.2, # 新增：單筆最大倉位 20%
    slippage_bps=5,
    transaction_cost_bps=5,
    hold_days=5 # 新增：固定持有天數
):
    """
    Runs a backtest with Liquidation, Time-Stop, and Sorted Entries.
    """
    # 初始化風控模組 (含倉位上限)
    rm = RiskManager(target_risk=target_risk, max_position_pct=max_position_pct)

    # --- Signal Generation ---
    all_data = all_data.sort_index()
    # Entry: RSI < 10 & Price > SMA200 (T-1 generated, executed at T Open)
    all_data['entry_signal'] = (all_data['RSI_2'].shift(1) < 10) & (all_data['close'].shift(1) > all_data['SMA_200'].shift(1))

    # Exit 1: Technical Exit (RSI > 50)
    all_data['tech_exit_signal'] = all_data['RSI_2'].shift(1) > 50

    cash = initial_capital
    # Positions structure: {symbol: {'shares': float, 'entry_price': float, 'days_held': int}}
    positions = {}
    equity = pd.Series(index=all_data.index.unique().sort_values())

    # --- Main Backtest Loop ---
    for date, daily_data in all_data.groupby(level=0):

        # 0. Get Market Regime for Today
        # 假設所有標的當天的 regime_signal 是一樣的 (來自 Macro)
        current_regime = 0
        if 'regime_signal' in daily_data.columns:
            current_regime = daily_data['regime_signal'].iloc[0]

        # 1. Value portfolio at start of day (Mark-to-Market)
        portfolio_value = cash
        for symbol, pos_data in positions.items():
            if symbol in daily_data['symbol'].values:
                price_data = daily_data[daily_data['symbol'] == symbol]['open']
                current_price = price_data.iloc[0] if not price_data.empty else pos_data['entry_price']
                portfolio_value += pos_data['shares'] * current_price
        equity[date] = portfolio_value

        # 2. Process Exits (含緊急清倉與時間止損)
        symbols_to_exit = []

        # [Defense Upgrade] 緊急清倉機制 (Liquidation)
        if current_regime == 2:
            # 如果是崩盤模式，清空所有持倉
            symbols_to_exit = list(positions.keys())
        else:
            # 正常模式：檢查技術指標與時間止損
            for symbol, pos_data in positions.items():
                # 更新持倉天數
                pos_data['days_held'] += 1

                should_sell = False

                # A. 時間止損 (Time Stop)
                if pos_data['days_held'] >= hold_days:
                    should_sell = True

                # B. 技術出場 (RSI > 50)
                if not should_sell and symbol in daily_data['symbol'].values:
                    exit_signal_row = daily_data[daily_data['symbol'] == symbol]
                    if not exit_signal_row.empty and exit_signal_row['tech_exit_signal'].iloc[0]:
                        should_sell = True

                if should_sell:
                    symbols_to_exit.append(symbol)

        # 執行賣出
        for symbol in symbols_to_exit:
            if symbol not in positions: continue # 避免重複刪除

            # 取得賣出價格 (Open)
            exit_price_data = daily_data[daily_data['symbol'] == symbol]['open']

            # 若數據缺失 (停牌)，則無法賣出，保留至隔日
            if not exit_price_data.empty:
                exit_price = exit_price_data.iloc[0]
                shares = positions.pop(symbol)['shares']

                # 結算資金
                exit_price_adj = exit_price * (1 - slippage_bps / 10000)
                proceeds = shares * exit_price_adj
                costs = proceeds * (transaction_cost_bps / 10000)
                cash += proceeds - costs

        # 3. Process Entries (僅在非崩盤時買入)
        if rm.apply_regime_filter(current_regime):
            entry_signals = daily_data[daily_data['entry_signal']]

            if not entry_signals.empty:
                # [Fix] 訊號排序：優先買入 RSI 最低 (超賣最嚴重) 的股票
                entry_signals = entry_signals.sort_values(by='RSI_2', ascending=True)

                for _, row in entry_signals.iterrows():
                    symbol = row['symbol']
                    if symbol in positions:
                        continue

                    entry_price = row['open']
                    atr = row['ATR_14']

                    if atr > 0 and entry_price > 0:
                        # 使用新的倉位計算 (含上限)
                        shares = rm.calculate_position_size(portfolio_value, entry_price, atr)

                        if shares > 0:
                            entry_price_adj = entry_price * (1 + slippage_bps / 10000)
                            cost_of_trade = shares * entry_price_adj
                            transaction_fees = cost_of_trade * (transaction_cost_bps / 10000)
                            total_cost = cost_of_trade + transaction_fees

                            if cash >= total_cost:
                                cash -= total_cost
                                positions[symbol] = {
                                    'shares': shares,
                                    'entry_price': entry_price_adj,
                                    'days_held': 0 # 初始化持倉天數
                                }

    return equity.dropna()

def analyze_performance(equity_curve, output_dir, filename_prefix, title, benchmark_curve=None, benchmark_label='Benchmark'):
    """Calculates and saves performance metrics and plots."""
    def calculate_metrics(curve):
        if curve is None or curve.empty or curve.iloc[0] == 0:
            return 0, 0, 0, -1
        returns = curve.pct_change().fillna(0)
        total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
        # CAGR handling for periods < 1 year
        days = (curve.index[-1] - curve.index[0]).days
        years = days / 365.25
        if years > 0:
            cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1
        else:
            cagr = total_return

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
