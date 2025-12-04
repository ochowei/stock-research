# V5.2/ml_pipeline/backtesting_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from risk_manager import RiskManager

def run_backtest(
    all_data,
    initial_capital=100000.0,
    target_risk=0.01,
    max_position_pct=0.2,
    slippage_bps=5,
    transaction_cost_bps=5,
    hold_days=5,
    use_regime_filter=True, # [New] 開關 L1 防禦
    force_equal_weight=False # [New] 強制等權重 (V5.1 模式)
):
    """
    Runs a backtest with Liquidation, Time-Stop, and Sorted Entries.
    Includes fixes for Data Contamination and Look-Ahead Bias.
    """
    # 初始化風控模組 (含倉位上限)
    rm = RiskManager(target_risk=target_risk, max_position_pct=max_position_pct)

    # --- Data Preparation & Signal Generation (Fixes Applied) ---
    all_data = all_data.sort_index()
    grouped = all_data.groupby('symbol')
    
    # 取得 T-1 的關鍵指標
    all_data['prev_RSI_2'] = grouped['RSI_2'].shift(1)
    all_data['prev_close'] = grouped['close'].shift(1)
    all_data['prev_SMA_200'] = grouped['SMA_200'].shift(1)
    all_data['prev_ATR_14'] = grouped['ATR_14'].shift(1)

    # 生成進場訊號 (使用 T-1 資訊)
    all_data['entry_signal'] = (all_data['prev_RSI_2'] < 10) & \
                               (all_data['prev_close'] > all_data['prev_SMA_200'])

    # 生成技術出場訊號 (使用 T-1 資訊)
    all_data['tech_exit_signal'] = all_data['prev_RSI_2'] > 50

    # --- Backtest Loop Initialization ---
    cash = initial_capital
    positions = {}
    equity = pd.Series(index=all_data.index.unique().sort_values())

    # --- Main Backtest Loop ---
    for date, daily_data in all_data.groupby(level=0):

        # 0. Get Market Regime
        current_regime = 0
        if 'regime_signal' in daily_data.columns:
            current_regime = daily_data['regime_signal'].iloc[0]

        # 1. Value portfolio
        portfolio_value = cash
        for symbol, pos_data in positions.items():
            if symbol in daily_data['symbol'].values:
                price_data = daily_data[daily_data['symbol'] == symbol]['open']
                current_price = price_data.iloc[0] if not price_data.empty else pos_data['entry_price']
                portfolio_value += pos_data['shares'] * current_price
            else:
                portfolio_value += pos_data['shares'] * pos_data['entry_price']
        
        equity[date] = portfolio_value

        # 2. Process Exits
        symbols_to_exit = []
        for pos_data in positions.values():
            pos_data['days_held'] += 1

        # [Defense Upgrade] 緊急清倉機制
        # 如果 use_regime_filter 為 False，則忽略 Crash 訊號
        is_crash = (current_regime == 2) and use_regime_filter
        
        if is_crash:
            symbols_to_exit = list(positions.keys())
        else:
            for symbol, pos_data in positions.items():
                should_sell = False
                if pos_data['days_held'] >= hold_days:
                    should_sell = True
                if not should_sell and symbol in daily_data['symbol'].values:
                    row = daily_data[daily_data['symbol'] == symbol]
                    if not row.empty and row['tech_exit_signal'].iloc[0]:
                        should_sell = True
                if should_sell:
                    symbols_to_exit.append(symbol)

        # Execute Exits
        for symbol in symbols_to_exit:
            if symbol not in positions: continue
            exit_price_data = daily_data[daily_data['symbol'] == symbol]['open']
            if not exit_price_data.empty:
                exit_price = exit_price_data.iloc[0]
                shares = positions.pop(symbol)['shares']
                exit_price_adj = exit_price * (1 - slippage_bps / 10000)
                proceeds = shares * exit_price_adj
                costs = proceeds * (transaction_cost_bps / 10000)
                cash += proceeds - costs

        # 3. Process Entries
        # 如果 use_regime_filter 為 False，則忽略過濾條件，始終允許進場
        can_enter = True
        if use_regime_filter:
            can_enter = rm.apply_regime_filter(current_regime)

        if can_enter:
            entry_signals = daily_data[daily_data['entry_signal']]
            if not entry_signals.empty:
                entry_signals = entry_signals.sort_values(by='prev_RSI_2', ascending=True)

                for _, row in entry_signals.iterrows():
                    symbol = row['symbol']
                    if symbol in positions: continue

                    entry_price = row['open']
                    atr = row['prev_ATR_14']

                    if pd.notna(atr) and atr > 0 and entry_price > 0:
                        
                        # [Logic Switch] V5.1 (Equal Weight) vs V5.2 (Risk Parity)
                        if force_equal_weight:
                            # 模擬 V5.1：固定分配 (例如 max_position_pct=0.2 即 20%)
                            # 我們透過給予一個極大的 target_risk (如 100%)，
                            # 讓 RiskManager 的 min(vol_size, cap_size) 邏輯總是選中 cap_size
                            shares = rm.calculate_position_size(portfolio_value, entry_price, atr=1e-9) 
                            # 注意：這裡傳入極小 ATR 會導致 vol_shares 極大，進而觸發 Max Cap 限制
                            # 但為了安全，建議直接計算：
                            target_equity = portfolio_value * max_position_pct
                            shares = int(target_equity / entry_price)
                        else:
                            # V5.2 標準：ATR 波動率管理
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
                                    'days_held': 0
                                }

    return equity.dropna()

# analyze_performance 函數保持不變，略...
def analyze_performance(equity_curve, output_dir, filename_prefix, title, benchmark_curve=None, benchmark_label='Benchmark'):
    # ... (保持原樣)
    def calculate_metrics(curve):
        if curve is None or curve.empty or curve.iloc[0] == 0:
            return 0, 0, 0, -1
        returns = curve.pct_change().fillna(0)
        total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
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