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
    hold_days=5
):
    """
    Runs a backtest with Liquidation, Time-Stop, and Sorted Entries.
    Includes fixes for Data Contamination and Look-Ahead Bias.
    """
    # 初始化風控模組 (含倉位上限)
    rm = RiskManager(target_risk=target_risk, max_position_pct=max_position_pct)

    # --- Data Preparation & Signal Generation (Fixes Applied) ---
    # 確保數據按時間排序，但在計算指標時必須按 symbol 分組
    all_data = all_data.sort_index()
    
    # 使用 groupby 確保 shift 只發生在同一檔股票內
    # 這是為了避免 "Data Contamination" (拿 A 股票的昨天預測 B 股票的今天)
    grouped = all_data.groupby('symbol')
    
    # 取得 T-1 的關鍵指標 (用於 T 日開盤決策)
    all_data['prev_RSI_2'] = grouped['RSI_2'].shift(1)
    all_data['prev_close'] = grouped['close'].shift(1)
    all_data['prev_SMA_200'] = grouped['SMA_200'].shift(1)
    
    # [Fix Look-Ahead Bias] 使用 T-1 的 ATR 來計算 T Open 的部位大小
    # 因為在 T Open 時，我們還不知道 T 日的 High/Low/Close (無法算出當日 ATR)
    all_data['prev_ATR_14'] = grouped['ATR_14'].shift(1)

    # 生成進場訊號 (使用 T-1 資訊)
    # 邏輯: RSI(T-1) < 10 且 Price(T-1) > SMA200(T-1)
    all_data['entry_signal'] = (all_data['prev_RSI_2'] < 10) & \
                               (all_data['prev_close'] > all_data['prev_SMA_200'])

    # 生成技術出場訊號 (使用 T-1 資訊)
    # 邏輯: RSI(T-1) > 50
    all_data['tech_exit_signal'] = all_data['prev_RSI_2'] > 50

    # --- Backtest Loop Initialization ---
    cash = initial_capital
    # Positions structure: {symbol: {'shares': float, 'entry_price': float, 'days_held': int}}
    positions = {}
    equity = pd.Series(index=all_data.index.unique().sort_values())

    # --- Main Backtest Loop ---
    # 按日期遍歷 (模擬真實時間流逝)
    for date, daily_data in all_data.groupby(level=0):

        # 0. Get Market Regime for Today
        # 假設所有標的當天的 regime_signal 是一樣的 (來自 Macro，且已對齊)
        current_regime = 0
        if 'regime_signal' in daily_data.columns:
            current_regime = daily_data['regime_signal'].iloc[0]

        # 1. Value portfolio at start of day (Mark-to-Market)
        portfolio_value = cash
        for symbol, pos_data in positions.items():
            if symbol in daily_data['symbol'].values:
                # 嘗試取得今日開盤價作為估值 (最保守估計)
                price_data = daily_data[daily_data['symbol'] == symbol]['open']
                current_price = price_data.iloc[0] if not price_data.empty else pos_data['entry_price']
                portfolio_value += pos_data['shares'] * current_price
            else:
                # 若今日無數據 (停牌)，沿用進場價或昨日收盤價 (此處簡化沿用 Entry)
                portfolio_value += pos_data['shares'] * pos_data['entry_price']
        
        equity[date] = portfolio_value

        # 2. Process Exits (含緊急清倉與時間止損)
        symbols_to_exit = []

        # 更新持倉天數 (在檢查出場前先 +1，因為是開盤檢查)
        for pos_data in positions.values():
            pos_data['days_held'] += 1

        # [Defense Upgrade] 緊急清倉機制 (Liquidation)
        if current_regime == 2:
            # 如果是崩盤模式 (Crash)，清空所有持倉
            symbols_to_exit = list(positions.keys())
        else:
            # 正常模式：檢查技術指標與時間止損
            for symbol, pos_data in positions.items():
                should_sell = False

                # A. 時間止損 (Time Stop)
                if pos_data['days_held'] >= hold_days:
                    should_sell = True

                # B. 技術出場 (RSI > 50)
                # 檢查該股票今日是否有出場訊號
                if not should_sell and symbol in daily_data['symbol'].values:
                    row = daily_data[daily_data['symbol'] == symbol]
                    if not row.empty and row['tech_exit_signal'].iloc[0]:
                        should_sell = True

                if should_sell:
                    symbols_to_exit.append(symbol)

        # 執行賣出操作
        for symbol in symbols_to_exit:
            if symbol not in positions: continue # 避免重複操作

            # 取得賣出價格 (以 Open 賣出)
            exit_price_data = daily_data[daily_data['symbol'] == symbol]['open']

            # 若數據缺失 (停牌)，則無法賣出，保留至隔日
            if not exit_price_data.empty:
                exit_price = exit_price_data.iloc[0]
                shares = positions.pop(symbol)['shares']

                # 結算資金 (扣除滑價與手續費)
                exit_price_adj = exit_price * (1 - slippage_bps / 10000)
                proceeds = shares * exit_price_adj
                costs = proceeds * (transaction_cost_bps / 10000)
                cash += proceeds - costs

        # 3. Process Entries (僅在非崩盤時買入)
        if rm.apply_regime_filter(current_regime):
            # 篩選出今日有進場訊號的股票
            entry_signals = daily_data[daily_data['entry_signal']]

            if not entry_signals.empty:
                # 訊號排序：優先買入 RSI(T-1) 最低 (超賣最嚴重) 的股票
                # 注意：這裡必須使用 prev_RSI_2 進行排序
                entry_signals = entry_signals.sort_values(by='prev_RSI_2', ascending=True)

                for _, row in entry_signals.iterrows():
                    symbol = row['symbol']
                    if symbol in positions:
                        continue # 已持有則不加碼

                    entry_price = row['open']
                    # [Fix] 使用 T-1 的 ATR (prev_ATR_14) 進行部位計算，避免前視偏差
                    atr = row['prev_ATR_14']

                    # 確保 ATR 有效且大於 0
                    if pd.notna(atr) and atr > 0 and entry_price > 0:
                        # 使用風控模組計算股數 (含倉位上限)
                        shares = rm.calculate_position_size(portfolio_value, entry_price, atr)

                        if shares > 0:
                            entry_price_adj = entry_price * (1 + slippage_bps / 10000)
                            cost_of_trade = shares * entry_price_adj
                            transaction_fees = cost_of_trade * (transaction_cost_bps / 10000)
                            total_cost = cost_of_trade + transaction_fees

                            # 檢查現金是否足夠
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