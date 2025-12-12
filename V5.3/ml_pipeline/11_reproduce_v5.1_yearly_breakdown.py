import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
from data_loader import DataLoader
from backtesting_utils import analyze_performance

# --- 基礎設定 ---
CONFIG = {
    'initial_capital': 100_000.0,
    'hold_days': 5,              # 死守 5 天
    'max_positions': 5,          # 持倉上限 5 檔
    'slippage': 0.0005,          # 滑價 5bps
    'transaction_cost': 0.0005   # 交易成本 5bps
}

def load_data(base_dir, track='custom'):
    features_path = os.path.join(base_dir, 'data', track, 'features', 'stock_features.parquet')
    if not os.path.exists(features_path):
        print(f"[Error] Data not found at {features_path}")
        return None
    print(f"Loading features from {features_path}...")
    df = pd.read_parquet(features_path)
    return df.sort_index()

def filter_data_by_tickers(df, tickers):
    if 'symbol' in df.index.names:
        valid_tickers = set(df.index.get_level_values('symbol').unique())
        target_tickers = [t for t in tickers if t in valid_tickers]
        return df[df.index.get_level_values('symbol').isin(target_tickers)].copy()
    elif 'symbol' in df.columns:
        valid_tickers = set(df['symbol'].unique())
        target_tickers = [t for t in tickers if t in valid_tickers]
        return df[df['symbol'].isin(target_tickers)].copy()
    return pd.DataFrame()

def get_spy_benchmark(start_date, end_date, initial_capital):
    print(f"  Downloading SPY Benchmark ({start_date.date()} - {end_date.date()})...")
    # yfinance 只能下載到最新日期，如果 end_date 是過去，需要切片
    spy = yf.download("SPY", start=start_date, end=end_date + pd.Timedelta(days=5), interval="1d", auto_adjust=True, progress=False)
    
    if spy.empty: return pd.Series()
    
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close'].iloc[:, 0]
        open_p = spy['Open'].iloc[:, 0]
    else:
        close = spy['Close']
        open_p = spy['Open']
    
    # 確保時間區間精確
    mask = (close.index >= start_date) & (close.index <= end_date)
    close = close[mask]
    
    if close.empty: return pd.Series()
    
    # 模擬 Buy & Hold (以第一天 Open 買入)
    # 若第一天 Open 無法取得，改用第一天 Close
    start_price = open_p[mask].iloc[0] if not open_p[mask].empty else close.iloc[0]
    shares = initial_capital / start_price
    equity = close * shares
    return equity

def run_backtest_for_period(df, start_date, end_date, config, year_label):
    """
    針對特定時間區段執行獨立回測
    """
    print(f"\n--- Processing Year: {year_label} ({start_date.date()} to {end_date.date()}) ---")
    
    # 1. 預先計算 T-1 訊號 (在全量數據上計算，確保邊界日的 T-1 數據存在)
    # 為了避免 SettingWithCopyWarning，先 copy
    df = df.copy()
    df['prev_RSI_2'] = df.groupby(level='symbol')['RSI_2'].shift(1)
    df['prev_SMA_200'] = df.groupby(level='symbol')['SMA_200'].shift(1)
    df['prev_close'] = df.groupby(level='symbol')['close'].shift(1)
    
    # 進場訊號
    df['entry_signal'] = (df['prev_RSI_2'] < 10) & (df['prev_close'] > df['prev_SMA_200'])
    
    # 2. 時間切片 (Slice)
    mask = (df.index.get_level_values('timestamp') >= start_date) & \
           (df.index.get_level_values('timestamp') <= end_date)
    period_df = df[mask].copy()
    
    if period_df.empty:
        print(f"[Warning] No data found for {year_label}.")
        return pd.Series(), [], {}

    daily_data = period_df.reorder_levels(['timestamp', 'symbol']).sort_index()
    all_dates = daily_data.index.get_level_values('timestamp').unique().sort_values()
    
    # 3. 初始化回測變數 (資金重置)
    cash = config['initial_capital']
    positions = {} 
    equity_curve = []
    trade_log = []
    
    slippage = config['slippage']
    cost_rate = config['transaction_cost']
    max_pos = config['max_positions']

    # 4. 回測迴圈
    for date in all_dates:
        try:
            today_bar = daily_data.loc[date]
        except KeyError:
            continue
            
        # --- A. 出場 (Time Exit) ---
        symbols_to_sell = []
        for sym, pos in positions.items():
            pos['days_held'] += 1
            if pos['days_held'] >= config['hold_days']:
                if sym in today_bar.index:
                    price = today_bar.loc[sym]['open']
                    shares = pos['shares']
                    value = shares * price * (1 - slippage)
                    net_proceeds = value - (value * cost_rate)
                    cash += net_proceeds
                    
                    ret = (net_proceeds / (shares * pos['entry_price'])) - 1
                    trade_log.append({
                        'symbol': sym, 'entry_date': pos['entry_date'], 
                        'exit_date': date, 'return': ret, 'reason': 'Time_Exit'
                    })
                    symbols_to_sell.append(sym)
        
        for sym in symbols_to_sell: del positions[sym]
            
        # --- B. 權益更新 ---
        curr_equity = cash
        for sym, pos in positions.items():
            if sym in today_bar.index:
                price = today_bar.loc[sym]['close']
                curr_equity += pos['shares'] * price
            else:
                curr_equity += pos['shares'] * pos['entry_price']
        equity_curve.append({'timestamp': date, 'equity': curr_equity})
        
        # --- C. 進場 (Equal Weight) ---
        open_slots = max_pos - len(positions)
        if open_slots > 0:
            candidates = today_bar[today_bar['entry_signal']]
            if not candidates.empty:
                candidates = candidates.sort_values('prev_RSI_2', ascending=True)
                target_per_trade = curr_equity / max_pos
                
                for sym, row in candidates.iterrows():
                    if open_slots <= 0: break
                    if sym in positions: continue
                    
                    price = row['open']
                    if pd.isna(price) or price <= 0: continue
                    
                    max_buy_cost = min(cash, target_per_trade)
                    shares = int(max_buy_cost / (price * (1 + slippage + cost_rate)))
                    
                    if shares > 0:
                        cost = shares * price * (1 + slippage)
                        total_outlay = cost + (cost * cost_rate)
                        if cash >= total_outlay:
                            cash -= total_outlay
                            positions[sym] = {
                                'shares': shares, 
                                'entry_price': price * (1 + slippage),
                                'entry_date': date, 
                                'days_held': 0
                            }
                            open_slots -= 1

    equity_series = pd.DataFrame(equity_curve).set_index('timestamp')['equity']
    
    # 計算該年度指標
    if not equity_series.empty:
        total_ret = (equity_series.iloc[-1] / config['initial_capital']) - 1
        daily_ret = equity_series.pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() != 0 else 0
        dd = (equity_series / equity_series.cummax() - 1).min()
        win_rate = pd.DataFrame(trade_log)['return'].gt(0).mean() if trade_log else 0
        
        metrics = {
            'Total Return': total_ret,
            'Sharpe': sharpe,
            'MaxDD': dd,
            'Win Rate': win_rate,
            'Trades': len(trade_log)
        }
    else:
        metrics = {}

    return equity_series, trade_log, metrics

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== V5.1 Yearly Performance Breakdown (2024 & 2025) ===")
    
    # 1. 載入 Final Pool
    loader = DataLoader(SCRIPT_DIR, normal_file='final_asset_pool.json', toxic_file='final_toxic_asset_pool.json')
    target_tickers = loader.get_all_tickers()
    
    # 2. 載入數據
    df = load_data(SCRIPT_DIR, track='custom')
    if df is None: return
    df_subset = filter_data_by_tickers(df, target_tickers)
    
    # 3. 定義年份區間
    periods = [
        {'label': '2024 Full Year', 'start': pd.Timestamp('2024-01-01'), 'end': pd.Timestamp('2024-12-31')},
        {'label': '2025 YTD',       'start': pd.Timestamp('2025-01-01'), 'end': pd.Timestamp('2025-12-31')} # 到最新數據
    ]
    
    summary_list = []

    for p in periods:
        # A. 執行策略
        equity, trades, met = run_backtest_for_period(df_subset, p['start'], p['end'], CONFIG, p['label'])
        
        if not equity.empty:
            # B. 執行 SPY 基準
            spy_equity = get_spy_benchmark(p['start'], p['end'], CONFIG['initial_capital'])
            
            # 對齊
            common_idx = equity.index.intersection(spy_equity.index)
            if not common_idx.empty:
                # 重新歸一化以確保公平比較 (都從 10 萬開始)
                sub_strat = equity.loc[common_idx]
                sub_spy = spy_equity.loc[common_idx]
                sub_spy = sub_spy / sub_spy.iloc[0] * CONFIG['initial_capital']
                
                spy_ret = (sub_spy.iloc[-1] / CONFIG['initial_capital']) - 1
                spy_dd = (sub_spy / sub_spy.cummax() - 1).min()
                
                # 繪圖
                analyze_performance(
                    equity_curve=sub_strat,
                    output_dir=OUTPUT_DIR,
                    filename_prefix=f"v5.1_{p['label'].replace(' ', '_')}",
                    title=f"V5.1 Strict Hold ({p['label']}) vs SPY",
                    benchmark_curve=sub_spy,
                    benchmark_label='SPY'
                )
                
                # 紀錄結果
                summary_list.append({
                    'Period': p['label'],
                    'V5.1 Return': f"{met['Total Return']:.2%}",
                    'SPY Return': f"{spy_ret:.2%}",
                    'Alpha': f"{met['Total Return'] - spy_ret:.2%}",
                    'V5.1 Sharpe': f"{met['Sharpe']:.2f}",
                    'V5.1 MaxDD': f"{met['MaxDD']:.2%}",
                    'SPY MaxDD': f"{spy_dd:.2%}",
                    'Trades': met['Trades']
                })
            else:
                print(f"  [Error] No overlapping data with SPY for {p['label']}")
        else:
            print(f"  [Info] No trades for {p['label']}")

    # 4. 輸出總表
    if summary_list:
        res_df = pd.DataFrame(summary_list)
        print("\n" + "="*80)
        print(" YEARLY PERFORMANCE BREAKDOWN (Independent Simulations)")
        print("="*80)
        print(res_df.to_string(index=False))
        print("-" * 80)
        
        # 存檔
        res_df.to_csv(os.path.join(OUTPUT_DIR, 'v5.1_yearly_breakdown.csv'), index=False)
        print(f"Summary saved to {os.path.join(OUTPUT_DIR, 'v5.1_yearly_breakdown.csv')}")

if __name__ == "__main__":
    main()