import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
from data_loader import DataLoader
from backtesting_utils import analyze_performance

# --- 設定：2025 專屬回測 ---
START_DATE = '2025-01-01'
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
    """篩選出指定清單的數據 (支援 MultiIndex)"""
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
    """下載並計算 SPY 同期績效"""
    print(f"Downloading SPY Benchmark ({start_date} - {end_date})...")
    spy = yf.download("SPY", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    
    if spy.empty: return pd.Series()
    
    # 處理 yfinance 格式 (MultiIndex columns)
    if isinstance(spy.columns, pd.MultiIndex):
        close = spy['Close'].iloc[:, 0]
        open_p = spy['Open'].iloc[:, 0]
    else:
        close = spy['Close']
        open_p = spy['Open']
        
    # 模擬 Buy & Hold
    shares = initial_capital / open_p.iloc[0]
    equity = close * shares
    return equity

def run_strict_hold_backtest(df, config):
    """V5.1 極簡回測邏輯 (Strict Time Stop)"""
    print(f"Running Strict Hold Backtest (From {START_DATE})...")
    
    # 1. 預先計算 T-1 訊號
    df['prev_RSI_2'] = df.groupby(level='symbol')['RSI_2'].shift(1)
    df['prev_SMA_200'] = df.groupby(level='symbol')['SMA_200'].shift(1)
    df['prev_close'] = df.groupby(level='symbol')['close'].shift(1)
    
    # 進場訊號
    df['entry_signal'] = (df['prev_RSI_2'] < 10) & (df['prev_close'] > df['prev_SMA_200'])
    
    # 2. 時間過濾 (只保留 2025 之後的數據)
    df = df[df.index.get_level_values('timestamp') >= pd.Timestamp(START_DATE)]
    
    if df.empty:
        print("[Error] No data found after start date.")
        return pd.DataFrame(), pd.DataFrame()

    daily_data = df.reorder_levels(['timestamp', 'symbol']).sort_index()
    all_dates = daily_data.index.get_level_values('timestamp').unique().sort_values()
    
    cash = config['initial_capital']
    positions = {} 
    equity_curve = []
    trade_log = []
    
    slippage = config['slippage']
    cost_rate = config['transaction_cost']
    max_pos = config['max_positions']

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

    return pd.DataFrame(equity_curve).set_index('timestamp'), pd.DataFrame(trade_log)

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"=== V5.1 Performance Review (2025 YTD) ===")
    
    # 1. 載入 Final Pool
    loader = DataLoader(SCRIPT_DIR, normal_file='final_asset_pool.json', toxic_file='final_toxic_asset_pool.json')
    target_tickers = loader.get_all_tickers()
    
    # 2. 載入與篩選數據
    df = load_data(SCRIPT_DIR, track='custom')
    if df is None: return
    df_subset = filter_data_by_tickers(df, target_tickers)
    
    # 3. 執行策略回測
    equity, trades = run_strict_hold_backtest(df_subset, CONFIG)
    
    if not equity.empty:
        # 4. 取得 SPY 基準
        start_dt = equity.index.min()
        end_dt = equity.index.max() + pd.Timedelta(days=1) # 確保包含最後一天
        spy_curve = get_spy_benchmark(start_dt, end_dt, CONFIG['initial_capital'])
        
        # 對齊時間軸
        if not spy_curve.empty:
            spy_curve = spy_curve.reindex(equity.index, method='ffill')
            # 歸一化起點 (從 10萬開始)
            spy_curve = spy_curve / spy_curve.iloc[0] * CONFIG['initial_capital']

        # 5. 計算並顯示結果
        strat_ret = (equity.iloc[-1]['equity'] / 100000) - 1
        spy_ret = (spy_curve.iloc[-1] / 100000) - 1 if not spy_curve.empty else 0
        
        print("\n" + "="*30)
        print(f"2025 YTD Performance Summary")
        print("="*30)
        print(f"Strategy (V5.1): {strat_ret:>.2%}")
        print(f"Benchmark (SPY): {spy_ret:>.2%}")
        print(f"Alpha:           {strat_ret - spy_ret:>.2%}")
        print(f"Trades Count:    {len(trades)}")
        print("-" * 30)
        
        # 繪圖
        analyze_performance(
            equity_curve=equity['equity'],
            output_dir=OUTPUT_DIR,
            filename_prefix='v5.1_2025_ytd',
            title=f'V5.1 (Strict Hold) vs SPY - 2025 YTD',
            benchmark_curve=spy_curve,
            benchmark_label='SPY (Buy & Hold)'
        )
        print(f"Chart saved to {OUTPUT_DIR}/v5.1_2025_ytd_equity.png")
        
    else:
        print("[Error] No trades generated in 2025.")

if __name__ == "__main__":
    main()