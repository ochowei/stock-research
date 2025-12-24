import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from datetime import datetime

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 測試區間
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'
GAP_THRESHOLD = 0.005 # 0.5%

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return list(set([t.split(':')[-1].strip().replace('.', '-') for t in json.load(f)]))

def fetch_data(tickers):
    print(f"Downloading data for {len(tickers)} tickers...")
    df = yf.download(tickers, start=START_DATE, end=END_DATE, interval='1d', auto_adjust=True, progress=True, threads=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        df['Ticker'] = tickers[0]
        df = df.reset_index()
        
    return df

def run_blind_limit_test(df):
    """核心回測邏輯"""
    df = df.sort_values(['Ticker', 'Date']).copy()
    
    # 1. 計算基礎特徵
    df['Prev_Close'] = df.groupby('Ticker')['Close'].shift(1)
    df['Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # 計算 ATR
    df['TR'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Prev_Close']))
    df['ATR'] = df.groupby('Ticker')['TR'].transform(lambda x: x.rolling(14).mean().shift(1))
    
    # 2. 篩選訊號
    signals = df[df['Gap'] > GAP_THRESHOLD].copy()
    
    if signals.empty:
        print("No signals found.")
        return None

    # [新增] 取得總標的數量 (用於計算平均)
    total_tickers = df['Ticker'].nunique()
    print(f"Total MOO Signals: {len(signals)} across {total_tickers} tickers")

    results = []

    # --- 測試場景 ---
    scenarios = [
        {'name': 'MOO (Baseline)', 'type': 'moo', 'val': 0},
        
        {'name': 'Limit +0.3%', 'type': 'fixed', 'val': 0.003},
        {'name': 'Limit +0.5%', 'type': 'fixed', 'val': 0.005},
        {'name': 'Limit +1.0%', 'type': 'fixed', 'val': 0.010},
        
        {'name': 'Limit +0.1 ATR', 'type': 'atr', 'val': 0.1},
        {'name': 'Limit +0.2 ATR', 'type': 'atr', 'val': 0.2},
        {'name': 'Limit +0.3 ATR', 'type': 'atr', 'val': 0.3},
    ]
    
    for sc in scenarios:
        temp = signals.copy()
        
        # A. 計算掛單
        if sc['type'] == 'moo':
            temp['Entry_Price'] = temp['Open']
            temp['Filled'] = True
        elif sc['type'] == 'fixed':
            temp['Entry_Price'] = temp['Open'] * (1 + sc['val'])
            temp['Filled'] = temp['High'] >= temp['Entry_Price']
        elif sc['type'] == 'atr':
            temp['Entry_Price'] = temp['Open'] + (sc['val'] * temp['ATR'])
            temp['Filled'] = temp['High'] >= temp['Entry_Price']
            
        # B. 計算回報
        slippage = 0.001 if sc['type'] == 'moo' else 0.0
        raw_ret = (temp['Entry_Price'] - temp['Close']) / temp['Entry_Price']
        temp['Return'] = np.where(temp['Filled'], raw_ret - slippage, 0.0)
        
        # C. 統計指標
        filled_count = int(temp['Filled'].sum())
        fill_rate = filled_count / len(temp)
        
        # [新增] 平均每檔股票交易次數
        avg_count_per_ticker = filled_count / total_tickers
        
        filled_trades = temp[temp['Filled']]
        win_rate = (filled_trades['Return'] > 0).mean() if not filled_trades.empty else 0
        avg_ret = filled_trades['Return'].mean() if not filled_trades.empty else 0
        total_profit = temp['Return'].sum()
        
        results.append({
            'Scenario': sc['name'],
            'Total Trades': filled_count,
            'Avg Trades/Ticker': avg_count_per_ticker, # [新增]
            'Fill Rate': fill_rate,
            'Win Rate': win_rate,
            'Avg Ret': avg_ret,
            'Total Return': total_profit
        })
        
    return pd.DataFrame(results)

def main():
    print("=== EXP-08: Blind Limit Optimization ===")
    tickers = load_tickers()
    if not tickers: return
    
    df = fetch_data(tickers)
    res = run_blind_limit_test(df)
    
    if res is not None:
        print("\n" + "="*110)
        print("RESULT SUMMARY (Short Gap Up > 0.5%)")
        print("="*110)
        
        # 格式化輸出
        print(res.to_string(index=False, formatters={
            'Fill Rate': "{:.2%}".format,
            'Win Rate': "{:.2%}".format,
            'Avg Ret': "{:.2%}".format,
            'Total Return': "{:.2%}".format,
            'Total Trades': "{:,}".format,
            'Avg Trades/Ticker': "{:.1f}".format # 顯示到小數第一位
        }))
        
        # 存檔
        res.to_csv(os.path.join(OUTPUT_DIR, 'exp_08_blind_limit_results.csv'), index=False)
        print("\nResults saved.")

if __name__ == '__main__':
    main()