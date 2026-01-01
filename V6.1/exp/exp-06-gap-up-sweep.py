import os
import json
import pandas as pd
import numpy as np
import yfinance as yf

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_START = '2015-01-01'
TEST_END    = '2025-12-31'

# 測試的 Gap Up 門檻 (正值)
THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

def load_tagged_tickers():
    """讀取兩個 Pool 並加上標籤"""
    pools = {
        'Asset': '2025_final_asset_pool.json',
        'Toxic': '2025_final_toxic_asset_pool.json'
    }
    
    ticker_map = {}
    all_tickers = []
    
    for label, filename in pools.items():
        path = os.path.join(RESOURCE_DIR, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                clean_list = [t.split(':')[-1].strip().replace('.', '-') for t in raw]
                all_tickers.extend(clean_list)
                for t in clean_list:
                    ticker_map[t] = label
    
    return sorted(list(set(all_tickers))), ticker_map

def fetch_data(tickers):
    print(f"Downloading data for {len(tickers)} tickers...")
    try:
        data = yf.download(
            tickers, start=TRAIN_START, end=TEST_END, 
            interval='1d', auto_adjust=True, progress=True, threads=True
        )
        if isinstance(data.columns, pd.MultiIndex):
            data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        else:
            data['Ticker'] = tickers[0]
            data = data.reset_index()
        return data
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return pd.DataFrame()

def analyze_gap_up_sweep(df, ticker_map):
    all_results = []
    
    # 預先計算
    df = df.sort_values(['Ticker', 'Date'])
    df['Prev_Close'] = df.groupby('Ticker')['Close'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # Fade Return: 賣出(Open) - 買回(Close)
    # 正值代表賣對了 (Open > Close)
    # 負值代表賣錯了 (Open < Close，賣飛了)
    df['Fade_Ret'] = (df['Open'] - df['Close']) / df['Open']

    for th in THRESHOLDS:
        print(f"Analyzing Gap Up > {th*100:.1f}% ...")
        
        for ticker, group in df.groupby('Ticker'):
            pool_type = ticker_map.get(ticker, 'Unknown')
            
            # 篩選條件：Gap >= Threshold
            # 過濾掉極端值 (> 30% 可能是併購或錯誤)
            mask = (group['Gap_Pct'] >= th) & (group['Gap_Pct'] < 0.30)
            events = group[mask]
            
            count = len(events)
            if count < 5: continue
            
            # 勝率：Fade_Ret > 0 的比例 (代表開盤賣比收盤賣划算)
            win_rate = (events['Fade_Ret'] > 0).mean()
            avg_fade = events['Fade_Ret'].mean()
            
            all_results.append({
                'Ticker': ticker,
                'Pool': pool_type,
                'Threshold': th,
                'Events': count,
                'Win Rate': win_rate,
                'Avg Fade Return': avg_fade
            })
            
    return pd.DataFrame(all_results)

def main():
    tickers, ticker_map = load_tagged_tickers()
    if not tickers: return
    
    df_data = fetch_data(tickers)
    if df_data.empty: return
    
    results = analyze_gap_up_sweep(df_data, ticker_map)
    
    # 聚合報告
    print("\n" + "="*80)
    print(" >>> Gap Up Optimization: To Sell or Not To Sell? <<<")
    print(" (Win Rate > 50% means 'Selling at Open' is usually better than Holding)")
    print(" (Avg Fade Return > 0 means 'Selling at Open' saves money)")
    print("="*80)
    
    summary = results.groupby(['Threshold', 'Pool']).agg({
        'Win Rate': 'mean',
        'Avg Fade Return': 'mean',
        'Events': 'sum'
    }).reset_index()
    
    # 格式化
    summary['Avg Fade Return'] = summary['Avg Fade Return'].apply(lambda x: f"{x*100:.4f}%")
    summary['Win Rate'] = summary['Win Rate'].apply(lambda x: f"{x*100:.2f}%")
    summary['Threshold'] = summary['Threshold'].apply(lambda x: f"{x*100:.1f}%")
    
    print(summary.to_string(index=False))
    
    csv_path = os.path.join(OUTPUT_DIR, 'exp_06_gap_up_sweep.csv')
    results.to_csv(csv_path, index=False)
    print(f"\n[Saved] Detailed report: {csv_path}")

if __name__ == "__main__":
    main()