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

# 參數設定
# 這裡設定我們關心的 "Gap Down" 門檻，依照您的需求設為 -0.5%
TARGET_THRESHOLD = -0.01 
TRAIN_START = '2015-01-01'
TEST_END    = '2025-12-31'

def load_target_pool():
    """只讀取 2025_final_asset_pool.json"""
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
        # 清洗格式 (例如 "NYSE:UBER" -> "UBER")
        tickers = [t.split(':')[-1].strip().replace('.', '-') for t in raw]
        
    return sorted(list(set(tickers)))

def fetch_data(tickers):
    print(f"Downloading data for {len(tickers)} tickers ({TRAIN_START} ~ {TEST_END})...")
    try:
        data = yf.download(
            tickers, start=TRAIN_START, end=TEST_END, 
            interval='1d', auto_adjust=True, progress=True, threads=True
        )
        if isinstance(data.columns, pd.MultiIndex):
            data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        else:
            # 單一 ticker 處理
            data['Ticker'] = tickers[0]
            data = data.reset_index()
        return data
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return pd.DataFrame()

def analyze_individual_ticker(df):
    """針對每一檔 Ticker 計算 Gap Down < -0.5% 後的表現"""
    
    stats_list = []
    
    # 確保資料排序
    df = df.sort_values(['Ticker', 'Date'])
    
    # 計算需要的欄位
    # GroupBy Ticker 避免跨股票 shift
    df['Prev_Close'] = df.groupby('Ticker')['Close'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # Intraday Return: 開盤買，收盤賣 (Buy Open, Sell Close)
    df['Intra_Ret'] = (df['Close'] - df['Open']) / df['Open']
    
    # Intraday Max Drawdown: (最低 - 開盤) / 開盤 (看盤中最大浮虧)
    df['Intra_DD'] = (df['Low'] - df['Open']) / df['Open']

    # 針對每一檔股票進行統計
    for ticker, group in df.groupby('Ticker'):
        # 篩選條件：Gap 低於設定門檻 (例如 -0.5%)
        # 同時過濾掉極端數據 (例如 -30% 可能是拆股或錯誤)
        mask = (group['Gap_Pct'] <= TARGET_THRESHOLD) & (group['Gap_Pct'] > -0.30)
        events = group[mask]
        
        count = len(events)
        if count < 10: # 樣本太少不計入
            continue
            
        # 勝率：收盤價 > 開盤價
        win_count = (events['Intra_Ret'] > 0).sum()
        win_rate = win_count / count
        
        # 平均報酬 (期望值)
        avg_ret = events['Intra_Ret'].mean()
        median_ret = events['Intra_Ret'].median()
        
        # 風險指標
        avg_dd = events['Intra_DD'].mean() # 平均盤中最大跌幅
        worst_dd = events['Intra_DD'].min() # 歷史最大盤中跌幅
        
        # Kelly Criterion (簡易版參考) - 幫助判斷下注比例
        # p = win_rate, b = odds (avg_win / avg_loss)
        wins = events[events['Intra_Ret'] > 0]['Intra_Ret']
        losses = events[events['Intra_Ret'] <= 0]['Intra_Ret']
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        
        kelly = 0
        if avg_loss > 0:
            kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)

        stats_list.append({
            'Ticker': ticker,
            'Events': count,
            'Win Rate': win_rate,
            'Avg Return': avg_ret,
            'Median Return': median_ret,
            'Avg Intraday DD': avg_dd,
            'Kelly Score': kelly
        })
        
    return pd.DataFrame(stats_list)

def main():
    # 1. Load List
    tickers = load_target_pool()
    print(f"Target Pool: {len(tickers)} tickers loaded from 2025_final_asset_pool.json")
    
    if not tickers:
        return

    # 2. Fetch Data
    df_all = fetch_data(tickers)
    if df_all.empty:
        return

    # 3. Analyze
    print(f"\nAnalyzing Gap Down < {TARGET_THRESHOLD*100}% behavior for each stock...")
    results = analyze_individual_ticker(df_all)
    
    # 4. Sort & Save
    # 依照「平均報酬」排序，找出最適合做 Buy the Dip 的股票
    results = results.sort_values('Avg Return', ascending=False)
    
    # 顯示前 20 名
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nTop 20 Best 'Buy the Dip' Candidates:")
    print(results.head(20).to_string(index=False, float_format="%.4f"))
    
    print("\nBottom 10 Worst Candidates (Don't touch these on dips):")
    print(results.tail(10).to_string(index=False, float_format="%.4f"))

    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, 'exp_06_individual_gap_down_stats.csv')
    results.to_csv(csv_path, index=False)
    print(f"\n[Saved] Full report saved to: {csv_path}")

if __name__ == "__main__":
    main()