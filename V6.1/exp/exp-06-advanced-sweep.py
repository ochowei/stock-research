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

# 測試區間
TRAIN_START = '2015-01-01'
TEST_END    = '2025-12-31'

# 我們要測試的 Gap Down 門檻 (您想知道 -1% 是否比較好，我們多測幾個)
THRESHOLDS = [-0.005, -0.010, -0.015, -0.020, -0.030]

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
                # 清洗 Ticker
                clean_list = [t.split(':')[-1].strip().replace('.', '-') for t in raw]
                all_tickers.extend(clean_list)
                
                for t in clean_list:
                    ticker_map[t] = label
    
    # 去重
    unique_tickers = sorted(list(set(all_tickers)))
    return unique_tickers, ticker_map

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

def analyze_sweep(df, ticker_map):
    """針對多個 Threshold 進行掃描分析"""
    
    all_results = []
    
    # 預先計算基礎欄位
    df = df.sort_values(['Ticker', 'Date'])
    df['Prev_Close'] = df.groupby('Ticker')['Close'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Intra_Ret'] = (df['Close'] - df['Open']) / df['Open']
    df['Intra_DD'] = (df['Low'] - df['Open']) / df['Open']

    # 開始掃描不同的 Threshold
    for th in THRESHOLDS:
        print(f"Analyzing Threshold: {th*100:.1f}% ...")
        
        for ticker, group in df.groupby('Ticker'):
            pool_type = ticker_map.get(ticker, 'Unknown')
            
            # 篩選條件：Gap <= Threshold 且 Gap > -30% (濾掉極端異常)
            mask = (group['Gap_Pct'] <= th) & (group['Gap_Pct'] > -0.30)
            events = group[mask]
            
            count = len(events)
            if count < 5: continue # 樣本過少忽略
            
            win_rate = (events['Intra_Ret'] > 0).mean()
            avg_ret = events['Intra_Ret'].mean()
            avg_dd = events['Intra_DD'].mean()
            
            all_results.append({
                'Ticker': ticker,
                'Pool': pool_type,
                'Threshold': th,
                'Events': count,
                'Win Rate': win_rate,
                'Avg Return': avg_ret,
                'Avg Intraday DD': avg_dd
            })
            
    return pd.DataFrame(all_results)

def main():
    # 1. Load Data
    tickers, ticker_map = load_tagged_tickers()
    if not tickers: return
    
    print(f"Total Tickers: {len(tickers)} (Covering Asset & Toxic Pools)")
    
    df_data = fetch_data(tickers)
    if df_data.empty: return
    
    # 2. Analyze
    results = analyze_sweep(df_data, ticker_map)
    
    # 3. Output 1: 總體聚合報告 (回答：哪個數字比較好？)
    # 我們計算每個 Threshold 下，所有股票的平均表現
    print("\n" + "="*60)
    print(" >>> PART 1: Threshold Optimization (Global Average) <<<")
    print("="*60)
    
    summary = results.groupby(['Threshold', 'Pool']).agg({
        'Win Rate': 'mean',
        'Avg Return': 'mean',
        'Events': 'sum' # 總交易次數
    }).reset_index()
    
    # 格式化輸出
    summary['Avg Return'] = summary['Avg Return'].apply(lambda x: f"{x*100:.4f}%")
    summary['Win Rate'] = summary['Win Rate'].apply(lambda x: f"{x*100:.2f}%")
    summary['Threshold'] = summary['Threshold'].apply(lambda x: f"{x*100:.1f}%")
    
    print(summary.to_string(index=False))
    
    # 4. Output 2: 詳細個股報告 (回答：詳細標註)
    # 這裡我們挑選 -1.0% (或是您看完報告後覺得最好的那個) 來做詳細存檔，
    # 為了方便，我們把所有 Threshold 的明細都存檔，您可以用 Excel 篩選
    csv_path = os.path.join(OUTPUT_DIR, 'exp_06_advanced_sweep_details.csv')
    
    # 排序：先看平均回報高的
    results.sort_values(['Threshold', 'Avg Return'], ascending=[False, False], inplace=True)
    results.to_csv(csv_path, index=False)
    print(f"\n[Saved] Detailed report saved to: {csv_path}")
    
    # 5. Output 3: 針對 -1% 的特別聚合 (回答：加總/平均結果)
    target_th = -0.01
    subset = results[results['Threshold'] == target_th]
    
    if not subset.empty:
        print("\n" + "="*60)
        print(f" >>> PART 2: Aggregated Stats for Threshold {target_th*100}% <<<")
        print("="*60)
        
        # 依 Pool 分組平均
        grp_stats = subset.groupby('Pool').agg({
            'Win Rate': 'mean',
            'Avg Return': 'mean',
            'Avg Intraday DD': 'mean',
            'Events': 'mean' # 平均每檔股票發生幾次
        })
        print(grp_stats)

        # 總體平均
        total_stats = subset.agg({
            'Win Rate': 'mean',
            'Avg Return': 'mean',
            'Avg Intraday DD': 'mean'
        })
        print("\n[Total Average]")
        print(total_stats)

if __name__ == "__main__":
    main()