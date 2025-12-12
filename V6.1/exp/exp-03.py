import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import time
import matplotlib.pyplot as plt
import seaborn as sns

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', '..', 'V6.0', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 載入標的 (與 EXP-02 相同邏輯)
def load_tickers():
    files = ['2025_final_asset_pool.json', '2025_final_toxic_asset_pool.json']
    tickers = []
    for f in files:
        path = os.path.join(RESOURCE_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as json_file:
                raw = json.load(json_file)
                clean = [t.split(':')[-1].strip().replace('.', '-') for t in raw]
                tickers.extend(clean)
    # 排除指數
    indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT']
    return sorted(list(set([t for t in tickers if t not in indices])))

def run_microstructure_analysis():
    tickers = load_tickers()
    print(f"Analyzing {len(tickers)} tickers over the last 60 days (5m data)...")
    
    # 下載數據 (含盤前盤後)
    # 注意：一次下載大量 tickers 的 5m 數據可能會很久或失敗，建議分批或針對重點股
    # 這裡演示針對前 20 檔高波動標的 + 重點關注股進行快速驗證
    # 實盤可放寬
    sample_tickers = tickers[:50] 
    
    try:
        data = yf.download(
            sample_tickers, 
            period="60d", 
            interval="5m", 
            prepost=True, 
            group_by='ticker',
            auto_adjust=True,
            threads=True
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return

    results = []

    for ticker in sample_tickers:
        try:
            if len(sample_tickers) > 1:
                df = data[ticker].copy()
            else:
                df = data.copy()
            
            if df.empty: continue
            
            # 處理時區
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')

            # 依日期分組
            days = df.groupby(df.index.date)
            
            # 轉換為列表以便存取 T-1
            sorted_dates = sorted(list(days.groups.keys()))
            
            for i in range(1, len(sorted_dates)):
                date_prev = sorted_dates[i-1]
                date_curr = sorted_dates[i]
                
                df_prev = days.get_group(date_prev)
                df_curr = days.get_group(date_curr)
                
                # --- 1. 計算 T-1 尾盤動能 (Tail Momentum) ---
                # 正規交易時間 15:30 ~ 16:00
                regular_prev = df_prev.between_time('09:30', '16:00')
                if regular_prev.empty: continue
                
                # 尾盤切片
                tail_start = regular_prev.index[-1].replace(hour=15, minute=30, second=0)
                tail_data = regular_prev[tail_start:]
                
                if tail_data.empty: 
                    tail_mom = 0
                else:
                    tail_open = tail_data['Open'].iloc[0]
                    tail_close = tail_data['Close'].iloc[-1]
                    tail_mom = (tail_close - tail_open) / tail_open

                prev_close = regular_prev['Close'].iloc[-1]

                # --- 2. 計算 T 盤前特徵 (Pre-market Fade) ---
                # 盤前時段 04:00 ~ 09:30
                pre_data = df_curr.between_time('04:00', '09:30')
                
                if pre_data.empty:
                    pre_fade = 0
                    pre_gap = 0 # 無盤前數據無法準確判斷 Gap 結構，暫略
                else:
                    pre_high = pre_data['High'].max()
                    pre_last = pre_data['Close'].iloc[-1]
                    # Fade: 最高點回落幅度
                    pre_fade = (pre_high - pre_last) / pre_high if pre_high > 0 else 0
                
                # --- 3. 計算 T 日內表現 (Target) ---
                regular_curr = df_curr.between_time('09:30', '16:00')
                if regular_curr.empty: continue
                
                open_curr = regular_curr['Open'].iloc[0]
                close_curr = regular_curr['Close'].iloc[-1]
                
                gap_pct = (open_curr - prev_close) / prev_close
                day_ret = (close_curr - open_curr) / open_curr
                
                # 僅關注「跳空高開」的案例 (Gap > 0.5%)
                if gap_pct > 0.005:
                    results.append({
                        'Ticker': ticker,
                        'Date': date_curr,
                        'Gap_%': gap_pct * 100,
                        'Tail_Mom_%': tail_mom * 100,
                        'Pre_Fade_%': pre_fade * 100,
                        'Day_Ret_%': day_ret * 100,
                        'Win': 1 if day_ret < 0 else 0 # 賣出策略，跌就是贏
                    })
                    
        except Exception as e:
            continue

    # --- 分析結果 ---
    if not results:
        print("No gap events found.")
        return

    df_res = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print(f"EXP-V6.1-03: Microstructure Analysis (Last 60 Days)")
    print(f"Sample Size: {len(df_res)} gap events")
    print("="*60)
    
    # 1. 基準勝率
    base_win_rate = df_res['Win'].mean()
    print(f"Baseline Win Rate (Gap > 0.5%): {base_win_rate:.2%}")
    
    # 2. 加入 Tail Momentum 濾網 (昨日尾盤急拉 > 0.5%)
    mask_tail = df_res['Tail_Mom_%'] > 0.5
    win_tail = df_res[mask_tail]['Win'].mean()
    print(f"With Strong Tail (>0.5%):       {win_tail:.2%} (n={mask_tail.sum()})")
    
    # 3. 加入 Pre-market Fade 濾網 (盤前已回落 > 1.0%)
    mask_fade = df_res['Pre_Fade_%'] > 1.0
    win_fade = df_res[mask_fade]['Win'].mean()
    print(f"With Pre-market Fade (>1.0%):   {win_fade:.2%} (n={mask_fade.sum()})")
    
    # 4. 雙重濾網
    mask_dual = mask_tail & mask_fade
    win_dual = df_res[mask_dual]['Win'].mean()
    print(f"With Dual Filter:               {win_dual:.2%} (n={mask_dual.sum()})")
    
    # 儲存
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'exp_03_microstructure.csv'), index=False)
    print(f"\nDetailed data saved to {OUTPUT_DIR}/exp_03_microstructure.csv")

if __name__ == '__main__':
    run_microstructure_analysis()