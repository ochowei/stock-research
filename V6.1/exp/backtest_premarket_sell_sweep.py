import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dt_time
import warnings

# --- 設定 ---
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [修改點] 設定要測試的門檻列表
THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.03]  # 0.5% ~ 3.0%
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# --- 工具函數 ---

def load_holding_tickers():
    """讀取 Holding Pool (監控清單)"""
    path = os.path.join(RESOURCE_DIR, HOLDING_POOL_FILE)
    if not os.path.exists(path):
        path = path.replace('V6.1', 'V6.0') # Fallback check
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw]))
    else:
        print(f"[Warning] 找不到 {HOLDING_POOL_FILE}，使用預設清單")
        return ['NVDA', 'TSLA', 'AAPL', 'AMD', 'PLTR', 'MSTR']

def fetch_data(tickers):
    print(f"1. 下載日線資料 (基準)...")
    df_daily = yf.download(tickers, period="3mo", interval="1d", auto_adjust=True, progress=False, threads=True)
    
    print(f"2. 下載盤前分時資料 (最近59天, 5分K)...")
    df_intra = yf.download(tickers, period="59d", interval="5m", prepost=True, auto_adjust=True, progress=True, threads=True)
    
    return df_daily, df_intra

def backtest_sell_limit_sweep(ticker, daily_data, intra_data):
    """
    回測核心：一次測試多個 Thresholds
    """
    # 提取單一股票數據
    if isinstance(intra_data.columns, pd.MultiIndex):
        try:
            df = intra_data.xs(ticker, axis=1, level=1).copy()
        except KeyError: return []
    else:
        df = intra_data.copy()
    
    df = df.dropna()
    if df.empty: return []

    try:
        df.index = df.index.tz_convert('America/New_York')
    except TypeError:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    d_data = daily_data.xs(ticker, axis=1, level=1).copy() if isinstance(daily_data.columns, pd.MultiIndex) else daily_data.copy()
    d_data.index = pd.to_datetime(d_data.index).date
    
    results = []
    dates = sorted(list(set(df.index.date)))
    
    for d in dates:
        # 1. 取得昨日收盤與今日收盤
        try:
            loc = d_data.index.get_loc(d)
            if loc == 0: continue
            prev_close = float(d_data.iloc[loc-1]['Close'])
            close_today = float(d_data.iloc[loc]['Close']) 
        except: continue
            
        # 2. 取得當日數據
        day_bars = df[df.index.date == d]
        if day_bars.empty: continue
        
        market_open_time = dt_time(9, 30)
        pre_market = day_bars[day_bars.index.time < market_open_time]
        regular_market = day_bars[day_bars.index.time >= market_open_time]
        
        if regular_market.empty: continue
        open_price = float(regular_market.iloc[0]['Open'])
        
        # 基準：死抱到收盤的報酬
        ret_hold = (close_today - prev_close) / prev_close

        # [修改點] 針對每個 Threshold 跑一次邏輯
        for th in THRESHOLDS:
            target_price = prev_close * (1 + th)
            
            # --- 策略 A: 盤前掛單 ---
            exec_price_pre = None
            filled_in_pre = False
            
            # 檢查盤前
            if not pre_market.empty:
                mask = pre_market['High'] >= target_price
                if mask.any():
                    filled_in_pre = True
                    # 成交在 Target (或者該 Bar Open 更高)
                    first_bar = pre_market[mask].iloc[0]
                    exec_price_pre = max(target_price, first_bar['Open'])

            # 盤前沒成交，看開盤 (Open)
            if not filled_in_pre:
                if open_price >= target_price:
                    exec_price_pre = open_price
                else:
                    # 都沒成交 -> 持有到收盤
                    exec_price_pre = close_today 

            ret_pre = (exec_price_pre - prev_close) / prev_close
            
            results.append({
                'Date': d,
                'Ticker': ticker,
                'Threshold': th,         # 標記這是哪個門檻的結果
                'Pre_Filled': filled_in_pre,
                'Ret_Strategy': ret_pre, # 策略報酬
                'Ret_Hold': ret_hold     # 基準報酬 (重複存沒關係，方便groupby)
            })
        
    return results

def generate_report(trades):
    if not trades:
        print("沒有產生交易紀錄。")
        return
        
    df = pd.DataFrame(trades)
    
    print(f"\n=== [V6.1 參數掃描] 持倉止盈策略門檻分析 ===")
    print(f"測試門檻: {[f'{t*100}%' for t in THRESHOLDS]}")
    print("-" * 100)
    
    # 1. 總表分析
    # 依 Threshold 分組統計
    summary = []
    
    # 計算基準 (Hold till Close) 的平均報酬，這對所有 threshold 都一樣
    base_avg_ret = df['Ret_Hold'].mean()
    
    for th in THRESHOLDS:
        sub_df = df[df['Threshold'] == th]
        
        avg_ret = sub_df['Ret_Strategy'].mean()
        fill_rate = sub_df['Pre_Filled'].mean()
        
        # 勝率 (比 Hold 好的比例)
        win_rate = (sub_df['Ret_Strategy'] > sub_df['Ret_Hold']).mean()
        
        summary.append({
            'Threshold': f"{th*100:>4.1f}%",
            'Avg Daily Ret': avg_ret,
            'Lift (vs Hold)': avg_ret - base_avg_ret,
            'Fill Rate': fill_rate,
            'Win Rate': win_rate
        })
        
    res_df = pd.DataFrame(summary)
    
    print(f"基準策略 (Hold till Close) Avg Daily Return: {base_avg_ret*100:.4f}%")
    print("-" * 100)
    
    # 格式化輸出
    header = f"{'Threshold':<10} {'Avg Daily Ret':<15} {'Lift (Alpha)':<15} {'Fill Rate (Pre)':<18} {'Win Rate (vs Hold)':<20}"
    print(header)
    print("-" * 100)
    
    for _, row in res_df.iterrows():
        print(f"{row['Threshold']:<10} {row['Avg Daily Ret']*100:>6.4f}%        {row['Lift (vs Hold)']*100:>6.4f}%        {row['Fill Rate']:>6.1%}            {row['Win Rate']:>6.1%}")
        
    print("-" * 100)
    
    # 2. 最佳建議
    best_row = res_df.loc[res_df['Avg Daily Ret'].idxmax()]
    print(f"\n🏆 最佳表現門檻: {best_row['Threshold']}")
    print(f"   平均日報酬: {best_row['Avg Daily Ret']*100:.4f}% (比死抱多賺 {best_row['Lift (vs Hold)']*100:.4f}%)")
    print(f"   盤前成交率: {best_row['Fill Rate']:.1%}")
    
    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, 'premarket_sell_sweep_report.csv')
    res_df.to_csv(csv_path, index=False)
    print(f"\n[Saved] 分析報告已儲存: {csv_path}")

def main():
    tickers = load_holding_tickers()
    if not tickers: return
    
    print(f"監控持倉: {len(tickers)} 檔")
    df_daily, df_intra = fetch_data(tickers)
    
    all_res = []
    uniq_tickers = df_intra.columns.levels[1] if isinstance(df_intra.columns, pd.MultiIndex) else [tickers[0]]
    
    print("開始執行參數掃描...")
    for t in uniq_tickers:
        res = backtest_sell_limit_sweep(t, df_daily, df_intra)
        all_res.extend(res)
        
    generate_report(all_res)

if __name__ == '__main__':
    main()