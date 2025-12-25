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
# 假設資源檔路徑
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 參數
GAP_THRESHOLD = 0.005  # 0.5%
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# --- 工具函數 ---

def load_holding_tickers():
    """讀取 Holding Pool (監控清單)"""cd 
    path = os.path.join(RESOURCE_DIR, HOLDING_POOL_FILE)
    # 相容性檢查
    if not os.path.exists(path):
        path = path.replace('V6.1', 'V6.0')
    
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw]))
    else:
        print(f"[Warning] 找不到 {HOLDING_POOL_FILE}，使用預設清單")
        return ['NVDA', 'TSLA', 'AAPL', 'AMD', 'PLTR', 'MSTR']

def fetch_data(tickers):
    """取得日線 (計算昨收) 與 分時線 (模擬盤前掛單)"""
    print(f"1. 下載日線資料 (基準)...")
    df_daily = yf.download(tickers, period="3mo", interval="1d", auto_adjust=True, progress=False, threads=True)
    
    print(f"2. 下載盤前分時資料 (最近59天, 5分K)...")
    df_intra = yf.download(tickers, period="59d", interval="5m", prepost=True, auto_adjust=True, progress=True, threads=True)
    
    return df_daily, df_intra

def backtest_sell_limit(ticker, daily_data, intra_data):
    """
    回測核心：持有股票，比較不同賣出策略
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

    # 時區轉換
    try:
        df.index = df.index.tz_convert('America/New_York')
    except TypeError:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')

    # 準備日線 (查找 Prev Close)
    d_data = daily_data.xs(ticker, axis=1, level=1).copy() if isinstance(daily_data.columns, pd.MultiIndex) else daily_data.copy()
    d_data.index = pd.to_datetime(d_data.index).date
    
    results = []
    dates = sorted(list(set(df.index.date)))
    
    for d in dates:
        # 1. 取得昨日收盤價 (持有成本基準)
        try:
            loc = d_data.index.get_loc(d)
            if loc == 0: continue
            prev_close = float(d_data.iloc[loc-1]['Close'])
            close_today = float(d_data.iloc[loc]['Close']) # 若持有到收盤的價格
        except: continue
            
        # 2. 設定賣出目標 (Sell Limit)
        target_price = prev_close * (1 + GAP_THRESHOLD)
        
        # 3. 取得當日數據
        day_bars = df[df.index.date == d]
        if day_bars.empty: continue
        
        market_open_time = dt_time(9, 30)
        pre_market = day_bars[day_bars.index.time < market_open_time]
        regular_market = day_bars[day_bars.index.time >= market_open_time]
        
        if regular_market.empty: continue
        open_price = float(regular_market.iloc[0]['Open'])

        # --- 策略 A: 盤前掛單 (Pre-market Sell Limit) ---
        # 假設：我们在盤前就掛 Sell Limit @ Target
        
        exec_price_pre = None
        filled_in_pre = False
        
        # 檢查盤前是否觸發
        if not pre_market.empty:
            # 如果盤前最高價 >= Target，假設成交
            mask = pre_market['High'] >= target_price
            if mask.any():
                filled_in_pre = True
                first_bar = pre_market[mask].iloc[0]
                # 成交價邏輯：如果是限價單(Limit)，成交在 Target 或更好
                # 但保守起見，如果它跳空過 Target，我們算它成交在 max(Target, Open_of_bar)
                # 不過通常 Sell Limit 就是成交在 Target (除非流動性極佳)
                # 這裡假設成交在 Target (鎖定獲利)
                exec_price_pre = target_price 
                
                # 修正：如果該根 K 棒的 Open 遠高於 Target (例如消息面大漲)
                # 我們的 Limit 單會以較佳價格成交嗎？會的。
                if first_bar['Open'] > target_price:
                    exec_price_pre = first_bar['Open']

        # 如果盤前沒成交，進入盤中 (Open)
        # 此時單子還掛著。如果 Open > Target，會以 Open 成交
        if not filled_in_pre:
            if open_price >= target_price:
                exec_price_pre = open_price
            else:
                # 盤前沒賣掉，開盤也沒到 -> 策略失敗，繼續持有到收盤 (或是盤中觸發?)
                # 這裡簡單假設：若沒Gap則持有到收盤
                # (或者您可以模擬盤中觸發，但這裡主要比對盤前優勢)
                exec_price_pre = close_today 

        # --- 策略 B: 堅持等到開盤 (Wait for Open) ---
        # 邏輯：看到開盤價才決定賣不賣
        
        exec_price_wait = None
        if open_price >= target_price:
            exec_price_wait = open_price # 成功 Gap Up，賣出
        else:
            exec_price_wait = close_today # 沒 Gap，持有到收盤
            
        # --- 比較基準 ---
        # Buy & Hold (Hold till Close): 價格 = close_today
        
        # 計算相對於昨收的當日報酬 (Day Return)
        # 用來衡量「今天賺了多少 %」
        ret_pre = (exec_price_pre - prev_close) / prev_close
        ret_wait = (exec_price_wait - prev_close) / prev_close
        ret_hold = (close_today - prev_close) / prev_close
        
        results.append({
            'Date': d,
            'Ticker': ticker,
            'Prev_Close': prev_close,
            'Target': target_price,
            'Close': close_today,
            'Open': open_price,
            'Pre_Filled': filled_in_pre,
            'Ret_Pre_Limit': ret_pre,
            'Ret_Wait_Open': ret_wait,
            'Ret_Hold_Close': ret_hold
        })
        
    return results

def generate_report(trades):
    if not trades:
        print("沒有產生交易紀錄。")
        return
        
    df = pd.DataFrame(trades)
    
    print(f"\n=== [V6.1 修正版] 持倉止盈策略回測 (最近 60 天) ===")
    print(f"情境: 持有股票，目標獲利 +{GAP_THRESHOLD*100}% (Sell Limit)")
    print("-" * 80)
    
    # 統計平均每日報酬 (Average Daily Return on Holdings)
    # 這代表「如果有這檔股票，採用此策略平均每天能多賺/少賠多少」
    avg_pre = df['Ret_Pre_Limit'].mean()
    avg_wait = df['Ret_Wait_Open'].mean()
    avg_hold = df['Ret_Hold_Close'].mean()
    
    print(f"{'Strategy':<30} {'Avg Daily Return':<15} {'Win Rate (vs Hold)':<20}")
    print("-" * 80)
    print(f"{'1. Pre-market Limit Sell':<30} {avg_pre*100:>6.4f}% {'-':<20}")
    print(f"{'2. Wait for Open Sell':<30} {avg_wait*100:>6.4f}% {(df['Ret_Wait_Open'] > df['Ret_Hold_Close']).mean():.1%}")
    print(f"{'3. Hold till Close (Base)':<30} {avg_hold*100:>6.4f}% {'-':<20}")
    
    print("\n[關鍵差異分析]")
    
    # 1. 盤前偷跑成功率 (Pre-market Fill Rate)
    # 多少比例的日子，我們在盤前就順利止盈出場了？
    fill_rate = df['Pre_Filled'].mean()
    print(f"👉 盤前掛單成交率 (Fill Rate): {fill_rate:.1%}")
    
    # 2. 盤前賣對了嗎？ (Pre-market vs Close)
    # 在盤前成交的日子裡，賣出價是否高於收盤價？(賣在高點 vs 賣飛)
    filled_df = df[df['Pre_Filled'] == True]
    if not filled_df.empty:
        sold_higher = (filled_df['Ret_Pre_Limit'] > filled_df['Ret_Hold_Close']).mean()
        print(f"👉 在盤前成交的日子裡，有 {sold_higher:.1%} 的機率賣得比收盤價好 (成功止盈)。")
        
        # 額外比較：盤前賣 vs 開盤賣
        # 有時候盤前衝高，開盤就掉下來了 (Fade)。這就是盤前掛單的最大優勢。
        better_than_open = (filled_df['Ret_Pre_Limit'] > filled_df['Ret_Wait_Open']).mean()
        print(f"👉 相比等到開盤，盤前掛單有 {better_than_open:.1%} 的機率賣得更高 (避開開盤下跌)。")
    
    # 3. 盤前掛單的風險 (Missed Upside)
    # 賣掉後股票繼續噴出 (Gap and Go)
    missed_gains = filled_df[filled_df['Ret_Hold_Close'] > filled_df['Ret_Pre_Limit']]
    if not missed_gains.empty:
        avg_miss = (missed_gains['Ret_Hold_Close'] - missed_gains['Ret_Pre_Limit']).mean()
        print(f"👉 風險: 在 {len(missed_gains)} 次交易中賣飛了，錯失平均 {avg_miss*100:.2f}% 的後續漲幅。")

    csv_path = os.path.join(OUTPUT_DIR, 'premarket_sell_limit_backtest.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n詳細報表已儲存: {csv_path}")

def main():
    tickers = load_holding_tickers()
    if not tickers:
        return
    
    print(f"監控持倉: {len(tickers)} 檔 (e.g., {tickers[:3]})")
    df_daily, df_intra = fetch_data(tickers)
    
    all_res = []
    uniq_tickers = df_intra.columns.levels[1] if isinstance(df_intra.columns, pd.MultiIndex) else [tickers[0]]
    
    print("開始回測...")
    for t in uniq_tickers:
        res = backtest_sell_limit(t, df_daily, df_intra)
        all_res.extend(res)
        
    generate_report(all_res)

if __name__ == '__main__':
    main()