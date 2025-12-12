import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import time

# --- 1. 設定與參數 ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 來源檔案 (新增 Sensitive Pool)
ASSET_POOL_FILE = '2025_final_asset_pool.json'
TOXIC_POOL_FILE = '2025_final_toxic_asset_pool.json'
SENSITIVE_POOL_FILE = '2025_final_crypto_sensitive_pool.json'

# 動能股黑名單
MOMENTUM_BLACKLIST = [
    'NVDA', 'APP', 'NET', 'ANET', 'AMD', 'MSFT', 'GOOG', 'AMZN', 
    'LLY', 'NVO', 'V', 'MCD', 'IBM', 'QCOM', 'SMCI', 'PLTR', 'COIN', 'MSTR'
    # 注意：TSLA 已移至 Sensitive Pool，這裡可以保留以防萬一，或從黑名單移除讓它受控於 Sensitive 邏輯
]

# 策略參數
DEFAULT_GAP_THRESHOLD = 0.005  # 0.5%
FADE_THRESHOLD_PCT = 0.010     # 1.0%
CRYPTO_YELLOW_THRESHOLD = 0.01 # 1%
CRYPTO_RED_THRESHOLD = 0.05    # 5%

# --- 2. 工具函數 ---

def load_tickers_from_json(filename):
    path = os.path.join(RESOURCE_DIR, filename)
    if not os.path.exists(path):
        print(f"[Info] 找不到檔案 {filename}，將建立空清單。")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        cleaned_list = [t.split(':')[-1].strip().replace('.', '-') for t in raw_list]
        return list(set(cleaned_list))
    except Exception as e:
        print(f"[Error] 無法讀取清單 {filename}: {e}")
        return []

def get_crypto_sentiment():
    """回傳: (漲跌幅, 狀態, Emoji)"""
    if datetime.now().weekday() != 0: 
        return 0.0, "Weekday", "⚪"

    try:
        print("[System] 正在檢查 ETH 週末走勢 (Crypto Filter)...")
        df = yf.download("ETH-USD", period="5d", interval="1h", progress=False)
        
        if df.empty: return 0.0, "NoData", "⚪"
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')

        now_price = float(df['Close'].iloc[-1])
        
        today = datetime.now().date()
        last_friday = today - timedelta(days=3)
        target_time = pd.Timestamp(f"{last_friday} 16:00").tz_localize('America/New_York')
        
        try:
            idx = df.index.get_indexer([target_time], method='nearest')[0]
            fri_price = float(df['Close'].iloc[idx])
        except:
            fri_price = float(df['Close'].iloc[0])
        
        if fri_price == 0: return 0.0, "Error", "⚪"

        ret = (now_price - fri_price) / fri_price
        
        if ret > CRYPTO_RED_THRESHOLD:
            return ret, "RED", "🔴"
        elif ret > CRYPTO_YELLOW_THRESHOLD:
            return ret, "YELLOW", "🟡"
        else:
            return ret, "GREEN", "🟢"

    except Exception as e:
        print(f"[Warning] Crypto 檢查失敗: {e}")
        return 0.0, "Error", "⚪"

def get_market_data(tickers):
    # (同前，省略重複代碼，保持與上一版相同)
    # ... 為了節省篇幅，這裡請直接使用上一版的 get_market_data 函數 ...
    # ... 核心邏輯是抓取日線計算 ATR 和 盤前數據 ...
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載 {len(tickers)} 檔股票數據...")
    data_map = {}
    try:
        df_daily = yf.download(tickers, period="1mo", interval="1d", progress=False)
        if isinstance(df_daily.columns, pd.MultiIndex): 
            closes, highs, lows = df_daily['Close'], df_daily['High'], df_daily['Low']
        else:
            closes, highs, lows = df_daily[['Close']], df_daily[['High']], df_daily[['Low']]
    except: return {}

    try:
        df_intraday = yf.download(tickers, period="5d", interval="1m", prepost=True, progress=False)
        if df_intraday.index.tz is None:
            df_intraday.index = df_intraday.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df_intraday.index = df_intraday.index.tz_convert('America/New_York')
        current_date = df_intraday.index[-1].date()
    except: return {}

    for ticker in tickers:
        try:
            if ticker not in closes.columns: continue
            h, l, c = highs[ticker].dropna(), lows[ticker].dropna(), closes[ticker].dropna()
            if len(c) < 15: continue
            prev_close = float(c.iloc[-1])
            tr = h - l 
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr / prev_close if prev_close > 0 else 0

            if ticker in df_intraday['Close'].columns:
                series_c = df_intraday['Close'][ticker]
                series_h = df_intraday['High'][ticker] if 'High' in df_intraday.columns else series_c
                today_mask = series_c.index.date == current_date
                today_close = series_c[today_mask]
                today_high = series_h[today_mask]
                if not today_close.empty:
                    curr_price = float(today_close.iloc[-1])
                    pre_high = float(today_high.max())
                else:
                    curr_price, pre_high = np.nan, np.nan
            else:
                curr_price, pre_high = np.nan, np.nan

            if pd.notna(pre_high) and pre_high > 0 and pd.notna(curr_price):
                pre_fade = (pre_high - curr_price) / pre_high
            else:
                pre_fade = 0.0

            data_map[ticker] = {'prev_close': prev_close, 'curr_price': curr_price, 'pre_high': pre_high, 'pre_fade': pre_fade, 'atr_pct': atr_pct}
        except: continue
    return data_map

def generate_live_dashboard():
    print(f"\n>>> V6.1 Gap Strategy Dashboard (Multi-List Support)")
    print(f">>> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 載入三份清單
    pool_toxic = load_tickers_from_json(TOXIC_POOL_FILE)
    pool_asset = load_tickers_from_json(ASSET_POOL_FILE)
    pool_sensitive = load_tickers_from_json(SENSITIVE_POOL_FILE)
    
    all_tickers = list(set(pool_toxic + pool_asset + pool_sensitive))
    valid_tickers = [t for t in all_tickers if t not in MOMENTUM_BLACKLIST]
    
    print(f"清單概況:")
    print(f"  - Asset Pool (標準): {len(pool_asset)} 檔")
    print(f"  - Toxic Pool (高毒): {len(pool_toxic)} 檔")
    print(f"  - Sensitive Pool (連動): {len(pool_sensitive)} 檔")
    print(f"  - 監控總數: {len(valid_tickers)} 檔")

    # 2. Crypto 濾網檢查
    eth_ret, eth_status, eth_light = get_crypto_sentiment()
    
    print(f"\n[Market Context]")
    if eth_status != "Weekday":
        print(f"  ETH Weekend Return: {eth_ret*100:+.2f}% {eth_light}")
        if eth_status == "RED":
            print(f"  ⚠️ [CRITICAL] ETH 暴漲 > 5%！Toxic & Sensitive Pools 暫停交易！")
        elif eth_status == "YELLOW":
            print(f"  ⚠️ [WARNING] ETH 轉強 (>1%)。高風險資產建議保守操作。")
        else:
            print(f"  ✅ [SAFE] ETH 平穩。全清單正常交易。")
    else:
        print(f"  (非週一，跳過 Crypto 濾網)")

    # 3. 取得數據
    market_data = get_market_data(valid_tickers)
    
    report_data = []
    
    for ticker in valid_tickers:
        if ticker not in market_data: continue
        data = market_data[ticker]
        
        curr_price = data['curr_price']
        prev_close = data['prev_close']
        
        if pd.isna(curr_price) or prev_close <= 0: continue
        gap_pct = (curr_price - prev_close) / prev_close
        if gap_pct <= 0: continue
            
        # 分類判斷
        if ticker in pool_toxic:
            category = "Toxic"
            cat_code = "T"
        elif ticker in pool_sensitive:
            category = "Sensitive"
            cat_code = "S" # Sensitive
        else:
            category = "Asset"
            cat_code = "A" # Asset (Standard)
            
        atr_pct = data['atr_pct']
        pre_fade = data['pre_fade']
        
        # A. 動態門檻
        # Toxic 和 Sensitive 都使用較嚴格的門檻
        if category in ["Toxic", "Sensitive"]:
            dynamic_threshold = max(DEFAULT_GAP_THRESHOLD, 0.3 * atr_pct)
        else:
            dynamic_threshold = DEFAULT_GAP_THRESHOLD
            
        # B. 判斷訊號
        status = "WAIT"
        score = 0
        
        if gap_pct > dynamic_threshold:
            # C. 應用 Crypto 濾網
            # 針對 Toxic 和 Sensitive 同步套用濾網
            if category in ["Toxic", "Sensitive"] and eth_status == "RED":
                status = "✋ HOLD (ETH)"
                score = -1
            elif category in ["Toxic", "Sensitive"] and eth_status == "YELLOW":
                if pre_fade > FADE_THRESHOLD_PCT:
                    status = "⚠️ RISKY SELL"
                    score = 1
                else:
                    status = "WAIT (Yellow)"
            else:
                if pre_fade > FADE_THRESHOLD_PCT:
                    status = "🔴 STRONG SELL"
                    score = 3
                else:
                    status = "🔴 SELL"
                    score = 2
        
        report_data.append({
            'Ticker': ticker,
            'Cat': cat_code,
            'Gap%': gap_pct,
            'Thres%': dynamic_threshold,
            'Fade%': pre_fade,
            'ATR%': atr_pct,
            'Price': curr_price,
            'Status': status,
            'Score': score
        })
            
    # 4. 輸出報表
    if not report_data:
        print("\n無 Gap > 0 標的。")
        return

    df = pd.DataFrame(report_data)
    df.sort_values(by=['Score', 'Gap%'], ascending=[False, False], inplace=True)
    
    print("\n" + "="*85)
    print(f"{'Ticker':<6} {'Cat':<3} {'Gap%':>7} {'Thres%':>7} {'Fade%':>7} {'ATR%':>6} {'Price':>8} {'Status':<15}")
    print("-" * 85)
    
    for _, row in df.iterrows():
        mark = ">>" if row['Score'] >= 2 else "  "
        print(f"{mark} {row['Ticker']:<6} {row['Cat']:<3} "
              f"{row['Gap%']*100:>6.2f}% {row['Thres%']*100:>6.2f}% "
              f"{row['Fade%']*100:>6.2f}% {row['ATR%']*100:>5.1f}% "
              f"{row['Price']:>8.2f} {row['Status']:<15}")
    print("="*85)

    # 5. 存檔
    outfile = os.path.join(OUTPUT_DIR, f'gap_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(outfile, index=False)
    print(f"\n[Saved] {outfile}")

if __name__ == '__main__':
    try:
        generate_live_dashboard()
    except KeyboardInterrupt:
        print("\nStopped.")