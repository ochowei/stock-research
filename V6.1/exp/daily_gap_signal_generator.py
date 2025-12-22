import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, date
import json
import time
import logging
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# --- 0. 抑制 yfinance 的雜訊 ---
# yfinance 的錯誤訊息有時會直接 print 到 stderr，這裡將其 logger 級別調高，只顯示 Critical
logger = logging.getLogger('yfinance')
logger.setLevel(logging.CRITICAL)

# --- 1. 設定與參數 ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 來源檔案
ASSET_POOL_FILE = '2025_final_asset_pool.json'
TOXIC_POOL_FILE = '2025_final_toxic_asset_pool.json'
SENSITIVE_POOL_FILE = '2025_final_crypto_sensitive_pool.json'

# 動能股黑名單 (Momentum Blacklist)
MOMENTUM_BLACKLIST = [
    'NVDA', 'APP', 'NET', 'ANET', 'AMD', 'MSFT', 'GOOG', 'AMZN', 
    'LLY', 'NVO', 'V', 'MCD', 'IBM', 'QCOM', 'SMCI', 'PLTR', 'COIN', 'MSTR'
]

# 策略參數
DEFAULT_GAP_THRESHOLD = 0.005  # 0.5%
FADE_THRESHOLD_PCT = 0.010     # 1.0%
CRYPTO_YELLOW_THRESHOLD = 0.01 # 1%
CRYPTO_RED_THRESHOLD = 0.05    # 5%

# 下載設定
BATCH_SIZE = 20  # 每次下載 20 檔，避免 Timeout
MAX_RETRIES = 2  # 失敗重試次數

# --- 2. 工具函數 ---

def load_tickers_from_json(filename):
    path = os.path.join(RESOURCE_DIR, filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        cleaned_list = [t.split(':')[-1].strip().replace('.', '-') for t in raw_list]
        return list(set(cleaned_list))
    except Exception as e:
        print(f"[Error] 無法讀取清單 {filename}: {e}")
        return []

def get_calendar_status(target_date=None):
    """判斷指定日期 (預設今日) 是否為 TOTM 或 Pre-Holiday"""
    if target_date is None:
        target_date = datetime.now().date()
    
    start_date = target_date - timedelta(days=35)
    end_date = target_date + timedelta(days=35)
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    
    df = pd.DataFrame(index=dates)
    
    # 計算 TOTM
    date_series = df.index.to_series()
    groups = date_series.groupby(date_series.dt.to_period('M'))
    totm_dates = []
    for period, dates_in_month in groups:
        days = dates_in_month.index
        if len(days) < 4: continue
        totm_dates.append(days[-1])
        totm_dates.extend(days[:3])
        
    is_totm = target_date in [d.date() for d in totm_dates]

    # 計算 Pre-Holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    is_pre_holiday = False
    
    # 找 target_date 在交易日曆的 index
    try:
        loc = dates.get_loc(pd.Timestamp(target_date))
        if loc < len(dates) - 1:
            next_trade_day = dates[loc + 1].date()
            # 檢查下一個交易日之前是否有假期
            # 簡單邏輯：若下一個交易日 > target_date + 1 (且非週末)，通常意味著中間有假期
            # 但更準確的是直接比對 holidays
            # 這裡採用：如果明天不是交易日，且明天是 Holiday (或明天週六但週五是 Holiday)
            # 為了簡化且準確：檢查 target_date + 1 是否為 holiday
            tomorrow = target_date + timedelta(days=1)
            if tomorrow in holidays:
                is_pre_holiday = True
            # 或者：如果下一個交易日跟今天差超過 3 天 (週末是 3 天，長週末是 4 天)
            elif (next_trade_day - target_date).days > 3:
                is_pre_holiday = True
    except KeyError:
        pass 

    status_parts = []
    if is_totm: status_parts.append("TOTM(月初)")
    if is_pre_holiday: status_parts.append("Pre-Holiday(節前)")
    
    status_str = " + ".join(status_parts) if status_parts else "Normal(一般日)"
    
    return is_totm, is_pre_holiday, status_str

def get_crypto_sentiment():
    """回傳: (漲跌幅, 狀態, Emoji)"""
    if datetime.now().weekday() != 0: 
        return 0.0, "Weekday", "⚪"

    try:
        # 消除 FutureWarning: auto_adjust=False
        df = yf.download("ETH-USD", period="5d", interval="1h", progress=False, auto_adjust=False)
        
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

    except Exception:
        return 0.0, "Error", "⚪"

def download_data_in_batches(tickers, period, interval, prepost=False):
    """
    分批下載數據以避免 Timeout
    """
    all_data = []
    total = len(tickers)
    
    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        # 簡單進度顯示
        # print(f"  Downloading batch {i//BATCH_SIZE + 1}/{(total-1)//BATCH_SIZE + 1} ({len(batch)} tickers)...")
        
        for attempt in range(MAX_RETRIES):
            try:
                # 關鍵修正：加入 auto_adjust=False, threads=True (加速)
                df = yf.download(
                    batch, 
                    period=period, 
                    interval=interval, 
                    prepost=prepost, 
                    progress=False, 
                    auto_adjust=False,
                    threads=True
                )
                
                # yfinance 有時回傳空 DataFrame
                if not df.empty:
                    all_data.append(df)
                break # 成功則跳出重試
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1) # 稍作休息後重試
                else:
                    print(f"  [Warning] Batch failed: {batch[0]}... {e}")

    if not all_data:
        return pd.DataFrame()
        
    # 合併數據
    # 注意：yfinance 的 MultiIndex 行為
    # 如果只有一個 batch 且只有一檔股票，結構可能不同，這裡嘗試通用的 concat
    try:
        # 如果是多個 batch，需要合併
        if len(all_data) == 1:
            return all_data[0]
        
        # 針對 Column 進行合併 (Date index 是相同的)
        # yfinance download 多檔股票時，Columns 是 (Price Type, Ticker)
        # 我們需要水平合併 (axis=1)
        full_df = pd.concat(all_data, axis=1)
        return full_df
    except Exception as e:
        print(f"  [Error] Data merge failed: {e}")
        return pd.DataFrame()

# --- 替換部分開始 ---

def get_market_data(tickers):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載 {len(tickers)} 檔股票數據...")
    data_map = {}
    
    # 1. 下載數據 (為了簡化與穩定，直接抓 5 天的 1m 數據來找最新價格，抓 1mo 日線找昨收)
    # 下載日線 (Batch)
    try:
        df_daily = download_data_in_batches(tickers, period="1mo", interval="1d")
    except Exception as e:
        print(f"[Error] 日線下載失敗: {e}")
        return {}

    # 下載盤前 (Batch)
    try:
        df_intraday = download_data_in_batches(tickers, period="5d", interval="1m", prepost=True)
        if not df_intraday.empty:
            if df_intraday.index.tz is None:
                df_intraday.index = df_intraday.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df_intraday.index = df_intraday.index.tz_convert('America/New_York')
    except Exception as e:
        print(f"[Error] 分時數據下載失敗: {e}")
        return {}

    # 3. 整合數據
    for ticker in tickers:
        try:
            # --- A. 取得 Prev Close (昨收) ---
            # 處理 MultiIndex 或 Single Index
            if isinstance(df_daily.columns, pd.MultiIndex):
                if ticker not in df_daily['Close'].columns:
                    # print(f"  [Skip] {ticker} 無日線數據")
                    continue
                c = df_daily['Close'][ticker].dropna()
                h = df_daily['High'][ticker].dropna()
                l = df_daily['Low'][ticker].dropna()
            else:
                if ticker not in df_daily.columns: # 針對單一股票結構可能不同，這裡簡化判斷
                     # 如果只有一檔股票且沒有 MultiIndex，columns 可能直接是 'Close', 'Open'...
                     # 但 download_data_in_batches 試圖合併，通常會有 MultiIndex
                     pass
                c = df_daily['Close'].dropna()
                h = df_daily['High'].dropna()
                l = df_daily['Low'].dropna()

            if len(c) < 2: 
                continue

            prev_close = float(c.iloc[-1])
            
            # ATR 計算 (14日)
            tr = h - l 
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr / prev_close if prev_close > 0 else 0

            # --- B. 取得 Current Price (現價) ---
            # 邏輯：優先找 Intraday 最後一筆，如果沒有則用日線最後一筆
            curr_price = np.nan
            pre_high = np.nan
            
            # 嘗試從 Intraday 獲取
            if not df_intraday.empty:
                # 處理 Columns
                if isinstance(df_intraday.columns, pd.MultiIndex):
                     if ticker in df_intraday['Close'].columns:
                        series_c = df_intraday['Close'][ticker].dropna()
                        series_h = df_intraday['High'][ticker].dropna() if 'High' in df_intraday.columns else series_c
                        
                        if not series_c.empty:
                            curr_price = float(series_c.iloc[-1])
                            # 盤前高點邏輯 (簡單取最後一天的高點)
                            last_date = series_c.index[-1].date()
                            today_mask = series_c.index.date == last_date
                            pre_high = float(series_h[today_mask].max())
            
            # 如果 Intraday 沒抓到，回退使用日線 Close (代表尚未開盤或資料延遲)
            if pd.isna(curr_price):
                curr_price = prev_close
                
            # --- Pre-Fade 計算 ---
            pre_fade = 0.0
            if pd.notna(pre_high) and pre_high > 0 and pd.notna(curr_price):
                if pre_high > curr_price:
                    pre_fade = (pre_high - curr_price) / pre_high

            data_map[ticker] = {
                'prev_close': prev_close, 
                'curr_price': curr_price, 
                'pre_high': pre_high, 
                'pre_fade': pre_fade, 
                'atr_pct': atr_pct
            }
        except Exception as e:
            # print(f"  [Error] 處理 {ticker} 時發生錯誤: {e}")
            continue
            
    return data_map

def generate_live_dashboard():
    print(f"\n>>> V6.1 Gap Strategy Dashboard (Holding Monitor)")
    print(f">>> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 設定檔案路徑
    HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

    # 2. 載入清單
    pool_holding = load_tickers_from_json(HOLDING_POOL_FILE)
    pool_toxic = load_tickers_from_json(TOXIC_POOL_FILE)
    pool_sensitive = load_tickers_from_json(SENSITIVE_POOL_FILE)
    
    # 直接監控所有持倉，不使用黑名單過濾
    valid_tickers = pool_holding
    
    print(f"清單概況:")
    print(f"  - Holding Pool: {len(pool_holding)} 檔")
    
    if not valid_tickers:
        print("[Error] 持倉清單為空或讀取失敗。")
        return

    # 3. 取得環境狀態
    is_totm, is_pre_holiday, cal_status_str = get_calendar_status()
    eth_ret, eth_status, eth_light = get_crypto_sentiment()
    
    print(f"\n[Market Context]")
    print(f"  📅 Calendar: {cal_status_str}")
    if eth_status != "Weekday":
        print(f"  🪙 Crypto: ETH {eth_ret*100:+.2f}% {eth_light}")

    # 4. 取得數據
    market_data = get_market_data(valid_tickers)
    
    if not market_data:
        print("\n[Error] 無法獲取市場數據。")
        return

    report_data = []
    
    for ticker in valid_tickers:
        if ticker not in market_data: 
            # 記錄無數據的標的
            # report_data.append({'Ticker': ticker, 'Status': 'No Data', 'Score': -99})
            continue
            
        data = market_data[ticker]
        curr_price = data['curr_price']
        prev_close = data['prev_close']
        
        if prev_close <= 0: continue
        
        # 計算漲跌幅
        gap_pct = (curr_price - prev_close) / prev_close
        
        # [關鍵修改] 這裡移除了 "if gap_pct <= 0: continue"，讓所有股票都能顯示
        
        # 分類
        if ticker in pool_toxic: cat_code = "T"; category = "Toxic"
        elif ticker in pool_sensitive: cat_code = "S"; category = "Sensitive"
        else: cat_code = "A"; category = "Asset"
            
        atr_pct = data['atr_pct']
        pre_fade = data['pre_fade']
        
        # 門檻
        if category in ["Toxic", "Sensitive"]:
            dynamic_threshold = max(DEFAULT_GAP_THRESHOLD, 0.3 * atr_pct)
        else:
            dynamic_threshold = DEFAULT_GAP_THRESHOLD

        trigger_price = prev_close * (1 + dynamic_threshold)

        # 狀態判斷
        status = "Watching"
        score = 0
        
        if gap_pct > dynamic_threshold:
            status = "🔴 GAP UP"
            score = 2
            # 簡單的過濾邏輯顯示
            if category == "Asset" and (is_totm or is_pre_holiday): status += " (Skip)"
        elif gap_pct < -0.02:
            status = "🟢 GAP DOWN"
            score = -1
        elif abs(gap_pct) <= 0.001:
            status = "Flat"
            
        report_data.append({
            'Ticker': ticker, 'Cat': cat_code,
            'Gap%': gap_pct, 'Thres%': dynamic_threshold,
            'Fade%': pre_fade, 'ATR%': atr_pct,
            'Price': curr_price, 'TrigPx': trigger_price,
            'Status': status, 'Score': score
        })
            
    # 5. 輸出報表
    if not report_data:
        print("\n無數據可顯示。")
        return

    df = pd.DataFrame(report_data)
    # 依照漲跌幅排序
    df.sort_values(by=['Gap%'], ascending=False, inplace=True)
    
    print("\n" + "="*105) 
    print(f"{'Ticker':<6} {'Cat':<3} {'Gap%':>7} {'Thres%':>7} {'Fade%':>7} {'ATR%':>6} {'Price':>8} {'TrigPx':>8} {'Status':<20}")
    print("-" * 105)
    
    for _, row in df.iterrows():
        # 處理可能的 NaN
        gap_val = row['Gap%'] if pd.notna(row['Gap%']) else 0
        fade_val = row['Fade%'] if pd.notna(row['Fade%']) else 0
        
        mark = "  "
        if gap_val > row['Thres%']: mark = ">>"
        
        print(f"{mark} {row['Ticker']:<6} {row['Cat']:<3} "
              f"{gap_val*100:>6.2f}% {row['Thres%']*100:>6.2f}% "
              f"{fade_val*100:>6.2f}% {row['ATR%']*100:>5.1f}% "
              f"{row['Price']:>8.2f} {row['TrigPx']:>8.2f} {row['Status']:<20}")
    print("="*105)

    outfile = os.path.join(OUTPUT_DIR, f'holding_monitor_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(outfile, index=False)
    print(f"\n[Saved] {outfile}")

# --- 替換部分結束 ---

if __name__ == '__main__':
    try:
        generate_live_dashboard()
    except KeyboardInterrupt:
        print("\nStopped.")