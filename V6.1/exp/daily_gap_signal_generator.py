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

def get_market_data(tickers):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載 {len(tickers)} 檔股票數據 (分批處理)...")
    data_map = {}
    
    # 1. 下載日線 (Batch)
    try:
        df_daily = download_data_in_batches(tickers, period="1mo", interval="1d")
        
        if df_daily.empty: return {}
        
        # 處理 MultiIndex
        if isinstance(df_daily.columns, pd.MultiIndex): 
            # 這是標準情況
            closes = df_daily['Close']
            highs = df_daily['High']
            lows = df_daily['Low']
        else:
            # 單一股票情況 (yfinance 有時會降維)
            # 為了通用性，手動轉回 DataFrame
            closes = pd.DataFrame({tickers[0]: df_daily['Close']})
            highs = pd.DataFrame({tickers[0]: df_daily['High']})
            lows = pd.DataFrame({tickers[0]: df_daily['Low']})
            
    except Exception as e:
        print(f"[Error] 日線下載失敗: {e}")
        return {}

    # 2. 下載盤前 (Batch)
    try:
        df_intraday = download_data_in_batches(tickers, period="5d", interval="1m", prepost=True)
        
        if df_intraday.empty: 
            # 盤前數據失敗不應阻擋主流程，回傳已有的日線數據即可
            # 但需標記無盤前
            pass
        else:
            if df_intraday.index.tz is None:
                df_intraday.index = df_intraday.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df_intraday.index = df_intraday.index.tz_convert('America/New_York')
            
        current_date = df_intraday.index[-1].date() if not df_intraday.empty else date.today()
        
    except Exception as e:
        print(f"[Error] 分時數據下載失敗: {e}")
        return {}

    # 3. 整合數據
    for ticker in tickers:
        try:
            # --- 日線處理 ---
            if ticker not in closes.columns: 
                # 可能下載失敗或 Delisted
                continue
            
            c = closes[ticker].dropna()
            h = highs[ticker].dropna()
            l = lows[ticker].dropna()
            
            if len(c) < 15: continue
            prev_close = float(c.iloc[-1])
            
            # ATR 計算
            tr = h - l 
            atr = tr.rolling(14).mean().iloc[-1]
            atr_pct = atr / prev_close if prev_close > 0 else 0

            # --- 盤前處理 ---
            curr_price = np.nan
            pre_high = np.nan
            
            if not df_intraday.empty and ticker in df_intraday['Close'].columns:
                series_c = df_intraday['Close'][ticker]
                # 嘗試獲取 High，若無則用 Close
                if 'High' in df_intraday.columns and ticker in df_intraday['High'].columns:
                    series_h = df_intraday['High'][ticker]
                else:
                    series_h = series_c
                
                # 篩選今日
                today_mask = series_c.index.date == current_date
                today_close = series_c[today_mask]
                today_high = series_h[today_mask]
                
                if not today_close.empty:
                    curr_price = float(today_close.iloc[-1])
                    pre_high = float(today_high.max())

            # --- Pre-Fade 計算 ---
            pre_fade = 0.0
            if pd.notna(pre_high) and pre_high > 0 and pd.notna(curr_price):
                pre_fade = (pre_high - curr_price) / pre_high

            data_map[ticker] = {
                'prev_close': prev_close, 
                'curr_price': curr_price, 
                'pre_high': pre_high, 
                'pre_fade': pre_fade, 
                'atr_pct': atr_pct
            }
        except Exception:
            continue
            
    return data_map

def generate_live_dashboard():
    print(f"\n>>> V6.1 Gap Strategy Dashboard (Optimized)")
    print(f">>> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 載入清單
    pool_toxic = load_tickers_from_json(TOXIC_POOL_FILE)
    pool_asset = load_tickers_from_json(ASSET_POOL_FILE)
    pool_sensitive = load_tickers_from_json(SENSITIVE_POOL_FILE)
    
    all_tickers = list(set(pool_toxic + pool_asset + pool_sensitive))
    # 排除黑名單
    valid_tickers = [t for t in all_tickers if t not in MOMENTUM_BLACKLIST]
    
    print(f"清單概況:")
    print(f"  - Asset Pool (A): {len(pool_asset)} 檔")
    print(f"  - Toxic Pool (T): {len(pool_toxic)} 檔")
    print(f"  - Sensitive Pool (S): {len(pool_sensitive)} 檔")
    print(f"  - 監控總數: {len(valid_tickers)} 檔")

    # 2. 取得環境狀態
    is_totm, is_pre_holiday, cal_status_str = get_calendar_status()
    eth_ret, eth_status, eth_light = get_crypto_sentiment()
    
    print(f"\n[Market Context]")
    print(f"  📅 Calendar: {cal_status_str}")
    
    if is_totm:
        print(f"     👉 Asset Pool: ⚠️ 暫停交易 (月初法人買盤)")
        print(f"     👉 Toxic Pool: 🔥 積極交易 (資金再平衡效應)")
    if is_pre_holiday:
        print(f"     👉 All Pools : ⚠️ 節前量縮 (小心假訊號)")

    if eth_status != "Weekday":
        print(f"  🪙 Crypto: ETH {eth_ret*100:+.2f}% {eth_light}")
        if eth_status == "RED":
            print(f"     👉 Toxic/Sensitive: ⛔ 暫停交易 (ETH > 5% 暴漲)")
    else:
        print(f"  🪙 Crypto: 平日模式 (無週末濾網)")

    # 3. 取得數據 (已優化)
    market_data = get_market_data(valid_tickers)
    
    # 檢查是否有數據回傳
    if not market_data:
        print("\n[Error] 無法獲取任何市場數據，請檢查網路連線或代碼清單。")
        return

    report_data = []
    
    for ticker in valid_tickers:
        if ticker not in market_data: continue
        data = market_data[ticker]
        
        curr_price = data['curr_price']
        prev_close = data['prev_close']
        
        # 過濾掉無效數據
        if pd.isna(curr_price) or prev_close <= 0: continue
        
        gap_pct = (curr_price - prev_close) / prev_close
        
        # 只看 Gap Up
        if gap_pct <= 0: continue
            
        # 分類與邏輯
        if ticker in pool_toxic:
            category = "Toxic"; cat_code = "T"
        elif ticker in pool_sensitive:
            category = "Sensitive"; cat_code = "S"
        else:
            category = "Asset"; cat_code = "A"
            
        atr_pct = data['atr_pct']
        pre_fade = data['pre_fade']
        
        # 動態門檻
        if category in ["Toxic", "Sensitive"]:
            dynamic_threshold = max(DEFAULT_GAP_THRESHOLD, 0.3 * atr_pct)
        else:
            dynamic_threshold = DEFAULT_GAP_THRESHOLD
            
        # 訊號判斷
        status = "WAIT"
        score = 0
        
        if gap_pct > dynamic_threshold:
            if category in ["Toxic", "Sensitive"] and eth_status == "RED":
                status = "✋ HOLD (ETH)"; score = -2
            elif category == "Asset" and (is_totm or is_pre_holiday):
                status = "✋ SKIP (Calendar)"; score = -1
            elif category in ["Toxic", "Sensitive"] and is_totm:
                if pre_fade > FADE_THRESHOLD_PCT:
                    status = "🔥🔥 TOTM SELL"; score = 4
                else:
                    status = "🔥 TOTM (Fade?)"; score = 2
            else:
                if category in ["Toxic", "Sensitive"] and eth_status == "YELLOW":
                    if pre_fade > FADE_THRESHOLD_PCT:
                        status = "⚠️ RISKY SELL"; score = 1
                    else:
                        status = "WAIT (Yellow)"; score = 0
                elif category in ["Toxic", "Sensitive"] and is_pre_holiday:
                     if pre_fade > FADE_THRESHOLD_PCT:
                        status = "⚠️ Holiday SELL"; score = 1
                     else:
                        status = "WAIT (Holiday)"; score = 0
                else:
                    if pre_fade > FADE_THRESHOLD_PCT:
                        status = "🔴 STRONG SELL"; score = 3
                    else:
                        status = "🔴 SELL"; score = 2
        
        report_data.append({
            'Ticker': ticker, 'Cat': cat_code,
            'Gap%': gap_pct, 'Thres%': dynamic_threshold,
            'Fade%': pre_fade, 'ATR%': atr_pct,
            'Price': curr_price, 'Status': status, 'Score': score
        })
            
    # 4. 輸出報表
    if not report_data:
        print("\n無 Gap > 0 標的。")
        return

    df = pd.DataFrame(report_data)
    df.sort_values(by=['Score', 'Gap%'], ascending=[False, False], inplace=True)
    
    print("\n" + "="*95)
    print(f"{'Ticker':<6} {'Cat':<3} {'Gap%':>7} {'Thres%':>7} {'Fade%':>7} {'ATR%':>6} {'Price':>8} {'Status':<20}")
    print("-" * 95)
    
    for _, row in df.iterrows():
        mark = ">>" if row['Score'] >= 2 else "  "
        if row['Score'] < 0: mark = "XX"
        
        print(f"{mark} {row['Ticker']:<6} {row['Cat']:<3} "
              f"{row['Gap%']*100:>6.2f}% {row['Thres%']*100:>6.2f}% "
              f"{row['Fade%']*100:>6.2f}% {row['ATR%']*100:>5.1f}% "
              f"{row['Price']:>8.2f} {row['Status']:<20}")
    print("="*95)

    outfile = os.path.join(OUTPUT_DIR, f'gap_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(outfile, index=False)
    print(f"\n[Saved] {outfile}")

if __name__ == '__main__':
    try:
        generate_live_dashboard()
    except KeyboardInterrupt:
        print("\nStopped.")