import os
import sys
import json
import time
import logging
import joblib
import warnings
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from pandas.tseries.offsets import BDay

# --- 設定 ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 指向資源目錄
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 模型路徑設定 (三模型)
SELL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')
MOM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'momentum_model.joblib')
DIP_MODEL_PATH = os.path.join(OUTPUT_DIR, 'dip_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案設定
ASSET_POOL_FILE = '2025_final_asset_pool.json'
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'
TOXIC_POOL_FILE = '2025_final_toxic_asset_pool.json' # 雖然用動態偵測，但仍可讀取作為參考

# 策略參數
DEFAULT_GAP_THRESHOLD = 0.005 # 0.5% (預設 Gap Up 賣出門檻)
RIP_THRESHOLD = 0.03          # 3.0% (Sell Rip 賣出門檻)
DIP_THRESHOLD = 0.03          # 3.0% (Buy Dip 買進門檻 - 取絕對值)
AI_CONFIDENCE_LV = 0.50       # Sell AI 信心門檻
MOMENTUM_THRESHOLD = 0.53     # Momentum AI 信心門檻 (53%)
DIP_CONFIDENCE_LV = 0.50      # Dip AI 信心門檻

# [新增] Crypto 濾網參數
CRYPTO_TICKER = 'ETH-USD'
CORR_WINDOW = 60          # 計算相關係數的窗口 (60天)
CORR_THRESHOLD = 0.50     # 相關係數門檻 (大於此值視為高度連動)
ETH_PUMP_THRESHOLD = 0.05 # ETH 短期漲幅門檻 (5% 視為 Risk-On)

# 美股主要假期 (2025-2026) 用於 Pre-Holiday 判斷
US_HOLIDAYS = [
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', 
    '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25',
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', '2026-05-25',
    '2026-06-19', '2026-07-03', '2026-09-07', '2026-11-26', '2026-12-25'
]

# --- 工具函數 ---

def load_tickers_and_tags():
    """
    讀取 Asset Pool 與 Holding Pool，並回傳聯集列表與標籤對照表。
    """
    tags_map = {}
    
    # 1. 讀取 Final Asset Pool (潛在機會)
    path_asset = os.path.join(RESOURCE_DIR, ASSET_POOL_FILE)
    if os.path.exists(path_asset):
        try:
            with open(path_asset, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                for t in raw:
                    clean_t = t.split(':')[-1].strip().replace('.', '-')
                    if clean_t not in tags_map: tags_map[clean_t] = set()
                    tags_map[clean_t].add('Asset')
        except Exception as e:
            print(f"[Error] Failed to load Asset Pool: {e}")
    
    # 2. 讀取 Holding Pool (庫存監控)
    path_holding = os.path.join(RESOURCE_DIR, HOLDING_POOL_FILE)
    if os.path.exists(path_holding):
        try:
            with open(path_holding, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                for t in raw:
                    clean_t = t.split(':')[-1].strip().replace('.', '-')
                    if clean_t not in tags_map: tags_map[clean_t] = set()
                    tags_map[clean_t].add('Held')
        except Exception as e:
            print(f"[Error] Failed to load Holding Pool: {e}")
            
    # 3. 讀取 Toxic Pool (標記參考用)
    path_toxic = os.path.join(RESOURCE_DIR, TOXIC_POOL_FILE)
    if os.path.exists(path_toxic):
        try:
            with open(path_toxic, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                for t in raw:
                    clean_t = t.split(':')[-1].strip().replace('.', '-')
                    if clean_t not in tags_map: tags_map[clean_t] = set()
                    tags_map[clean_t].add('Toxic')
        except Exception as e:
            pass

    # 4. 產生去重列表
    all_tickers = sorted(list(tags_map.keys()))
    return all_tickers, tags_map

def get_current_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        if df.empty: return 20.0
        return float(df['Close'].iloc[-1])
    except:
        return 20.0

def get_calendar_status():
    """
    實作 Survey 定義的日曆效應:
    1. TOTM (月初效應): 月底最後 1 天 ~ 月初前 3 天
    2. Pre-Holiday (節假日前夕): 假期前 1 個交易日
    """
    try:
        today = datetime.now().date()
        status_msg = "Normal (一般日)"
        is_bullish = False
        
        # --- 判斷 1: Pre-Holiday ---
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5: 
            next_day += timedelta(days=1)
            
        if next_day.strftime('%Y-%m-%d') in US_HOLIDAYS:
            return "Pre-Holiday (Bullish) 🏖️", True

        # --- 判斷 2: TOTM (Turn of the Month) ---
        current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(0)
        if current_month_end.date() < today: 
             current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(1)
        
        last_trading_day = current_month_end.date()
        next_month_start = current_month_end + BDay(1)
        first_3_days = [ (next_month_start + BDay(i)).date() for i in range(3) ]
        
        if today == last_trading_day:
            return "TOTM (Month End) 🚀", True
        elif today in first_3_days:
            return "TOTM (Month Start) 🚀", True

        return status_msg, is_bullish
        
    except Exception as e:
        print(f"[Warning] Calendar check failed: {e}")
        return "Unknown", False

# [新增] Crypto Context 函式
def get_crypto_context():
    """
    取得 Crypto 市場狀態與 ETH 歷史數據
    Returns:
        is_crypto_bullish (bool): ETH 是否顯著上漲 (Risk-On)
        eth_history (pd.Series): ETH 的歷史收盤價 (用於計算相關性)
        msg (str): 狀態描述
    """
    try:
        # 下載 ETH 數據 (抓 6 個月以確保有足夠數據計算 60 日相關性)
        df_eth = yf.download(CRYPTO_TICKER, period="6mo", interval="1d", progress=False, auto_adjust=True)
        
        if df_eth.empty:
            return False, None, "Crypto Data Missing"

        # 1. 判斷短期漲幅 (這裡取過去 2 天的累積漲幅，模擬週末效應)
        # 如果是週一執行，這裡通常能捕捉到週末的變化
        if len(df_eth) < 3:
             return False, df_eth['Close'], "Insufficient Data"

        last_close = float(df_eth['Close'].iloc[-1])
        prev_2d_close = float(df_eth['Close'].iloc[-3]) 
        eth_change = (last_close - prev_2d_close) / prev_2d_close
        
        is_bullish = False
        status_msg = f"Normal ({eth_change:.1%})"
        
        # 簡單判定：漲幅 > 5% 視為狂熱
        if eth_change > ETH_PUMP_THRESHOLD:
            is_bullish = True
            status_msg = f"🚀 PUMPING ({eth_change:.1%})"
        elif eth_change < -ETH_PUMP_THRESHOLD:
            status_msg = f"🥶 DUMPING ({eth_change:.1%})"
            
        return is_bullish, df_eth['Close'], status_msg

    except Exception as e:
        print(f"[Warning] Crypto check failed: {e}")
        return False, None, "Error"

# [新增] 相關係數計算函式
def calculate_correlation(stock_series, eth_series, window=60):
    """
    計算個股與 ETH 的最近期相關係數
    """
    try:
        if eth_series is None or len(stock_series) < window:
            return 0.0
            
        # 對齊索引 (Intersection)
        # yfinance download 的 index 都有時區，需確保一致 (通常都為 UTC 或 America/New_York)
        # 這裡做一個簡單的 timezone naive 處理以防萬一
        stock_series.index = stock_series.index.tz_convert(None)
        eth_series_naive = eth_series.copy()
        eth_series_naive.index = eth_series_naive.index.tz_convert(None)
        
        common_idx = stock_series.index.intersection(eth_series_naive.index)
        
        if len(common_idx) < window:
            return 0.0
            
        s_ret = stock_series.loc[common_idx].pct_change()
        e_ret = eth_series_naive.loc[common_idx].pct_change()
        
        # 計算滾動相關係數
        corr = s_ret.rolling(window=window).corr(e_ret).iloc[-1]
        
        if np.isnan(corr): return 0.0
        return corr
    except Exception as e:
        return 0.0

def download_data(tickers, max_retries=3):
    """
    下載數據 (含 Retry 機制)
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retrying download (Attempt {attempt+1}/{max_retries})...")
            
            # [修改] 為了計算相關性，將日線數據長度拉長到 6mo
            data = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=True, threads=True)
            intra = yf.download(tickers, period="5d", interval="1m", prepost=True, progress=False, auto_adjust=True, threads=True)
            
            if not data.empty and not intra.empty:
                return data, intra
            else:
                print(f"[Warning] Downloaded data is empty. Waiting...")
                time.sleep(2)
                
        except Exception as e:
            print(f"[Error] Download failed: {e}")
            time.sleep(2)
            
    print("[Error] Max retries reached. Returning empty data.")
    return pd.DataFrame(), pd.DataFrame()

def calculate_metrics(ticker, df_daily, df_intra, vix_val):
    try:
        # 清理並確保數值型別
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df_daily.columns:
                df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
        
        df_daily = df_daily.dropna()
        if len(df_daily) < 20: return None

        prev_close = float(df_daily['Close'].iloc[-1])
        curr_price = prev_close # Default
        
        # 嘗試從 Intraday 獲取最新價格
        if isinstance(df_intra.columns, pd.MultiIndex):
            if ticker in df_intra.columns.levels[1]:
                df_m = df_intra.xs(ticker, axis=1, level=1).dropna()
                if not df_m.empty:
                    curr_price = float(df_m['Close'].iloc[-1])
        else:
            df_m = df_intra.dropna()
            if not df_m.empty:
                curr_price = float(df_m['Close'].iloc[-1])

        gap_pct = (curr_price - prev_close) / prev_close
        atr = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14).iloc[-1]
        atr_pct = atr / prev_close
        rsi = ta.rsi(df_daily['Close'], length=14).iloc[-1]
        
        vol_ma20 = df_daily['Volume'].rolling(20).mean().iloc[-2]
        vol_last = df_daily['Volume'].iloc[-1]
        vol_ratio = vol_last / vol_ma20 if vol_ma20 > 0 else 1.0
        
        # Dist_MA20
        ma19_prev = df_daily['Close'].tail(19).mean()
        ma20_sim = (ma19_prev * 19 + curr_price) / 20
        dist_ma20 = (curr_price / ma20_sim) - 1

        features = pd.DataFrame([[rsi, atr_pct, vol_ratio, gap_pct, vix_val, dist_ma20]], 
                                columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20'])
        
        return {
            'price': curr_price,
            'prev_close': prev_close,
            'gap_pct': gap_pct,
            'features': features,
            'atr_pct': atr_pct
        }
    except Exception as e:
        return None

# --- 主程式 ---

def generate_report():
    # 0. 取得環境狀態 (Calendar & Crypto)
    cal_status, is_bullish_cal = get_calendar_status()
    is_crypto_bullish, eth_history, crypto_msg = get_crypto_context()
    
    # [動態調整] 若為 TOTM 或 Pre-Holiday，提高 Gap 門檻至 1.0%
    current_gap_threshold = DEFAULT_GAP_THRESHOLD
    if is_bullish_cal:
        current_gap_threshold = 0.01 # 1.0%

    print(f"\n>>> 6.1.3 Daily Gap & Dip Scanner (With Crypto Filter)")
    print(f">>> Target: Holdings + Asset Pool")
    print(f">>> Thresholds: Sell Rip > {RIP_THRESHOLD:.1%}, Buy Dip < -{DIP_THRESHOLD:.1%}")
    print(f">>> Gap Up Threshold: {current_gap_threshold:.1%} (Default: {DEFAULT_GAP_THRESHOLD:.1%})")
    print(f">>> AI Thresholds: Mom > {MOMENTUM_THRESHOLD:.0%}, Dip > {DIP_CONFIDENCE_LV:.0%}")
    print(f">>> [Market Context] 📅 Calendar: {cal_status}")
    print(f">>> [Market Context] 🪙 Crypto:   {crypto_msg}")
    
    if is_bullish_cal:
        print(f">>> [Strategy Adjustment] Calendar Bullish! Raising Sell Threshold.")
    if is_crypto_bullish:
        print(f">>> [Strategy Adjustment] Crypto Pump Detected! Activating Correlation Filter (> {CORR_THRESHOLD}).")
        
    print(f">>> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 載入模型
    sell_model = None
    mom_model = None
    dip_model = None
    
    try:
        if os.path.exists(SELL_MODEL_PATH): sell_model = joblib.load(SELL_MODEL_PATH)
    except: pass
    try:
        if os.path.exists(MOM_MODEL_PATH): mom_model = joblib.load(MOM_MODEL_PATH)
    except: pass
    try:
        if os.path.exists(DIP_MODEL_PATH): dip_model = joblib.load(DIP_MODEL_PATH)
    except: pass

    # 2. 載入清單
    tickers, tags_map = load_tickers_and_tags()
    if not tickers: return

    print(f"Scanning {len(tickers)} tickers...")
    curr_vix = get_current_vix()
    
    # 3. 下載數據
    daily_data, intra_data = download_data(tickers)
    if daily_data.empty: return

    # 4. 處理數據格式
    try:
        if isinstance(daily_data.columns, pd.MultiIndex):
            # 這是 yfinance 新版格式 (Ticker, Date)
            # 為了方便 groupby，轉換為 DataFrame
            daily_data = daily_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        else:
            # 單一 ticker
            daily_data['Ticker'] = tickers[0]
            daily_data = daily_data.reset_index()

        daily_map = {t: g.set_index('Date') for t, g in daily_data.groupby('Ticker')}
    except Exception as e: 
        print(f"[Error] Data formatting failed: {e}")
        return
    
    results = []
    
    # 5. 計算訊號
    for t in tickers:
        if t not in daily_map: continue
        
        # 取得個股歷史 Close (用於計算 Crypto 相關性)
        stock_close_series = daily_map[t]['Close']
        
        metrics = calculate_metrics(t, daily_map[t], intra_data, curr_vix)
        if not metrics: continue
        
        gap = metrics['gap_pct']
        price = metrics['price']
        feats = metrics['features']
        
        # 計算 Crypto 相關性 (Dynamic Beta)
        crypto_corr = 0.0
        if eth_history is not None:
            crypto_corr = calculate_correlation(stock_close_series, eth_history, window=CORR_WINDOW)
            
        # 判斷是否為敏感股
        is_crypto_sensitive = (crypto_corr > CORR_THRESHOLD)
        
        # AI 預測
        sell_prob_str = "-"
        if sell_model:
            try:
                sell_prob = sell_model.predict_proba(feats.iloc[:, :5])[0][1]
                sell_prob_str = f"{sell_prob:.0%}"
            except: pass
            
        mom_prob_str = "-"
        mom_prob = 0.0
        if mom_model:
            try:
                mom_prob = mom_model.predict_proba(feats.iloc[:, :5])[0][1]
                mom_prob_str = f"{mom_prob:.0%}"
            except: pass

        dip_prob_str = "-"
        dip_prob = 0.0
        if dip_model:
            try:
                dip_prob = dip_model.predict_proba(feats)[0][1]
                dip_prob_str = f"{dip_prob:.0%}"
            except: pass

        # --- 核心策略邏輯 6.1.3 ---
        status = "Flat"
        action = "WAIT"
        
        if gap > RIP_THRESHOLD:
            if mom_prob > MOMENTUM_THRESHOLD:
                status = "🚀 ROCKET"
                action = "HOLD/BUY"
            else:
                # [Crypto Filter Check]
                if is_crypto_bullish and is_crypto_sensitive:
                    status = f"⚠️ ETH FILTER ({crypto_corr:.2f})"
                    action = "WAIT/SKIP"
                else:
                    status = "🔴 SELL RIP"
                    action = "STRONG SELL"
                    if is_bullish_cal:
                        status += "(Bull)"
                        action = "TRIM ONLY"
                
        elif gap > current_gap_threshold:
            if mom_prob > MOMENTUM_THRESHOLD:
                status = "🟢 MOMENTUM"
                action = "HOLD"
            else:
                # [Crypto Filter Check]
                if is_crypto_bullish and is_crypto_sensitive:
                    status = f"⚠️ ETH FILTER ({crypto_corr:.2f})"
                    action = "WAIT/SKIP"
                else:
                    status = "🔴 GAP UP"
                    action = "SELL/TRIM"
                    if is_bullish_cal:
                        status += "(Bull)"
        
        # TOTM 區間的特殊處理
        elif gap > DEFAULT_GAP_THRESHOLD and is_bullish_cal:
            status = "🟡 HOLD (TOTM)"
            action = "RIDE GAP"
                
        elif gap < -DIP_THRESHOLD:
            if dip_prob > DIP_CONFIDENCE_LV:
                status = "🟢 SMART DIP"
                action = "BUY OPEN"
                if is_bullish_cal:
                    status += " (Aggr)"
            else:
                status = "🔵 WEAK DIP"
                action = "WATCH"
                
        elif gap < -DEFAULT_GAP_THRESHOLD:
            status = "🟡 GAP DOWN"
            action = "HOLD"
        else:
            status = "⚪ Flat"
            action = "-"
        
        t_tags = tags_map.get(t, set())
        is_held = 'Held' in t_tags
        tag_str = "[HOLD]" if is_held else "" 
        
        results.append({
            'Ticker': t,
            'Tag': tag_str,
            'Gap%': gap,
            'Price': price,
            'Status': status,
            'Action': action,
            'Sell%': sell_prob_str, 
            'Mom%': mom_prob_str,
            'Dip%': dip_prob_str,
            'ATR%': metrics['atr_pct'],
            'Corr': f"{crypto_corr:.2f}" # 新增欄位
        })

    results.sort(key=lambda x: x['Gap%'], reverse=True)
    
    print("\n" + "=" * 125)
    # [修改] Header 增加 Corr
    header = f"{'Ticker':<8} {'Tag':<6} {'Gap%':>8} {'Price':>10} {'Status':<20} {'Action':<12} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6} {'Corr':>6}"
    print(header)
    print("-" * 125)
    
    for r in results:
        marker = ""
        if "SMART DIP" in r['Status']: 
            marker = " <--- 🟢 AI APPROVED BUY"
        if "[HOLD]" in r['Tag'] and ("GAP UP" in r['Status'] or "SELL RIP" in r['Status']): 
            marker = " <--- 🔴 SELL SIGNAL"
            if is_bullish_cal:
                 marker = " <--- ⚠️ EXIT CAUTION (TOTM)"
        if "ROCKET" in r['Status'] or "MOMENTUM" in r['Status']:
             marker = " <--- 🔥 HIGH MOMENTUM"
        if "[HOLD]" in r['Tag'] and "SMART DIP" in r['Status']:
            marker = " <--- 🟢 ADD POSITION"
        
        if "HOLD (TOTM)" in r['Status']:
             marker = " <--- 🛡️ SAVED BY TOTM"
             
        # [新增] ETH Filter 標記
        if "ETH FILTER" in r['Status']:
             marker = " <--- 🛡️ SAVED BY ETH"

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Gap%']*100:>7.2f}% {r['Price']:>10.2f} {r['Status']:<20} {r['Action']:<12} {r['Sell%']:>6} {r['Mom%']:>6} {r['Dip%']:>6} {r['ATR%']*100:>5.1f}% {r['Corr']:>6}{marker}")
        
    print("=" * 125)
    print(f"Total Scanned: {len(results)}")
    
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_with_crypto_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()