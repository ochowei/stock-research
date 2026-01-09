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

# [變更] 模型路徑設定 (三模型)
SELL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')
MOM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'momentum_model.joblib')
DIP_MODEL_PATH = os.path.join(OUTPUT_DIR, 'dip_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案設定
ASSET_POOL_FILE = '2025_final_asset_pool.json'
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# 策略參數
DEFAULT_GAP_THRESHOLD = 0.005 # 0.5% (預設 Gap Up 賣出門檻)
RIP_THRESHOLD = 0.03          # 3.0% (Sell Rip 賣出門檻)
DIP_THRESHOLD = 0.03          # 3.0% (Buy Dip 買進門檻 - 取絕對值)
AI_CONFIDENCE_LV = 0.50       # Sell AI 信心門檻
MOMENTUM_THRESHOLD = 0.53     # Momentum AI 信心門檻 (53%)
DIP_CONFIDENCE_LV = 0.50      # Dip AI 信心門檻

# [新增] 美股主要假期 (2025-2026) 用於 Pre-Holiday 判斷
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
    else:
        print(f"[Warning] Asset pool not found: {path_asset}")

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
    else:
        print(f"[Warning] Holding pool not found: {path_holding}")

    # 3. 產生去重列表
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
    [修改] 實作 Survey 定義的日曆效應:
    1. TOTM (月初效應): 月底最後 1 天 ~ 月初前 3 天
    2. Pre-Holiday (節假日前夕): 假期前 1 個交易日
    """
    try:
        today = datetime.now().date()
        # today = date(2025, 1, 31) # Debug: 測試 TOTM
        
        status_msg = "Normal (一般日)"
        is_bullish = False
        
        # --- 判斷 1: Pre-Holiday ---
        # 檢查明天(或下個交易日)是否為假期
        next_day = today + timedelta(days=1)
        # 如果明天是週末，往後推到週一
        while next_day.weekday() >= 5: 
            next_day += timedelta(days=1)
            
        if next_day.strftime('%Y-%m-%d') in US_HOLIDAYS:
            return "Pre-Holiday (Bullish) 🏖️", True

        # --- 判斷 2: TOTM (Turn of the Month) ---
        # 建立當月與下個月的交易日曆範圍 (使用 pandas BDay)
        # 取得當月最後一個交易日
        current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(0)
        if current_month_end.date() < today: # 如果今天已經過了當月最後交易日(理論上不可能，除非資料延遲)，抓下個月
             current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(1)
        
        # 取得當月最後 1 個交易日
        last_trading_day = current_month_end.date()
        
        # 取得下個月前 3 個交易日
        next_month_start = current_month_end + BDay(1)
        first_3_days = [ (next_month_start + BDay(i)).date() for i in range(3) ]
        
        # 判斷今天是否在 TOTM 窗口
        if today == last_trading_day:
            return "TOTM (Month End) 🚀", True
        elif today in first_3_days:
            return "TOTM (Month Start) 🚀", True

        return status_msg, is_bullish
        
    except Exception as e:
        print(f"[Warning] Calendar check failed: {e}")
        return "Unknown", False

def download_data(tickers, max_retries=3):
    """
    [修改] 加入 Retry 機制
    """
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retrying download (Attempt {attempt+1}/{max_retries})...")
            
            data = yf.download(tickers, period="3mo", interval="1d", progress=False, auto_adjust=True, threads=True)
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
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df_daily.columns:
                df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
        
        df_daily = df_daily.dropna()
        if len(df_daily) < 20: return None

        prev_close = float(df_daily['Close'].iloc[-1])
        curr_price = prev_close
        pre_high = prev_close
        
        # 處理 Intraday Data
        if isinstance(df_intra.columns, pd.MultiIndex):
            if ticker in df_intra.columns.levels[1]:
                df_m = df_intra.xs(ticker, axis=1, level=1).dropna()
                if not df_m.empty:
                    curr_price = float(df_m['Close'].iloc[-1])
                    today_mask = df_m.index.date == df_m.index[-1].date()
                    if any(today_mask):
                        pre_high = float(df_m.loc[today_mask, 'High'].max())
                    else:
                        pre_high = curr_price
        else:
            df_m = df_intra.dropna()
            if not df_m.empty:
                curr_price = float(df_m['Close'].iloc[-1])
                today_mask = df_m.index.date == df_m.index[-1].date()
                if any(today_mask):
                    pre_high = float(df_m.loc[today_mask, 'High'].max())
                else:
                    pre_high = curr_price

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
    # 0. 取得日曆狀態 (Survey Logic)
    cal_status, is_bullish_cal = get_calendar_status()
    
    # [動態調整] 若為 TOTM 或 Pre-Holiday，提高 Gap 門檻至 1.0%
    current_gap_threshold = DEFAULT_GAP_THRESHOLD
    if is_bullish_cal:
        current_gap_threshold = 0.01 # 1.0%

    print(f"\n>>> V6.1.3 Daily Gap & Dip Scanner (Calendar Aware)")
    print(f">>> Target: Holdings + Asset Pool")
    print(f">>> Thresholds: Sell Rip > {RIP_THRESHOLD:.1%}, Buy Dip < -{DIP_THRESHOLD:.1%}")
    # 顯示當前使用的 Gap Threshold
    print(f">>> Gap Up Threshold: {current_gap_threshold:.1%} (Default: {DEFAULT_GAP_THRESHOLD:.1%})")
    print(f">>> AI Thresholds: Mom > {MOMENTUM_THRESHOLD:.0%}, Dip > {DIP_CONFIDENCE_LV:.0%}")
    print(f">>> [Market Context] 📅 Calendar: {cal_status}")
    if is_bullish_cal:
        print(f">>> [Strategy Adjustment] Bullish Window Detected! Raising Sell Threshold to avoid early exit.")
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
            daily_data = daily_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
        else:
            daily_data['Ticker'] = tickers[0]
            daily_data = daily_data.reset_index()

        daily_map = {t: g.set_index('Date') for t, g in daily_data.groupby('Ticker')}
    except: return
    
    results = []
    
    # 5. 計算訊號
    for t in tickers:
        if t not in daily_map: continue
        
        metrics = calculate_metrics(t, daily_map[t], intra_data, curr_vix)
        if not metrics: continue
        
        gap = metrics['gap_pct']
        price = metrics['price']
        feats = metrics['features']
        
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

        # --- 核心策略邏輯 V6.1.3 (含 Survey Calendar Logic) ---
        status = "Flat"
        action = "WAIT"
        
        if gap > RIP_THRESHOLD:
            if mom_prob > MOMENTUM_THRESHOLD:
                status = "🚀 ROCKET"
                action = "HOLD/BUY"
            else:
                status = "🔴 SELL RIP"
                action = "STRONG SELL"
                # [Survey Logic] 若為強勢窗口，Sell Rip 也建議減碼而非清倉
                if is_bullish_cal:
                    status += "(Bull)"
                    action = "TRIM ONLY"
                
        elif gap > current_gap_threshold: # 使用動態調整後的門檻
            if mom_prob > MOMENTUM_THRESHOLD:
                status = "🟢 MOMENTUM"
                action = "HOLD"
            else:
                status = "🔴 GAP UP"
                action = "SELL/TRIM"
                if is_bullish_cal:
                    status += "(Bull)"
        
        # 處理那些在原本門檻(0.5%)與新門檻(1.0%)之間的股票
        elif gap > DEFAULT_GAP_THRESHOLD and is_bullish_cal:
            status = "🟡 HOLD (TOTM)"
            action = "RIDE GAP"
                
        elif gap < -DIP_THRESHOLD:
            if dip_prob > DIP_CONFIDENCE_LV:
                status = "🟢 SMART DIP"
                action = "BUY OPEN"
                if is_bullish_cal:
                    status += " (Aggr)" # 窗口期可以積極一點
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
            'ATR%': metrics['atr_pct']
        })

    results.sort(key=lambda x: x['Gap%'], reverse=True)
    
    print("\n" + "=" * 115)
    header = f"{'Ticker':<8} {'Tag':<6} {'Gap%':>8} {'Price':>10} {'Status':<16} {'Action':<12} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6}"
    print(header)
    print("-" * 115)
    
    for r in results:
        marker = ""
        if "SMART DIP" in r['Status']: 
            marker = " <--- 🟢 AI APPROVED BUY"
        if "[HOLD]" in r['Tag'] and ("GAP UP" in r['Status'] or "SELL RIP" in r['Status']): 
            marker = " <--- 🔴 SELL SIGNAL"
            # 若在強勢窗口，提示可能會賣飛
            if is_bullish_cal:
                 marker = " <--- ⚠️ EXIT CAUTION (TOTM)"
        if "ROCKET" in r['Status'] or "MOMENTUM" in r['Status']:
             marker = " <--- 🔥 HIGH MOMENTUM"
        if "[HOLD]" in r['Tag'] and "SMART DIP" in r['Status']:
            marker = " <--- 🟢 ADD POSITION"
        
        # 特別標註被 TOTM 豁免的股票
        if "HOLD (TOTM)" in r['Status']:
             marker = " <--- 🛡️ SAVED BY TOTM"

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Gap%']*100:>7.2f}% {r['Price']:>10.2f} {r['Status']:<16} {r['Action']:<12} {r['Sell%']:>6} {r['Mom%']:>6} {r['Dip%']:>6} {r['ATR%']*100:>5.1f}%{marker}")
        
    print("=" * 115)
    print(f"Total Scanned: {len(results)}")
    
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()