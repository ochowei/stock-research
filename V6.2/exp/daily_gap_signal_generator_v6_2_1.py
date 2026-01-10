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

# [V6.2 整合] 引用 production_daily_plan_v6_2 的模組邏輯
try:
    from production_daily_plan_v6_2 import get_regime_decision, clean_ticker
except ImportError:
    print("[Error] 找不到 production_daily_plan_v6_2.py，請確認檔名是否正確並位於同一目錄。")
    sys.exit(1)

# --- 設定 ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 模型路徑設定
SELL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')
MOM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'momentum_model.joblib')
DIP_MODEL_PATH = os.path.join(OUTPUT_DIR, 'dip_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案設定
ASSET_POOL_FILE = '2025_final_asset_pool.json'
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# 策略參數
DEFAULT_GAP_THRESHOLD = 0.005 
RIP_THRESHOLD = 0.03          
DIP_THRESHOLD = 0.03          
MOMENTUM_THRESHOLD = 0.53     
DIP_CONFIDENCE_LV = 0.50      

# 美股主要假期
US_HOLIDAYS = [
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26', 
    '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
]

# --- 工具函數 ---

def load_tickers_and_tags():
    tags_map = {}
    for filename, tag in [(ASSET_POOL_FILE, 'Asset'), (HOLDING_POOL_FILE, 'Held')]:
        path = os.path.join(RESOURCE_DIR, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    for t in raw:
                        clean_t = clean_ticker(t)
                        if clean_t not in tags_map: tags_map[clean_t] = set()
                        tags_map[clean_t].add(tag)
            except Exception as e: print(f"[Error] Failed to load {filename}: {e}")
    return sorted(list(tags_map.keys())), tags_map

def get_current_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1]) if not df.empty else 20.0
    except: return 20.0

def get_calendar_status():
    try:
        today = datetime.now().date()
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5: next_day += timedelta(days=1)
        if next_day.strftime('%Y-%m-%d') in US_HOLIDAYS: return "Pre-Holiday (Bullish) 🏖️", True
        current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(0)
        last_trading_day = current_month_end.date()
        next_month_start = current_month_end + BDay(1)
        first_3_days = [(next_month_start + BDay(i)).date() for i in range(3)]
        if today == last_trading_day: return "TOTM (Month End) 🚀", True
        elif today in first_3_days: return "TOTM (Month Start) 🚀", True
        return "Normal (一般日)", False
    except: return "Unknown", False

def download_data(tickers):
    try:
        daily = yf.download(tickers, period="3mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True)
        intra = yf.download(tickers, period="5d", interval="1m", group_by='ticker', prepost=True, progress=False, auto_adjust=True)
        return daily, intra
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_metrics(ticker, df_daily, df_intra, vix_val):
    try:
        df_daily = df_daily.dropna(subset=['Close'])
        if len(df_daily) < 20: return None
        prev_close = float(df_daily['Close'].iloc[-1])
        curr_price = prev_close
        if not df_intra.empty:
            df_m = df_intra.dropna(subset=['Close'])
            if not df_m.empty: curr_price = float(df_m['Close'].iloc[-1])

        gap_pct = (curr_price - prev_close) / prev_close
        atr_pct = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14).iloc[-1] / prev_close
        rsi = ta.rsi(df_daily['Close'], length=14).iloc[-1]
        vol_ratio = df_daily['Volume'].iloc[-1] / df_daily['Volume'].rolling(20).mean().iloc[-2]
        ma20_sim = ((df_daily['Close'].tail(19).mean() * 19) + curr_price) / 20
        dist_ma20 = (curr_price / ma20_sim) - 1

        features = pd.DataFrame([[rsi, atr_pct, vol_ratio, gap_pct, vix_val, dist_ma20]], 
                                columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20'])
        return {'price': curr_price, 'prev_close': prev_close, 'gap_pct': gap_pct, 'features': features, 'atr_pct': atr_pct}
    except: return None

# --- 主程式 ---

def generate_report():
    cal_status, is_bullish_cal = get_calendar_status()
    current_gap_threshold = 0.01 if is_bullish_cal else DEFAULT_GAP_THRESHOLD
    curr_vix = get_current_vix()

    print(f"\n>>> V6.2 Ultra Daily Gap Scanner (Modularized + Sorted)")
    print(f">>> [Market Context] 📅 Calendar: {cal_status} | VIX: {curr_vix:.1f}")
    print("-" * 125)
    
    # 載入模型
    models = {
        'sell': joblib.load(SELL_MODEL_PATH) if os.path.exists(SELL_MODEL_PATH) else None,
        'mom': joblib.load(MOM_MODEL_PATH) if os.path.exists(MOM_MODEL_PATH) else None,
        'dip': joblib.load(DIP_MODEL_PATH) if os.path.exists(DIP_MODEL_PATH) else None
    }

    tickers, tags_map = load_tickers_and_tags()
    daily_data, intra_data = download_data(tickers)
    
    results = []
    
    for t in tickers:
        try:
            df_t_daily = daily_data[t] if len(tickers) > 1 else daily_data
            df_t_intra = intra_data[t] if not intra_data.empty else pd.DataFrame()
            
            # Regime 判定
            regime_status, er_val, _ = get_regime_decision(df_t_daily, t)
            
            metrics = calculate_metrics(t, df_t_daily, df_t_intra, curr_vix)
            if not metrics: continue
            gap, price, feats = metrics['gap_pct'], metrics['price'], metrics['features']
            
            # AI 預測
            probs = {}
            for k, m in models.items():
                try:
                    p = m.predict_proba(feats if k == 'dip' else feats.iloc[:, :5])[0][1]
                    probs[k] = f"{p:.0%}"
                    probs[f'{k}_val'] = p
                except: probs[k], probs[f'{k}_val'] = "-", 0.0

            # 決策邏輯
            status, action = "Flat", "-"
            if regime_status == "🛑 BLOCK":
                status, action = "🛑 BLOCK (Trendy)", "SKIP"
            else:
                if gap > RIP_THRESHOLD:
                    if probs.get('mom_val', 0) > MOMENTUM_THRESHOLD: status, action = "🚀 ROCKET", "HOLD/BUY"
                    else:
                        status, action = "🔴 SELL RIP", "STRONG SELL"
                        if is_bullish_cal: status, action = "🔴 SELL(Bull)", "TRIM ONLY"
                elif gap > current_gap_threshold:
                    if probs.get('mom_val', 0) > MOMENTUM_THRESHOLD: status, action = "🟢 MOMENTUM", "HOLD"
                    else: status, action = "🔴 GAP UP", "SELL/TRIM"
                elif gap > DEFAULT_GAP_THRESHOLD and is_bullish_cal: status, action = "🟡 HOLD (TOTM)", "RIDE GAP"
                elif gap < -DIP_THRESHOLD:
                    if probs.get('dip_val', 0) > DIP_CONFIDENCE_LV: status, action = "🟢 SMART DIP", "BUY OPEN"
                    else: status, action = "🔵 WEAK DIP", "WATCH"
                elif gap < -DEFAULT_GAP_THRESHOLD: status, action = "🟡 GAP DOWN", "HOLD"

            is_held = 'Held' in tags_map.get(t, set())
            results.append({
                'Ticker': t, 'Tag': "[HOLD]" if is_held else "", 'Regime': regime_status,
                'Gap%': gap, 'Price': price, 'Status': status, 'Action': action,
                'Sell%': probs.get('sell', '-'), 'Mom%': probs.get('mom', '-'), 
                'Dip%': probs.get('dip', '-'), 'ATR%': metrics['atr_pct'], 'ER': er_val
            })
        except: continue

    # --- 排序與輸出邏輯 ---
    def get_sort_priority(r):
        if r['Action'] in ['SELL/TRIM', 'STRONG SELL', 'BUY OPEN', 'TRIM ONLY']: return 0
        if r['Action'] in ['HOLD', 'RIDE GAP', 'WATCH']: return 1
        if r['Regime'] == "✅ PASS": return 2
        return 3

    results.sort(key=lambda x: (get_sort_priority(x), -abs(x['Gap%'])))

    # 定義標題
    header = f"{'Ticker':<8} {'Tag':<6} {'Regime':<12} {'Gap%':>8} {'Status':<16} {'Action':<12} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6}"
    print(header)
    print("-" * 125)
    
    last_priority = -1
    for r in results:
        curr_priority = get_sort_priority(r)
        if curr_priority != last_priority:
            if curr_priority == 1: print("-" * 45 + " [ 持有 / 觀察區 ] " + "-" * 62)
            if curr_priority == 2: print("-" * 45 + " [ 盤整無訊號區 ] " + "-" * 62)
            if curr_priority == 3: print("-" * 45 + " [ 🛑 趨勢過強禁止區 ] " + "-" * 57)
            last_priority = curr_priority

        marker = ""
        if "BLOCK" in r['Status']: marker = " <--- 🛡️ REGIME FILTERED"
        elif "SMART DIP" in r['Status']: marker = " <--- 🟢 AI APPROVED BUY"
        elif "[HOLD]" in r['Tag'] and ("GAP UP" in r['Status'] or "SELL" in r['Status']): 
            marker = " <--- ⚠️ EXIT SIGNAL" if not is_bullish_cal else " <--- ⚠️ EXIT CAUTION (TOTM)"
        elif "ROCKET" in r['Status']: marker = " <--- 🔥 HIGH MOMENTUM"

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Regime']:<12} {r['Gap%']*100:>7.2f}% {r['Status']:<16} {r['Action']:<12} {r['Sell%']:>6} {r['Mom%']:>6} {r['Dip%']:>6} {r['ATR%']*100:>5.1f}%{marker}")
    
    print("-" * 125)
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Total Scanned: {len(results)} | [Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()