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
# SELL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')
# New V2 Model
SELL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_v2_model.joblib')
SECTOR_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'exp_07_v2_encoder.joblib')
SECTOR_LIST_PATH = os.path.join(OUTPUT_DIR, 'exp_07_v2_sectors.joblib')
SECTOR_MAP_FILE = os.path.join(RESOURCE_DIR, 'ticker_sectors.json')

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

def load_sectors():
    if os.path.exists(SECTOR_MAP_FILE):
        with open(SECTOR_MAP_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_current_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1]) if not df.empty else 20.0
    except: return 20.0

def get_spy_gap():
    try:
        df = yf.download("SPY", period="5d", interval="1d", progress=False)
        if len(df) < 2: return 0.0
        prev_close = float(df['Close'].iloc[-2])
        curr_open = float(df['Open'].iloc[-1]) # Use Open for real-time gap, or Close if checking previous
        # Assuming we are running this pre-market or at open.
        # But wait, df['Open'].iloc[-1] might be today's open if market is open.
        # If pre-market, yf might not have today's candle yet unless prepost=True.
        # Let's assume we run this when data is available.
        # For robustness, if today's date matches system date, use it.
        return (curr_open - prev_close) / prev_close
    except: return 0.0

def get_calendar_status():
    try:
        today = datetime.now().date()
        next_day = today + timedelta(days=1)
        while next_day.weekday() >= 5: next_day += timedelta(days=1)
        if next_day.strftime('%Y-%m-%d') in US_HOLIDAYS: return "Pre-Holiday (Bullish) 🏖️", True, False, False
        current_month_end = pd.Timestamp(today) + pd.tseries.offsets.BMonthEnd(0)
        last_trading_day = current_month_end.date()
        next_month_start = current_month_end + BDay(1)
        first_3_days = [(next_month_start + BDay(i)).date() for i in range(3)]

        is_month_start = 1 if today in first_3_days else 0
        is_month_end = 1 if today == last_trading_day else 0
        is_totm = 1 if (is_month_start or is_month_end) else 0 # Simplified TOTM

        status = "Normal (一般日)"
        is_bullish = False
        if today == last_trading_day:
            status = "TOTM (Month End) 🚀"
            is_bullish = True
        elif today in first_3_days:
            status = "TOTM (Month Start) 🚀"
            is_bullish = True

        return status, is_bullish, is_totm
    except: return "Unknown", False, 0

def download_data(tickers):
    try:
        # Include SPY for reference if needed, but we fetch it separately for Gap
        daily = yf.download(tickers, period="3mo", interval="1d", group_by='ticker', progress=False, auto_adjust=True)
        intra = yf.download(tickers, period="5d", interval="1m", group_by='ticker', prepost=True, progress=False, auto_adjust=True)
        return daily, intra
    except Exception as e:
        print(f"[Error] Download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

def calculate_metrics(ticker, df_daily, df_intra, vix_val, spy_gap, sector_map, encoder, top_sectors):
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

        # Dist MA20
        # For prediction, we use "shifted" features (values known at T-1 close).
        # Except Gap which is known at T Open.
        # In generator:
        # rsi is from iloc[-1] (T-1 Close).
        # atr_pct is from iloc[-1] (T-1 Close).
        # vol_ratio is Volume(T-1) / MeanVolume(T-2..T-21).
        # These match "Prev_*" features in training.

        # Calculate Dist MA20 (Prev)
        # Prev_Dist_MA20 = (Close(T-1) / MA20(T-1)) - 1
        # Generator calculates `dist_ma20` using `curr_price` (T Open/Current).
        # But Training used `Prev_Dist_MA20`.
        # So we should calculate Dist MA20 based on Prev Close.
        ma20_prev = df_daily['Close'].rolling(20).mean().iloc[-1]
        dist_ma20_prev = (prev_close / ma20_prev) - 1

        # RS Gap
        rs_gap = gap_pct - spy_gap

        # TOTM
        _, _, is_totm = get_calendar_status()

        # Sector
        sector = sector_map.get(ticker, 'Unknown')
        sector_clean = sector if sector in top_sectors else 'Other'
        # Encode
        # We need a DataFrame to use the encoder if it expects named columns?
        # Or just pass [[sector_clean]]
        # The encoder was trained on a DataFrame column 'Sector_Clean'.
        # We should create a DataFrame with same column name to be safe or just numpy array if fit on array.
        # In training: `encoder.fit_transform(df[['Sector_Clean']])` -> Input was DataFrame.
        sector_encoded = encoder.transform(pd.DataFrame({'Sector_Clean': [sector_clean]}))

        # Construct Feature Vector
        # Training Cols: 'Prev_RSI_14', 'Prev_ATR_Pct', 'Prev_Vol_Ratio', 'Gap_Pct', 'VIX',
        #                'Prev_Dist_MA20', 'RS_Gap', 'Is_TOTM' + sector_cols

        base_feats = [rsi, atr_pct, vol_ratio, gap_pct, vix_val, dist_ma20_prev, rs_gap, is_totm]
        # Flatten sector_encoded (it's 1xN)
        sector_feats = sector_encoded[0].tolist()

        full_feats = base_feats + sector_feats

        # Feature names are needed for some models or debugging, but predict usually takes array.
        # However, to be safe with sklearn warning about feature names:
        # We can reconstruct the DataFrame if we knew column names.
        # But joblib loaded model should work with array if order is correct.

        features_array = np.array(full_feats).reshape(1, -1)

        return {'price': curr_price, 'prev_close': prev_close, 'gap_pct': gap_pct, 'features': features_array, 'atr_pct': atr_pct}
    except Exception as e:
        # print(f"Error calc metrics for {ticker}: {e}")
        return None

# --- 主程式 ---

def generate_report():
    cal_status, is_bullish_cal, is_totm = get_calendar_status()
    current_gap_threshold = 0.01 if is_bullish_cal else DEFAULT_GAP_THRESHOLD
    curr_vix = get_current_vix()
    spy_gap = get_spy_gap()

    print(f"\n>>> V6.2 Ultra Daily Gap Scanner (Modularized + Sorted)")
    print(f">>> [Market Context] 📅 Calendar: {cal_status} | VIX: {curr_vix:.1f} | SPY Gap: {spy_gap*100:.2f}%")
    print("-" * 135)
    
    # 載入模型
    models = {
        'sell': joblib.load(SELL_MODEL_PATH) if os.path.exists(SELL_MODEL_PATH) else None,
        'mom': joblib.load(MOM_MODEL_PATH) if os.path.exists(MOM_MODEL_PATH) else None,
        'dip': joblib.load(DIP_MODEL_PATH) if os.path.exists(DIP_MODEL_PATH) else None
    }

    # Load Encoder and Sectors for Sell Model
    encoder = None
    top_sectors = []
    if os.path.exists(SECTOR_ENCODER_PATH):
        encoder = joblib.load(SECTOR_ENCODER_PATH)
    if os.path.exists(SECTOR_LIST_PATH):
        top_sectors = joblib.load(SECTOR_LIST_PATH)

    sector_map = load_sectors()

    tickers, tags_map = load_tickers_and_tags()
    daily_data, intra_data = download_data(tickers)
    
    results = []
    
    for t in tickers:
        try:
            df_t_daily = daily_data[t] if len(tickers) > 1 else daily_data
            df_t_intra = intra_data[t] if not intra_data.empty else pd.DataFrame()
            
            # Regime 判定
            regime_status, er_val, _ = get_regime_decision(df_t_daily, t)
            
            metrics = calculate_metrics(t, df_t_daily, df_t_intra, curr_vix, spy_gap, sector_map, encoder, top_sectors)
            if not metrics: continue
            gap, price, feats = metrics['gap_pct'], metrics['price'], metrics['features']
            
            # AI 預測
            probs = {}
            for k, m in models.items():
                try:
                    if k == 'sell':
                         # New V2 model expects all new features
                         p = m.predict_proba(feats)[0][1]
                    elif k == 'dip':
                        # Dip model expects 6 features: RSI, ATR, Vol, Gap, VIX, Dist_MA20
                        # These are the first 6 in our base_feats
                        p = m.predict_proba(feats[:, :6])[0][1]
                    else:
                        # Momentum model (or others) likely expects first 5: RSI, ATR, Vol, Gap, VIX
                        p = m.predict_proba(feats[:, :5])[0][1]

                    probs[k] = f"{p:.0%}"
                    probs[f'{k}_val'] = p
                except Exception as e:
                    # print(f"Pred error {k} {t}: {e}")
                    probs[k], probs[f'{k}_val'] = "-", 0.0

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
        except Exception as e:
            # print(f"Loop error {t}: {e}")
            continue

    # --- 排序與輸出邏輯 ---
    def get_sort_priority(r):
        if r['Action'] in ['SELL/TRIM', 'STRONG SELL', 'BUY OPEN', 'TRIM ONLY']: return 0
        if r['Action'] in ['HOLD', 'RIDE GAP', 'WATCH']: return 1
        if r['Regime'] == "✅ PASS": return 2
        return 3

    results.sort(key=lambda x: (get_sort_priority(x), -abs(x['Gap%'])))

    # 定義標題
    header = f"{'Ticker':<8} {'Tag':<6} {'Regime':<12} {'Gap%':>8} {'Limit':>9} {'Status':<16} {'Action':<12} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6}"
    print(header)
    print("-" * 135)
    
    last_priority = -1
    for r in results:
        curr_priority = get_sort_priority(r)
        if curr_priority != last_priority:
            if curr_priority == 1: print("-" * 45 + " [ 持有 / 觀察區 ] " + "-" * 72)
            if curr_priority == 2: print("-" * 45 + " [ 盤整無訊號區 ] " + "-" * 72)
            if curr_priority == 3: print("-" * 45 + " [ 🛑 趨勢過強禁止區 ] " + "-" * 67)
            last_priority = curr_priority

        marker = ""
        if "BLOCK" in r['Status']: marker = " <--- 🛡️ REGIME FILTERED"
        elif "SMART DIP" in r['Status']: marker = " <--- 🟢 AI APPROVED BUY"
        elif "[HOLD]" in r['Tag'] and ("GAP UP" in r['Status'] or "SELL" in r['Status']): 
            marker = " <--- ⚠️ EXIT SIGNAL" if not is_bullish_cal else " <--- ⚠️ EXIT CAUTION (TOTM)"
        elif "ROCKET" in r['Status']: marker = " <--- 🔥 HIGH MOMENTUM"

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Regime']:<12} {r['Gap%']*100:>7.2f}% {r['Price']:>9.2f} {r['Status']:<16} {r['Action']:<12} {r['Sell%']:>6} {r['Mom%']:>6} {r['Dip%']:>6} {r['ATR%']*100:>5.1f}%{marker}")
    
    print("-" * 135)
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"Total Scanned: {len(results)} | [Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()