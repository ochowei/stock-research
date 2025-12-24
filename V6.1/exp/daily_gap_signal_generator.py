import os
import sys
import json
import time
import logging
import joblib
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# --- 設定 ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# [修改點 1] 指定讀取 Holding Pool
TARGET_POOL_FILE = '2025_holding_asset_pool.json'

# 參數
GAP_THRESHOLD = 0.005      # 0.5%
AI_CONFIDENCE_LV = 0.50    # AI 門檻

# --- 工具函數 ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, TARGET_POOL_FILE)
    if not os.path.exists(path):
        # Fallback 找 V6.0
        path = path.replace('V6.1', 'V6.0')
        if not os.path.exists(path):
            print(f"[Error] Cannot find {TARGET_POOL_FILE}")
            return []
            
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw]))
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return []

def get_calendar_status():
    target_date = datetime.now().date()
    # 簡化版日曆狀態
    return "Normal(一般日)" 

def get_current_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1])
    except:
        return 20.0

def download_data(tickers):
    # 下載足夠的歷史數據以計算 RSI(14)
    data = yf.download(tickers, period="3mo", interval="1d", progress=False, auto_adjust=True, threads=True)
    
    # 取得最新盤前/盤中數據 (1m) 用於計算即時 Gap/Fade
    intra = yf.download(tickers, period="5d", interval="1m", prepost=True, progress=False, auto_adjust=True, threads=True)
    
    return data, intra

def calculate_metrics(ticker, df_daily, df_intra, vix_val):
    """計算所有欄位所需的數值"""
    try:
        # 強制轉數值
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
        
        df_daily = df_daily.dropna()
        if len(df_daily) < 20: return None

        prev_close = float(df_daily['Close'].iloc[-1])
        
        # 取得即時價格 (Intraday Last) & 盤前高點 (Pre-Market High)
        curr_price = prev_close
        pre_high = prev_close
        
        if ticker in df_intra.columns.levels[1]: # MultiIndex check
            df_m = df_intra.xs(ticker, axis=1, level=1).dropna()
            if not df_m.empty:
                curr_price = float(df_m['Close'].iloc[-1])
                # 簡單抓當日最高當作 Pre-High (近似)
                today_mask = df_m.index.date == df_m.index[-1].date()
                if any(today_mask):
                    pre_high = float(df_m.loc[today_mask, 'High'].max())
                else:
                    pre_high = curr_price

        # 1. 基礎指標
        gap_pct = (curr_price - prev_close) / prev_close
        atr = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14).iloc[-1]
        atr_pct = atr / prev_close
        
        # Fade% = (High - Curr) / High (僅在 Gap Up 時有意義，但也算出數值)
        fade_pct = 0.0
        if pre_high > 0:
            fade_pct = (pre_high - curr_price) / pre_high

        # 2. AI 特徵 (Exp-07)
        rsi = ta.rsi(df_daily['Close'], length=14).iloc[-1]
        
        vol_ma20 = df_daily['Volume'].rolling(20).mean().iloc[-2] # T-1 的 MA
        vol_last = df_daily['Volume'].iloc[-1]
        vol_ratio = vol_last / vol_ma20 if vol_ma20 > 0 else 1.0
        
        features = pd.DataFrame([[rsi, atr_pct, vol_ratio, gap_pct, vix_val]], 
                                columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX'])
        
        return {
            'price': curr_price,
            'prev_close': prev_close,
            'gap_pct': gap_pct,
            'fade_pct': fade_pct,
            'atr_pct': atr_pct,
            'features': features
        }
    except Exception as e:
        return None

# --- 主程式 ---

def generate_report():
    print(f">>> V6.1 Gap Strategy Dashboard (Holding Monitor)")
    print(f">>> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 載入模型
    ai_model = None
    try:
        if os.path.exists(MODEL_PATH):
            ai_model = joblib.load(MODEL_PATH)
    except: pass

    # 2. 載入清單
    tickers = load_tickers()
    print(f"清單概況:\n  - {TARGET_POOL_FILE.replace('.json','')}: {len(tickers)} 檔")
    
    cal_status = get_calendar_status()
    print(f"\n[Market Context]\n  📅 Calendar: {cal_status}")
    
    curr_vix = get_current_vix()
    
    # 3. 下載數據
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載 {len(tickers)} 檔股票數據...")
    daily_data, intra_data = download_data(tickers)
    
    # 4. 處理 Single/Multi Index
    if isinstance(daily_data.columns, pd.MultiIndex):
        daily_data = daily_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        daily_data['Ticker'] = tickers[0]
        daily_data = daily_data.reset_index()

    # 轉成 Dict 方便存取
    daily_map = {t: g.set_index('Date') for t, g in daily_data.groupby('Ticker')}
    
    results = []
    
    # 5. 計算 Loop
    for t in tickers:
        if t not in daily_map: continue
        
        metrics = calculate_metrics(t, daily_map[t], intra_data, curr_vix)
        if not metrics: continue
        
        gap = metrics['gap_pct']
        threshold = GAP_THRESHOLD # 預設 0.5%
        
        # 決定 Status
        status = "Watching"
        if gap > threshold: status = "🔴 GAP UP"
        elif gap < -threshold: status = "🟢 GAP DOWN"
        elif abs(gap) < 0.002: status = "Flat"
        
        # AI Predict (僅在有訊號時跑，或者全部跑也可以，這裡為了填表全部跑)
        ai_prob_str = "N/A"
        ai_dec = ""
        
        if ai_model and abs(gap) > 0.003: # 只對稍有波動的跑 AI
            try:
                prob = ai_model.predict_proba(metrics['features'])[0][1]
                ai_prob_str = f"{prob:.0%}"
                if prob > AI_CONFIDENCE_LV:
                    ai_dec = "GO"
                else:
                    ai_dec = "SKIP"
            except: pass
            
        results.append({
            'Ticker': t,
            'Cat': 'A', # 假設 Holding 都是 Asset
            'Gap%': gap,
            'Thres%': threshold,
            'Fade%': metrics['fade_pct'],
            'ATR%': metrics['atr_pct'],
            'Price': metrics['price'],
            'TrigPx': metrics['prev_close'] * (1 + (threshold if gap > 0 else -threshold)),
            'Status': status,
            'AI Prob': ai_prob_str,
            'Decision': ai_dec
        })

    # 6. 排序與列印 (模仿舊版排版)
    results.sort(key=lambda x: x['Gap%'], reverse=True)
    
    print("\n" + "=" * 105)
    # 格式化字串 (增加 AI 欄位)
    header = f"{'Ticker':<6} {'Cat':<4} {'Gap%':>6} {'Thres%':>6} {'Fade%':>6} {'ATR%':>5} {'Price':>8} {'TrigPx':>8} {'Status':<12} {'AI Prob':>7} {'Decision':<8}"
    print(header)
    print("-" * 105)
    
    for r in results:
        # 顏色處理 (簡單版)
        row_str = f"{r['Ticker']:<6} {r['Cat']:<4} {r['Gap%']*100:>5.2f}% {r['Thres%']*100:>5.2f}% " \
                  f"{r['Fade%']*100:>5.2f}% {r['ATR%']*100:>4.1f}% {r['Price']:>8.2f} {r['TrigPx']:>8.2f} " \
                  f"{r['Status']:<12} {r['AI Prob']:>7} {r['Decision']:<8}"
        print(row_str)
        
    print("=" * 105)
    
    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, f'holding_monitor_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()