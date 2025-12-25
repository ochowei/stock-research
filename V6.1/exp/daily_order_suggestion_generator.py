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

# --- 設定 ---
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 指定讀取 Holding Pool
TARGET_POOL_FILE = '2025_holding_asset_pool.json'

# [策略參數]
TAKE_PROFIT_PCT = 0.005  # +0.5% 止盈
AI_CONFIDENCE_LV = 0.50

# --- 工具函數 ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, TARGET_POOL_FILE)
    if not os.path.exists(path):
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
    # 模仿原版 Dashboard 的日曆狀態
    return "Normal(一般日)" 

def get_current_vix():
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1])
    except:
        return 20.0

def download_data(tickers):
    # 下載日線 (計算 ATR, RSI, 昨收)
    data = yf.download(tickers, period="1mo", interval="1d", progress=False, auto_adjust=True, threads=True)
    # 下載盤前數據 (檢查是否 Hit)
    intra = yf.download(tickers, period="5d", interval="5m", prepost=True, progress=False, auto_adjust=True, threads=True)
    return data, intra

def calculate_metrics(ticker, df_daily, df_intra, vix_val):
    """計算所有顯示欄位"""
    try:
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df_daily.columns:
                df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
        
        df_daily = df_daily.dropna()
        if len(df_daily) < 5: return None

        prev_close = float(df_daily['Close'].iloc[-1])
        target_price = prev_close * (1 + TAKE_PROFIT_PCT)
        
        # 取得即時價格 & 盤前高點
        curr_price = prev_close
        pre_high = prev_close
        
        if ticker in df_intra.columns.levels[1]:
            df_m = df_intra.xs(ticker, axis=1, level=1).dropna()
            if not df_m.empty:
                curr_price = float(df_m['Close'].iloc[-1])
                # 抓今日盤前最高
                last_date = df_m.index[-1].date()
                today_mask = df_m.index.date == last_date
                if any(today_mask):
                    pre_high = float(df_m.loc[today_mask, 'High'].max())
                else:
                    pre_high = curr_price

        # 基礎指標
        gap_pct = (curr_price - prev_close) / prev_close
        atr = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14).iloc[-1]
        atr_pct = atr / prev_close
        
        # Fade% (回吐幅度)
        fade_pct = 0.0
        if pre_high > 0:
            fade_pct = (pre_high - curr_price) / pre_high

        # 狀態判斷 (HIT logic)
        status = "Waiting"
        if pre_high >= target_price:
            status = "✅ HIT(Pre)"
        elif curr_price >= target_price:
            status = "✅ HIT(Now)"
        elif gap_pct < -0.01:
             status = "Weak"
        
        # AI 特徵
        rsi = ta.rsi(df_daily['Close'], length=14).iloc[-1]
        vol_ma20 = df_daily['Volume'].rolling(20).mean().iloc[-2]
        vol_ratio = df_daily['Volume'].iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
        
        features = pd.DataFrame([[rsi, atr_pct, vol_ratio, gap_pct, vix_val]], 
                                columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX'])
        
        return {
            'prev_close': prev_close,
            'target_price': target_price,
            'curr_price': curr_price,
            'gap_pct': gap_pct,
            'fade_pct': fade_pct,
            'atr_pct': atr_pct,
            'status': status,
            'features': features
        }
    except Exception as e:
        return None

# --- 主程式 ---

def generate_report():
    print(f">>> V6.1 Gap Strategy Dashboard (Order Suggestion Mode)")
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
    
    if isinstance(daily_data.columns, pd.MultiIndex):
        daily_data = daily_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        daily_data['Ticker'] = tickers[0]
        daily_data = daily_data.reset_index()

    daily_map = {t: g.set_index('Date') for t, g in daily_data.groupby('Ticker')}
    
    results = []
    
    # 4. 計算 Loop
    for t in tickers:
        if t not in daily_map: continue
        
        metrics = calculate_metrics(t, daily_map[t], intra_data, curr_vix)
        if not metrics: continue
        
        # AI Predict
        ai_prob_str = "-"
        ai_dec = ""
        if ai_model:
            try:
                prob = ai_model.predict_proba(metrics['features'])[0][1]
                ai_prob_str = f"{prob:.0%}"
                if prob > 0.6: ai_dec = "Bull"
                elif prob < 0.4: ai_dec = "Bear"
            except: pass
            
        results.append({
            'Ticker': t,
            'Gap%': metrics['gap_pct'],
            'Price': metrics['curr_price'],
            'Target': metrics['target_price'],  # 這是 Limit Price
            'PrevCls': metrics['prev_close'],
            'Fade%': metrics['fade_pct'],
            'ATR%': metrics['atr_pct'],
            'Status': metrics['status'],
            'AI Prob': ai_prob_str,
            'Decision': ai_dec
        })

    # 5. 排序與列印
    results.sort(key=lambda x: x['Gap%'], reverse=True)
    
    print("\n" + "=" * 115)
    # [調整] 欄位對應 Dashboard 格式
    # 原版: Ticker Gap% Price GapUp GapDn Fade% ATR% Status AI Prob Decision
    # 新版: Ticker Gap% Price Target PrevCls Fade% ATR% Status AI Prob View
    header = f"{'Ticker':<6} {'Gap%':>7} {'Price':>8} {'Target':>8} {'PrevCls':>8} {'Fade%':>6} {'ATR%':>5} {'Status':<12} {'AI Prob':>7} {'View':<8}"
    print(header)
    print("-" * 115)
    
    for r in results:
        row_str = f"{r['Ticker']:<6} {r['Gap%']*100:>6.2f}% {r['Price']:>8.2f} " \
                  f"{r['Target']:>8.2f} {r['PrevCls']:>8.2f} " \
                  f"{r['Fade%']*100:>5.2f}% {r['ATR%']*100:>4.1f}% " \
                  f"{r['Status']:<12} {r['AI Prob']:>7} {r['Decision']:<8}"
        print(row_str)
        
    print("=" * 115)
    print(f"註: Target = PrevCls * (1 + {TAKE_PROFIT_PCT:.1%}). Status '✅ HIT' 代表盤前或現在已達標，建議止盈。")

    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, f'order_suggestions_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()