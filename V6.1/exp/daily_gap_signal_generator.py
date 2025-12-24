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

# 抑制警告與非關鍵日誌
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# --- 1. 全域設定 (Configuration) ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案路徑
FILES = {
    'ASSET': '2025_final_asset_pool.json',
    'TOXIC': '2025_final_toxic_asset_pool.json',
    'CRYPTO': '2025_final_crypto_sensitive_pool.json',
    'HOLDING': '2025_holding_asset_pool.json'
}

# 策略參數
GAP_THRESHOLD = 0.005      # 0.5% (基礎門檻)
PROFIT_THRESHOLD = 0.002   # 0.2% (AI 訓練目標)
AI_CONFIDENCE_LV = 0.50    # AI 預測機率門檻 (大於此值才算 GO)

# 下載設定
BATCH_SIZE = 20
MAX_RETRIES = 2

# --- 2. 輔助工具 (Helpers) ---

def load_tickers(key):
    """從 resource 讀取股票清單"""
    filename = FILES.get(key)
    if not filename: return []
    path = os.path.join(RESOURCE_DIR, filename)
    if not os.path.exists(path):
        # 嘗試回退到 V6.0 目錄查找 (相容性)
        alt_path = path.replace('V6.1', 'V6.0')
        if os.path.exists(alt_path):
            path = alt_path
        else:
            return []
            
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        # 清洗 Ticker 格式 (去除 'NYSE:', '.' 轉 '-')
        return list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw_list]))
    except Exception as e:
        print(f"[Error] Load {filename} failed: {e}")
        return []

def get_calendar_status(target_date=None):
    """判斷日曆效應 (TOTM, Pre-Holiday)"""
    if target_date is None:
        target_date = datetime.now().date()
    
    start_date = target_date - timedelta(days=40)
    end_date = target_date + timedelta(days=40)
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    dates = pd.date_range(start=start_date, end=end_date, freq=us_bd)
    
    # TOTM: 月末 1 天 + 月初 3 天
    df = pd.DataFrame(index=dates)
    date_series = df.index.to_series()
    groups = date_series.groupby(date_series.dt.to_period('M'))
    totm_dates = []
    for _, days in groups:
        if len(days) < 4: continue
        totm_dates.append(days[-1].date())
        totm_dates.extend([d.date() for d in days[:3]])
        
    is_totm = target_date in totm_dates

    # Pre-Holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    is_pre_holiday = (target_date + timedelta(days=1)) in holidays
    
    # 狀態字串
    status = []
    if is_totm: status.append("TOTM(月初)")
    if is_pre_holiday: status.append("Holiday(節前)")
    
    return is_totm, is_pre_holiday, " + ".join(status) if status else "Normal"

def get_crypto_sentiment():
    """判斷 ETH 週末情緒 (僅週一有效)"""
    if datetime.now().weekday() != 0:
        return 0.0, "Weekday"

    try:
        df = yf.download("ETH-USD", period="5d", interval="1h", progress=False, auto_adjust=False)
        if df.empty: return 0.0, "NoData"
        
        # 處理時區
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')
            
        now_price = float(df['Close'].iloc[-1])
        
        # 找上週五 16:00
        today = datetime.now().date()
        last_friday = today - timedelta(days=3)
        target_ts = pd.Timestamp(f"{last_friday} 16:00").tz_localize('America/New_York')
        
        # 找最近的時間點
        idx = df.index.get_indexer([target_ts], method='nearest')[0]
        fri_price = float(df['Close'].iloc[idx])
        
        ret = (now_price - fri_price) / fri_price
        return ret, "Weekend_Move"
    except:
        return 0.0, "Error"

def get_current_vix():
    """獲取即時 VIX"""
    try:
        df = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        return float(df['Close'].iloc[-1])
    except:
        return 20.0 # Fallback

# --- 3. 核心數據處理 (Data Processing) ---

def download_data_batch(tickers):
    """分批下載日線與分時線"""
    daily_data = {}
    intra_data = {}
    
    total = len(tickers)
    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i+BATCH_SIZE]
        print(f"  Fetching batch {i+1}-{min(i+BATCH_SIZE, total)}...", end='\r')
        
        try:
            # 日線 (用於計算 RSI, ATR, Vol MA)
            d = yf.download(batch, period="2mo", interval="1d", progress=False, auto_adjust=True, threads=True)
            # 分時 (用於抓最新盤前價/開盤價)
            m = yf.download(batch, period="5d", interval="1m", prepost=True, progress=False, auto_adjust=True, threads=True)
            
            # 整理數據結構
            if not d.empty:
                # 處理 Single/Multi Index
                if isinstance(d.columns, pd.MultiIndex):
                    d = d.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
                else:
                    d['Ticker'] = batch[0]
                    d = d.reset_index()
                
                # 存入 Dict
                for t, group in d.groupby('Ticker'):
                    daily_data[t] = group.set_index('Date').sort_index()

            if not m.empty:
                if isinstance(m.columns, pd.MultiIndex):
                    m = m.stack(level=1, future_stack=True).rename_axis(['Datetime', 'Ticker']).reset_index()
                else:
                    m['Ticker'] = batch[0]
                    m = m.reset_index()
                
                for t, group in m.groupby('Ticker'):
                    # 轉換時區統一為 NY
                    df_m = group.set_index('Datetime')
                    if df_m.index.tz is None:
                        df_m.index = df_m.index.tz_localize('UTC').tz_convert('America/New_York')
                    else:
                        df_m.index = df_m.index.tz_convert('America/New_York')
                    intra_data[t] = df_m.sort_index()
                    
        except Exception as e:
            print(f"Batch failed: {e}")
            continue
            
    print(f"\n  Data fetch complete. Daily: {len(daily_data)}, Intraday: {len(intra_data)}")
    return daily_data, intra_data

def prepare_ai_features(ticker, df_daily, df_intra, vix_val):
    """
    為 EXP-07 模型準備特徵
    Features: ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX']
    """
    try:
        # 強制轉數值
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
        
        df_daily = df_daily.dropna()
        if len(df_daily) < 25: return None, 0, 0 # 資料不足

        # 1. 取得最新價格 (Real-time Gap)
        # 若有分時數據，取最後一筆 (可能是盤前或開盤)
        if ticker in df_intra and not df_intra[ticker].empty:
            curr_price = float(df_intra[ticker]['Close'].iloc[-1])
        else:
            # 若無分時，暫時用日線收盤 (這在盤後跑沒問題，盤前跑會失真)
            curr_price = float(df_daily['Close'].iloc[-1])
            
        prev_close = float(df_daily['Close'].iloc[-1])
        # 若 curr_price == prev_close (尚未開盤)，則 Gap 為 0
        
        gap_pct = (curr_price - prev_close) / prev_close

        # 2. 計算技術指標 (基於 T-1 歷史數據)
        # EXP-07 訓練時是用 "T-1 的指標" 來預測 "T 的獲利"
        # 因此我們直接在日線上算，取最後一筆即可
        
        # RSI 14
        rsi_series = ta.rsi(df_daily['Close'], length=14)
        rsi_val = rsi_series.iloc[-1]
        
        # ATR 14
        atr_series = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14)
        atr_val = atr_series.iloc[-1]
        atr_pct = atr_val / prev_close
        
        # Vol Ratio (昨日量 / 前20日均量)
        # 注意: 訓練時邏輯是 df['Prev_Vol'] / df['Vol_MA20'].shift(1)
        # 此處 iloc[-1] 是 Prev_Vol (T-1)
        vol = df_daily['Volume']
        vol_last = vol.iloc[-1]
        
        # MA20 (不包含 T-1 的 rolling mean? 需小心)
        # rolling(20).mean() 在 T-1 時包含了 T-1。
        # shift(1) 代表 T-2 的 rolling mean。
        vol_ma = vol.rolling(20).mean()
        vol_ma_ref = vol_ma.iloc[-2] # T-2 的 MA20
        
        vol_ratio = vol_last / vol_ma_ref if vol_ma_ref > 0 else 1.0
        
        # 3. 組合特徵 DataFrame
        features = pd.DataFrame([[rsi_val, atr_pct, vol_ratio, gap_pct, vix_val]], 
                                columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX'])
        
        return features, gap_pct, curr_price
        
    except Exception as e:
        # print(f"Feature calc error for {ticker}: {e}")
        return None, 0, 0

# --- 4. 主程式 (Dashboard Generator) ---

def generate_live_dashboard():
    print("="*60)
    print(f"🚀 V6.1 Gap Signal Generator (AI Enhanced)")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. 載入模型
    ai_model = None
    if os.path.exists(MODEL_PATH):
        try:
            ai_model = joblib.load(MODEL_PATH)
            print(f"[System] AI Model loaded: EXP-07 XGBoost")
        except Exception as e:
            print(f"[Warning] Failed to load model: {e}")
    else:
        print(f"[Warning] Model not found at {MODEL_PATH}. Running in Classic Mode.")

    # 2. 載入清單
    # 合併 Asset Pool 與 Toxic Pool，並移除重複
    pool_asset = load_tickers('ASSET')
    pool_toxic = load_tickers('TOXIC')
    pool_holding = load_tickers('HOLDING') # 選項: 加入持倉監控
    
    all_tickers = list(set(pool_asset + pool_toxic + pool_holding))
    print(f"[System] Monitoring {len(all_tickers)} tickers.")
    
    # 3. 獲取環境數據
    is_totm, is_holiday, cal_status = get_calendar_status()
    crypto_ret, crypto_status = get_crypto_sentiment()
    curr_vix = get_current_vix()
    
    print(f"\n[Market Context]")
    print(f"  📅 Calendar: {cal_status}")
    print(f"  🌊 VIX     : {curr_vix:.2f}")
    if crypto_status == "Weekend_Move":
        print(f"  🪙 Crypto  : ETH Weekend {crypto_ret*100:+.2f}%")
        
    # 4. 下載股票數據
    print("\n[Data Fetching]")
    daily_map, intra_map = download_data_batch(all_tickers)
    
    # 5. 分析訊號
    signals = []
    
    print("\n[Analysis]")
    for ticker in all_tickers:
        if ticker not in daily_map: continue
        
        # 準備特徵與計算 Gap
        features, gap_pct, price = prepare_ai_features(
            ticker, daily_map[ticker], intra_map, curr_vix
        )
        
        if features is None: continue
        
        # 基礎濾網: Gap > 0.5%
        if gap_pct <= GAP_THRESHOLD: continue
        
        # 取得 Pool 屬性
        is_toxic = ticker in pool_toxic
        pool_tag = "Toxic" if is_toxic else "Asset"
        
        # --- AI 預測 ---
        ai_prob = 0.5
        ai_rec = "N/A"
        
        if ai_model:
            try:
                # XGBoost predict
                ai_prob = ai_model.predict_proba(features)[0][1] # Class 1 Prob
                ai_rec = "✅ GO" if ai_prob > AI_CONFIDENCE_LV else "❌ SKIP"
            except:
                ai_rec = "Err"
        
        # --- 規則濾網 (Context Rules) ---
        rule_action = "PASS"
        rule_reason = ""
        
        # Rule 1: Crypto Filter (僅針對 Toxic)
        if is_toxic and crypto_status == "Weekend_Move" and crypto_ret > 0.05:
            rule_action = "BLOCK"
            rule_reason = "Crypto Surge"
            
        # Rule 2: Calendar Filter (針對 Asset)
        if not is_toxic:
            if is_holiday:
                rule_action = "BLOCK"
                rule_reason = "Pre-Holiday"
            elif is_totm:
                rule_action = "BLOCK"
                rule_reason = "TOTM Flow"
                
        # --- 整合決策 ---
        final_decision = "WAIT"
        if rule_action == "BLOCK":
            final_decision = f"⛔ {rule_reason}"
        elif ai_model and ai_prob < AI_CONFIDENCE_LV:
            final_decision = "📉 AI Low Conf"
        else:
            final_decision = "🚀 ACTION"
            
        # 收集結果
        signals.append({
            'Ticker': ticker,
            'Pool': pool_tag,
            'Price': price,
            'Gap%': gap_pct,
            'RSI': features['RSI_14'].iloc[0],
            'ATR%': features['ATR_Pct'].iloc[0],
            'VolRatio': features['Vol_Ratio'].iloc[0],
            'AI_Prob': ai_prob,
            'Decision': final_decision
        })
        
    # 6. 輸出報表
    if not signals:
        print("\nNo Gap signals detected (> 0.5%).")
        return

    df_res = pd.DataFrame(signals)
    df_res.sort_values('AI_Prob', ascending=False, inplace=True)
    
    # 顯示
    print("\n" + "="*95)
    print(f"{'Ticker':<6} {'Pool':<6} {'Price':>8} {'Gap%':>7} {'RSI':>4} {'ATR%':>5} {'VolR':>5} {'AI Prob':>8} {'Decision':<15}")
    print("-" * 95)
    
    for _, r in df_res.iterrows():
        # 顏色標記 (如果環境支援 ANSI)
        dec = r['Decision']
        prob_str = f"{r['AI_Prob']:.0%}"
        
        # 簡單打印
        print(f"{r['Ticker']:<6} {r['Pool']:<6} {r['Price']:>8.2f} {r['Gap%']*100:>6.2f}% "
              f"{r['RSI']:>4.0f} {r['ATR%']*100:>4.1f}% {r['VolRatio']:>5.1f} {prob_str:>8} {dec:<15}")
              
    print("="*95)
    
    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\nReport saved to: {csv_path}")

if __name__ == '__main__':
    try:
        generate_live_dashboard()
    except KeyboardInterrupt:
        print("\nStopped by user.")