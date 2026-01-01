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
# 指向資源目錄
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource') 
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案設定
ASSET_POOL_FILE = '2025_final_asset_pool.json'
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# 策略參數
GAP_THRESHOLD = 0.005      # 0.5% (Gap Up 賣出門檻) [恢復原值]
RIP_THRESHOLD = 0.03       # 3.0% (Sell Rip 賣出門檻 - 新增，與 Dip 對稱)
DIP_THRESHOLD = 0.03       # 3.0% (Buy Dip 買進門檻 - 取絕對值)
AI_CONFIDENCE_LV = 0.50    # AI 信心門檻

# --- 工具函數 ---

def load_tickers_and_tags():
    """
    讀取 Asset Pool 與 Holding Pool，並回傳聯集列表與標籤對照表。
    Returns:
        tickers (list): 去重後的完整代號列表
        tags (dict): { 'AMD': {'Asset', 'Held'}, 'TSLA': {'Asset'}, ... }
    """
    tags_map = {}
    
    # 1. 讀取 Final Asset Pool (潛在機會)
    path_asset = os.path.join(RESOURCE_DIR, ASSET_POOL_FILE)
    if os.path.exists(path_asset):
        try:
            with open(path_asset, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                for t in raw:
                    # 清洗 Ticker: "NYSE:AMD" -> "AMD", "BRK.B" -> "BRK-B"
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

def download_data(tickers):
    # 下載日線 (計算 ATR, RSI)
    data = yf.download(tickers, period="3mo", interval="1d", progress=False, auto_adjust=True, threads=True)
    # 下載盤前/盤中 (計算 Gap) - 這裡抓 5 天是為了包含週五到週一的狀況
    intra = yf.download(tickers, period="5d", interval="1m", prepost=True, progress=False, auto_adjust=True, threads=True)
    return data, intra

def calculate_metrics(ticker, df_daily, df_intra, vix_val):
    try:
        # 強制轉數值
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if c in df_daily.columns:
                df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce')
        
        df_daily = df_daily.dropna()
        if len(df_daily) < 20: return None

        prev_close = float(df_daily['Close'].iloc[-1])
        
        # 取得即時價格 (Intraday Last) & 盤前高點
        curr_price = prev_close
        pre_high = prev_close
        
        # 處理 Intraday Data (抓取最新價格)
        if isinstance(df_intra.columns, pd.MultiIndex):
            # 檢查 ticker 是否在 columns level 1 中
            if ticker in df_intra.columns.levels[1]:
                df_m = df_intra.xs(ticker, axis=1, level=1).dropna()
                if not df_m.empty:
                    curr_price = float(df_m['Close'].iloc[-1])
                    # 抓盤前最高 (近似)
                    today_mask = df_m.index.date == df_m.index[-1].date()
                    if any(today_mask):
                        pre_high = float(df_m.loc[today_mask, 'High'].max())
                    else:
                        pre_high = curr_price
        else:
            # 單一 ticker 情況
            df_m = df_intra.dropna()
            if not df_m.empty:
                curr_price = float(df_m['Close'].iloc[-1])
                today_mask = df_m.index.date == df_m.index[-1].date()
                if any(today_mask):
                    pre_high = float(df_m.loc[today_mask, 'High'].max())
                else:
                    pre_high = curr_price

        # 計算 Gap %
        gap_pct = (curr_price - prev_close) / prev_close
        
        # 技術指標
        atr = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14).iloc[-1]
        atr_pct = atr / prev_close
        rsi = ta.rsi(df_daily['Close'], length=14).iloc[-1]
        
        # AI 特徵準備
        vol_ma20 = df_daily['Volume'].rolling(20).mean().iloc[-2]
        vol_last = df_daily['Volume'].iloc[-1]
        vol_ratio = vol_last / vol_ma20 if vol_ma20 > 0 else 1.0
        
        features = pd.DataFrame([[rsi, atr_pct, vol_ratio, gap_pct, vix_val]], 
                                columns=['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX'])
        
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
    print(f"\n>>> V6.2 Daily Gap & Dip Scanner (Union Mode)")
    print(f">>> Target: Holdings + Asset Pool")
    print(f">>> Thresholds: Sell Rip > {RIP_THRESHOLD:.1%}, Gap Up > {GAP_THRESHOLD:.1%}, Buy Dip < -{DIP_THRESHOLD:.1%}")
    print(f">>> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 載入模型 (選用)
    ai_model = None
    try:
        if os.path.exists(MODEL_PATH):
            ai_model = joblib.load(MODEL_PATH)
            # print("[Info] AI Model loaded.")
    except: pass

    # 2. 載入清單 (Union)
    tickers, tags_map = load_tickers_and_tags()
    
    if not tickers:
        print("[Error] No tickers found.")
        return

    print(f"Scanning {len(tickers)} tickers...")
    curr_vix = get_current_vix()
    
    # 3. 下載數據
    daily_data, intra_data = download_data(tickers)
    
    # 4. 處理數據格式
    if isinstance(daily_data.columns, pd.MultiIndex):
        daily_data = daily_data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        daily_data['Ticker'] = tickers[0]
        daily_data = daily_data.reset_index()

    daily_map = {t: g.set_index('Date') for t, g in daily_data.groupby('Ticker')}
    
    results = []
    
    # 5. 計算訊號
    for t in tickers:
        if t not in daily_map: continue
        
        metrics = calculate_metrics(t, daily_map[t], intra_data, curr_vix)
        if not metrics: continue
        
        gap = metrics['gap_pct']
        price = metrics['price']
        
        # --- 核心策略邏輯 V6.2 ---
        status = "Flat"
        action = "WAIT"
        
        if gap > RIP_THRESHOLD:  # [新增] Gap > 3.0%
            status = "🔴 SELL RIP"
            action = "STRONG SELL"
        elif gap > GAP_THRESHOLD: # Gap > 0.5%
            status = "🔴 GAP UP"
            action = "SELL/TRIM"
        elif gap < -DIP_THRESHOLD: # Gap < -3.0%
            status = "🟢 BUY DIP"
            action = "BUY OPEN"
        elif gap < -GAP_THRESHOLD:
            status = "🟡 GAP DOWN"
            action = "HOLD"
        else:
            status = "⚪ Flat"
            action = "-"
        
        # AI 預測 (輔助 Sell 決策，目前 AI 針對 Gap Up 訓練)
        ai_prob_str = "-"
        if ai_model and abs(gap) > 0.003:
            try:
                prob = ai_model.predict_proba(metrics['features'])[0][1]
                ai_prob_str = f"{prob:.0%}"
            except: pass
        
        # 標籤處理
        t_tags = tags_map.get(t, set())
        is_held = 'Held' in t_tags
        # 若不是庫存，顯示空白或 ASSET，減少視覺干擾
        tag_str = "[HOLD]" if is_held else "" 
        
        results.append({
            'Ticker': t,
            'Tag': tag_str,
            'Gap%': gap,
            'Price': price,
            'Status': status,
            'Action': action,
            'AI_Prob': ai_prob_str,
            'ATR%': metrics['atr_pct']
        })

    # 6. 排序與過濾
    # 排序邏輯：Gap 越大排越上面 (Gap Up / Sell Rip)，Gap 越小排越下面 (Deep Dip)
    results.sort(key=lambda x: x['Gap%'], reverse=True)
    
    print("\n" + "=" * 95)
    header = f"{'Ticker':<8} {'Tag':<6} {'Gap%':>8} {'Price':>10} {'Status':<12} {'Action':<10} {'AI Prob':>8} {'ATR%':>6}"
    print(header)
    print("-" * 95)
    
    significant_signals = 0
    
    for r in results:
        # [修改] 移除過濾器，顯示所有結果
        # if abs(r['Gap%']) < 0.005: continue
            
        significant_signals += 1
        
        # 視覺提示 (Alert Markers)
        marker = ""
        
        # 情況 A: 發現新的抄底機會 (Buy Dip)
        if "BUY DIP" in r['Status']: 
            marker = " <--- 🟢 BUY OPPORTUNITY"
        
        # 情況 B: 持股出現 Gap Up 或 Sell Rip (要賣)
        if "[HOLD]" in r['Tag'] and ("GAP UP" in r['Status'] or "SELL RIP" in r['Status']): 
            marker = " <--- 🔴 SELL SIGNAL"
        
        # 情況 C: 持股大跌 (可能加碼)
        if "[HOLD]" in r['Tag'] and "BUY DIP" in r['Status']:
            marker = " <--- 🟢 ADD POSITION"

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Gap%']*100:>7.2f}% {r['Price']:>10.2f} {r['Status']:<12} {r['Action']:<10} {r['AI_Prob']:>8} {r['ATR%']*100:>5.1f}%{marker}")
        
    print("=" * 95)
    print(f"Total Scanned: {len(results)}")
    
    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()