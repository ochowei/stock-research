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

# [變更] 模型路徑設定 (三模型)
SELL_MODEL_PATH = os.path.join(OUTPUT_DIR, 'exp_07_model.joblib')
MOM_MODEL_PATH = os.path.join(OUTPUT_DIR, 'momentum_model.joblib')
DIP_MODEL_PATH = os.path.join(OUTPUT_DIR, 'dip_model.joblib')  # [新增] Dip 模型路徑

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 檔案設定
ASSET_POOL_FILE = '2025_final_asset_pool.json'
HOLDING_POOL_FILE = '2025_holding_asset_pool.json'

# 策略參數
GAP_THRESHOLD = 0.005      # 0.5% (Gap Up 賣出門檻)
RIP_THRESHOLD = 0.03       # 3.0% (Sell Rip 賣出門檻)
DIP_THRESHOLD = 0.03       # 3.0% (Buy Dip 買進門檻 - 取絕對值)
AI_CONFIDENCE_LV = 0.50    # Sell AI 信心門檻
MOMENTUM_THRESHOLD = 0.53  # Momentum AI 信心門檻 (53%)
DIP_CONFIDENCE_LV = 0.50   # [新增] Dip AI 信心門檻

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
        
        # [新增] Dist_MA20 (乖離率) 計算 - 為了 Dip Model
        # 邏輯：模型訓練時使用 (MA19_Prev * 19 + Open) / 20 作為當日模擬 MA20
        # 這裡我們用 curr_price 代替 Open (即時模擬)
        ma19_prev = df_daily['Close'].tail(19).mean() # 取最後 19 天的收盤平均
        ma20_sim = (ma19_prev * 19 + curr_price) / 20
        dist_ma20 = (curr_price / ma20_sim) - 1

        # [變更] 加入 Dist_MA20 到特徵 DataFrame (共 6 個特徵)
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
    print(f"\n>>> V6.3 Daily Gap & Dip Scanner (Tri-Model Enhanced)")
    print(f">>> Target: Holdings + Asset Pool")
    print(f">>> Thresholds: Sell Rip > {RIP_THRESHOLD:.1%}, Gap Up > {GAP_THRESHOLD:.1%}, Buy Dip < -{DIP_THRESHOLD:.1%}")
    print(f">>> AI Thresholds: Mom > {MOMENTUM_THRESHOLD:.0%}, Dip > {DIP_CONFIDENCE_LV:.0%}")
    print(f">>> Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # 1. 載入模型 (三模型)
    sell_model = None
    mom_model = None
    dip_model = None
    
    try:
        if os.path.exists(SELL_MODEL_PATH):
            sell_model = joblib.load(SELL_MODEL_PATH)
            # print("[Info] Sell Model loaded.")
    except Exception as e:
        print(f"[Warning] Failed to load Sell Model: {e}")
        
    try:
        if os.path.exists(MOM_MODEL_PATH):
            mom_model = joblib.load(MOM_MODEL_PATH)
            # print("[Info] Momentum Model loaded.")
    except Exception as e:
        print(f"[Warning] Failed to load Momentum Model: {e}")

    try:
        if os.path.exists(DIP_MODEL_PATH):
            dip_model = joblib.load(DIP_MODEL_PATH)
            # print("[Info] Dip Model loaded.")
    except Exception as e:
        print(f"[Warning] Failed to load Dip Model: {e}")

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
        feats = metrics['features']
        
        # --- AI 預測 ---
        # 取得 Sell Model 機率 (注意：舊模型只需前 5 個特徵)
        sell_prob_str = "-"
        if sell_model:
            try:
                # 只取前 5 個欄位餵給 Sell Model
                sell_prob = sell_model.predict_proba(feats.iloc[:, :5])[0][1]
                sell_prob_str = f"{sell_prob:.0%}"
            except: pass
            
        # 取得 Momentum Model 機率 (注意：舊模型只需前 5 個特徵)
        mom_prob_str = "-"
        mom_prob = 0.0
        if mom_model:
            try:
                # 只取前 5 個欄位餵給 Mom Model
                mom_prob = mom_model.predict_proba(feats.iloc[:, :5])[0][1]
                mom_prob_str = f"{mom_prob:.0%}"
            except: pass

        # [新增] 取得 Dip Model 機率 (使用完整 6 個特徵)
        dip_prob_str = "-"
        dip_prob = 0.0
        if dip_model:
            try:
                # Dip Model 需要包含 Dist_MA20
                dip_prob = dip_model.predict_proba(feats)[0][1]
                dip_prob_str = f"{dip_prob:.0%}"
            except: pass

        # --- 核心策略邏輯 V6.3 (含 Dip) ---
        status = "Flat"
        action = "WAIT"
        
        if gap > RIP_THRESHOLD:  # Gap > 3.0% (Sell Rip)
            # [邏輯升級] 檢查動能
            if mom_prob > MOMENTUM_THRESHOLD:
                status = "🚀 ROCKET"
                action = "HOLD/BUY"
            else:
                status = "🔴 SELL RIP"
                action = "STRONG SELL"
                
        elif gap > GAP_THRESHOLD: # Gap > 0.5% (Gap Up)
            # [邏輯升級] 檢查動能
            if mom_prob > MOMENTUM_THRESHOLD:
                status = "🟢 MOMENTUM"
                action = "HOLD"
            else:
                status = "🔴 GAP UP"
                action = "SELL/TRIM"
                
        elif gap < -DIP_THRESHOLD: # Gap < -3.0% (Buy Dip)
            # [邏輯升級] 檢查 Dip Model 信心
            if dip_prob > DIP_CONFIDENCE_LV:
                status = "🟢 SMART DIP"
                action = "BUY OPEN"
            else:
                status = "🔵 WEAK DIP"
                action = "WATCH"
                
        elif gap < -GAP_THRESHOLD:
            status = "🟡 GAP DOWN"
            action = "HOLD"
        else:
            status = "⚪ Flat"
            action = "-"
        
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
            'Sell%': sell_prob_str, 
            'Mom%': mom_prob_str,
            'Dip%': dip_prob_str,   # [新增]
            'ATR%': metrics['atr_pct']
        })

    # 6. 排序與過濾
    # 排序邏輯：Gap 越大排越上面 (Gap Up / Sell Rip)，Gap 越小排越下面 (Deep Dip)
    results.sort(key=lambda x: x['Gap%'], reverse=True)
    
    print("\n" + "=" * 115)
    # [變更] 新增 Dip% 欄位，調整寬度
    header = f"{'Ticker':<8} {'Tag':<6} {'Gap%':>8} {'Price':>10} {'Status':<12} {'Action':<12} {'Sell%':>6} {'Mom%':>6} {'Dip%':>6} {'ATR%':>6}"
    print(header)
    print("-" * 115)
    
    significant_signals = 0
    
    for r in results:
        significant_signals += 1
        
        # 視覺提示 (Alert Markers)
        marker = ""
        
        # 情況 A: 發現新的抄底機會 (Buy Dip)
        if "SMART DIP" in r['Status']: 
            marker = " <--- 🟢 AI APPROVED BUY"
        
        # 情況 B: 持股出現 Gap Up 或 Sell Rip (要賣)
        # 排除 ROCKET/MOMENTUM 狀態
        if "[HOLD]" in r['Tag'] and ("GAP UP" in r['Status'] or "SELL RIP" in r['Status']): 
            marker = " <--- 🔴 SELL SIGNAL"
            
        # 情況 C: 強勢股續抱提示
        if "ROCKET" in r['Status'] or "MOMENTUM" in r['Status']:
             marker = " <--- 🔥 HIGH MOMENTUM"
        
        # 情況 D: 持股大跌 (可能加碼)
        if "[HOLD]" in r['Tag'] and "SMART DIP" in r['Status']:
            marker = " <--- 🟢 ADD POSITION"

        print(f"{r['Ticker']:<8} {r['Tag']:<6} {r['Gap%']*100:>7.2f}% {r['Price']:>10.2f} {r['Status']:<12} {r['Action']:<12} {r['Sell%']:>6} {r['Mom%']:>6} {r['Dip%']:>6} {r['ATR%']*100:>5.1f}%{marker}")
        
    print("=" * 115)
    print(f"Total Scanned: {len(results)}")
    
    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, f'daily_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n[Saved] {csv_path}")

if __name__ == '__main__':
    generate_report()