import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import time

# --- 1. 設定與參數 ---

# 基礎路徑設定 (自動抓取相對路徑)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', '..', 'V6.0', 'resource') # 指向 V6.0 資源資料夾
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 來源檔案
ASSET_POOL_FILE = '2025_final_asset_pool.json'
TOXIC_POOL_FILE = '2025_final_toxic_asset_pool.json'

# 動能股黑名單 (這些股票跳空高開通常是噴出，不適合賣出)
MOMENTUM_BLACKLIST = [
    'NVDA', 'APP', 'NET', 'ANET', 'AMD', 'TSLA', 'MSFT', 'GOOG', 'AMZN', 
    'LLY', 'NVO', 'V', 'MCD', 'IBM', 'QCOM', 'SMCI', 'PLTR', 'COIN', 'MSTR'
]

# 策略參數 (V6.1 最佳化設定)
GAP_THRESHOLD_PCT = 0.005  # Gap > 0.5% 觸發
FADE_THRESHOLD_PCT = 0.010 # Fade > 1.0% 為強力訊號

# --- 2. 工具函數 ---

def load_tickers_from_json(filename):
    path = os.path.join(RESOURCE_DIR, filename)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_list = json.load(f)
        # 清洗 "NYSE:MP" -> "MP", "BRK.B" -> "BRK-B"
        cleaned_list = [t.split(':')[-1].strip().replace('.', '-') for t in raw_list]
        return list(set(cleaned_list))
    except Exception as e:
        print(f"[Error] 無法讀取清單 {filename}: {e}")
        return []

def get_market_data(tickers):
    """
    抓取即時數據：昨收、最新價、盤前最高價
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 正在下載 {len(tickers)} 檔股票的盤前數據...")
    
    data_map = {}
    
    # A. 抓取昨收 (Prev Close) - 使用日線
    try:
        df_daily = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False)
        # 處理 MultiIndex 或 Single Ticker
        closes = df_daily['Close'] if len(tickers) > 1 else pd.DataFrame({tickers[0]: df_daily['Close']})
        # 取最後一筆非 NaN 的值作為昨收
        prev_closes = closes.ffill().iloc[-1]
    except Exception as e:
        print(f"[Error] 無法取得昨收價: {e}")
        return {}

    # B. 抓取盤前數據 (Intraday 1m)
    try:
        # 下載包含盤前盤後的 1分K
        df_intraday = yf.download(tickers, period="5d", interval="1m", prepost=True, auto_adjust=True, progress=False)
        
        # 取得今天的日期 (美東時間)
        if df_intraday.empty:
            print("[Error] 下載的盤前數據為空")
            return {}
            
        # 轉換時區以確保日期正確 (yfinance 預設 UTC 或 America/New_York)
        if df_intraday.index.tz is None:
            df_intraday.index = df_intraday.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df_intraday.index = df_intraday.index.tz_convert('America/New_York')
            
        current_date = df_intraday.index[-1].date()
        
        # 針對每一檔股票處理
        for ticker in tickers:
            try:
                # 1. 取得昨收
                prev_close = prev_closes[ticker] if len(tickers) > 1 else prev_closes.iloc[0]
                
                # 2. 取得該股票的分時數據
                if len(tickers) > 1:
                    if ticker not in df_intraday['Close'].columns:
                        continue
                    series_close = df_intraday['Close'][ticker].dropna()
                    series_high = df_intraday['High'][ticker].dropna() if 'High' in df_intraday.columns else series_close
                else:
                    series_close = df_intraday['Close'].dropna()
                    series_high = df_intraday['High'].dropna() if 'High' in df_intraday.columns else series_close

                if series_close.empty:
                    continue

                # 3. 篩選「今日盤前」數據 (04:00 AM 以後)
                today_mask = series_close.index.date == current_date
                today_close = series_close[today_mask]
                today_high = series_high[today_mask]
                
                if today_close.empty:
                    curr_price = np.nan
                    pre_high = np.nan
                else:
                    curr_price = today_close.iloc[-1]
                    pre_high = today_high.max()

                # 4. 計算 Pre-Fade %
                # 公式: (盤前最高 - 目前價格) / 盤前最高
                if pd.notna(pre_high) and pre_high > 0 and pd.notna(curr_price):
                    pre_fade = (pre_high - curr_price) / pre_high
                else:
                    pre_fade = 0.0

                data_map[ticker] = {
                    'prev_close': prev_close,
                    'curr_price': curr_price,
                    'pre_high': pre_high,
                    'pre_fade': pre_fade
                }
                
            except Exception as e:
                # 個別股票錯誤不中斷迴圈
                continue
                
    except Exception as e:
        print(f"[Error] 無法取得盤前價: {e}")

    return data_map

def generate_live_dashboard():
    print(f"\n>>> 啟動 V6.1 Gap 策略實盤儀表板")
    print(f"> 執行時間建議: 美股開盤前 15~30 分鐘 (TW 21:00 / 22:00)")
    print("-" * 60)
    
    # 1. 載入清單
    pool_toxic = load_tickers_from_json(TOXIC_POOL_FILE)
    pool_asset = load_tickers_from_json(ASSET_POOL_FILE)
    
    # 合併並過濾黑名單
    all_tickers = list(set(pool_toxic + pool_asset))
    valid_tickers = [t for t in all_tickers if t not in MOMENTUM_BLACKLIST]
    
    print(f"監控標的: {len(valid_tickers)} 檔 (已排除黑名單 {len(MOMENTUM_BLACKLIST)} 檔)")
    
    # 2. 取得數據
    market_data = get_market_data(valid_tickers)
    
    report_data = []
    
    for ticker in valid_tickers:
        if ticker not in market_data:
            continue
            
        data = market_data[ticker]
        prev_close = data['prev_close']
        curr_price = data['curr_price']
        pre_high = data['pre_high']
        pre_fade = data['pre_fade']
        
        if pd.isna(curr_price) or prev_close <= 0:
            continue
            
        # 計算 Gap %
        gap_pct = (curr_price - prev_close) / prev_close
        
        # 判斷狀態
        category = "Toxic" if ticker in pool_toxic else "Standard"
        
        status = "WAIT"
        signal_score = 0 # 用於排序
        
        if gap_pct > GAP_THRESHOLD_PCT:
            if pre_fade > FADE_THRESHOLD_PCT:
                status = "🔴 STRONG SELL"
                signal_score = 2
            else:
                status = "🔴 SELL"
                signal_score = 1
                
        # 僅顯示 Gap > 0 的股票 (或是接近門檻的)
        if gap_pct > 0.0:
            report_data.append({
                'Ticker': ticker,
                'Category': category,
                'Prev Close': prev_close,
                'Curr Price': curr_price,
                'Gap %': gap_pct,
                'Pre High': pre_high,
                'Fade %': pre_fade,
                'Status': status,
                'Score': signal_score
            })
    
    # 3. 轉為 DataFrame 並排序
    if not report_data:
        print("目前沒有任何股票 Gap > 0。")
        return

    df = pd.DataFrame(report_data)
    
    # 排序：訊號強度 > Gap幅度
    df.sort_values(by=['Score', 'Gap %'], ascending=[False, False], inplace=True)
    
    # 4. 輸出美化報表
    print("\n" + "="*100)
    print(f"【V6.1 盤前訊號】 (Gap > {GAP_THRESHOLD_PCT*100}% | Fade > {FADE_THRESHOLD_PCT*100}%)")
    print("-" * 100)
    print(f"{'Ticker':<8} {'Category':<10} {'Gap %':>8} {'Fade %':>8} {'Price':>8} {'PreHigh':>8} {'Status':<15}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        # 格式化顯示
        gap_str = f"{row['Gap %']*100:+.2f}%"
        fade_str = f"{row['Fade %']*100:.2f}%"
        
        # 顏色/標記
        mark = ">>" if row['Score'] > 0 else "  "
        
        print(f"{mark} {row['Ticker']:<5} {row['Category']:<10} "
              f"{gap_str:>8} {fade_str:>8} "
              f"{row['Curr Price']:>8.2f} {row['Pre High']:>8.2f} "
              f"{row['Status']:<15}")
              
    print("="*100)
    
    # 5. 存檔
    output_file = os.path.join(OUTPUT_DIR, f'gap_signals_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(output_file, index=False)
    print(f"\n[Saved] 完整數據已儲存: {output_file}")

if __name__ == '__main__':
    try:
        generate_live_dashboard()
    except KeyboardInterrupt:
        print("\n程式已手動停止。")
    except Exception as e:
        print(f"\n[Critical Error] {e}")