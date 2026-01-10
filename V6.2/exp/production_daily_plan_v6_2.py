import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# --- 設定 ---
RESOURCE_DIR = "../resource"
OUTPUT_DIR = "daily_plans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 核心模組化邏輯 (可被引用) ---

def clean_ticker(ticker):
    """修正 yfinance 代號格式"""
    if "BRK.B" in ticker: return "BRK-B"
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def calculate_er(series, window=5):
    """計算考夫曼效率比率 (Efficiency Ratio)"""
    if len(series) < window + 1:
        return 0.0
    subset = series.tail(window + 1)
    net_change = abs(subset.iloc[-1] - subset.iloc[0])
    sum_abs_change = subset.diff().abs().sum()
    if sum_abs_change == 0:
        return 0.0
    return net_change / sum_abs_change

def get_regime_decision(df_daily, ticker, window=5, threshold=0.6):
    """
    判斷該標的是否處於過度趨勢 (BLOCK) 或震盪 (PASS)
    """
    try:
        if df_daily is None or df_daily.empty or len(df_daily) < window + 1:
            return "UNKNOWN", 0.0, "Insufficient Data"
        
        er_value = calculate_er(df_daily['Close'], window=window)
        is_trend_regime = er_value > threshold
        
        status = "🛑 BLOCK" if is_trend_regime else "✅ PASS"
        action = "SKIP (Too Trendy)" if is_trend_regime else "Trade Gaps"
        
        return status, er_value, action
    except Exception as e:
        return "ERROR", 0.0, f"Error: {str(e)}"

# --- 執行邏輯 ---

def load_tickers():
    """
    載入所有資產池，包含：
    1. Final Asset Pool (潛在標的)
    2. Holding Asset Pool (目前持倉)
    3. Default (宏觀/熱門標的)
    """
    tickers = []
    
    # 1. 載入 Final Asset Pool
    path_asset = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(path_asset):
        with open(path_asset, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
            
    # 2. [新增] 載入 Holding Asset Pool (確保過濾持倉標的)
    path_holding = os.path.join(RESOURCE_DIR, "2025_holding_asset_pool.json")
    if os.path.exists(path_holding):
        with open(path_holding, 'r', encoding='utf-8') as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
    
    # 3. 預設熱門/宏觀標的 (確保不錯過指數波動)
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'MARA', 'PLTR', 'HOOD', 'UPST']
    tickers.extend(default)
    
    # 去重與清洗格式
    cleaned = sorted(list(set([clean_ticker(t) for t in tickers])))
    print(f"📋 Loaded {len(cleaned)} tickers from pools and defaults.")
    return cleaned

def generate_daily_plan():
    print(f"🚀 V6.2 Ultra: Generating Daily Battle Plan...")
    tickers = load_tickers()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📥 Fetching data for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, period="1mo", progress=True, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"❌ Critical Download Error: {e}")
        return

    plan_rows = []
    for ticker in tickers:
        try:
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker].dropna(subset=['Close'])
            else:
                df = data.dropna(subset=['Close'])
            
            if df.empty: continue

            status, er_val, action = get_regime_decision(df, ticker)
            
            if status != "UNKNOWN":
                plan_rows.append({
                    'Ticker': ticker,
                    'Latest_Date': df.index[-1].strftime("%Y-%m-%d"),
                    'Close': round(df['Close'].iloc[-1], 2),
                    'ER_5day': round(er_val, 4),
                    'Status': status,
                    'Action': action
                })
        except Exception:
            continue
            
    plan_df = pd.DataFrame(plan_rows)
    if plan_df.empty: 
        print("❌ No valid data processed.")
        return

    plan_df['Sort_Key'] = plan_df['Status'].apply(lambda x: 0 if "PASS" in x else 1)
    plan_df = plan_df.sort_values(by=['Sort_Key', 'ER_5day']).drop(columns=['Sort_Key'])
    
    csv_path = os.path.join(OUTPUT_DIR, f"V6.2_Ultra_Plan_{today_str}.csv")
    plan_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"🏆 V6.2 Ultra Daily Plan Generated: {csv_path}")
    print("="*50)

    print("\n✅ Top 5 SAFE Tickers (Perfect Chop):")
    safe_df = plan_df[plan_df['Status'] == "✅ PASS"]
    print(safe_df.head(5)[['Ticker', 'ER_5day', 'Close']].to_string(index=False))
    
    print("\n🛑 Top 5 DANGEROUS Tickers (Extreme Trend - DO NOT SHORT):")
    danger_df = plan_df[plan_df['Status'] == "🛑 BLOCK"].sort_values('ER_5day', ascending=False)
    print(danger_df.head(5)[['Ticker', 'ER_5day', 'Close']].to_string(index=False))

    print("\n💡 使用說明:")
    print("1. 開盤前檢查此表或執行 daily_gap_signal_generator.py。")
    print("2. 若標的 Status = 'BLOCK'，代表趨勢太強，無論跳空多大都應 SKIP。")
    print("3. 若 Status = 'PASS'，則結合開盤跳空進行操作 (Gap > 0.5% Short / < -0.5% Long)。")

if __name__ == "__main__":
    generate_daily_plan()