import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- 設定 ---
# 假設腳本在 V6.2/production/ 下，資源在 ../resource/
RESOURCE_DIR = "../resource"
OUTPUT_DIR = "daily_plans"

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_ticker(ticker):
    """修正 yfinance 代號格式"""
    if "BRK.B" in ticker: return "BRK-B"
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def load_tickers():
    """載入所有資產池 (包含之前的黑名單)"""
    tickers = []
    
    # 1. 載入 Final Pool
    path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            # 兼容 list 或 dict 格式
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
    
    # 2. 載入 Toxic Pool (V6.2 Ultra 也可以考慮做 Toxic，但要注意風險)
    # 這裡我們先專注於 Final Pool，因為這是 Ultra 回測的主要對象
    # 若您想激進一點，可以解開下面這行
    # path_toxic = os.path.join(RESOURCE_DIR, "2025_final_toxic_asset_pool.json")
    # if os.path.exists(path_toxic):
    #     with open(path_toxic, 'r') as f:
    #         data = json.load(f)
    #         raw = data if isinstance(data, list) else list(data.keys())
    #         tickers.extend(raw)

    # 3. 手動補充熱門標的 (確保不錯過)
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'MARA', 'PLTR', 'HOOD', 'UPST']
    tickers.extend(default)
    
    # 去重與清洗
    cleaned = list(set([clean_ticker(t) for t in tickers]))
    print(f"📋 Loaded {len(cleaned)} tickers for analysis.")
    return cleaned

def calculate_er(series, window=5):
    """
    計算考夫曼效率比率 (Efficiency Ratio)
    ER = |Price_t - Price_t-n| / Sum(|Price_i - Price_i-1|)
    """
    if len(series) < window + 1:
        return 0.0
    
    # 取最近 window+1 筆數據
    subset = series.tail(window + 1)
    
    net_change = abs(subset.iloc[-1] - subset.iloc[0])
    sum_abs_change = subset.diff().abs().sum()
    
    if sum_abs_change == 0:
        return 0.0
        
    return net_change / sum_abs_change

def generate_daily_plan():
    print(f"🚀 V6.2 Ultra: Generating Daily Battle Plan...")
    
    tickers = load_tickers()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # 下載數據 (取最近 30 天即可，計算 ER 只需 5 天)
    print("📥 Fetching latest market data...")
    # yfinance 批次下載較快
    try:
        data = yf.download(tickers, period="1mo", progress=True, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"❌ Critical Error downloading data: {e}")
        return

    plan_rows = []
    
    print("⚙️ Analyzing Regime (ER Filter)...")
    
    for ticker in tickers:
        try:
            # 處理 Multi-index DataFrame
            if len(tickers) > 1:
                df = data[ticker]
            else:
                df = data
            
            # 移除空值行
            df = df.dropna(subset=['Close'])
            
            if len(df) < 10:
                continue
            
            latest_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2]
            latest_date = df.index[-1].strftime("%Y-%m-%d")
            
            # --- 核心指標計算 ---
            er_5 = calculate_er(df['Close'], window=5)
            
            # --- V6.2 Ultra 邏輯 ---
            # 閾值: 0.6
            is_trend_regime = er_5 > 0.6
            
            status = "🛑 BLOCK" if is_trend_regime else "✅ PASS"
            action_guide = "SKIP (Too Trendy)" if is_trend_regime else "Trade Gaps"
            
            plan_rows.append({
                'Ticker': ticker,
                'Latest_Date': latest_date,
                'Close': round(latest_close, 2),
                'ER_5day': round(er_5, 4),
                'Status': status,
                'Action': action_guide
            })
            
        except Exception:
            # 某些 ticker 可能下載失敗或數據不足，忽略
            continue
            
    # 轉為 DataFrame 並排序
    plan_df = pd.DataFrame(plan_rows)
    
    if plan_df.empty:
        print("❌ No data processed.")
        return

    # 排序：先看 PASS 的，再按 ER 由小到大 (ER 越小越震盪，越安全)
    plan_df['Sort_Key'] = plan_df['Status'].apply(lambda x: 0 if "PASS" in x else 1)
    plan_df = plan_df.sort_values(by=['Sort_Key', 'ER_5day'])
    plan_df = plan_df.drop(columns=['Sort_Key'])
    
    # 輸出 CSV
    csv_filename = f"V6.2_Ultra_Plan_{today_str}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    plan_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"🏆 V6.2 Ultra Daily Plan Generated: {csv_path}")
    print("="*50)
    
    # 顯示摘要 (Top Safe & Top Danger)
    print("\n✅ Top 5 SAFE Tickers (Perfect Chop):")
    print(plan_df[plan_df['Status'] == "✅ PASS"].head(5)[['Ticker', 'ER_5day', 'Close']].to_string(index=False))
    
    print("\n🛑 Top 5 DANGEROUS Tickers (Extreme Trend - DO NOT SHORT):")
    # 顯示 ER 最高的 5 個 (最危險)
    print(plan_df[plan_df['Status'] == "🛑 BLOCK"].sort_values('ER_5day', ascending=False).head(5)[['Ticker', 'ER_5day', 'Close']].to_string(index=False))

    print("\n💡 使用說明:")
    print("1. 開盤前檢查此表。")
    print("2. 若 Status = 'BLOCK'，無論跳空多大，**絕對不做**。")
    print("3. 若 Status = 'PASS'，則觀察開盤跳空：")
    print("   - Gap > 0.5% -> Short")
    print("   - Gap < -0.5% -> Long")

if __name__ == "__main__":
    generate_daily_plan()