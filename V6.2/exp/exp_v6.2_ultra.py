import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

# --- 設定 ---
RESOURCE_DIR = "../resource"
OUTPUT_DIR = "output"
START_DATE = "2023-01-01"
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_ticker(ticker):
    if "BRK.B" in ticker: return "BRK-B"
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def load_tickers():
    tickers = []
    path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'MARA', 'PLTR', 'HOOD', 'ONDS', 'UPST']
    tickers.extend(default)
    return list(set([clean_ticker(t) for t in tickers]))

def calculate_metrics(df):
    df = df.copy()
    # 策略訊號
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['R_day'] = (df['Close'] - df['Open']) / df['Open']
    df['Signal_Short'] = df['Gap'] > 0.005
    df['Signal_Long'] = df['Gap'] < -0.005
    
    # 濾網因子: ER
    window = 5
    net = (df['Close'] - df['Close'].shift(window)).abs()
    sum_abs = df['Close'].diff().abs().rolling(window).sum()
    df['Metric_ER'] = net / (sum_abs + 1e-9)
    
    return df

def run_ultra_backtest(tickers):
    print(f"🚀 執行 V6.2 Ultra 驗證 (Pool: {len(tickers)})...")
    
    pnl_records = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
            if len(df) < 200: continue
            
            df = calculate_metrics(df)
            
            # 模擬交易
            for date, row in df.iterrows():
                daily_pnl = 0.0
                if row['Signal_Short']: daily_pnl = -1 * row['R_day']
                elif row['Signal_Long']: daily_pnl = row['R_day']
                
                if daily_pnl == 0: continue
                
                # 定義濾網狀態
                is_trend_regime = row['Metric_ER'] > 0.6
                
                # 1. V6.1 Base (無濾網)
                pnl_base = daily_pnl
                
                # 2. V6.2 Pro (黑名單 + 趨勢濾網)
                # 這裡無法精確重現 IC 黑名單，直接引用 Pro 邏輯：假設它是「乖寶寶」才做
                # 但為了對比 Ultra，我們先跳過 Pro 的精確重現，專注於 Ultra 的邏輯
                
                # 3. V6.2 Ultra (無黑名單 + 趨勢濾網)
                # 邏輯：所有股票都做，但只要遇到趨勢就閃
                pnl_ultra = 0.0
                if not is_trend_regime:
                    pnl_ultra = daily_pnl

                pnl_records.append({
                    'Date': date,
                    'Ticker': ticker,
                    'V6.1_Base': pnl_base,
                    'V6.2_Ultra': pnl_ultra
                })
                
        except Exception:
            pass

    # 分析
    res_df = pd.DataFrame(pnl_records)
    if res_df.empty: return

    daily_res = res_df.groupby('Date')[['V6.1_Base', 'V6.2_Ultra']].mean()
    cum_res = daily_res.cumsum()
    
    stats = []
    for col in daily_res.columns:
        sharpe = daily_res[col].mean() / (daily_res[col].std() + 1e-9) * np.sqrt(252)
        total_ret = cum_res[col].iloc[-1]
        win_rate = (daily_res[col] > 0).mean()
        stats.append({'Strategy': col, 'Sharpe': sharpe, 'Total_Return': total_ret, 'Win_Rate': win_rate})
        
    stats_df = pd.DataFrame(stats)
    
    print("\n🏆 --- V6.2 Ultra vs Base ---")
    print(stats_df.round(4).to_string(index=False))
    
    # 畫圖
    plt.figure(figsize=(10, 6))
    plt.plot(cum_res.index, cum_res['V6.1_Base'], label='V6.1 Base', color='gray', alpha=0.5)
    plt.plot(cum_res.index, cum_res['V6.2_Ultra'], label='V6.2 Ultra (Trend Filter Only)', color='purple', linewidth=2)
    plt.title('Is "Trend Filter Only" Better Than Blacklist?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_v6.2_ultra_equity.png"))
    print(f"\n📈 圖表: {OUTPUT_DIR}/exp_v6.2_ultra_equity.png")

if __name__ == "__main__":
    tickers = load_tickers()
    run_ultra_backtest(tickers)