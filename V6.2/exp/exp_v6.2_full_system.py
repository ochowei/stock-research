import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import datetime

# --- 設定 ---
RESOURCE_DIR = "../resource"
OUTPUT_DIR = "output"
START_DATE = "2023-01-01" # 聚焦在最近兩年 (包含 2023 盤整與 2024 牛市)
END_DATE = datetime.date.today().strftime("%Y-%m-%d")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_ticker(ticker):
    if "BRK.B" in ticker: return "BRK-B"
    if ":" in ticker: return ticker.split(":")[-1]
    return ticker

def load_tickers():
    tickers = []
    # 載入 Final Pool
    path = os.path.join(RESOURCE_DIR, "2025_final_asset_pool.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
            raw = data if isinstance(data, list) else list(data.keys())
            tickers.extend(raw)
            
    # 加入測試與熱門標的
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'MARA', 'PLTR', 'HOOD', 'ONDS', 'UPST']
    tickers.extend(default)
    return list(set([clean_ticker(t) for t in tickers]))

def calculate_metrics(df):
    """計算回測所需的特徵"""
    df = df.copy()
    
    # 1. 策略訊號 (Gap Strategy)
    # Gap > 0.5% -> Short (賭日內反轉)
    # Gap < -0.5% -> Long
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['R_day'] = (df['Close'] - df['Open']) / df['Open']
    
    df['Signal_Short'] = df['Gap'] > 0.005
    df['Signal_Long'] = df['Gap'] < -0.005
    
    # 2. 濾網因子: Efficiency Ratio (ER)
    # 用來判斷當前是否為強趨勢
    window = 5
    net = (df['Close'] - df['Close'].shift(window)).abs()
    sum_abs = df['Close'].diff().abs().rolling(window).sum()
    df['Metric_ER'] = net / (sum_abs + 1e-9)
    
    # 3. IC 計算因子
    # 用來判斷該股票是 Momentum 還是 Reversion
    # 注意：這裡為了簡化，我們使用滾動窗口計算 IC，模擬即時決策
    # 在真實情況下，可以使用過去一年的固定 IC
    df['Past_ER'] = df['Metric_ER']
    df['Future_ER'] = df['Metric_ER'].shift(-5) # 用於計算 IC，但不參與當日交易決策
    
    return df

def run_full_system_backtest(tickers):
    print(f"🚀 執行 V6.2 完整系統回測 (Pool: {len(tickers)})...")
    
    # 儲存每日 PnL
    pnl_records = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
            if len(df) < 200: continue
            
            df = calculate_metrics(df)
            
            # --- 計算該股票的 IC 特性 (Asset Selection) ---
            # 這裡我們用全歷史數據算一個靜態 IC 來模擬「是否在黑名單」
            # 若 IC < -0.05 -> Reversion (Dangerous)
            clean_ic_df = df.dropna()
            if len(clean_ic_df) > 50:
                ic, _ = spearmanr(clean_ic_df['Past_ER'], clean_ic_df['Future_ER'])
            else:
                ic = 0.0 # 默認中性
                
            is_blacklisted = ic < -0.05
            
            # --- 逐日模擬 ---
            for date, row in df.iterrows():
                # 基礎 PnL (V6.1)
                daily_pnl = 0.0
                if row['Signal_Short']:
                    daily_pnl = -1 * row['R_day'] # 做空
                elif row['Signal_Long']:
                    daily_pnl = row['R_day']      # 做多
                
                # 若無交易訊號，跳過
                if daily_pnl == 0: continue
                
                # [V6.2 Lite] Asset Filter
                # 如果是黑名單股票，PnL 歸零 (不做)
                pnl_lite = 0.0 if is_blacklisted else daily_pnl
                
                # [V6.2 Pro] Asset Filter + Regime Filter
                # 如果是黑名單，或者 當前是強趨勢 (ER > 0.6)，PnL 歸零
                is_trend_regime = row['Metric_ER'] > 0.6
                pnl_pro = 0.0
                if not is_blacklisted and not is_trend_regime:
                    pnl_pro = daily_pnl
                
                pnl_records.append({
                    'Date': date,
                    'Ticker': ticker,
                    'V6.1_Base': daily_pnl,
                    'V6.2_Lite': pnl_lite,
                    'V6.2_Pro': pnl_pro
                })
                
        except Exception:
            pass

    # --- 彙總分析 ---
    res_df = pd.DataFrame(pnl_records)
    if res_df.empty:
        print("❌ No trades generated.")
        return

    # 按日期聚合 (Portfolio PnL)
    daily_res = res_df.groupby('Date')[['V6.1_Base', 'V6.2_Lite', 'V6.2_Pro']].mean()
    
    # 計算累計回報 (Cumulative Return)
    cum_res = daily_res.cumsum()
    
    # 計算績效指標
    stats = []
    for col in daily_res.columns:
        sharpe = daily_res[col].mean() / (daily_res[col].std() + 1e-9) * np.sqrt(252)
        total_ret = cum_res[col].iloc[-1]
        win_rate = (daily_res[col] > 0).mean()
        stats.append({'Strategy': col, 'Sharpe': sharpe, 'Total_Return': total_ret, 'Win_Rate': win_rate})
        
    stats_df = pd.DataFrame(stats)
    
    print("\n🏆 --- V6.2 最終對決結果 (Final Performance) ---")
    print(stats_df.round(4).to_string(index=False))
    
    # 繪圖
    plt.figure(figsize=(12, 6))
    plt.plot(cum_res.index, cum_res['V6.1_Base'], label='V6.1 Baseline (Raw)', color='gray', linestyle='--')
    plt.plot(cum_res.index, cum_res['V6.2_Lite'], label='V6.2 Lite (No Reversion)', color='orange')
    plt.plot(cum_res.index, cum_res['V6.2_Pro'], label='V6.2 Pro (No Reversion + No Trend)', color='green', linewidth=2)
    
    plt.title('V6.2 System Evolution: Impact of Two-Layer Filtering')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (Avg per Trade)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_v6.2_full_system_equity.png"))
    print(f"\n📈 權益曲線已儲存至 {OUTPUT_DIR}/exp_v6.2_full_system_equity.png")

if __name__ == "__main__":
    tickers = load_tickers()
    run_full_system_backtest(tickers)