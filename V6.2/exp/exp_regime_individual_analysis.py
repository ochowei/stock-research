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
START_DATE = "2020-01-01"
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
            
    # 加入測試標的
    default = ['SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AMD', 'COIN', 'MSTR', 'GME', 'AMC', 'TLT']
    tickers.extend(default)
    return list(set([clean_ticker(t) for t in tickers]))

def calculate_efficiency_ratio(close_series, window=5):
    """計算考夫曼效率比率 (ER)"""
    net_change = (close_series - close_series.shift(window)).abs()
    sum_abs_change = close_series.diff().abs().rolling(window).sum()
    return net_change / (sum_abs_change + 1e-9)

def analyze_ticker_predictability(ticker):
    """
    分析單一標的的「體制可預測性」
    """
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, multi_level_index=False)
        if len(df) < 250: return None
        
        df = df.copy()
        
        # 1. 因子 (X): 過去 5 天的趨勢強度 (慣性)
        df['X_Prev_ER'] = calculate_efficiency_ratio(df['Close'], window=5)
        
        # 2. 因子 (X): 布林通道擠壓 (Squeeze)
        ma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['X_Squeeze'] = (ma20 + 2*std20 - (ma20 - 2*std20)) / ma20
        
        # 3. 因子 (X): 日夜相關性 (Day-Night Corr)
        r_day = (df['Close'] - df['Open']) / df['Open']
        r_night = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['X_Corr'] = r_day.rolling(15).corr(r_night)

        # 4. 目標 (Y): 未來 5 天的趨勢強度
        future_net = (df['Close'].shift(-5) - df['Close']).abs()
        future_sum = df['Close'].diff().abs().rolling(5).sum().shift(-5)
        df['Y_Future_ER'] = future_net / (future_sum + 1e-9)
        
        df = df.dropna()
        if len(df) < 100: return None

        # --- 計算 IC (Information Coefficient) ---
        # 使用 Spearman Rank Correlation (抗極端值)
        
        # IC_Inertia: "現在是趨勢，未來是不是也是趨勢？"
        ic_inertia, _ = spearmanr(df['X_Prev_ER'], df['Y_Future_ER'])
        
        # IC_Squeeze: "現在擠壓(數值小)，未來會不會噴出(趨勢大)？"
        # 注意：Squeeze 小代表變盤，所以預期是負相關
        ic_squeeze, _ = spearmanr(df['X_Squeeze'], df['Y_Future_ER'])
        
        # IC_Structure: "日夜同向(Corr高)，未來會不會是趨勢？"
        ic_corr, _ = spearmanr(df['X_Corr'], df['Y_Future_ER'])
        
        return {
            'Ticker': ticker,
            'IC_Inertia': ic_inertia, # 越高越好 (慣性強)
            'IC_Squeeze': ic_squeeze, # 越負越好 (擠壓後噴出)
            'IC_Corr': ic_corr,       # 越高越好 (結構延續)
            'Samples': len(df)
        }
        
    except Exception as e:
        return None

def run_analysis(tickers):
    print(f"🔬 開始個別標的分析 (Pool: {len(tickers)})...")
    results = []
    
    for t in tickers:
        res = analyze_ticker_predictability(t)
        if res:
            results.append(res)
            
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        print("❌ No results generated.")
        return

    # 排序：根據 IC_Inertia (慣性) 由大到小
    df_res = df_res.sort_values('IC_Inertia', ascending=False)
    
    print("\n🏆 --- Top 10 Most Predictable Tickers (By Inertia) ---")
    print("解釋: IC > 0.1 代表該股票具有顯著的「動能慣性」，即「一旦發動趨勢就會持續」。")
    print(df_res[['Ticker', 'IC_Inertia', 'IC_Squeeze', 'IC_Corr']].head(10).to_string(index=False))
    
    print("\n📉 --- Bottom 10 (Mean Reverting / Random) ---")
    print("解釋: IC 接近 0 或負值，代表無慣性或均值回歸。")
    print(df_res[['Ticker', 'IC_Inertia', 'IC_Squeeze', 'IC_Corr']].tail(10).to_string(index=False))

    # 統計分佈
    positive_ratio = (df_res['IC_Inertia'] > 0.1).mean()
    print(f"\n📊 統計摘要:")
    print(f"   具有顯著慣性 (IC > 0.1) 的標的比例: {positive_ratio:.2%}")
    print(f"   平均 IC_Inertia: {df_res['IC_Inertia'].mean():.4f}")
    
    # 儲存
    output_path = os.path.join(OUTPUT_DIR, "exp_regime_individual_analysis.csv")
    df_res.to_csv(output_path, index=False)
    print(f"\n✅ 詳細報告已儲存至: {output_path}")
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    plt.hist(df_res['IC_Inertia'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(0.1, color='red', linestyle='--', label='Predictable Threshold (0.1)')
    plt.axvline(0, color='gray', linestyle='-')
    plt.title('Distribution of Regime Predictability (IC_Inertia)')
    plt.xlabel('Spearman Correlation (Current ER vs Future ER)')
    plt.ylabel('Count of Tickers')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "exp_regime_individual_dist.png"))
    print(f"📈 分佈圖已儲存至: {OUTPUT_DIR}/exp_regime_individual_dist.png")

if __name__ == "__main__":
    tickers = load_tickers()
    run_analysis(tickers)