import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- 設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_START = '2023-01-01'
TEST_START = '2024-01-01'
TEST_END   = '2025-12-31'

# 測試範圍: 0.5% ~ 5.0%, 間隔 0.5%
THRESHOLDS = {f'Gap > {i/10:.1f}%': i/1000 for i in range(5, 55, 5)}

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        return list(set([t.split(':')[-1].strip().replace('.', '-') for t in json.load(f)]))

def fetch_data(tickers):
    print(f"Downloading data for {len(tickers)} tickers...")
    df = yf.download(tickers, start=DATA_START, end=TEST_END, interval='1d', auto_adjust=True, progress=True, threads=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        df['Ticker'] = tickers[0]
        df = df.reset_index()
    return df

def run_sweep(df):
    results = []
    df['Prev_Close'] = df.groupby('Ticker')['Close'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # 計算日內做空回報 (Open - Close) / Open
    df['Fade_Ret'] = (df['Open'] - df['Close']) / df['Open']

    # 排序以確保 Equity Curve 計算正確 (雖然這是 Trade-based)
    df = df.sort_values('Date')

    print(f"\n{'Threshold':<15} {'Count':<8} {'Win Rate':<10} {'Avg Ret':<10} {'Total Ret':<10} {'Max DD':<10} {'Sharpe':<8}")
    print("-" * 90)

    plot_data = {'Threshold': [], 'Sharpe': [], 'Total Return': [], 'Win Rate': [], 'Max Drawdown': []}

    for name, thres in THRESHOLDS.items():
        # 篩選訊號
        signals = df[df['Gap_Pct'] > thres].copy()
        
        if signals.empty:
            continue
            
        # 統計指標
        count = len(signals)
        win_rate = (signals['Fade_Ret'] > 0).mean()
        avg_ret = signals['Fade_Ret'].mean()
        total_ret = signals['Fade_Ret'].sum() # 單利加總
        
        # --- [新增] 計算 Max Drawdown ---
        # 這裡計算的是 "Strategy Equity Curve" (假設每次都全倉 or 固定比例複利)
        # 為了更貼近實際感受，我們用 (1 + Ret).cumprod()
        equity_curve = (1 + signals['Fade_Ret']).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # 夏普 (簡易估算)
        std_ret = signals['Fade_Ret'].std()
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        
        print(f"{name:<15} {count:<8} {win_rate:6.2%}     {avg_ret:6.3%}     {total_ret:6.2f}     {max_dd:7.2%}    {sharpe:5.2f}")

        plot_data['Threshold'].append(thres * 100) # 轉成 %
        plot_data['Sharpe'].append(sharpe)
        plot_data['Total Return'].append(total_ret)
        plot_data['Win Rate'].append(win_rate)
        plot_data['Max Drawdown'].append(max_dd)

    return plot_data

def main():
    print(f"=== EXP-02-C: Wide Spectrum Sweep (0.5% - 5.0%) ===")
    tickers = load_tickers()
    if not tickers: return
    
    df = fetch_data(tickers)
    data = run_sweep(df)
    
    # 繪圖
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # 總回報 (紅線)
    ax1.plot(data['Threshold'], data['Total Return'], 'r-o', label='Total Return')
    ax1.set_xlabel('Gap Threshold (%)')
    ax1.set_ylabel('Total Return (Sum)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.grid(True, alpha=0.3)
    
    # Max Drawdown (綠色虛線) - 這是新增的
    ax2 = ax1.twinx()
    ax2.plot(data['Threshold'], data['Max Drawdown'], 'g--^', label='Max Drawdown')
    ax2.set_ylabel('Max Drawdown', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # 標題與儲存
    plt.title('Gap Threshold Analysis: Return vs Risk (Max DD)')
    
    # 合併圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp_02_wide_sweep.png'))
    print(f"\nChart saved to output/exp_02_wide_sweep.png")

if __name__ == '__main__':
    main()