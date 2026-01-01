import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
import warnings
import matplotlib.pyplot as plt

# --- 設定 ---
warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'dip_model.joblib')

TEST_START = '2024-01-01'
TEST_END   = '2025-12-31'
GAP_THRESHOLD = -0.03   # Deep Dip
TX_COST = 0.002         # 交易成本 0.2%

# [關鍵] 測試不同的單筆資金上限
# 1.0 = 100% (All-in), 0.2 = 20%, 0.1 = 10%, 0.05 = 5%
POS_SIZES = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0] 

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    all_tickers = tickers + ['^VIX']
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=TEST_START, end=TEST_END, interval='1d', auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        data['Ticker'] = all_tickers[0]
        data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()
    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    stock_df = data[data['Ticker'] != '^VIX']
    return stock_df, vix_df

def build_features_and_predict(df, vix_df, model):
    df = df.sort_index()
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # VIX T-1
    vix_shifted = vix_df.shift(1).rename(columns={'VIX': 'VIX_Prev'})
    df = df.join(vix_shifted, how='left')
    df['VIX'] = df['VIX_Prev'].ffill().bfill().fillna(20.0)

    # Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    if len(df) < 20: return pd.DataFrame()
    try:
        df['RSI_14'] = ta.rsi(df['Close'].shift(1), length=14)
        df['ATR_14'] = ta.atr(df['High'].shift(1), df['Low'].shift(1), df['Close'].shift(1), length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
        df['MA20_Prev'] = df['Close'].shift(1).rolling(19).mean() 
        df['MA20_Sim'] = (df['MA20_Prev'] * 19 + df['Open']) / 20
        df['Dist_MA20'] = (df['Open'] / df['MA20_Sim']) - 1
    except: return pd.DataFrame()

    df['Vol_MA20'] = df['Volume'].shift(1).rolling(20).mean()
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']
    
    # Filter & Predict
    df = df.dropna(subset=['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Dist_MA20'])
    df = df[df['Gap_Pct'] < GAP_THRESHOLD]
    
    if df.empty: return df
    
    features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20']
    try:
        df['Pred'] = model.predict(df[features])
    except:
        df['Pred'] = 0
    return df

def run_portfolio_simulation(full_df, max_pos_size):
    """
    模擬資金池管理:
    - 每日訊號數量 N
    - 權重 = min(1/N, max_pos_size)
    - 若 N * max_pos_size < 1，則部分持有現金 (Cash Drag)
    """
    # 只取 Smart Dip 訊號
    df = full_df[full_df['Pred'] == 1].copy()
    
    # 計算單日個股報酬 (不設停損，只扣成本)
    df['Final_Return'] = ((df['Close'] - df['Open']) / df['Open']) - TX_COST
    
    # 根據日期分組
    daily_groups = df.groupby(df.index)
    
    portfolio_returns = {}
    all_dates = pd.date_range(TEST_START, TEST_END)
    
    for date in all_dates:
        if date in daily_groups.groups:
            group = daily_groups.get_group(date)
            n_signals = len(group)
            
            # 資金分配邏輯
            # 如果只有 1 檔，且上限是 0.1，則只投入 10% 資金，90% 現金 (報酬為 0)
            # 如果有 20 檔，且上限是 0.1，理論需 200%，但只能 100%，所以每檔 1/20 = 0.05
            weight = min(max_pos_size, 1.0 / n_signals)
            
            # 當日總回報 = 平均股票回報 * (投入資金比例)
            # 投入資金比例 = weight * n_signals
            avg_stock_ret = group['Final_Return'].mean()
            invested_capital = weight * n_signals
            
            daily_port_ret = avg_stock_ret * invested_capital
            portfolio_returns[date] = daily_port_ret
        else:
            portfolio_returns[date] = 0.0
            
    daily_series = pd.Series(portfolio_returns)
    equity = (1 + daily_series).cumprod()
    
    total_ret = (equity.iloc[-1] - 1) * 100
    mdd = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    sharpe = (daily_series.mean() / daily_series.std()) * np.sqrt(252) if daily_series.std() != 0 else 0
    
    return total_ret, mdd, sharpe, equity

def main():
    print("=== Position Sizing (Portfolio) Simulation ===")
    print(f"Strategy: Smart Dip (SL=None), Cost={TX_COST:.1%}")
    
    if not os.path.exists(MODEL_PATH):
        print("[Error] Model not found.")
        return
        
    model = joblib.load(MODEL_PATH)
    tickers = load_tickers()
    stock_raw, vix_raw = fetch_data(tickers)
    
    print("Preparing data...")
    dfs = []
    for t, g in stock_raw.groupby('Ticker'):
        d = build_features_and_predict(g.set_index('Date'), vix_raw, model)
        if not d.empty: dfs.append(d)
    
    if not dfs: return
    full_df = pd.concat(dfs).sort_index()
    print(f"Total Smart Trades: {len(full_df[full_df['Pred']==1])}")
    
    results = []
    equity_curves = {}
    
    print(f"\n{'Max Pos %':<12} {'Total Ret':<15} {'MDD':<15} {'Sharpe':<10}")
    print("-" * 55)
    
    for size in POS_SIZES:
        ret, mdd, shp, eq = run_portfolio_simulation(full_df, size)
        label = f"{size*100:.0f}%"
        print(f"{label:<12} {ret:>14.2f}% {mdd:>14.2f}% {shp:>9.2f}")
        results.append({'Size': label, 'Ret': ret, 'MDD': mdd, 'Sharpe': shp})
        equity_curves[label] = eq
        
    # 簡單推薦
    # 尋找 Sharpe 最高 或者 MDD 可接受(-20%以內) 且回報最高的設定
    best_sharpe = max(results, key=lambda x: x['Sharpe'])
    
    print("\n" + "="*55)
    print(f"🏆 Best Efficiency (Sharpe): {best_sharpe['Size']}")
    print(f"   Return: {best_sharpe['Ret']:.2f}% | MDD: {best_sharpe['MDD']:.2f}%")
    print("="*55)
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    for label, eq in equity_curves.items():
        if label in ['100%', '50%', '20%', '10%', '5%']: # 只畫幾個代表性的
            plt.plot(eq, label=f"Size {label} (MDD {results[[r['Size'] for r in results].index(label)]['MDD']:.0f}%)")
    
    plt.title('Impact of Position Sizing on Portfolio Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    chart_path = os.path.join(OUTPUT_DIR, 'portfolio_sizing_comparison.png')
    plt.savefig(chart_path)
    print(f"\n[Saved] Chart: {chart_path}")

if __name__ == '__main__':
    main()