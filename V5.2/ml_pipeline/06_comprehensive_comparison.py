import pandas as pd
import numpy as np
import os
import json
import yfinance as yf
import matplotlib.pyplot as plt
from backtesting_utils import run_backtest, analyze_performance

# --- 輔助函數：載入 V5.1 數據 ---
def load_v5_1_equity(benchmark_path, all_dates, initial_capital=100000.0):
    """從 V5.1 交易紀錄重建資金曲線"""
    try:
        trades = pd.read_csv(benchmark_path, parse_dates=['entry_date', 'exit_date'])
    except FileNotFoundError:
        print(f"Warning: V5.1 Benchmark file not found at {benchmark_path}")
        return pd.Series(dtype=float)

    # 計算每日報酬 (簡單平均)
    daily_returns = trades.groupby('exit_date')['return'].mean()
    daily_returns = daily_returns.reindex(all_dates, fill_value=0)

    # 重建淨值
    equity_curve = pd.Series(index=all_dates, dtype=float)
    equity_curve.iloc[0] = initial_capital
    for i in range(1, len(equity_curve)):
        equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + daily_returns.iloc[i])
    
    return equity_curve

# --- 輔助函數：載入 V5.2 數據 (通用) ---
def load_v5_2_data(features_path, regime_signals_path, ticker_filter=None):
    """載入特徵與訊號，可選擇性過濾標的"""
    try:
        features_df = pd.read_parquet(features_path)
        regime_signals_df = pd.read_parquet(regime_signals_path)
    except FileNotFoundError as e:
        print(f"Error loading V5.2 data: {e}")
        return None

    # 過濾標的 (如果有的話)
    if ticker_filter:
        features_df = features_df[features_df.index.get_level_values('symbol').isin(ticker_filter)]

    # 合併訊號
    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')
    df.rename(columns={'signal': 'regime_signal'}, inplace=True)
    df['regime_signal'] = df['regime_signal'].ffill()
    df = df.set_index('timestamp')
    return df

# --- 輔助函數：下載 Buy & Hold 基準 ---
def get_spy_benchmark(start_date, end_date, initial_capital=100000.0):
    print(f"Downloading SPY benchmark ({start_date.date()} to {end_date.date()})...")
    
    # [Fix] 加入 auto_adjust=True 並處理輸出格式
    df = yf.download("SPY", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    
    if df.empty:
        return pd.Series(dtype=float)
        
    # [Fix] 確保取得的是 Series (單一序列) 而非 DataFrame
    # 處理 yfinance 可能回傳的多層索引 (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close_data = df['Close']['SPY']
            open_data = df['Open']['SPY']
        except KeyError:
            # 萬一結構只有一層 (舊版行為)
            close_data = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
            open_data = df['Open'].iloc[:, 0] if df['Open'].ndim > 1 else df['Open']
    else:
        close_data = df['Close']
        open_data = df['Open']

    # 強制轉換為 Series 並移除空值
    close_data = close_data.squeeze()
    open_data = open_data.squeeze()

    # 計算買入持有淨值
    if len(open_data) > 0 and open_data.iloc[0] > 0:
        shares = initial_capital / open_data.iloc[0]
        equity = close_data * shares
        return equity
    else:
        return pd.Series(dtype=float)

def calculate_metrics(curve):
    """計算單一曲線的績效指標"""
    if curve is None or curve.empty:
        return {k: 0 for k in ['Total Return', 'CAGR', 'Sharpe', 'MaxDD']}
    
    # [Safety] 確保輸入是 Series
    if isinstance(curve, pd.DataFrame):
        curve = curve.iloc[:, 0]

    returns = curve.pct_change().fillna(0)
    total_ret = (curve.iloc[-1] / curve.iloc[0]) - 1
    
    days = (curve.index[-1] - curve.index[0]).days
    years = days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # [Fix] 安全計算 Sharpe (處理 std 為 0 或 Series 的情況)
    std_val = returns.std()
    if isinstance(std_val, pd.Series): # 防呆
        std_val = std_val.iloc[0]
        
    sharpe = returns.mean() / std_val * np.sqrt(252) if std_val != 0 else 0
    
    roll_max = curve.cummax()
    drawdown = (curve - roll_max) / roll_max
    max_dd = drawdown.min()
    
    return {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'MaxDD': max_dd,
        'Final Equity': curve.iloc[-1]
    }

def main():
    # --- 路徑設定 ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) # V5.2/
    REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))    # Root/
    
    # 輸入檔案
    FEATURES_PATH = os.path.join(PROJECT_ROOT, 'features', 'stock_features.parquet')
    REGIME_PATH = os.path.join(PROJECT_ROOT, 'signals', 'regime_signals.parquet')
    ASSET_POOL_PATH = os.path.join(SCRIPT_DIR, 'asset_pool.json')
    V5_1_TRADES_PATH = os.path.join(REPO_ROOT, 'V5.1', 'ml_pipeline', 'analysis', 'minimalist_trades_fixed.csv')
    
    # 輸出目錄
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Starting Comprehensive Strategy Comparison ===")

    # 1. 準備共用變數
    with open(ASSET_POOL_PATH, 'r') as f:
        asset_pool_raw = json.load(f)
    custom_tickers = [t.split(':')[1].replace('.', '-') for t in asset_pool_raw]
    
    # 2. 生成 V5.2 Custom 曲線 (Risk-Managed)
    print("\n[1/4] Running V5.2 Custom Backtest...")
    df_custom = load_v5_2_data(FEATURES_PATH, REGIME_PATH, ticker_filter=custom_tickers)
    equity_custom = run_backtest(df_custom)
    
    # 取得共同時間軸
    if equity_custom.empty:
        print("Error: V5.2 Custom backtest failed. Aborting.")
        return
    
    common_index = equity_custom.index.sort_values()
    start_date = common_index[0]
    end_date = common_index[-1]

    # 3. 生成 V5.2 Index 曲線 (Stress Test)
    print("\n[2/4] Running V5.2 Index Backtest...")
    df_index = load_v5_2_data(FEATURES_PATH, REGIME_PATH) 
    equity_index = run_backtest(df_index)
    
    # [Fix] 使用 .ffill() 和 .bfill() 取代 method 參數
    equity_index = equity_index.reindex(common_index).ffill().bfill()

    # 4. 生成 V5.1 Benchmark 曲線 (Aggressive)
    print("\n[3/4] Reconstructing V5.1 Strategy Curve...")
    equity_v5_1 = load_v5_1_equity(V5_1_TRADES_PATH, common_index)
    
    # 5. 下載 Buy & Hold (SPY) 曲線
    print("\n[4/4] Fetching SPY Buy & Hold...")
    equity_spy = get_spy_benchmark(start_date, end_date)
    # [Fix] 使用 .ffill() 和 .bfill()
    equity_spy = equity_spy.reindex(common_index).ffill().bfill()
    
    # 歸一化 SPY 初始資金
    if not equity_spy.empty and equity_spy.iloc[0] > 0:
        equity_spy = equity_spy * (100000.0 / equity_spy.iloc[0])

    # --- 整合與分析 ---
    print("\nGenerating Report and Plots...")
    
    strategies = {
        'V5.2 Custom (Risk-Aware)': equity_custom,
        'V5.2 Index (Stress Test)': equity_index,
        'V5.1 Strategy (Aggressive)': equity_v5_1,
        'Buy & Hold (SPY)': equity_spy
    }
    
    # 建立比較表
    metrics_list = []
    for name, curve in strategies.items():
        if curve is None or curve.empty:
            print(f"Warning: Curve for {name} is empty.")
            continue
            
        m = calculate_metrics(curve)
        m['Strategy'] = name
        metrics_list.append(m)
        
    df_metrics = pd.DataFrame(metrics_list)
    # 調整欄位順序
    cols = ['Strategy', 'Total Return', 'CAGR', 'Sharpe', 'MaxDD', 'Final Equity']
    # 確保 columns 存在
    cols = [c for c in cols if c in df_metrics.columns]
    df_metrics = df_metrics[cols]
    
    # 格式化輸出
    df_display = df_metrics.copy()
    try:
        df_display['Total Return'] = df_display['Total Return'].apply(lambda x: f"{x:.2%}")
        df_display['CAGR'] = df_display['CAGR'].apply(lambda x: f"{x:.2%}")
        df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
        df_display['MaxDD'] = df_display['MaxDD'].apply(lambda x: f"{x:.2%}")
        df_display['Final Equity'] = df_display['Final Equity'].apply(lambda x: f"${x:,.0f}")
    except Exception as e:
        print(f"Formatting warning: {e}")
    
    print("\n=== Comprehensive Performance Report ===")
    print(df_display.to_string(index=False))
    
    # 存檔 CSV
    df_display.to_csv(os.path.join(OUTPUT_DIR, 'comprehensive_report.csv'), index=False)
    
    # 繪圖
    plt.figure(figsize=(12, 7))
    colors = {
        'V5.2 Custom (Risk-Aware)': '#1f77b4', # Blue
        'V5.2 Index (Stress Test)': '#2ca02c', # Green
        'V5.1 Strategy (Aggressive)': '#ff7f0e', # Orange
        'Buy & Hold (SPY)': '#7f7f7f'          # Gray
    }
    
    for name, curve in strategies.items():
        if curve is not None and not curve.empty:
            # 歸一化繪圖
            start_val = curve.iloc[0]
            if start_val > 0:
                norm_curve = curve / start_val
                plt.plot(norm_curve.index, norm_curve, label=name, color=colors.get(name, 'black'), linewidth=1.5)
            
    plt.title('Strategy Comparison: Risk-Aware vs Aggressive vs Market', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'comprehensive_equity.png')
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    print(f"Report saved to {os.path.join(OUTPUT_DIR, 'comprehensive_report.csv')}")

if __name__ == "__main__":
    main()