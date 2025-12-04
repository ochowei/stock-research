import pandas as pd
import numpy as np
import os
import json
import yfinance as yf
import matplotlib.pyplot as plt
from backtesting_utils import run_backtest, analyze_performance
# [New] 引入 DataLoader
from data_loader import DataLoader

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
        # 確保 ticker_filter 與 index 中的 symbol 格式一致
        valid_symbols = set(features_df.index.get_level_values('symbol').unique())
        selected_symbols = [t for t in ticker_filter if t in valid_symbols]
        
        if not selected_symbols:
            print("Warning: No matching tickers found for filter.")
            return None
            
        features_df = features_df[features_df.index.get_level_values('symbol').isin(selected_symbols)]

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
    
    df = yf.download("SPY", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
    
    if df.empty:
        return pd.Series(dtype=float)
        
    # 處理 yfinance 可能回傳的多層索引
    if isinstance(df.columns, pd.MultiIndex):
        try:
            close_data = df['Close']['SPY']
            open_data = df['Open']['SPY']
        except KeyError:
            close_data = df['Close'].iloc[:, 0]
            open_data = df['Open'].iloc[:, 0]
    else:
        close_data = df['Close']
        open_data = df['Open']

    close_data = close_data.squeeze()
    open_data = open_data.squeeze()

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
    
    if isinstance(curve, pd.DataFrame): curve = curve.iloc[:, 0]

    returns = curve.pct_change().fillna(0)
    total_ret = (curve.iloc[-1] / curve.iloc[0]) - 1
    
    days = (curve.index[-1] - curve.index[0]).days
    years = days / 365.25
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    std_val = returns.std()
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
    V5_1_TRADES_PATH = os.path.join(REPO_ROOT, 'V5.1', 'ml_pipeline', 'analysis', 'minimalist_trades_fixed.csv')
    
    # 輸出目錄
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Starting Comprehensive Strategy Comparison (Internal & External) ===")

    # 1. 準備清單 (使用 DataLoader)
    loader = DataLoader(SCRIPT_DIR)
    
    # 定義內部比較場景
    custom_scenarios = {
        'V5.2 Custom (Normal)': loader.get_normal_tickers(),
        'V5.2 Custom (Toxic Stress)': loader.get_toxic_tickers(),
        'V5.2 Custom (Merged Real)': loader.get_all_tickers()
    }
    
    strategies = {}

    # 2. 執行 V5.2 Custom 內部回測 (Normal, Toxic, Merged)
    print("\n[Phase 1] Running V5.2 Custom Internal Scenarios...")
    
    # 為了優化效能，我們可以先載入全部數據，再進行過濾
    # 但為了保持 load_v5_2_data 的通用性，這裡我們針對每個 scenario 呼叫一次
    # (如果記憶體足夠，載入一次 full data 再 filter 會更快)
    
    for name, tickers in custom_scenarios.items():
        print(f"  Running: {name} ({len(tickers)} tickers)...")
        df = load_v5_2_data(FEATURES_PATH, REGIME_PATH, ticker_filter=tickers)
        if df is not None and not df.empty:
            eq = run_backtest(df)
            if not eq.empty:
                strategies[name] = eq
        else:
            print(f"    Warning: No data for {name}")

    # 3. 執行 V5.2 Index 回測 (Stress Test)
    print("\n[Phase 2] Running V5.2 Index Backtest (External Stress)...")
    df_index = load_v5_2_data(FEATURES_PATH, REGIME_PATH) 
    # Index 通常沒有 ticker filter，讀取全部 (假設 features 包含 index 數據)
    # 注意：如果您的 features.parquet 混合了 custom 和 index 數據，這裡可能需要區分
    # 假設 V5.2 Index 是分開下載的，這裡可能需要確認 load_v5_2_data 的路徑是否正確
    # 若 Index 數據在不同資料夾，需調整 load_v5_2_data 邏輯。
    # 假設目前 features.parquet 包含所有需要的數據 (或您已在步驟 01/02 合併)
    
    equity_index = run_backtest(df_index)
    if not equity_index.empty:
        strategies['V5.2 Index (Market)'] = equity_index

    # 取得共同時間軸 (以 Merged 或 Normal 為主)
    if not strategies:
        print("Error: No strategies generated.")
        return
        
    ref_strategy = list(strategies.keys())[0]
    common_index = strategies[ref_strategy].index.sort_values()
    start_date = common_index[0]
    end_date = common_index[-1]

    # 對齊所有策略的時間軸
    for name in strategies:
        strategies[name] = strategies[name].reindex(common_index).ffill().bfill()

    # 4. 載入 V5.1 Benchmark (Aggressive)
    print("\n[Phase 3] Loading V5.1 Benchmark...")
    equity_v5_1 = load_v5_1_equity(V5_1_TRADES_PATH, common_index)
    if not equity_v5_1.empty:
        strategies['V5.1 (Aggressive)'] = equity_v5_1

    # 5. 下載 SPY Benchmark
    print("\n[Phase 4] Loading SPY Benchmark...")
    equity_spy = get_spy_benchmark(start_date, end_date)
    if not equity_spy.empty:
        equity_spy = equity_spy.reindex(common_index).ffill().bfill()
        # 歸一化
        equity_spy = equity_spy * (100000.0 / equity_spy.iloc[0])
        strategies['Buy & Hold (SPY)'] = equity_spy

    # --- 整合與分析 ---
    print("\nGenerating Comprehensive Report...")
    
    # 建立比較表
    metrics_list = []
    for name, curve in strategies.items():
        m = calculate_metrics(curve)
        m['Strategy'] = name
        metrics_list.append(m)
        
    df_metrics = pd.DataFrame(metrics_list)
    cols = ['Strategy', 'Total Return', 'CAGR', 'Sharpe', 'MaxDD', 'Final Equity']
    df_metrics = df_metrics[cols]
    
    # 格式化輸出
    df_display = df_metrics.copy()
    try:
        df_display['Total Return'] = df_display['Total Return'].apply(lambda x: f"{x:.2%}")
        df_display['CAGR'] = df_display['CAGR'].apply(lambda x: f"{x:.2%}")
        df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
        df_display['MaxDD'] = df_display['MaxDD'].apply(lambda x: f"{x:.2%}")
        df_display['Final Equity'] = df_display['Final Equity'].apply(lambda x: f"${x:,.0f}")
    except Exception:
        pass
    
    print("\n=== Final Comprehensive Report ===")
    print(df_display.to_string(index=False))
    
    df_display.to_csv(os.path.join(OUTPUT_DIR, 'comprehensive_report.csv'), index=False)
    
    # 繪圖
    plt.figure(figsize=(14, 8))
    
    # 定義顏色與樣式
    styles = {
        'V5.2 Custom (Normal)':       {'color': '#1f77b4', 'lw': 2.0, 'ls': '-'}, # Blue
        'V5.2 Custom (Toxic Stress)': {'color': '#d62728', 'lw': 2.0, 'ls': '-'}, # Red (Risk!)
        'V5.2 Custom (Merged Real)':  {'color': '#2ca02c', 'lw': 2.5, 'ls': '-'}, # Green (Main)
        'V5.2 Index (Market)':        {'color': '#9467bd', 'lw': 1.5, 'ls': '--'}, # Purple
        'V5.1 (Aggressive)':          {'color': '#ff7f0e', 'lw': 1.5, 'ls': ':'},  # Orange
        'Buy & Hold (SPY)':           {'color': 'gray',    'lw': 1.5, 'ls': '-.', 'alpha': 0.6}
    }
    
    for name, curve in strategies.items():
        if curve is not None and not curve.empty:
            # 歸一化
            start_val = curve.iloc[0]
            if start_val > 0:
                norm_curve = curve / start_val
                
                # 取得樣式
                style = styles.get(name, {'color': 'black', 'lw': 1.0, 'ls': '-'})
                
                plt.plot(norm_curve.index, norm_curve, label=name, 
                         color=style.get('color'), linewidth=style.get('lw'), 
                         linestyle=style.get('ls'), alpha=style.get('alpha', 1.0))
            
    plt.title('V5.2 Comprehensive Analysis: Internal Stress & External Benchmarks', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity (Start=1.0)')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'comprehensive_equity.png')
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    print(f"Report saved to {os.path.join(OUTPUT_DIR, 'comprehensive_report.csv')}")

if __name__ == "__main__":
    main()