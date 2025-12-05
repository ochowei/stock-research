import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
from backtesting_utils import run_backtest
from data_loader import DataLoader

# --- 輔助函數：計算績效指標 ---
def calculate_metrics(curve):
    if curve is None or curve.empty:
        return {k: 0 for k in ['Total Return', 'CAGR', 'Sharpe', 'MaxDD']}
    
    # 確保是 Series
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

# --- 輔助函數：載入 V5 格式數據 ---
def load_data(base_dir):
    features_path = os.path.join(base_dir, 'features', 'stock_features.parquet')
    regime_path = os.path.join(base_dir, 'signals', 'regime_signals.parquet')

    try:
        print(f"Loading features from {features_path}...")
        features_df = pd.read_parquet(features_path)
        print(f"Loading regime signals from {regime_path}...")
        regime_signals_df = pd.read_parquet(regime_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Hint: Ensure you have run '02_build_features.py' and '03_build_regime_filter.py' first.")
        return None

    # 合併特徵與訊號
    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')
    df.rename(columns={'signal': 'regime_signal'}, inplace=True)
    df['regime_signal'] = df['regime_signal'].ffill()
    df = df.set_index('timestamp')
    
    return df

# --- 輔助函數：獲取 SPY 基準 ---
def get_spy_benchmark(start_date, end_date, initial_capital=100000.0):
    print(f"Downloading SPY Benchmark ({start_date.date()} to {end_date.date()})...")
    try:
        df = yf.download("SPY", start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        if df.empty: return pd.Series(dtype=float)
        
        # 處理 yfinance 多層索引
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0]
            open_price = df['Open'].iloc[:, 0]
        else:
            close = df['Close']
            open_price = df['Open']
            
        # 計算 Buy & Hold 權益
        shares = initial_capital / open_price.iloc[0]
        equity = close * shares
        return equity
    except Exception as e:
        print(f"Error downloading SPY: {e}")
        return pd.Series(dtype=float)

def filter_data(master_df, tickers):
    valid = [t for t in tickers if t in master_df['symbol'].unique()]
    if not valid: return pd.DataFrame()
    return master_df[master_df['symbol'].isin(valid)].copy()

def run_and_report_for_group(group_name, loader, master_df, output_dir):
    """執行單一組別的回測與報告生成"""
    
    print(f"\n{'='*20} Running Backtests for: {group_name.upper()} Group {'='*20}")
    
    # --- 1. 載入資產池 ---
    target_tickers = loader.get_all_tickers()
    if not target_tickers:
        print(f"No tickers found for {group_name} group. Skipping.")
        return

    data_pool = filter_data(master_df, target_tickers)
    if data_pool.empty:
        print(f"Error: No data found for {group_name} tickers.")
        return

    # --- 2. 執行基準回測 ---
    strategies = {}
    
    # Benchmark A: V5.1 Aggressive
    print(f"\n[{group_name}] Running Benchmark A: V5.1 Aggressive...")
    eq_v5_1 = run_backtest(data_pool, force_equal_weight=True, use_regime_filter=False, use_liquidation=False, use_time_stop=True)
    if not eq_v5_1.empty: strategies['V5.1 Aggressive'] = eq_v5_1
    
    # Benchmark B: V5.2 Risk-Aware
    print(f"[{group_name}] Running Benchmark B: V5.2 Risk-Aware...")
    eq_v5_2 = run_backtest(data_pool, force_equal_weight=False, use_regime_filter=True, use_liquidation=True, use_time_stop=True)
    if not eq_v5_2.empty: strategies['V5.2 Risk-Aware'] = eq_v5_2
    
    # Benchmark C: Market Baseline (SPY Buy & Hold)
    if strategies:
        ref_idx = list(strategies.values())[0].index
        start_dt, end_dt = ref_idx.min(), ref_idx.max()
        eq_spy = get_spy_benchmark(start_dt, end_dt)
        if not eq_spy.empty:
            strategies['SPY (Buy & Hold)'] = eq_spy.reindex(ref_idx, method='ffill')

    # --- 3. 產出報告 ---
    if not strategies:
        print(f"No strategies were successfully backtested for {group_name}. No report generated.")
        return

    print(f"\n[{group_name}] Generating Benchmark Report...")
    metrics_list = []
    
    plt.figure(figsize=(14, 8))
    colors = {'V5.1 Aggressive': '#ff7f0e', 'V5.2 Risk-Aware': '#2ca02c', 'SPY (Buy & Hold)': 'gray'}
    styles = {'V5.1 Aggressive': '--', 'V5.2 Risk-Aware': '-', 'SPY (Buy & Hold)': '-.'}
    
    for name, curve in strategies.items():
        m = calculate_metrics(curve)
        m['Strategy'] = name
        metrics_list.append(m)
        norm = curve / curve.iloc[0]
        plt.plot(norm.index, norm, label=name, color=colors.get(name, 'blue'), linestyle=styles.get(name, '-'))

    df_res = pd.DataFrame(metrics_list)
    cols = ['Strategy', 'Total Return', 'CAGR', 'Sharpe', 'MaxDD', 'Final Equity']
    
    df_fmt = df_res.copy()
    for c in ['Total Return', 'CAGR', 'MaxDD']: df_fmt[c] = df_fmt[c].apply(lambda x: f"{x:.2%}")
    df_fmt['Sharpe'] = df_fmt['Sharpe'].apply(lambda x: f"{x:.2f}")
    df_fmt['Final Equity'] = df_fmt['Final Equity'].apply(lambda x: f"${x:,.0f}")
    
    print(f"\n--- {group_name.upper()} Results ---")
    print(df_fmt[cols].to_string(index=False))
    
    # --- 4. 存檔 ---
    prefix = f"{group_name}_"
    csv_path = os.path.join(output_dir, f'{prefix}baseline_performance.csv')
    df_fmt[cols].to_csv(csv_path, index=False)
    
    plt.title(f'V5.3 Benchmarks ({group_name.capitalize()} Pool): V5.1 vs V5.2 vs Market')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{prefix}baseline_comparison.png'))
    plt.close() # 關閉當前圖形，避免重疊
    
    print(f"[{group_name}] Results saved to {output_dir} with prefix '{prefix}'")


def main():
    # --- 1. 設定路徑與載入主數據 ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== V5.3 Step 1.2: Dual-Track Benchmark Backtesting ===")

    master_df = load_data(SCRIPT_DIR)
    if master_df is None: return

    # --- 2. 設定雙軌回測 ---
    backtest_groups = {
        "origin": DataLoader(SCRIPT_DIR,
                             normal_file='origin_asset_pool.json',
                             toxic_file='origin_toxic_asset_pool.json'),
        "cleaned": DataLoader(SCRIPT_DIR,
                              normal_file='asset_pool.json',
                              toxic_file='toxic_asset_pool.json')
    }

    # --- 3. 循環執行 ---
    for group_name, loader in backtest_groups.items():
        run_and_report_for_group(group_name, loader, master_df, OUTPUT_DIR)

    print(f"\n{'='*20} Dual-Track Backtesting Complete {'='*20}")
    print(f"All reports saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()