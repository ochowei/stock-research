import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import DataLoader
from backtesting_utils import run_backtest

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

def load_master_data(features_path, regime_signals_path):
    """
    一次性載入所有數據 (Master Dataframe)，不進行過濾。
    """
    try:
        print(f"Loading features from {features_path}...")
        features_df = pd.read_parquet(features_path)
        print(f"Loading regime signals from {regime_signals_path}...")
        regime_signals_df = pd.read_parquet(regime_signals_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    # 合併 L1 訊號
    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')
    df.rename(columns={'signal': 'regime_signal'}, inplace=True)
    df['regime_signal'] = df['regime_signal'].ffill() # 填補空值
    df = df.set_index('timestamp')
    
    print(f"Master Data Loaded. Shape: {df.shape}")
    return df

def filter_data_by_tickers(master_df, target_tickers):
    """
    從 Master Data 中篩選出指定的 tickers。
    """
    # 檢查 master_df 中有哪些 tickers 是存在的
    available_symbols = set(master_df['symbol'].unique())
    valid_tickers = [t for t in target_tickers if t in available_symbols]
    
    if not valid_tickers:
        print("Warning: No matching tickers found in data.")
        return pd.DataFrame()
        
    print(f"Filtering {len(valid_tickers)} symbols...")
    filtered_df = master_df[master_df['symbol'].isin(valid_tickers)].copy()
    return filtered_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    FEATURES_PATH = os.path.join(project_root, 'features', 'stock_features.parquet')
    REGIME_SIGNALS_PATH = os.path.join(project_root, 'signals', 'regime_signals.parquet')
    OUTPUT_DIR = os.path.join(script_dir, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 初始化 Loader 與載入主數據
    loader = DataLoader(script_dir)
    master_df = load_master_data(FEATURES_PATH, REGIME_SIGNALS_PATH)
    
    if master_df is None or master_df.empty:
        print("Critical Error: Master data load failed.")
        return

    # 2. 定義要比較的場景 (Scenarios)
    scenarios = [
        ('Normal Pool', loader.get_normal_tickers(), '#1f77b4'), # Blue
        ('Toxic Pool (Stress)', loader.get_toxic_tickers(), '#d62728'), # Red
        ('Merged Pool (Real)', loader.get_all_tickers(), '#2ca02c')  # Green
    ]
    
    results = {}
    
    print("\n=== Starting Comparative Backtest ===")
    
    # 3. 迴圈執行回測
    for name, tickers, color in scenarios:
        print(f"\n>> Running Scenario: {name}")
        df_subset = filter_data_by_tickers(master_df, tickers)
        
        if not df_subset.empty:
            equity_curve = run_backtest(df_subset)
            
            # 確保有數據才紀錄
            if not equity_curve.empty:
                results[name] = {
                    'equity': equity_curve,
                    'color': color
                }
            else:
                print(f"  [Warning] Backtest returned empty curve for {name}.")
        else:
            print(f"  [Warning] No data found for {name}.")

    # 4. 整合分析與繪圖
    if not results:
        print("No results to analyze.")
        return

    print("\nGenerating Comparative Report...")
    metrics_list = []
    
    plt.figure(figsize=(12, 7))
    
    # 找出共同起始點以進行歸一化比較 (以 Merged 或 Normal 為主)
    # 這裡簡單採用各曲線自己的起始點歸一化
    for name, data in results.items():
        curve = data['equity']
        
        # 計算指標
        m = calculate_metrics(curve)
        m['Scenario'] = name
        metrics_list.append(m)
        
        # 繪圖 (Normalized to 1.0)
        norm_curve = curve / curve.iloc[0]
        plt.plot(norm_curve.index, norm_curve, label=name, color=data['color'], linewidth=1.5)

    # 格式化表格
    df_metrics = pd.DataFrame(metrics_list)
    cols = ['Scenario', 'Total Return', 'CAGR', 'Sharpe', 'MaxDD', 'Final Equity']
    df_metrics = df_metrics[cols]
    
    # 顯示與存檔
    print("\n" + df_metrics.to_string(index=False))
    df_metrics.to_csv(os.path.join(OUTPUT_DIR, 'internal_stress_test_report.csv'), index=False)
    
    # 存圖
    plt.title('V5.2 Internal Stress Test: Normal vs Toxic vs Merged', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(OUTPUT_DIR, 'internal_stress_test_chart.png')
    plt.savefig(plot_path)
    print(f"\n[Output] Chart saved to: {plot_path}")
    print(f"[Output] Report saved to: {os.path.join(OUTPUT_DIR, 'internal_stress_test_report.csv')}")

if __name__ == "__main__":
    main()