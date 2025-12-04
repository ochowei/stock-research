# V5.2/ml_pipeline/07_ablation_study.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from backtesting_utils import run_backtest
from data_loader import DataLoader

# --- 輔助函數：計算指標 ---
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

# --- 輔助函數：載入數據 ---
def load_master_data(features_path, regime_signals_path):
    try:
        print(f"Loading features from {features_path}...")
        features_df = pd.read_parquet(features_path)
        print(f"Loading regime signals from {regime_signals_path}...")
        regime_signals_df = pd.read_parquet(regime_signals_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None

    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')
    df.rename(columns={'signal': 'regime_signal'}, inplace=True)
    df['regime_signal'] = df['regime_signal'].ffill()
    df = df.set_index('timestamp')
    
    return df

def filter_data_by_tickers(master_df, target_tickers):
    available_symbols = set(master_df['symbol'].unique())
    valid_tickers = [t for t in target_tickers if t in available_symbols]
    if not valid_tickers:
        return pd.DataFrame()
    return master_df[master_df['symbol'].isin(valid_tickers)].copy()

def main():
    # --- 1. 環境設定 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    FEATURES_PATH = os.path.join(project_root, 'features', 'stock_features.parquet')
    REGIME_SIGNALS_PATH = os.path.join(project_root, 'signals', 'regime_signals.parquet')
    OUTPUT_DIR = os.path.join(script_dir, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== V5.2 Ablation Study (Feature Importance Analysis) ===")

    # --- 2. 準備數據 (使用 Merged Pool 代表真實場景) ---
    loader = DataLoader(script_dir)
    master_df = load_master_data(FEATURES_PATH, REGIME_SIGNALS_PATH)
    
    if master_df is None:
        return

    print("Filtering data for Merged Pool (Normal + Toxic)...")
    merged_tickers = loader.get_all_tickers()
    data_pool = filter_data_by_tickers(master_df, merged_tickers)
    
    if data_pool.empty:
        print("Error: No data found for specified tickers.")
        return

    # --- 3. 定義實驗場景 (Scenarios) ---
    # 每個場景關閉一個特定功能，觀察績效變化
    scenarios = {
        'V5.2 Full System': {
            'use_time_stop': True, 'use_position_cap': True, 
            'use_signal_sorting': True, 'use_liquidation': True, 'use_regime_filter': True
        },
        'No Liquidation (只停買)': {
            'use_time_stop': True, 'use_position_cap': True, 
            'use_signal_sorting': True, 'use_liquidation': False, 'use_regime_filter': True
        },
        'No Sorting (隨機選股)': {
            'use_time_stop': True, 'use_position_cap': True, 
            'use_signal_sorting': False, 'use_liquidation': True, 'use_regime_filter': True
        },
        'No Position Cap (重倉)': {
            'use_time_stop': True, 'use_position_cap': False, 
            'use_signal_sorting': True, 'use_liquidation': True, 'use_regime_filter': True
        },
        'No Time Stop (死抱)': {
            'use_time_stop': False, 'use_position_cap': True, 
            'use_signal_sorting': True, 'use_liquidation': True, 'use_regime_filter': True
        },
        'No Regime Filter (無大盤濾網)': {
            'use_time_stop': True, 'use_position_cap': True, 
            'use_signal_sorting': True, 'use_liquidation': False, 'use_regime_filter': False
        }
    }

    results = []
    equity_curves = {}

    # --- 4. 執行回測迴圈 ---
    for name, config in scenarios.items():
        print(f"Running Scenario: {name}...")
        try:
            # 傳遞所有開關參數
            equity = run_backtest(
                data_pool, 
                force_equal_weight=False, # 保持 ATR Sizing 邏輯
                **config
            )
            
            if not equity.empty:
                metrics = calculate_metrics(equity)
                metrics['Scenario'] = name
                metrics['Config'] = str(config)
                results.append(metrics)
                equity_curves[name] = equity
            else:
                print(f"  [Warning] Empty equity curve for {name}")
                
        except Exception as e:
            print(f"  [Error] Failed to run {name}: {e}")

    # --- 5. 彙整結果與計算貢獻值 ---
    if not results:
        print("No results generated.")
        return

    df_res = pd.DataFrame(results)
    
    # 格式化顯示
    df_display = df_res[['Scenario', 'Total Return', 'CAGR', 'Sharpe', 'MaxDD', 'Final Equity']].copy()
    
    # 抓出基準值 (Full System)
    base_row = df_display[df_display['Scenario'] == 'V5.2 Full System']
    if not base_row.empty:
        base_sharpe = base_row['Sharpe'].iloc[0]
        base_ret = base_row['Total Return'].iloc[0]
        base_dd = base_row['MaxDD'].iloc[0]
        
        # 計算與基準的差異 (Impact)
        df_display['Sharpe Impact'] = df_display['Sharpe'] - base_sharpe
        df_display['Return Impact'] = df_display['Total Return'] - base_ret
        df_display['MaxDD Impact'] = df_display['MaxDD'] - base_dd
    
    # 輸出表格
    print("\n=== Ablation Study Results ===")
    # 格式化百分比
    for col in ['Total Return', 'CAGR', 'MaxDD', 'Return Impact', 'MaxDD Impact']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
    
    df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
    if 'Sharpe Impact' in df_display.columns:
        df_display['Sharpe Impact'] = df_display['Sharpe Impact'].apply(lambda x: f"{x:+.2f}")
        
    df_display['Final Equity'] = df_display['Final Equity'].apply(lambda x: f"${x:,.0f}")

    # 調整欄位順序
    cols_order = ['Scenario', 'Total Return', 'Sharpe', 'MaxDD', 'Sharpe Impact', 'MaxDD Impact']
    print(df_display[cols_order].to_string(index=False))
    
    # 存檔
    csv_path = os.path.join(OUTPUT_DIR, 'ablation_study_report.csv')
    df_display.to_csv(csv_path, index=False)
    print(f"\nReport saved to: {csv_path}")

    # --- 6. 繪圖 (Normalized) ---
    plt.figure(figsize=(14, 8))
    
    # 設定顏色或樣式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (name, curve) in enumerate(equity_curves.items()):
        if not curve.empty:
            norm_curve = curve / curve.iloc[0]
            # Full System 加粗
            lw = 3.0 if name == 'V5.2 Full System' else 1.5
            alpha = 1.0 if name == 'V5.2 Full System' else 0.7
            plt.plot(norm_curve.index, norm_curve, label=name, lw=lw, alpha=alpha)

    plt.title('V5.2 Ablation Study: Feature Importance Analysis')
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    png_path = os.path.join(OUTPUT_DIR, 'ablation_study_chart.png')
    plt.savefig(png_path)
    print(f"Chart saved to: {png_path}")

if __name__ == "__main__":
    main()