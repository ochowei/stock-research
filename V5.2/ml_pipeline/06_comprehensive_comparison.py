# V5.2/ml_pipeline/06_comprehensive_comparison.py

import pandas as pd
import numpy as np
import os
import yfinance as yf
import matplotlib.pyplot as plt
from backtesting_utils import run_backtest
from data_loader import DataLoader

# --- 輔助函數：載入 V5.2 數據 (通用) ---
def load_v5_2_data(features_path, regime_signals_path, ticker_filter=None):
    try:
        features_df = pd.read_parquet(features_path)
        regime_signals_df = pd.read_parquet(regime_signals_path)
    except FileNotFoundError as e:
        print(f"Error loading V5.2 data: {e}")
        return None

    # 過濾標的
    if ticker_filter:
        valid_symbols = set(features_df.index.get_level_values('symbol').unique())
        selected_symbols = [t for t in ticker_filter if t in valid_symbols]
        if not selected_symbols:
            return None
        features_df = features_df[features_df.index.get_level_values('symbol').isin(selected_symbols)]

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
    if df.empty: return pd.Series(dtype=float)
    
    if isinstance(df.columns, pd.MultiIndex):
        close_data = df['Close'].iloc[:, 0] if 'SPY' not in df['Close'] else df['Close']['SPY']
        open_data = df['Open'].iloc[:, 0] if 'SPY' not in df['Open'] else df['Open']['SPY']
    else:
        close_data = df['Close']
        open_data = df['Open']

    if len(open_data) > 0 and open_data.iloc[0] > 0:
        shares = initial_capital / open_data.iloc[0]
        return close_data * shares
    return pd.Series(dtype=float)

def calculate_metrics(curve):
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
    max_dd = ((curve - roll_max) / roll_max).min()
    
    return {
        'Total Return': total_ret, 'CAGR': cagr, 'Sharpe': sharpe, 
        'MaxDD': max_dd, 'Final Equity': curve.iloc[-1]
    }

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    
    FEATURES_PATH = os.path.join(PROJECT_ROOT, 'features', 'stock_features.parquet')
    REGIME_PATH = os.path.join(PROJECT_ROOT, 'signals', 'regime_signals.parquet')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Starting Comprehensive Strategy Comparison (Re-running Logic) ===")
    loader = DataLoader(SCRIPT_DIR)
    
    # 1. 準備各種情境的標的清單
    scenarios = {
        'V5.2 Custom (Normal)': loader.get_normal_tickers(),
        'V5.2 Custom (Toxic Stress)': loader.get_toxic_tickers(),
        'V5.2 Custom (Merged Real)': loader.get_all_tickers()
    }
    
    strategies = {}

    # 2. 執行 V5.2 策略回測 (有風控, ATR Sizing)
    print("\n[Phase 1] Running V5.2 Strategies (With Risk Management)...")
    for name, tickers in scenarios.items():
        print(f"  Running: {name}...")
        df = load_v5_2_data(FEATURES_PATH, REGIME_PATH, ticker_filter=tickers)
        if df is not None and not df.empty:
            eq = run_backtest(
                df, 
                use_regime_filter=True,   # 開啟 L1 防禦
                force_equal_weight=False  # 使用 ATR Sizing
            )
            if not eq.empty: strategies[name] = eq

    # 3. 執行 V5.1 (Aggressive) 重現回測
    # 邏輯：使用 Merged Pool (或 Normal Pool)，但關閉風控，強制等權重
    print("\n[Phase 2] Re-running V5.1 Logic (Aggressive / No Filter)...")
    v5_1_tickers = loader.get_all_tickers() # 使用全部標的進行比較
    df_v5_1 = load_v5_2_data(FEATURES_PATH, REGIME_PATH, ticker_filter=v5_1_tickers)
    
    if df_v5_1 is not None and not df_v5_1.empty:
        eq_v5_1 = run_backtest(
            df_v5_1,
            use_regime_filter=False,  # 關閉 L1 防禦 (V5.1 Minimalist)
            force_equal_weight=True   # 強制等權重 (V5.1 Logic)
        )
        if not eq_v5_1.empty:
            strategies['V5.1 (Aggressive)'] = eq_v5_1

    # 4. 執行 V5.2 Index 回測
    print("\n[Phase 3] Running V5.2 Index Backtest...")
    df_index = load_v5_2_data(FEATURES_PATH, REGIME_PATH) # 載入全市場數據
    if df_index is not None and not df_index.empty:
        equity_index = run_backtest(df_index, use_regime_filter=True, force_equal_weight=False)
        if not equity_index.empty:
            strategies['V5.2 Index (Market)'] = equity_index

    # 5. 處理時間軸對齊與 SPY 基準
    if not strategies: return
    ref_key = list(strategies.keys())[0]
    common_idx = strategies[ref_key].index.sort_values()
    
    # 下載 SPY
    equity_spy = get_spy_benchmark(common_idx[0], common_idx[-1])
    if not equity_spy.empty:
        equity_spy = equity_spy.reindex(common_idx).ffill().bfill()
        equity_spy = equity_spy * (100000.0 / equity_spy.iloc[0])
        strategies['Buy & Hold (SPY)'] = equity_spy

    # 對齊所有策略
    for name in strategies:
        strategies[name] = strategies[name].reindex(common_idx).ffill().bfill()

    # --- 輸出報告與圖表 ---
    print("\nGenerating Report...")
    metrics_list = []
    for name, curve in strategies.items():
        m = calculate_metrics(curve)
        m['Strategy'] = name
        metrics_list.append(m)
        
    df_metrics = pd.DataFrame(metrics_list)
    cols = ['Strategy', 'Total Return', 'CAGR', 'Sharpe', 'MaxDD', 'Final Equity']
    df_display = df_metrics[cols].copy()
    
    # 格式化
    for col in ['Total Return', 'CAGR', 'MaxDD']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2%}")
    df_display['Sharpe'] = df_display['Sharpe'].apply(lambda x: f"{x:.2f}")
    df_display['Final Equity'] = df_display['Final Equity'].apply(lambda x: f"${x:,.0f}")

    print("\n=== Final Comprehensive Report (Re-run Logic) ===")
    print(df_display.to_string(index=False))
    df_display.to_csv(os.path.join(OUTPUT_DIR, 'comprehensive_report.csv'), index=False)

    # 繪圖
    plt.figure(figsize=(14, 8))
    styles = {
        'V5.2 Custom (Normal)':       {'color': '#1f77b4', 'lw': 2.0},
        'V5.2 Custom (Toxic Stress)': {'color': '#d62728', 'lw': 2.0},
        'V5.2 Custom (Merged Real)':  {'color': '#2ca02c', 'lw': 2.5},
        'V5.2 Index (Market)':        {'color': '#9467bd', 'lw': 1.5, 'ls': '--'},
        'V5.1 (Aggressive)':          {'color': '#ff7f0e', 'lw': 1.5, 'ls': ':'},
        'Buy & Hold (SPY)':           {'color': 'gray',    'lw': 1.5, 'ls': '-.', 'alpha': 0.6}
    }
    
    for name, curve in strategies.items():
        if not curve.empty:
            norm_curve = curve / curve.iloc[0]
            s = styles.get(name, {'color': 'black', 'lw': 1.0})
            plt.plot(norm_curve.index, norm_curve, label=name, **s)
            
    plt.title('V5.2 Comprehensive Analysis (All Logic Re-run Internally)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_equity.png'))
    print(f"\nAnalysis Complete. Outputs in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()