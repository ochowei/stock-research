import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import DataLoader
from backtesting_utils import run_backtest, analyze_performance

# --- V5.1 策略參數設定 (The Minimalist Baseline) ---
STRATEGY_CONFIG = {
    'initial_capital': 100_000.0,
    'hold_days': 5,              # 核心：固定持有 5 天
    'use_regime_filter': False,  # 核心：關閉 L1 市場狀態濾網
    'force_equal_weight': True,  # 核心：強制等權重 (20% per trade)
    'use_time_stop': True,       # 開啟時間止損
    'use_liquidation': False,    # 關閉緊急清倉
    'use_signal_sorting': True,  # 開啟 RSI 排序 (RSI 越低越優先)
    'max_position_pct': 0.2      # 等權重比例 (1/5)
}

def load_data(base_dir, track='custom'):
    """載入 V5.3 格式的特徵與訊號數據"""
    # 路徑指向 V5.3 的資料結構
    features_path = os.path.join(base_dir, 'data', track, 'features', 'stock_features.parquet')
    regime_path = os.path.join(base_dir, 'data', track, 'signals', 'regime_signals.parquet')
    
    if not os.path.exists(features_path) or not os.path.exists(regime_path):
        print(f"[Error] Data not found. Please ensure 02_build_features.py & 03_build_regime_filter.py are run.")
        return None

    print(f"Loading features from {features_path}...")
    features_df = pd.read_parquet(features_path)
    print(f"Loading regime signals from {regime_path}...")
    regime_signals_df = pd.read_parquet(regime_path)

    # 合併數據 (即使 V5.1 不用 Regime Filter，backtesting_utils 仍需要此欄位結構)
    features_df = features_df.reset_index()
    df = pd.merge(features_df, regime_signals_df, left_on='timestamp', right_index=True, how='left')
    
    # 重新命名與填補
    if 'signal' in df.columns:
        df.rename(columns={'signal': 'regime_signal'}, inplace=True)
    df['regime_signal'] = df['regime_signal'].ffill().fillna(0) # 預設為 0 (Normal)
    
    df = df.set_index('timestamp').sort_index()
    return df

def filter_data_by_tickers(df, tickers):
    """篩選出指定清單的數據"""
    valid_tickers = set(df['symbol'].unique())
    target_tickers = [t for t in tickers if t in valid_tickers]
    
    if not target_tickers:
        print("[Warning] No matching tickers found in data!")
        return pd.DataFrame()
        
    print(f"Filtering data for {len(target_tickers)} tickers...")
    return df[df['symbol'].isin(target_tickers)].copy()

def main():
    # --- 1. 環境設定 ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Reproducing V5.1 Baseline on V5.3 Final Data ===")
    print(f"Config: {STRATEGY_CONFIG}")

    # --- 2. 載入 Final 清單 (Normal + Toxic) ---
    # 指定讀取經過 audit 的 final 版本
    loader = DataLoader(
        SCRIPT_DIR, 
        normal_file='final_asset_pool.json', 
        toxic_file='final_toxic_asset_pool.json'
    )
    
    # 合併兩個池 (Final Merged Pool)
    target_tickers = loader.get_all_tickers()
    print(f"\nTarget Asset Pool: Final Merged (Normal + Toxic)")
    print(f"Total Tickers: {len(target_tickers)}")

    # --- 3. 載入市場數據 ---
    df = load_data(SCRIPT_DIR, track='custom')
    if df is None: return

    # 篩選數據
    df_subset = filter_data_by_tickers(df, target_tickers)
    if df_subset.empty: return

    # --- 4. 執行回測 ---
    print("\nRunning Backtest...")
    equity_curve = run_backtest(df_subset, **STRATEGY_CONFIG)

    # --- 5. 分析與產出 ---
    if not equity_curve.empty:
        print("\n=== V5.1 Final Pool Performance ===")
        # 使用 analyze_performance 產生圖表與 CSV
        analyze_performance(
            equity_curve=equity_curve,
            output_dir=OUTPUT_DIR,
            filename_prefix='v5.1_repro_final',
            title='V5.1 Baseline (Fixed 5D, EqWt) on Final Merged Pool',
            benchmark_label=None # 暫不需比較 Benchmark
        )
        
        # 簡單打印最終結果
        total_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        print(f"Total Return: {total_ret:.2%}")
        print(f"Final Equity: ${equity_curve.iloc[-1]:,.0f}")
        print(f"Output saved to: {os.path.join(OUTPUT_DIR, 'v5.1_repro_final_performance.csv')}")
    else:
        print("[Error] Backtest generated no trades.")

if __name__ == "__main__":
    main()
