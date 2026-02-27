import os
import sys
import json
import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt

# --- 1. Setup & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../../../'))
sys.path.append(ROOT_DIR)

RESOURCE_DIR = os.path.join(ROOT_DIR, 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Parameters ---
TRAIN_START = '2020-01-01'
TEST_END    = '2025-12-31'
GAP_THRESHOLD = 0.005

# --- 3. Utilities ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    # Extract ticker from " Exchange : Ticker " format
    tickers = sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))
    return tickers

def fetch_data_batched(tickers, batch_size=15):
    all_tickers = tickers + ['^VIX', 'QQQ', 'SPY']
    all_tickers = sorted(list(set(all_tickers)))

    print(f"Downloading data for {len(all_tickers)} tickers in batches of {batch_size}...")

    combined_df = pd.DataFrame()
    failed_tickers = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}: {batch}")

        try:
            time.sleep(random.uniform(1, 3))

            data = yf.download(
                batch, start=TRAIN_START, end=TEST_END,
                interval='1d', auto_adjust=True, progress=False, threads=True
            )

            if data.empty:
                print(f"    [Warning] Empty data for batch {batch}")
                failed_tickers.extend(batch)
                continue

            # Handle MultiIndex columns (Ticker, OHLCV)
            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.stack(level=1, future_stack=True)
                except TypeError:
                     data = data.stack(level=1)

                data = data.rename_axis(['Date', 'Ticker']).reset_index()
            else:
                if len(batch) == 1:
                    data['Ticker'] = batch[0]
                    data = data.reset_index()
                else:
                    print("    [Error] Unexpected data format for batch")
                    continue

            combined_df = pd.concat([combined_df, data], ignore_index=True)

        except Exception as e:
            print(f"    [Error] Batch failed: {e}")
            failed_tickers.extend(batch)

    if combined_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.tz_localize(None).dt.normalize()

    vix_df = combined_df[combined_df['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    stock_df = combined_df[combined_df['Ticker'] != '^VIX']

    vix_df = vix_df.resample('D').ffill()

    print(f"  - Stock Data Rows: {len(stock_df)}")
    print(f"  - VIX Data Rows: {len(vix_df)}")

    return stock_df, vix_df

def build_features(df):
    df = df.sort_index()

    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    # Gap Calculation
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Momentum Indicators (Shifted by 1 to prevent look-ahead)
    try:
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_14'] = df['RSI_14'].shift(1)
    except:
        df['RSI_14'] = np.nan

    # Volume Indicators
    try:
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        # Vol_Ratio: T-1 Volume relative to T-2 Baseline (Shifted MA)
        # We need to know if the volume LEADING UP TO the gap was extreme.
        # df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1) # Correct
        # But wait, df['Vol_MA20'].shift(1) is MA20 ending at T-2.
        # df['Prev_Vol'] is Volume at T-1.
        # So yes, this compares T-1 volume to T-2 baseline.
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)
    except:
        df['Vol_Ratio'] = np.nan

    df['Long_Ret'] = (df['Close'] - df['Open']) / df['Open']
    df['Short_Ret'] = (df['Open'] - df['Close']) / df['Open']

    df['Is_Gap_Up'] = df['Gap_Pct'] > GAP_THRESHOLD

    return df

def run_analysis(stock_df):
    print("\n--- Running Analysis ---")

    all_signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        df = group.set_index('Date').copy()
        if len(df) < 50: continue

        df = build_features(df)

        signals = df[df['Is_Gap_Up'] == True].copy()

        if not signals.empty:
            signals['Ticker'] = ticker
            all_signals.append(signals)

    if not all_signals:
        print("[Warning] No signals found.")
        return

    full_df = pd.concat(all_signals).sort_index()
    full_df = full_df.dropna(subset=['RSI_14', 'Vol_Ratio', 'Long_Ret', 'Short_Ret'])

    print(f"Total Gap Up Signals: {len(full_df)}")

    # Define Regimes
    # 1. High Mom (RSI > 70)
    high_mom_df = full_df[full_df['RSI_14'] > 70]

    # 2. High Mom + High Vol (Vol Ratio > 2.0)
    high_mom_high_vol = high_mom_df[high_mom_df['Vol_Ratio'] > 2.0]

    # 3. High Mom + Normal Vol (Vol Ratio <= 2.0)
    high_mom_norm_vol = high_mom_df[high_mom_df['Vol_Ratio'] <= 2.0]

    print(f"High Momentum Signals (RSI>70): {len(high_mom_df)}")
    print(f"  - High Vol (Ratio>2.0): {len(high_mom_high_vol)}")
    print(f"  - Normal Vol (Ratio<=2.0): {len(high_mom_norm_vol)}")

    def calculate_metrics(df, strategy_col, name):
        if df.empty:
            return {'Name': name, 'Count': 0, 'Win_Rate': 0, 'Avg_Ret': 0}

        win_rate = (df[strategy_col] > 0).mean()
        avg_ret = df[strategy_col].mean()
        return {
            'Name': name,
            'Count': len(df),
            'Win_Rate': win_rate,
            'Avg_Ret': avg_ret
        }

    stats = []
    # Baseline: High Mom Short (from EXP-06)
    stats.append(calculate_metrics(high_mom_df, 'Short_Ret', 'High_Mom_Short_Base'))
    # Experiment: High Vol Filter
    stats.append(calculate_metrics(high_mom_high_vol, 'Short_Ret', 'High_Mom_High_Vol_Short'))
    # Control: Normal Vol
    stats.append(calculate_metrics(high_mom_norm_vol, 'Short_Ret', 'High_Mom_Norm_Vol_Short'))

    stats_df = pd.DataFrame(stats)
    print("\n--- Performance Report ---")
    print(stats_df)

    stats_path = os.path.join(OUTPUT_DIR, 'performance_report.csv')
    stats_df.to_csv(stats_path, index=False)

    # Plot Equity Curves
    plt.figure(figsize=(12, 6))

    def get_curve(df, col):
        if df.empty: return pd.Series()
        ret = df.groupby(df.index)[col].mean().fillna(0)
        return (1 + ret).cumprod()

    curve_base = get_curve(high_mom_df, 'Short_Ret')
    curve_high_vol = get_curve(high_mom_high_vol, 'Short_Ret')
    curve_norm_vol = get_curve(high_mom_norm_vol, 'Short_Ret')

    if not curve_base.empty:
        plt.plot(curve_base.index, curve_base, label='Base (All RSI>70)', color='gray', linestyle='--')
    if not curve_high_vol.empty:
        plt.plot(curve_high_vol.index, curve_high_vol, label='High Vol (>2.0)', color='red', linewidth=2)
    if not curve_norm_vol.empty:
        plt.plot(curve_norm_vol.index, curve_norm_vol, label='Normal Vol (<=2.0)', color='blue')

    plt.title('Short Strategy Equity Curve: Volume Filter Impact')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, 'equity_curve.png')
    plt.savefig(plot_path)

def main():
    print("=== EXP-07: High Momentum Short Strategy - Volume Filter ===")
    tickers = load_tickers()
    if not tickers: return
    stock_df, _ = fetch_data_batched(tickers)
    if stock_df.empty: return
    run_analysis(stock_df)

if __name__ == '__main__':
    main()
