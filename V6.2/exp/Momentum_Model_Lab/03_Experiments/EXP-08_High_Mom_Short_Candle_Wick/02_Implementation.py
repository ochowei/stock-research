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
RSI_THRESHOLD = 70
SL_LEVELS = [0.002, 0.005, 0.01]  # 0.2%, 0.5%, 1.0%

# --- 3. Utilities ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
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
                continue

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
        return pd.DataFrame()

    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.tz_localize(None).dt.normalize()
    stock_df = combined_df[combined_df['Ticker'] != '^VIX']

    # We don't strictly need VIX for this experiment, just stock data for gaps/wicks
    print(f"  - Stock Data Rows: {len(stock_df)}")
    return stock_df

def build_features(df):
    df = df.sort_index()
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Calculate RSI (T-1)
    try:
        df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
    except:
        df['RSI_14'] = np.nan

    df['Is_Gap_Up'] = df['Gap_Pct'] > GAP_THRESHOLD

    # Candle Wick Calculation (Max excursion above Open)
    # If High > Open, this is the max loss for a short
    # If High < Open (GAP and CRAP immediately), max loss is 0 (entry price)
    df['Max_Up_Pct'] = (df['High'] - df['Open']) / df['Open']

    # Base Short Return (Open to Close)
    df['Short_Ret_No_SL'] = (df['Open'] - df['Close']) / df['Open']

    return df

def simulate_sl(df, sl_pct):
    """
    Simulate Short Trade with Stop Loss.
    If Max_Up_Pct >= sl_pct, we get stopped out at -sl_pct.
    Else, we get the full Short_Ret_No_SL.
    """
    # Vectorized condition
    # If hit SL, return is -sl_pct
    # Else, return is Short_Ret_No_SL

    conditions = [
        (df['Max_Up_Pct'] >= sl_pct), # Hit SL
        (df['Max_Up_Pct'] < sl_pct)   # Did not hit SL
    ]

    choices = [
        -sl_pct,
        df['Short_Ret_No_SL']
    ]

    return np.select(conditions, choices, default=np.nan)

def run_analysis(stock_df):
    print("\n--- Running Analysis ---")
    all_signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        df = group.set_index('Date').copy()
        if len(df) < 50: continue

        df = build_features(df)

        # Filter for High Momentum Gaps
        signals = df[
            (df['Is_Gap_Up'] == True) &
            (df['RSI_14'] > RSI_THRESHOLD)
        ].copy()

        if not signals.empty:
            signals['Ticker'] = ticker
            all_signals.append(signals)

    if not all_signals:
        print("[Warning] No signals found.")
        return

    full_df = pd.concat(all_signals).sort_index()
    full_df = full_df.dropna(subset=['RSI_14', 'Short_Ret_No_SL'])

    print(f"Total High Momentum Gap Signals: {len(full_df)}")

    # --- Simulate SL Strategies ---

    results = {}

    # Baseline
    results['No_SL'] = full_df['Short_Ret_No_SL']

    # SL Variants
    for sl in SL_LEVELS:
        col_name = f"Short_SL_{sl*100:.1f}%"
        full_df[col_name] = simulate_sl(full_df, sl)
        results[f"SL_{sl*100:.1f}%"] = full_df[col_name]

    # --- Metrics Calculation ---

    metrics = []

    for name, returns in results.items():
        win_rate = (returns > 0).mean()
        avg_ret = returns.mean()
        total_ret = returns.sum()
        count = len(returns)

        # Stop Out Rate (only for SL variants)
        stop_out_rate = 0.0
        if "SL" in name and "No" not in name:
            sl_val = float(name.split('_')[1].replace('%', '')) / 100
            # Check how many hit the exact loss amount (approx)
            # Or better, re-calculate using the logic
            stop_out_rate = (full_df['Max_Up_Pct'] >= sl_val).mean()

        metrics.append({
            "Strategy": name,
            "Count": count,
            "Win_Rate": win_rate,
            "Avg_Ret": avg_ret,
            "Total_Ret": total_ret,
            "Stop_Out_Rate": stop_out_rate
        })

    metrics_df = pd.DataFrame(metrics)
    print("\n--- Performance Report ---")
    print(metrics_df)

    stats_path = os.path.join(OUTPUT_DIR, 'performance_report.csv')
    metrics_df.to_csv(stats_path, index=False)

    # --- Equity Curve Plot ---
    plt.figure(figsize=(12, 6))

    for name, returns in results.items():
        curve = returns.groupby(returns.index).mean().fillna(0)
        cum_curve = (1 + curve).cumprod()
        plt.plot(cum_curve.index, cum_curve, label=name)

    plt.title(f'Equity Curve: High Momentum Short (RSI > {RSI_THRESHOLD}) with Stop Losses')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, 'equity_curve.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

def main():
    print("=== EXP-08: High Momentum Short - Candle Wick Analysis ===")
    tickers = load_tickers()
    if not tickers: return
    stock_df = fetch_data_batched(tickers)
    if stock_df.empty: return
    run_analysis(stock_df)

if __name__ == '__main__':
    main()
