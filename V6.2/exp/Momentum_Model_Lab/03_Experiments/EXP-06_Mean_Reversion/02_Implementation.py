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

    # Gap Calculation
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Momentum Indicators (Shifted by 1 to prevent look-ahead)
    try:
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_14'] = df['RSI_14'].shift(1)
    except:
        df['RSI_14'] = np.nan

    try:
        df['ROC_14'] = df['Close'].pct_change(periods=14)
        df['ROC_14'] = df['ROC_14'].shift(1)
    except:
        df['ROC_14'] = np.nan

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
    full_df = full_df.dropna(subset=['RSI_14', 'Long_Ret', 'Short_Ret'])

    print(f"Total Gap Up Signals: {len(full_df)}")

    high_mom_mask = full_df['RSI_14'] > 70
    low_mom_mask = full_df['RSI_14'] <= 70

    high_mom_df = full_df[high_mom_mask]
    low_mom_df = full_df[low_mom_mask]

    print(f"High Momentum Signals (RSI>70): {len(high_mom_df)}")
    print(f"Low Momentum Signals (RSI<=70): {len(low_mom_df)}")

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
    stats.append(calculate_metrics(full_df, 'Long_Ret', 'Baseline_Long_All'))
    stats.append(calculate_metrics(high_mom_df, 'Long_Ret', 'High_Mom_Long'))
    stats.append(calculate_metrics(high_mom_df, 'Short_Ret', 'High_Mom_Short'))
    stats.append(calculate_metrics(low_mom_df, 'Long_Ret', 'Low_Mom_Long'))

    stats_df = pd.DataFrame(stats)
    print("\n--- Performance Report ---")
    print(stats_df)

    stats_path = os.path.join(OUTPUT_DIR, 'performance_report.csv')
    stats_df.to_csv(stats_path, index=False)

    plt.figure(figsize=(12, 6))

    curve_short = high_mom_df.groupby(high_mom_df.index)['Short_Ret'].mean().fillna(0)
    curve_short = (1 + curve_short).cumprod()

    curve_long = high_mom_df.groupby(high_mom_df.index)['Long_Ret'].mean().fillna(0)
    curve_long = (1 + curve_long).cumprod()

    curve_base = full_df.groupby(full_df.index)['Long_Ret'].mean().fillna(0)
    curve_base = (1 + curve_base).cumprod()

    plt.plot(curve_short.index, curve_short, label='High Mom Short (Reversion)', color='green')
    plt.plot(curve_long.index, curve_long, label='High Mom Long (Continuation)', color='red')
    plt.plot(curve_base.index, curve_base, label='Baseline Long (All)', color='gray', linestyle='--')

    plt.title('Equity Curve: High Momentum Reversion vs Continuation')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, 'equity_curve.png')
    plt.savefig(plot_path)

def main():
    print("=== EXP-06: Mean Reversion Signal (Gap Fade) ===")
    tickers = load_tickers()
    if not tickers: return
    stock_df, _ = fetch_data_batched(tickers)
    if stock_df.empty: return
    run_analysis(stock_df)

if __name__ == '__main__':
    main()
