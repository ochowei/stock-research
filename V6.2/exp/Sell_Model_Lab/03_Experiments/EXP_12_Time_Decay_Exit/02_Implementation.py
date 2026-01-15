
import os
import sys
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Paths & Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct relative path to resource: V6.2/exp/Sell_Model_Lab/03_Experiments/EXP_12... -> ../../../../resource
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Re-use models from EXP-08
EXP_08_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '03_Experiments', 'EXP_08_Production_Integration', '03_Output'))
NON_TECH_MODEL_PATH = os.path.join(EXP_08_DIR, 'v6.3_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(EXP_08_DIR, 'v6.3_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(EXP_08_DIR, 'sector_map.json')

# Parameters
TEST_START = '2024-02-01' # Starting Feb to ensure 1h data availability (730 days limit)
GAP_THRESHOLD = 0.005 # 0.5%
SCORE_THRESHOLD = 0.5

# Feature Defs
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] Asset pool not found at {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_daily_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    all_tickers = tickers + benchmarks
    print(f"Downloading Daily data for {len(all_tickers)} tickers...")

    start_date = '2023-09-01'
    data = yf.download(all_tickers, start=start_date, end=None, interval='1d', auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns: pass
        data = data.reset_index()

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(benchmarks)]

    return stock_df, qqq_df

def prepare_benchmark_features(qqq_df):
    df = qqq_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['QQQ_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['QQQ_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['QQQ_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Close']]

def build_features(df, qqq_df, is_tech=False):
    df = df.sort_index().copy()

    # Calculate Base Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Tech Features
    if is_tech and not qqq_df.empty:
        common_idx = df.index.intersection(qqq_df.index)
        if len(common_idx) == 0: return pd.DataFrame()

        df_sub = df.loc[common_idx].copy()
        qqq_sub = qqq_df.loc[common_idx]

        df_sub['QQQ_Gap_Pct'] = qqq_sub['QQQ_Gap_Pct']
        df_sub['QQQ_RSI_14'] = qqq_sub['QQQ_RSI_14']
        df_sub['QQQ_Dist_MA20'] = qqq_sub['QQQ_Dist_MA20']

        # Sector Corr
        aligned_close = pd.concat([df_sub['Close'], qqq_sub['Close']], axis=1)
        aligned_close.columns = ['Stock_Close', 'QQQ_Close']
        corr_series = aligned_close['Stock_Close'].rolling(20).corr(aligned_close['QQQ_Close'])
        df_sub['Sector_Corr'] = corr_series.shift(1)

        df = df_sub

    return df

def fetch_intraday_data(tickers):
    """
    Fetches 1h data for the specified tickers.
    Returns a dictionary {ticker: dataframe}.
    """
    print(f"Downloading Intraday (1h) data for {len(tickers)} tickers...")

    # Batch download
    # yfinance download return multi-index if >1 ticker
    try:
        data = yf.download(tickers, period='730d', interval='1h', auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        print(f"Error downloading intraday: {e}")
        return {}

    intraday_map = {}

    if len(tickers) == 1:
        # Single ticker case, not multi-index columns usually, or just columns
        ticker = tickers[0]
        data['Ticker'] = ticker
        # Ensure timezone agnostic
        if 'Datetime' not in data.columns:
            data = data.reset_index()
        data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)
        intraday_map[ticker] = data
    else:
        # Multi-index: Level 0 = Price, Level 1 = Ticker
        # Or Level 1 = Ticker, Level 0 = Price (depending on pandas version/yf version)
        # Standard yf: columns are (Price, Ticker)

        # Stack to get Ticker column
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.stack(level=1, future_stack=True)
            except TypeError:
                data = data.stack(level=1)
            data = data.rename_axis(['Datetime', 'Ticker']).reset_index()

            data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)

            for t, group in data.groupby('Ticker'):
                intraday_map[t] = group.set_index('Datetime').sort_index()
        else:
            # Fallback
            print("Warning: Unexpected data format in intraday fetch.")

    return intraday_map

def get_hourly_exit(row, intraday_df, hours_held):
    """
    Finds the exit price after N hours.
    row: Signal row (Date, Ticker)
    intraday_df: DataFrame with Datetime index for this ticker
    hours_held: 1, 2, 3...
    """
    signal_date = row['Date'] # Timestamp

    # Filter intraday data for this date
    # Intraday index is Datetime
    day_data = intraday_df[intraday_df.index.normalize() == signal_date]

    if day_data.empty:
        return np.nan

    # Sort by time
    day_data = day_data.sort_index()

    # 09:30-10:30 is bar 0
    # 10:30-11:30 is bar 1
    # ...
    # We want to exit at the CLOSE of the Nth bar.
    # hours_held = 1 -> Exit at end of 1st bar (approx 10:30)

    idx = hours_held - 1

    if idx < 0: return np.nan
    if idx >= len(day_data):
        # If we asked for 5 hours but only 3 exist (half day?), take the last one
        return day_data.iloc[-1]['Close']

    return day_data.iloc[idx]['Close']

def main():
    print("=== EXP-12: Time-Decay Exit Optimization ===")

    # 1. Setup
    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    # 2. Daily Data & Signals
    stock_df, qqq_df = fetch_daily_data(tickers)
    qqq_feats = prepare_benchmark_features(qqq_df)

    if not os.path.exists(NON_TECH_MODEL_PATH) or not os.path.exists(TECH_MODEL_PATH):
        print("Models not found! Ensure EXP-08 ran successfully.")
        return

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    print("Generating Signals on Test Set...")
    signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        sector = sector_map.get(ticker, 'Unknown')
        is_tech = (sector == 'Technology')

        df = group.set_index('Date').copy()
        df = df[df.index >= '2023-11-01']

        if df.empty: continue

        try:
            feat_df = build_features(df, qqq_feats, is_tech=is_tech)
        except Exception:
            continue

        if feat_df.empty: continue
        feat_df = feat_df[feat_df.index >= TEST_START]
        candidates = feat_df[feat_df['Gap_Pct'] > GAP_THRESHOLD].copy()

        if candidates.empty: continue

        if is_tech:
            cols = BASE_FEATURES + TECH_FEATURES
            if not all(c in candidates.columns for c in cols): continue
            probs = tech_model.predict_proba(candidates[cols])[:, 1]
        else:
            cols = BASE_FEATURES
            if not all(c in candidates.columns for c in cols): continue
            probs = non_tech_model.predict_proba(candidates[cols])[:, 1]

        candidates['Probability'] = probs
        trades = candidates[candidates['Probability'] > SCORE_THRESHOLD].copy()
        trades['Ticker'] = ticker
        trades['Sector'] = sector
        trades['Date'] = trades.index

        signals.append(trades)

    all_signals = pd.concat(signals)
    print(f"Total Signals Generated: {len(all_signals)}")

    # 3. Intraday Data Fetch
    active_tickers = all_signals['Ticker'].unique().tolist()
    print(f"Active Tickers: {len(active_tickers)}")

    # Note: Fetching 500 tickers intraday might be slow.
    intraday_map = fetch_intraday_data(active_tickers)

    # 4. Simulation
    print("Simulating Hourly Exits...")

    # Strategy: Short at Open.
    # Return = (Open - Exit) / Open

    results = []

    # We will iterate and compute columns for each horizon
    # Horizons: 1h, 2h, 3h, 4h, 5h, Close (Daily Close)

    df_res = all_signals.copy()

    # Calculate Daily Close Return (Benchmark)
    df_res['Ret_MOC'] = (df_res['Open'] - df_res['Close']) / df_res['Open']

    horizons = [1, 2, 3, 4, 5]

    for h in horizons:
        col_name = f'Ret_{h}H'

        def get_ret(row):
            t = row['Ticker']
            if t not in intraday_map:
                return np.nan

            exit_price = get_hourly_exit(row, intraday_map[t], h)

            if pd.isna(exit_price):
                # If intraday data missing, fallback to Daily Close?
                # Better to keep NaN to filter later for fair comparison
                return np.nan

            return (row['Open'] - exit_price) / row['Open']

        df_res[col_name] = df_res.apply(get_ret, axis=1)

    # 5. Analysis
    # Filter rows where we have data for fair comparison?
    # Or just average available data.
    # Let's count missing
    print("\nData Completeness:")
    print(df_res[['Ret_MOC', 'Ret_1H', 'Ret_3H']].count())

    summary = []

    # Metric: Avg Return, Win Rate
    # Add MOC to list
    metrics_cols = [f'Ret_{h}H' for h in horizons] + ['Ret_MOC']

    for col in metrics_cols:
        valid_trades = df_res[col].dropna()
        if valid_trades.empty:
            continue

        win_rate = (valid_trades > 0).mean()
        avg_ret = valid_trades.mean()
        total_ret = valid_trades.sum()
        count = len(valid_trades)

        summary.append({
            'Strategy': col,
            'Win_Rate': win_rate,
            'Avg_Return': avg_ret,
            'Total_Return': total_ret,
            'Trade_Count': count
        })

    summary_df = pd.DataFrame(summary).sort_values('Total_Return', ascending=False)

    print("\n=== Results Summary ===")
    print(summary_df.to_string())

    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'time_decay_results.csv'), index=False)
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'signal_details.csv'), index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df['Strategy'], summary_df['Total_Return'], color='skyblue')
    plt.title('Total Return by Exit Strategy')
    plt.ylabel('Total Return (Sum of %)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'return_comparison.png'))

    print(f"\nOutputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
