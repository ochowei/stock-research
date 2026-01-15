
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import joblib
import time
import sys

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lab Utils Path
LAB_UTILS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '02_Lab_Utils'))
sys.path.append(LAB_UTILS_PATH)

try:
    from metrics import LabMetrics
except ImportError:
    class LabMetrics:
        TARGET_WIN_RATE = 0.55
        TARGET_AVG_RETURN = 0.0020
        @staticmethod
        def evaluate_experiment(df):
            return {}

# Date Range
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# Strategy Parameters
GAP_THRESHOLD = 0.005      # 0.5% Gap Threshold
PROFIT_THRESHOLD = 0.002   # 0.2% Profit Threshold

# Feature Definitions
BASE_FEATURES_TEMPLATE = ['Gap_Pct', 'RSI_PERIOD', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_CONTEXT = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20'] # Keep index RSI fixed at 14
NON_TECH_CONTEXT = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20']

# RSI Periods to Test
RSI_PERIODS = [2, 3, 4, 5, 7, 10, 14]

# Model Params (V6.2.4.RC)
TECH_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 8,
    'random_state': 42, 'verbosity': -1, 'n_jobs': 1
}
NON_TECH_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.02, 'max_depth': -1, 'num_leaves': 31,
    'random_state': 42, 'verbosity': -1, 'n_jobs': 1
}

# --- 2. Utility Functions ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NFLX', 'INTC', 'JPM', 'BAC', 'XOM', 'CVX', 'PG', 'KO', 'JNJ', 'PFE', 'PEP', 'CSCO']
    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_sectors(tickers):
    """Fetches sector information for tickers using yfinance with caching."""
    sector_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    if os.path.exists(sector_cache_path):
        with open(sector_cache_path, 'r') as f:
            return json.load(f)

    sector_map = {}
    print("Fetching sector information...")
    for i, t in enumerate(tickers):
        try:
            ticker_obj = yf.Ticker(t)
            sec = ticker_obj.info.get('sector', 'Unknown')
            sector_map[t] = sec
        except Exception:
            sector_map[t] = 'Unknown'
        time.sleep(0.1)

    with open(sector_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)
    return sector_map

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    # Use only a subset if too many tickers fail
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # Retry logic
    max_retries = 3
    data = pd.DataFrame()

    for i in range(max_retries):
        try:
            # Use 'repair=True' if available, otherwise standard
            data = yf.download(
                all_tickers, start=TRAIN_START, end=TEST_END,
                interval='1d', auto_adjust=True, progress=False, threads=True
            )
            if not data.empty:
                break
        except Exception as e:
            print(f"Download failed attempt {i+1}: {e}")
            time.sleep(2)

    if data.empty:
        print("Data download failed completely.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns: pass
        data = data.reset_index()

    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # Separate
    qqq = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stocks = data[~data['Ticker'].isin(benchmarks)]

    return stocks, qqq, spy

def safe_convert_numeric(df):
    """Safely converts columns to numeric, handling duplicates and types."""
    # Create copy
    df = df.copy()

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            # Ensure it is a Series
            if isinstance(df[col], pd.DataFrame):
                # Take first column if DataFrame
                df[col] = df[col].iloc[:, 0]

            # Force conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Additional check: If still object (due to all NaNs in object column?), force float
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(float)

    return df

def prepare_benchmark_features(df, prefix):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df = safe_convert_numeric(df)

    # Handle NaNs in close before calc
    df['Close'] = df['Close'].ffill()

    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Always keep Index RSI as 14 (Macro Context)
    # Check length
    if len(df) > 14:
        rsi = ta.rsi(df['Close'], length=14)
        if rsi is not None:
             df[f'{prefix}_RSI_14'] = rsi.shift(1)
        else:
             df[f'{prefix}_RSI_14'] = np.nan
    else:
        df[f'{prefix}_RSI_14'] = np.nan

    sum_prev_19 = df['Close'].rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[[f'{prefix}_Gap_Pct', f'{prefix}_RSI_14', f'{prefix}_Dist_MA20']]

def build_features(df, qqq_df, spy_df, rsi_period):
    """
    Builds features including the variable RSI period.
    """
    df = df.sort_index().copy()
    df = safe_convert_numeric(df)

    # Explicit cast to float for TA lib
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
         if col in df.columns:
             df[col] = df[col].astype(float)

    # Basic
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    try:
        # Variable RSI
        rsi_col_name = f'RSI_{rsi_period}'
        close_series = df['Close'].astype('float64')

        # Check if enough data
        if len(close_series) > rsi_period:
            rsi = ta.rsi(close_series, length=rsi_period)
            if rsi is not None:
                df[rsi_col_name] = rsi.shift(1)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

        # Other Base Features (Constant)
        high_s = df['High'].astype('float64')
        low_s = df['Low'].astype('float64')

        atr = ta.atr(high_s, low_s, close_series, length=14)
        if atr is not None:
             df['ATR_14'] = atr.shift(1)
        else:
             df['ATR_14'] = np.nan

        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
        df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

        close_filled = df['Close'].ffill()
        sum_prev_19 = close_filled.rolling(19).sum().shift(1)
        open_p = df['Open'].fillna(df['Close'])
        ma20_sim = (sum_prev_19 + open_p) / 20
        df['Dist_MA20'] = (open_p / ma20_sim) - 1

        df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        # Label
        df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
        df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
        df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

        # Join Context
        if not qqq_df.empty:
            df = df.join(qqq_df, how='left')
        if not spy_df.empty:
            df = df.join(spy_df, how='left')

    except Exception as e:
        # print(f"Feature build error: {e}")
        return pd.DataFrame()

    df = df.dropna()
    return df

def run_experiment(stock_raw, qqq_feat, spy_feat, sector_map):
    results = []

    if stock_raw.empty:
        print("Error: No stock data available.")
        return pd.DataFrame()

    for rsi_period in RSI_PERIODS:
        print(f"\nRunning for RSI Period: {rsi_period}")

        # Build dataset for this period
        all_data = []
        # Explicitly group by Ticker column
        for ticker, group in stock_raw.groupby('Ticker'):
            # Copy to avoid SettingWithCopy
            group = group.copy()

            # Robust Index Handling
            if group.index.name != 'Date':
                if 'Date' in group.columns:
                    df = group.set_index('Date')
                else:
                    # Reset index, rename 'index' to 'Date' if necessary
                    df = group.reset_index()
                    if 'index' in df.columns:
                        df = df.rename(columns={'index': 'Date'})

                    if 'Date' in df.columns:
                        df = df.set_index('Date')
                    else:
                        # Cannot find date column
                        continue
            else:
                df = group

            # Drop duplicates index
            df = df[~df.index.duplicated(keep='first')]

            if df.empty: continue

            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)

            feat_df = build_features(df, qqq_feat, spy_feat, rsi_period)
            if feat_df.empty: continue

            feat_df['Ticker'] = ticker
            feat_df['Sector'] = sector_map.get(ticker, 'Unknown')
            feat_df['Is_Tech'] = (feat_df['Sector'] == 'Technology').astype(int)

            signal_df = feat_df[feat_df['Is_Signal']].copy()
            if not signal_df.empty:
                all_data.append(signal_df)

        if not all_data:
            print("No signals generated.")
            continue

        full_df = pd.concat(all_data).sort_index()

        # Split
        train_df = full_df[full_df.index <= TRAIN_END]
        test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

        print(f"  Train Size: {len(train_df)}, Test Size: {len(test_df)}")

        if len(train_df) < 100 or len(test_df) < 10:
             print("  Not enough data to train/test.")
             continue

        # Features for this run
        current_base_features = [f.replace('RSI_PERIOD', f'RSI_{rsi_period}') for f in BASE_FEATURES_TEMPLATE]

        # Tech Model (Base + QQQ)
        tech_features = current_base_features + TECH_CONTEXT
        train_tech = train_df[train_df['Is_Tech'] == 1]
        test_tech = test_df[test_df['Is_Tech'] == 1]

        if len(train_tech) > 50:
            tech_model = LGBMClassifier(**TECH_PARAMS)
            tech_model.fit(
                train_tech[tech_features],
                train_tech['Label'],
                sample_weight=train_tech['Strategy_Ret'].abs() * 100
            )
            if len(test_tech) > 0:
                test_tech_preds = tech_model.predict_proba(test_tech[tech_features])[:, 1]
            else:
                test_tech_preds = np.array([])
        else:
            # print("  Warning: Not enough Tech training data.")
            test_tech_preds = np.zeros(len(test_tech))

        test_tech = test_tech.copy()
        test_tech['Pred_Prob'] = test_tech_preds

        # Non-Tech Model (Base + SPY)
        non_tech_features = current_base_features + NON_TECH_CONTEXT
        train_non = train_df[train_df['Is_Tech'] == 0]
        test_non = test_df[test_df['Is_Tech'] == 0]

        if len(train_non) > 50:
            non_tech_model = LGBMClassifier(**NON_TECH_PARAMS)
            non_tech_model.fit(
                train_non[non_tech_features],
                train_non['Label'],
                sample_weight=train_non['Strategy_Ret'].abs() * 100
            )
            if len(test_non) > 0:
                test_non_preds = non_tech_model.predict_proba(test_non[non_tech_features])[:, 1]
            else:
                test_non_preds = np.array([])
        else:
             # print("  Warning: Not enough Non-Tech training data.")
             test_non_preds = np.zeros(len(test_non))

        test_non = test_non.copy()
        test_non['Pred_Prob'] = test_non_preds

        combined_test = pd.concat([test_tech, test_non])

        # Apply standard threshold 0.5
        combined_test['Pred'] = (combined_test['Pred_Prob'] > 0.5).astype(int)

        # Metrics
        trades = combined_test[combined_test['Pred'] == 1]
        win_rate = (trades['Label'] == 1).mean() if len(trades) > 0 else 0
        avg_ret = trades['Strategy_Ret'].mean() if len(trades) > 0 else 0
        count = len(trades)

        print(f"  Result: WR {win_rate:.2%}, Ret {avg_ret:.4f}, Count {count}")

        results.append({
            'RSI_Period': rsi_period,
            'Win_Rate': win_rate,
            'Avg_Return': avg_ret,
            'Signal_Count': count
        })

    return pd.DataFrame(results)

def main():
    print("=== EXP-23: RSI Sensitivity Analysis ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)

    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)

    if stock_raw.empty:
        print("Critical Error: Stock data unavailable.")
        return

    # Pre-calculate benchmark context (constant)
    qqq_feat = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feat = prepare_benchmark_features(spy_raw, 'SPY')

    res_df = run_experiment(stock_raw, qqq_feat, spy_feat, sector_map)

    if res_df.empty:
        print("No results generated.")
        return

    # Save
    out_csv = os.path.join(OUTPUT_DIR, 'rsi_sensitivity_results.csv')
    res_df.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
    print(res_df)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['RSI_Period'], res_df['Win_Rate'], marker='o', label='Win Rate')
    plt.title('Win Rate vs RSI Period (V6.2.4 Framework)')
    plt.xlabel('RSI Period')
    plt.ylabel('Win Rate')
    plt.grid(True)
    plt.axhline(y=0.55, color='r', linestyle='--', label='Target 55%')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rsi_sensitivity_plot.png'))
    print("Plot saved.")

if __name__ == "__main__":
    main()
