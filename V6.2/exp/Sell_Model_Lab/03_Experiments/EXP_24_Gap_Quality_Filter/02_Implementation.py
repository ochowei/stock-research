
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
# V6.2.4.RC uses RSI 14
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_CONTEXT = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20']
NON_TECH_CONTEXT = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20']

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
        # Fallback list
        return ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NFLX', 'INTC']
    with open(path, 'r') as f:
        raw = json.load(f)
    # Clean tickers
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_sectors(tickers):
    """Fetches sector information for tickers using yfinance with caching."""
    sector_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    if os.path.exists(sector_cache_path):
        with open(sector_cache_path, 'r') as f:
            return json.load(f)

    sector_map = {}
    print("Fetching sector information...")
    # Batch processing could be better but keeping it simple
    for i, t in enumerate(tickers):
        if i % 10 == 0: print(f"  Processed {i}/{len(tickers)}...")
        try:
            ticker_obj = yf.Ticker(t)
            sec = ticker_obj.info.get('sector', 'Unknown')
            sector_map[t] = sec
        except Exception:
            sector_map[t] = 'Unknown'
        time.sleep(0.05) # Rate limit protection

    with open(sector_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)
    return sector_map

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    all_tickers = list(set(tickers + benchmarks))
    print(f"Downloading data for {len(all_tickers)} tickers...")

    batch_size = 10
    all_data_list = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1} ({len(batch)} tickers)...")

        for attempt in range(3):
            try:
                # Add delay to avoid rate limits
                time.sleep(1)
                data = yf.download(
                    batch, start=TRAIN_START, end=TEST_END,
                    interval='1d', auto_adjust=True, progress=False, threads=True
                )

                if not data.empty:
                    # Normalize MultiIndex columns immediately for this batch
                    if isinstance(data.columns, pd.MultiIndex):
                        try:
                            data = data.stack(level=1, future_stack=True)
                        except TypeError:
                            data = data.stack(level=1)
                        data = data.rename_axis(['Date', 'Ticker']).reset_index()
                    else:
                        # If single ticker, yf might not return MultiIndex in some versions
                        # Force Ticker column
                        if 'Ticker' not in data.columns:
                            data['Ticker'] = batch[0] # Assuming single ticker batch if not multiindex
                        data = data.reset_index()

                    all_data_list.append(data)
                    break
            except Exception as e:
                print(f"    Batch failed attempt {attempt+1}: {e}")
                time.sleep(2)
        else:
            print(f"    Skipping batch {batch} after 3 failures.")

    if not all_data_list:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Combine all batches
    full_data = pd.concat(all_data_list, ignore_index=True)

    if 'Date' not in full_data.columns and full_data.index.name == 'Date':
        full_data = full_data.reset_index()

    if 'Date' in full_data.columns:
        full_data['Date'] = pd.to_datetime(full_data['Date']).dt.tz_localize(None).dt.normalize()

    # Deduplicate just in case
    full_data = full_data.drop_duplicates(subset=['Date', 'Ticker'])

    print(f"Downloaded {len(full_data)} rows.")

    qqq = full_data[full_data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy = full_data[full_data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stocks = full_data[~full_data['Ticker'].isin(benchmarks)]

    return stocks, qqq, spy

def safe_convert_numeric(df):
    df = df.copy()
    # Remove duplicates
    df = df.loc[:, ~df.columns.duplicated()]

    cols_to_check = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_check:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].dtype == 'object':
                 df[col] = df[col].astype(float)
    return df

def prepare_benchmark_features(df, prefix):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df = safe_convert_numeric(df)
    df['Close'] = df['Close'].ffill()

    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

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

def build_features(df, qqq_df, spy_df):
    df = df.sort_index().copy()
    df = safe_convert_numeric(df)

    # Basic
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    try:
        # Check for NaNs in inputs before TA
        if df['Close'].isna().all(): return pd.DataFrame()

        # RSI 14
        close_series = pd.to_numeric(df['Close'], errors='coerce')
        # Fill NaNs for TA calculation to prevent error
        close_series_filled = close_series.ffill().bfill()

        rsi = ta.rsi(close_series_filled, length=14)
        if rsi is not None:
            df['RSI_14'] = rsi.shift(1)
        else:
            df['RSI_14'] = np.nan

        # ATR
        high_s = pd.to_numeric(df['High'], errors='coerce').ffill().bfill()
        low_s = pd.to_numeric(df['Low'], errors='coerce').ffill().bfill()
        # Recalculate filled close series
        close_series_filled = pd.to_numeric(df['Close'], errors='coerce').ffill().bfill()

        atr = ta.atr(high_s, low_s, close_series_filled, length=14)
        if atr is not None:
             df['ATR_14'] = atr.shift(1)
        else:
             df['ATR_14'] = np.nan

        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

        # Ensure Volume is numeric
        vol_s = pd.to_numeric(df['Volume'], errors='coerce')
        df['Vol_MA20'] = vol_s.rolling(20).mean().shift(1)
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

        close_filled = pd.to_numeric(df['Close'], errors='coerce').ffill()
        sum_prev_19 = close_filled.rolling(19).sum().shift(1)
        open_p = pd.to_numeric(df['Open'], errors='coerce').fillna(close_filled)
        ma20_sim = (sum_prev_19 + open_p) / 20
        df['Dist_MA20'] = (open_p / ma20_sim) - 1

        # Fill NaNs in features with 0 or mean to prevent signal drop
        # Ideally we should drop if we can't calculate, but here we see 'Adj Close' is issue?
        # Actually 'Adj Close' is not in our feature list explicitly, but might be lingering?
        # We only care about BASE_FEATURES

        # Check if we have 'Adj Close' in columns and drop it
        if 'Adj Close' in df.columns:
            df = df.drop(columns=['Adj Close'])

        # Ensure open/close/prev_close are numeric for Gap Pct
        open_n = pd.to_numeric(df['Open'], errors='coerce')
        prev_close_n = pd.to_numeric(df['Prev_Close'], errors='coerce')

        df['Gap_Pct'] = (open_n - prev_close_n) / prev_close_n

        # Label
        close_n = pd.to_numeric(df['Close'], errors='coerce')
        df['Strategy_Ret'] = (open_n - close_n) / open_n
        df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD

        # Debug signals
        # num_signals = df['Is_Signal'].sum()
        # if num_signals > 0:
        #     print(f"  Got {num_signals} signals")

        df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

        # Join Context
        if not qqq_df.empty:
            df = df.join(qqq_df, how='left')
        if not spy_df.empty:
            df = df.join(spy_df, how='left')

        # Forward fill context features if they are missing
        # Because we join on Date, and sometimes stocks have data when QQQ/SPY has slight mismatch or missing
        # But actually, QQQ/SPY should be full
        # Let's check if QQQ/SPY caused NaNs
        context_cols = [c for c in df.columns if 'QQQ_' in c or 'SPY_' in c]
        if context_cols:
             df[context_cols] = df[context_cols].ffill().bfill()

    except Exception as e:
        print(f"Feature build error: {e}")
        return pd.DataFrame()

    # Dropna check
    before_drop = len(df)

    # Check what columns have NaNs
    nan_cols = df.columns[df.isna().any()].tolist()

    # We only care if we lose Is_Signal rows
    sig_rows = df[df['Is_Signal']]
    if not sig_rows.empty:
         sig_rows_nans = sig_rows.columns[sig_rows.isna().any()].tolist()
         if sig_rows_nans:
              print(f"  Signal rows have NaNs in: {sig_rows_nans}")
              pass

    df = df.dropna()
    after_drop = len(df)

    if before_drop > 0 and after_drop == 0:
         pass # print(f"  Warning: Dropna removed all rows.")

    return df

def train_and_predict(stock_raw, qqq_feat, spy_feat, sector_map):
    print("Building datasets...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        group = group.copy()

        # Ensure Date index
        if group.index.name != 'Date':
            if 'Date' in group.columns:
                df = group.set_index('Date')
            else:
                df = group.reset_index()
                if 'index' in df.columns: df = df.rename(columns={'index': 'Date'})
                if 'Date' in df.columns: df = df.set_index('Date')
                else: continue
        else:
            df = group

        df = df[~df.index.duplicated(keep='first')]
        if df.empty: continue
        df.index = pd.to_datetime(df.index)

        feat_df = build_features(df, qqq_feat, spy_feat)
        if feat_df.empty: continue

        feat_df['Ticker'] = ticker
        feat_df['Sector'] = sector_map.get(ticker, 'Unknown')
        feat_df['Is_Tech'] = (feat_df['Sector'] == 'Technology').astype(int)

        # Filter for signals only to save memory, but need Train/Test split
        # We need all data for training
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("No signals generated.")
        return pd.DataFrame()

    full_df = pd.concat(all_data).sort_index()

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Total Signals: {len(full_df)}")
    print(f"Train Signals: {len(train_df)}")
    print(f"Test Signals:  {len(test_df)}")

    # --- Train Tech Model ---
    print("Training Tech Model...")
    train_tech = train_df[train_df['Is_Tech'] == 1]
    test_tech = test_df[test_df['Is_Tech'] == 1].copy() # Copy to avoid SettingWithCopy

    tech_features = BASE_FEATURES + TECH_CONTEXT

    if len(train_tech) > 50:
        tech_model = LGBMClassifier(**TECH_PARAMS)
        tech_model.fit(
            train_tech[tech_features],
            train_tech['Label'],
            sample_weight=train_tech['Strategy_Ret'].abs() * 100
        )
        if len(test_tech) > 0:
            test_tech['Pred_Prob'] = tech_model.predict_proba(test_tech[tech_features])[:, 1]
        else:
            test_tech['Pred_Prob'] = 0.0

        # Save feature importance
        imp = pd.DataFrame({'Feature': tech_features, 'Importance': tech_model.feature_importances_})
        imp.to_csv(os.path.join(OUTPUT_DIR, 'tech_feature_importance.csv'), index=False)
    else:
        print("Warning: Not enough Tech data.")
        test_tech['Pred_Prob'] = 0.0

    # --- Train Non-Tech Model ---
    print("Training Non-Tech Model...")
    train_non = train_df[train_df['Is_Tech'] == 0]
    test_non = test_df[test_df['Is_Tech'] == 0].copy()

    non_tech_features = BASE_FEATURES + NON_TECH_CONTEXT

    if len(train_non) > 50:
        non_tech_model = LGBMClassifier(**NON_TECH_PARAMS)
        non_tech_model.fit(
            train_non[non_tech_features],
            train_non['Label'],
            sample_weight=train_non['Strategy_Ret'].abs() * 100
        )
        if len(test_non) > 0:
            test_non['Pred_Prob'] = non_tech_model.predict_proba(test_non[non_tech_features])[:, 1]
        else:
            test_non['Pred_Prob'] = 0.0

        imp = pd.DataFrame({'Feature': non_tech_features, 'Importance': non_tech_model.feature_importances_})
        imp.to_csv(os.path.join(OUTPUT_DIR, 'non_tech_feature_importance.csv'), index=False)
    else:
        print("Warning: Not enough Non-Tech data.")
        test_non['Pred_Prob'] = 0.0

    # Combine
    combined_test = pd.concat([test_tech, test_non])
    combined_test['Pred'] = (combined_test['Pred_Prob'] > 0.5).astype(int)

    return combined_test

def analyze_gap_quality(df):
    print("\nAnalyzing Gap Quality...")

    # Filter for only predicted trades (V6.2.4.RC Baseline)
    trades = df[df['Pred'] == 1].copy()

    if trades.empty:
        print("No trades to analyze.")
        return

    # Baseline Metrics
    baseline_wr = (trades['Label'] == 1).mean()
    baseline_ret = trades['Strategy_Ret'].mean()
    print(f"Baseline (V6.2.4.RC) -> WR: {baseline_wr:.2%}, Avg Ret: {baseline_ret:.4f}, Count: {len(trades)}")

    # 1. Gap Size Analysis
    bins = [0.005, 0.01, 0.02, 0.03, 1.0]
    labels = ['0.5-1.0%', '1.0-2.0%', '2.0-3.0%', '>3.0%']
    trades['Gap_Bin'] = pd.cut(trades['Gap_Pct'], bins=bins, labels=labels)

    gap_stats = trades.groupby('Gap_Bin').agg({
        'Label': 'mean',
        'Strategy_Ret': 'mean',
        'Ticker': 'count'
    }).rename(columns={'Label': 'Win_Rate', 'Ticker': 'Count'})
    print("\n--- Performance by Gap Size ---")
    print(gap_stats)
    gap_stats.to_csv(os.path.join(OUTPUT_DIR, 'gap_size_analysis.csv'))

    # 2. Volume Ratio Analysis (Proxy for Relative Volume)
    # Bins: <1 (Low Vol), 1-2 (Normal), 2-3 (High), >3 (Extreme)
    vol_bins = [0, 1.0, 2.0, 3.0, 100.0]
    vol_labels = ['<1.0x', '1.0-2.0x', '2.0-3.0x', '>3.0x']
    trades['Vol_Bin'] = pd.cut(trades['Vol_Ratio'], bins=vol_bins, labels=vol_labels)

    vol_stats = trades.groupby('Vol_Bin').agg({
        'Label': 'mean',
        'Strategy_Ret': 'mean',
        'Ticker': 'count'
    }).rename(columns={'Label': 'Win_Rate', 'Ticker': 'Count'})
    print("\n--- Performance by Volume Ratio ---")
    print(vol_stats)
    vol_stats.to_csv(os.path.join(OUTPUT_DIR, 'vol_ratio_analysis.csv'))

    # 3. Combined Filter Test
    # Hypothesis: Avoid Large Gaps (>2%) AND Extreme Volume (>3x)
    print("\n--- Filter Tests ---")

    # Filter 1: Gap < 2%
    f1 = trades[trades['Gap_Pct'] <= 0.02]
    print(f"Filter: Gap <= 2.0% -> WR: {f1['Label'].mean():.2%}, Ret: {f1['Strategy_Ret'].mean():.4f}, Count: {len(f1)}")

    # Filter 2: Vol < 3.0
    f2 = trades[trades['Vol_Ratio'] <= 3.0]
    print(f"Filter: Vol <= 3.0x -> WR: {f2['Label'].mean():.2%}, Ret: {f2['Strategy_Ret'].mean():.4f}, Count: {len(f2)}")

    # Filter 3: Gap < 2% AND Vol < 3.0
    f3 = trades[(trades['Gap_Pct'] <= 0.02) & (trades['Vol_Ratio'] <= 3.0)]
    print(f"Filter: Gap <= 2% & Vol <= 3x -> WR: {f3['Label'].mean():.2%}, Ret: {f3['Strategy_Ret'].mean():.4f}, Count: {len(f3)}")

    # Filter 4: Contrarian - Large Gaps are Better?
    f4 = trades[trades['Gap_Pct'] > 0.02]
    print(f"Filter: Gap > 2.0% -> WR: {f4['Label'].mean():.2%}, Ret: {f4['Strategy_Ret'].mean():.4f}, Count: {len(f4)}")

    # Save full trade list for review
    trades.to_csv(os.path.join(OUTPUT_DIR, 'all_trades_with_bins.csv'))

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gap Bin Plot
    gap_stats['Win_Rate'].plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Win Rate by Gap Size')
    axes[0].set_ylabel('Win Rate')
    axes[0].axhline(y=baseline_wr, color='r', linestyle='--', label='Baseline')
    axes[0].legend()

    # Vol Bin Plot
    vol_stats['Win_Rate'].plot(kind='bar', ax=axes[1], color='lightgreen')
    axes[1].set_title('Win Rate by Volume Ratio')
    axes[1].set_ylabel('Win Rate')
    axes[1].axhline(y=baseline_wr, color='r', linestyle='--', label='Baseline')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gap_quality_analysis.png'))
    print("Plots saved.")

def main():
    print("=== EXP-24: Gap Quality Filter (Volume & Context) ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)

    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)

    if stock_raw.empty:
        print("Critical Error: Stock data unavailable.")
        return

    qqq_feat = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feat = prepare_benchmark_features(spy_raw, 'SPY')

    # Train and Predict (Baseline V6.2.4.RC)
    result_df = train_and_predict(stock_raw, qqq_feat, spy_feat, sector_map)

    if result_df.empty:
        print("No predictions generated.")
        return

    # Analyze
    analyze_gap_quality(result_df)

if __name__ == "__main__":
    main()
