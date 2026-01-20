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
GAP_THRESHOLD_BASE = 0.005      # 0.5% for General Model
GAP_THRESHOLD_HIGH_VOL = 0.02   # 2.0% for Specialized Model
PROFIT_THRESHOLD = 0.002

# Feature Definitions
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

# High Vol Model Params (Starting point: Robust settings)
HIGH_VOL_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 4, 'num_leaves': 15,
    'random_state': 42, 'verbosity': -1, 'n_jobs': 1
}

# --- 2. Utility Functions ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NFLX', 'INTC']
    with open(path, 'r') as f:
        raw = json.load(f)
    # Remove exchange prefixes like NASDAQ:
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_sectors(tickers):
    sector_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')
    if os.path.exists(sector_cache_path):
        with open(sector_cache_path, 'r') as f:
            return json.load(f)

    sector_map = {}
    print("Fetching sector information...")
    for i, t in enumerate(tickers):
        if i % 20 == 0: print(f"  Processed {i}/{len(tickers)}...")
        try:
            ticker_obj = yf.Ticker(t)
            sec = ticker_obj.info.get('sector', 'Unknown')
            sector_map[t] = sec
        except Exception:
            sector_map[t] = 'Unknown'
        time.sleep(0.05)
    with open(sector_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)
    return sector_map

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    # Filter tickers that we know are failing or problematic to save time/errors
    bad_tickers = ['FI', 'CELH', 'LTBR', 'STX', 'TMDX', 'DOCN', 'ANET', 'CCJ', 'QBTS']
    tickers = [t for t in tickers if t not in bad_tickers]

    all_tickers = list(set(tickers + benchmarks))
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # Check if we have a local cache to save time
    cache_path = os.path.join(OUTPUT_DIR, 'data_cache.joblib')
    if os.path.exists(cache_path):
        print("Loading data from cache...")
        return joblib.load(cache_path)

    batch_size = 20 # Increased batch size
    all_data_list = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        print(f"  Downloading batch {i//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1}...")
        for attempt in range(3):
            try:
                time.sleep(1)
                data = yf.download(
                    batch, start=TRAIN_START, end=TEST_END,
                    interval='1d', auto_adjust=True, progress=False, threads=True
                )
                if not data.empty:
                    # Fix for yfinance returning different structures
                    if isinstance(data.columns, pd.MultiIndex):
                        # If we have (Price, Ticker)
                        try:
                            # New yfinance structure
                            if data.columns.nlevels == 2:
                                data = data.stack(level=1, future_stack=True)
                            else:
                                data = data.stack(level=1)
                        except Exception as e:
                            # Fallback
                            data = data.stack(level=1)

                        data = data.rename_axis(['Date', 'Ticker']).reset_index()
                    else:
                        # Single ticker or flat structure
                        if 'Ticker' not in data.columns:
                            # If single ticker batch
                            data['Ticker'] = batch[0]
                        data = data.reset_index()

                    # Ensure columns exist
                    req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for c in req_cols:
                        if c not in data.columns:
                            # Sometimes yfinance returns 'Adj Close' instead of 'Close' if auto_adjust=False
                            # But we used auto_adjust=True
                            pass

                    all_data_list.append(data)
                    break
            except Exception as e:
                print(f"    Batch failed: {e}")
                time.sleep(2)

    if not all_data_list:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    full_data = pd.concat(all_data_list, ignore_index=True)
    if 'Date' in full_data.columns:
        full_data['Date'] = pd.to_datetime(full_data['Date']).dt.tz_localize(None).dt.normalize()

    # Clean duplicates
    full_data = full_data.drop_duplicates(subset=['Date', 'Ticker'])

    # Filter out empty or bad rows
    full_data = full_data.dropna(subset=['Close'])

    qqq = full_data[full_data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy = full_data[full_data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stocks = full_data[~full_data['Ticker'].isin(benchmarks)]

    print(f"  Downloaded {len(stocks)} rows of stock data.")

    # Save cache
    joblib.dump((stocks, qqq, spy), cache_path)

    return stocks, qqq, spy

def safe_convert_numeric(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors='coerce')
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
        df[f'{prefix}_RSI_14'] = rsi.shift(1) if rsi is not None else np.nan
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

    if len(df) < 50: return pd.DataFrame()
    if df['Close'].isna().all(): return pd.DataFrame()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    # Features
    close_filled = df['Close'].ffill().bfill()

    # RSI
    rsi = ta.rsi(close_filled, length=14)
    df['RSI_14'] = rsi.shift(1) if rsi is not None else np.nan

    # ATR
    high_s = df['High'].ffill().bfill()
    low_s = df['Low'].ffill().bfill()
    atr = ta.atr(high_s, low_s, close_filled, length=14)
    df['ATR_14'] = atr.shift(1) if atr is not None else np.nan
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

    # Vol Ratio
    vol_s = df['Volume'].fillna(0)
    df['Vol_MA20'] = vol_s.rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    # Dist MA20
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(close_filled)
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1

    # Gap
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Strategy Ret
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']

    # Base Signal (for General Model)
    df['Is_Signal_Base'] = df['Gap_Pct'] > GAP_THRESHOLD_BASE

    # High Vol Signal
    df['Is_Signal_HighVol'] = df['Gap_Pct'] > GAP_THRESHOLD_HIGH_VOL

    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

    # Context
    if not qqq_df.empty: df = df.join(qqq_df, how='left')
    if not spy_df.empty: df = df.join(spy_df, how='left')

    context_cols = [c for c in df.columns if 'QQQ_' in c or 'SPY_' in c]
    if context_cols: df[context_cols] = df[context_cols].ffill().bfill()

    return df.dropna()

def prepare_dataset(stock_raw, qqq_feat, spy_feat, sector_map):
    print("Building full dataset...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        group = group.copy()
        if group.index.name != 'Date':
            if 'Date' in group.columns: group = group.set_index('Date')
            else: continue

        group = group[~group.index.duplicated(keep='first')]
        if group.empty: continue
        group.index = pd.to_datetime(group.index)

        feat_df = build_features(group, qqq_feat, spy_feat)
        if feat_df.empty: continue

        feat_df['Ticker'] = ticker
        feat_df['Sector'] = sector_map.get(ticker, 'Unknown')
        feat_df['Is_Tech'] = (feat_df['Sector'] == 'Technology').astype(int)

        # We need rows that are signals in EITHER model
        signal_df = feat_df[feat_df['Is_Signal_Base']].copy() # Base is superset of HighVol usually

        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data: return pd.DataFrame()
    return pd.concat(all_data).sort_index()

def train_baseline(train_df, test_df):
    print("\n--- Training Baseline (V6.2.4.RC) ---")

    # Tech Model
    train_tech = train_df[train_df['Is_Tech'] == 1]
    test_tech = test_df[test_df['Is_Tech'] == 1].copy()
    tech_features = BASE_FEATURES + TECH_CONTEXT

    if len(train_tech) > 50:
        tech_model = LGBMClassifier(**TECH_PARAMS)
        tech_model.fit(
            train_tech[tech_features], train_tech['Label'],
            sample_weight=train_tech['Strategy_Ret'].abs() * 100
        )
        test_tech['Pred_Prob_Base'] = tech_model.predict_proba(test_tech[tech_features])[:, 1]
    else:
        test_tech['Pred_Prob_Base'] = 0.0

    # Non-Tech Model
    train_non = train_df[train_df['Is_Tech'] == 0]
    test_non = test_df[test_df['Is_Tech'] == 0].copy()
    non_tech_features = BASE_FEATURES + NON_TECH_CONTEXT

    if len(train_non) > 50:
        non_tech_model = LGBMClassifier(**NON_TECH_PARAMS)
        non_tech_model.fit(
            train_non[non_tech_features], train_non['Label'],
            sample_weight=train_non['Strategy_Ret'].abs() * 100
        )
        test_non['Pred_Prob_Base'] = non_tech_model.predict_proba(test_non[non_tech_features])[:, 1]
    else:
        test_non['Pred_Prob_Base'] = 0.0

    return pd.concat([test_tech, test_non])

def train_specialized(train_df, test_df):
    print("\n--- Training Specialized High-Vol Model ---")

    # Filter for High Volatility (Gap > 2%)
    train_hv = train_df[train_df['Is_Signal_HighVol']].copy()
    # Use ALL features (Union)
    features = list(set(BASE_FEATURES + TECH_CONTEXT + NON_TECH_CONTEXT))

    print(f"  Training on {len(train_hv)} High-Vol samples...")

    if len(train_hv) < 50:
        print("  Not enough data for specialized model.")
        test_df['Pred_Prob_Special'] = 0.0
        return test_df

    model = LGBMClassifier(**HIGH_VOL_PARAMS)
    model.fit(
        train_hv[features], train_hv['Label'],
        sample_weight=train_hv['Strategy_Ret'].abs() * 100
    )

    # Save Model
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'high_vol_model.joblib'))

    # Save Importance
    imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    imp = imp.sort_values('Importance', ascending=False)
    imp.to_csv(os.path.join(OUTPUT_DIR, 'high_vol_feature_importance.csv'), index=False)

    test_df = test_df.copy()
    test_df['Pred_Prob_Special'] = model.predict_proba(test_df[features])[:, 1]

    return test_df

def analyze_results(df):
    print("\n--- Comparative Analysis (Target: Gap > 2%) ---")

    # Filter Test Data for the Target Regime (Gap > 2%)
    target_df = df[df['Is_Signal_HighVol']].copy()

    if target_df.empty:
        print("No High-Vol samples in test set.")
        return

    # Baseline Performance
    target_df['Pred_Base'] = (target_df['Pred_Prob_Base'] > 0.5).astype(int)
    base_trades = target_df[target_df['Pred_Base'] == 1]

    base_wr = base_trades['Label'].mean()
    base_ret = base_trades['Strategy_Ret'].mean()
    base_count = len(base_trades)

    print(f"Baseline (V6.2.4.RC) on High Vol:")
    print(f"  Win Rate: {base_wr:.2%}")
    print(f"  Avg Ret:  {base_ret:.4f}")
    print(f"  Trades:   {base_count}")

    # Specialized Performance
    target_df['Pred_Special'] = (target_df['Pred_Prob_Special'] > 0.5).astype(int)
    spec_trades = target_df[target_df['Pred_Special'] == 1]

    spec_wr = spec_trades['Label'].mean()
    spec_ret = spec_trades['Strategy_Ret'].mean()
    spec_count = len(spec_trades)

    print(f"Specialized Model on High Vol:")
    print(f"  Win Rate: {spec_wr:.2%}")
    print(f"  Avg Ret:  {spec_ret:.4f}")
    print(f"  Trades:   {spec_count}")

    # Save Report
    report = {
        'Metric': ['Win Rate', 'Avg Return', 'Trade Count'],
        'Baseline': [base_wr, base_ret, base_count],
        'Specialized': [spec_wr, spec_ret, spec_count],
        'Delta': [spec_wr - base_wr, spec_ret - base_ret, spec_count - base_count]
    }
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, 'comparison_report.csv'), index=False)

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Win Rate
    ax[0].bar(['Baseline', 'Specialized'], [base_wr, spec_wr], color=['gray', 'blue'])
    ax[0].set_title('Win Rate (Gap > 2%)')
    ax[0].set_ylim(0.4, 0.65)
    ax[0].axhline(y=0.55, color='r', linestyle='--', label='Target')

    # Return
    ax[1].bar(['Baseline', 'Specialized'], [base_ret, spec_ret], color=['gray', 'green'])
    ax[1].set_title('Avg Return (Gap > 2%)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_plot.png'))

def main():
    print("=== EXP-25: Large Gap Specialization ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)

    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)
    if stock_raw.empty: return

    qqq_feat = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feat = prepare_benchmark_features(spy_raw, 'SPY')

    full_df = prepare_dataset(stock_raw, qqq_feat, spy_feat, sector_map)
    if full_df.empty:
        print("Dataset empty.")
        return

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Train Size: {len(train_df)}")
    print(f"Test Size:  {len(test_df)}")

    # 1. Train/Predict Baseline
    test_w_base = train_baseline(train_df, test_df)

    # 2. Train/Predict Specialized
    final_df = train_specialized(train_df, test_w_base)

    # 3. Analyze
    analyze_results(final_df)

if __name__ == "__main__":
    main()
