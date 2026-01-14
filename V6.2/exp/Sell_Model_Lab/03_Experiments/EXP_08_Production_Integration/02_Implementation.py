import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from lightgbm import LGBMClassifier
import joblib
import time
import datetime

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct path to resource from this experiment folder
# V6.2/exp/Sell_Model_Lab/03_Experiments/EXP_08.../02_Implementation.py
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Range (Full History for Production)
TRAIN_START = '2020-01-01'
TRAIN_END   = '2024-12-31' # Use data up to end of 2024 for training, or recent

# Feature Sets
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']

# Model Params (Hardcoded from EXP-06/07 findings)
NON_TECH_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42,
    'n_jobs': 1,
    'verbosity': -1
}

TECH_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.01,
    'num_leaves': 31, # depth=3 limits leaves anyway (2^3=8), but keeping default or matching depth
    'max_depth': 3,
    'random_state': 42,
    'n_jobs': 1,
    'verbosity': -1
}

# --- 2. Utility Functions ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_sectors(tickers):
    """Fetches sector information with caching."""
    # Reuse cache if available
    local_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    # Check if we can borrow from EXP-07 or EXP-06
    exp07_cache = os.path.join(BASE_DIR, '..', 'EXP_07_Tech_Features', '03_Output', 'sector_map.json')

    if os.path.exists(local_cache_path):
        with open(local_cache_path, 'r') as f: return json.load(f)

    if os.path.exists(exp07_cache):
        print("Using sector map from EXP-07...")
        with open(exp07_cache, 'r') as f:
            data = json.load(f)
            # Save to local for valid artifact
            with open(local_cache_path, 'w') as out:
                json.dump(data, out, indent=4)
            return data

    print("Fetching sectors...")
    sector_map = {}
    for i, t in enumerate(tickers):
        try:
            if i % 50 == 0: print(f" {i}/{len(tickers)}")
            ticker_obj = yf.Ticker(t)
            sector_map[t] = ticker_obj.info.get('sector', 'Unknown')
        except:
            sector_map[t] = 'Unknown'

    with open(local_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)
    return sector_map

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY', '^VIX']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # We grab data until "now" to have latest for verification, but filter for training
    data = yf.download(
        all_tickers, start=TRAIN_START,
        interval='1d', auto_adjust=True, progress=False, threads=True
    )

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

    if 'Close' in data.columns and data['Close'].isnull().all() and 'Adj Close' in data.columns:
        data['Close'] = data['Adj Close']

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
    df = df.sort_index()
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    # Base Features
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

    # Labeling
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    # 0.5% Gap Threshold for Labeling Validity
    df['Label'] = (df['Strategy_Ret'] > 0.002).astype(int)

    if is_tech:
        df = df.dropna(subset=BASE_FEATURES + TECH_FEATURES)
    else:
        df = df.dropna(subset=BASE_FEATURES)

    return df

def generate_production_script():
    content = r'''"""
Production Daily Plan V6.3
--------------------------
Generates daily sell signals using a Heterogeneous Ensemble:
1. Non-Tech Model: Base Features (5) + LightGBM (Unlimited Depth, LR 0.02)
2. Tech Model: Base + QQQ Features (9) + LightGBM (Depth 3, LR 0.01)

Usage: python production_daily_plan_v6_3.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'resource'))
MODEL_DIR = os.path.join(BASE_DIR, '03_Output') # Assuming models are here for now

# Check if running as standalone in root or exp
if not os.path.exists(RESOURCE_DIR):
    # Fallback for dev environment structure
    RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))

NON_TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.3_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.3_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(MODEL_DIR, 'sector_map.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']

GAP_THRESHOLD = 0.005

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] Asset pool not found at {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def get_sector(ticker, sector_map):
    return sector_map.get(ticker, 'Unknown')

def prepare_benchmark(qqq_df):
    df = qqq_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['QQQ_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['QQQ_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['QQQ_Dist_MA20'] = (open_p / ma20_sim) - 1
    return df

def build_features_latest(df, qqq_df, is_tech=False):
    """Builds features for the LAST row only (for prediction)"""
    # We need enough history for indicators (e.g. 50 days)
    if len(df) < 50: return None

    df = df.sort_index().copy()

    # Calculate Indicators on full history
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)

    # Use Open if available (Live), else use Close (Simulated Open for today?)
    # For daily plan generation, we typically run BEFORE market open or AT market open.
    # If Pre-market, we might fetch Pre-market open.
    # For now, we assume we have 'Open' (Live) or we use 'Close' of prev day as proxy if missing?
    # Actually, standard procedure:
    # We need today's Opening price to calculate Gap.
    # If generating plan BEFORE open, we can't calculate Gap yet.
    # This script likely generates "Potential Signals" or assumes we run it right after Open.
    # In V6.2/V6.3, we typically pass the 'Current Open' to the model.
    # Let's assume the data contains the latest candle with Open.

    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Extract latest row
    last_idx = df.index[-1]
    row = df.iloc[[-1]].copy()

    if is_tech:
        # QQQ Features
        # Match by date? Or just take latest QQQ?
        # Assuming QQQ data is aligned
        if last_idx not in qqq_df.index:
            # Try to use latest available QQQ
            qqq_row = qqq_df.iloc[[-1]]
        else:
            qqq_row = qqq_df.loc[[last_idx]]

        row['QQQ_Gap_Pct'] = qqq_row['QQQ_Gap_Pct'].values[0]
        row['QQQ_RSI_14'] = qqq_row['QQQ_RSI_14'].values[0]
        row['QQQ_Dist_MA20'] = qqq_row['QQQ_Dist_MA20'].values[0]

        # Sector Corr (Needs rolling calculation aligned)
        # We need the last 20 days aligned
        common_idx = df.index.intersection(qqq_df.index)
        df_sub = df.loc[common_idx]
        qqq_sub = qqq_df.loc[common_idx]

        aligned_close = pd.concat([df_sub['Close'], qqq_sub['Close']], axis=1)
        aligned_close.columns = ['Stock_Close', 'QQQ_Close']
        corr = aligned_close['Stock_Close'].rolling(20).corr(aligned_close['QQQ_Close']).shift(1)

        if np.isnan(corr.iloc[-1]): return None
        row['Sector_Corr'] = corr.iloc[-1]

    return row

def main():
    print("=== V6.3 Production Signal Generator ===")

    # Load Resources
    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    # Fetch Data (Live/Recent)
    # Fetching last 60 days to ensure enough for indicators
    print("Fetching market data...")
    data = yf.download(tickers + ['QQQ'], period='3mo', interval='1d', auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns: pass
        data = data.reset_index()

    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    stock_df = data[data['Ticker'] != 'QQQ']

    # Prepare QQQ
    qqq_prep = prepare_benchmark(qqq_df)

    # Identify Latest Date
    latest_date = stock_df['Date'].max()
    print(f"Generating signals for date: {latest_date.date()}")

    signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        # Filter group to last date
        if group['Date'].max() != latest_date:
            continue

        sector = get_sector(ticker, sector_map)
        is_tech = (sector == 'Technology')

        try:
            feat_row = build_features_latest(group.set_index('Date'), qqq_prep, is_tech)
        except Exception as e:
            continue

        if feat_row is None or feat_row.empty: continue

        # Check Gap Threshold
        if feat_row['Gap_Pct'].values[0] <= GAP_THRESHOLD:
            continue

        # Predict
        if is_tech:
            feats = TECH_FEATURES + BASE_FEATURES # Ensure order?
            # Actually Implementation used BASE + TECH order.
            # "ALL_FEATS = BASE_FEATURES + TECH_FEATURES"
            X = feat_row[BASE_FEATURES + TECH_FEATURES]
            prob = tech_model.predict_proba(X)[0][1]
        else:
            X = feat_row[BASE_FEATURES]
            prob = non_tech_model.predict_proba(X)[0][1]

        if prob > 0.5:
            signals.append({
                'Ticker': ticker,
                'Sector': sector,
                'Gap_Pct': feat_row['Gap_Pct'].values[0],
                'Probability': prob,
                'Model': 'Tech' if is_tech else 'Non-Tech'
            })

    # Output
    res_df = pd.DataFrame(signals).sort_values('Probability', ascending=False)
    print(f"\nGenerated {len(res_df)} signals.")
    print(res_df.head(10))

    out_path = os.path.join(MODEL_DIR, f"daily_plan_{latest_date.date()}.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(OUTPUT_DIR, 'production_daily_plan_v6_3.py'), 'w') as f:
        f.write(content)
    print("Generated production script.")

# --- 3. Main Workflow ---

def main():
    print("=== EXP-08: Production Integration ===")

    # 1. Load Data
    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, qqq_raw = fetch_data(tickers)

    qqq_feats = prepare_benchmark_features(qqq_raw)

    # 2. Train Non-Tech Model
    print("\n--- Training Non-Tech Model ---")
    non_tech_data = []
    for ticker, group in stock_raw.groupby('Ticker'):
        if sector_map.get(ticker) == 'Technology': continue

        df = group.set_index('Date').copy()
        df = df.dropna(subset=['Close'])
        if df.empty: continue

        # Build Base Features
        fdf = build_features(df, qqq_feats, is_tech=False)
        if not fdf.empty:
            # Filter for signal only for training?
            # Standard practice: Train on ALL data or just GAP data?
            # V6.2 usually trains on "Valid Gap Signals" to differentiate success/fail.
            sig_df = fdf[fdf['Gap_Pct'] > 0.005] # 0.5% Gap
            non_tech_data.append(sig_df)

    full_non_tech = pd.concat(non_tech_data)
    full_non_tech = full_non_tech[full_non_tech.index <= TRAIN_END]

    print(f"Non-Tech Samples: {len(full_non_tech)}")

    X_nt = full_non_tech[BASE_FEATURES]
    y_nt = full_non_tech['Label']
    w_nt = full_non_tech['Strategy_Ret'].abs() * 100

    model_nt = LGBMClassifier(**NON_TECH_PARAMS)
    model_nt.fit(X_nt, y_nt, sample_weight=w_nt)
    joblib.dump(model_nt, os.path.join(OUTPUT_DIR, 'v6.3_non_tech_model.joblib'))

    # 3. Train Tech Model
    print("\n--- Training Tech Model ---")
    tech_data = []
    for ticker, group in stock_raw.groupby('Ticker'):
        if sector_map.get(ticker) != 'Technology': continue

        df = group.set_index('Date').copy()
        df = df.dropna(subset=['Close'])
        if df.empty: continue

        # Build Tech Features
        fdf = build_features(df, qqq_feats, is_tech=True)
        if not fdf.empty:
            sig_df = fdf[fdf['Gap_Pct'] > 0.005]
            tech_data.append(sig_df)

    full_tech = pd.concat(tech_data)
    full_tech = full_tech[full_tech.index <= TRAIN_END]

    print(f"Tech Samples: {len(full_tech)}")

    # ALL_FEATS = BASE + TECH
    ALL_TECH_FEATS = BASE_FEATURES + TECH_FEATURES
    X_t = full_tech[ALL_TECH_FEATS]
    y_t = full_tech['Label']
    w_t = full_tech['Strategy_Ret'].abs() * 100

    model_t = LGBMClassifier(**TECH_PARAMS)
    model_t.fit(X_t, y_t, sample_weight=w_t)
    joblib.dump(model_t, os.path.join(OUTPUT_DIR, 'v6.3_tech_model.joblib'))

    # 4. Generate Script
    generate_production_script()

    print("\nSUCCESS: V6.3 Models and Script Generated.")

if __name__ == '__main__':
    main()
