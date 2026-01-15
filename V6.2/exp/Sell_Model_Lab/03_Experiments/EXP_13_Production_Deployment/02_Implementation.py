import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from lightgbm import LGBMClassifier
import joblib
import time

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for resource dir in standard V6.2 structure relative to this script
# V6.2/exp/Sell_Model_Lab/03_Experiments/EXP_13.../02_Implementation.py
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Range (Train on ALL available historical data up to now for Production)
TRAIN_START = '2020-01-01'
# We will use yfinance to fetch up to current date.
# We will not split train/test here, we want to maximize data for production models.
# But effectively we should probably stop at yesterday to avoid partial data.
# For simplicity in this script, we'll fetch everything and filter by date.
TRAIN_END = '2025-12-31' # Future date to capture everything

# Strategy Parameters
GAP_THRESHOLD = 0.005
PROFIT_THRESHOLD = 0.002

# Feature Sets
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

# Model Params (from EXP-06/07/11 Findings)
# Tech: Depth 3, LR 0.01
TECH_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 3,
    'num_leaves': 15, # 2^3 is 8, so 15 is ample
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

# Non-Tech: Unlimited Depth, LR 0.02
NON_TECH_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.02,
    'max_depth': -1,
    'num_leaves': 31,
    'random_state': 42,
    'n_jobs': -1,
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
    # Reuse cache from EXP-08 if possible (it had the full map)
    exp08_cache = os.path.join(BASE_DIR, '..', 'EXP_08_Production_Integration', '03_Output', 'sector_map.json')
    local_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    if os.path.exists(local_cache_path):
        with open(local_cache_path, 'r') as f:
            return json.load(f)

    if os.path.exists(exp08_cache):
        print(f"Loading sectors from EXP-08 cache...")
        with open(exp08_cache, 'r') as f:
            data = json.load(f)
        # Verify all tickers are present
        missing = [t for t in tickers if t not in data]
        if not missing:
            # Save locally
            with open(local_cache_path, 'w') as f:
                json.dump(data, f, indent=4)
            return data
        else:
             print(f"Cache missing {len(missing)} tickers. Updating...")
             # We will update 'data' with new tickers
    else:
        data = {}

    print("Fetching sector information...")
    for i, t in enumerate(tickers):
        if t in data: continue
        try:
            if i > 0 and i % 20 == 0: time.sleep(0.5)
            ticker_obj = yf.Ticker(t)
            data[t] = ticker_obj.info.get('sector', 'Unknown')
        except:
            data[t] = 'Unknown'

    with open(local_cache_path, 'w') as f:
        json.dump(data, f, indent=4)
    return data

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers, start=TRAIN_START, end=TRAIN_END,
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

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    if 'Close' in data.columns and data['Close'].isnull().all():
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
             data['Close'] = data['Adj Close']

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(benchmarks)]

    return stock_df, qqq_df, spy_df

def prepare_benchmark_features(bm_df, prefix):
    df = bm_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df[f'{prefix}_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[[f'{prefix}_Gap_Pct', f'{prefix}_RSI_14', f'{prefix}_Dist_MA20', 'Close']]

def build_features(df, bm_df, bm_prefix, bm_features):
    df = df.sort_index()
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    df.index = pd.to_datetime(df.index).normalize()
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    try:
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

        common_idx = df.index.intersection(bm_df.index)
        if len(common_idx) == 0: return pd.DataFrame()

        df_sub = df.loc[common_idx].copy()
        bm_sub = bm_df.loc[common_idx]

        for f in bm_features:
            if 'Gap' in f or 'RSI' in f or 'Dist' in f:
                df_sub[f] = bm_sub[f]

        aligned_close = pd.concat([df_sub['Close'], bm_sub['Close']], axis=1)
        corr = aligned_close.iloc[:,0].rolling(20).corr(aligned_close.iloc[:,1])

        # Decide name of correlation feature based on prefix
        corr_name = 'Sector_Corr' if bm_prefix == 'QQQ' else 'Market_Corr'
        df_sub[corr_name] = corr.shift(1)

        df = df_sub

    except Exception:
        return pd.DataFrame()

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

    req_cols = BASE_FEATURES + bm_features
    # Replace corr feature name in req_cols list locally
    corr_name = 'Sector_Corr' if bm_prefix == 'QQQ' else 'Market_Corr'
    req_cols = [c if 'Corr' not in c else corr_name for c in req_cols]

    return df.dropna(subset=req_cols)


def generate_production_script():
    script_content = r'''"""
Production Daily Plan V6.4
--------------------------
Generates daily sell signals using a Heterogeneous Ensemble:
1. Non-Tech Model: Base Features + SPY Context + LightGBM (Unlimited Depth, LR 0.02)
2. Tech Model: Base + QQQ Features + LightGBM (Depth 3, LR 0.01)

Usage: python production_daily_plan_v6_4.py
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
# Check for resource dir in standard V6.2 structure relative to this script
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
MODEL_DIR = BASE_DIR

NON_TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.4_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.4_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(MODEL_DIR, 'sector_map.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

GAP_THRESHOLD = 0.005

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        # Fallback search
        root_markers = ['V6.2', 'resource']
        parts = BASE_DIR.split(os.sep)
        if 'V6.2' in parts:
            idx = parts.index('V6.2')
            path = os.sep.join(parts[:idx+1] + ['resource', '2025_final_asset_pool.json'])
        else:
            print(f"[Error] Asset pool not found at {path}")
            return []

    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def get_sector(ticker, sector_map):
    return sector_map.get(ticker, 'Unknown')

def prepare_benchmark(bm_df, prefix):
    df = bm_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df[f'{prefix}_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1
    return df

def build_features_latest(df, bm_df, prefix, bm_features):
    """Builds features for the LAST row only"""
    if len(df) < 50: return None

    df = df.sort_index().copy()

    # Indicators
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
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

    last_idx = df.index[-1]
    row = df.iloc[[-1]].copy()

    # BM Features
    if last_idx not in bm_df.index:
        bm_row = bm_df.iloc[[-1]]
    else:
        bm_row = bm_df.loc[[last_idx]]

    for f in bm_features:
        if 'Corr' not in f:
            row[f] = bm_row[f].values[0]

    # Correlation
    common_idx = df.index.intersection(bm_df.index)
    df_sub = df.loc[common_idx]
    bm_sub = bm_df.loc[common_idx]

    aligned_close = pd.concat([df_sub['Close'], bm_sub['Close']], axis=1)
    corr = aligned_close.iloc[:,0].rolling(20).corr(aligned_close.iloc[:,1]).shift(1)

    corr_name = 'Sector_Corr' if prefix == 'QQQ' else 'Market_Corr'
    if np.isnan(corr.iloc[-1]): return None
    row[corr_name] = corr.iloc[-1]

    return row

def main():
    print("=== V6.4 Production Signal Generator ===")

    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    print("Fetching market data...")
    data = yf.download(tickers + ['QQQ', 'SPY'], period='3mo', interval='1d', auto_adjust=True, progress=False, threads=True)

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
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(['QQQ', 'SPY'])]

    # Prepare Benchmarks
    qqq_prep = prepare_benchmark(qqq_df, 'QQQ')
    spy_prep = prepare_benchmark(spy_df, 'SPY')

    latest_date = stock_df['Date'].max()
    print(f"Generating signals for date: {latest_date.date()}")

    signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        if group['Date'].max() != latest_date: continue

        sector = get_sector(ticker, sector_map)
        is_tech = (sector == 'Technology')

        try:
            if is_tech:
                feat_row = build_features_latest(group.set_index('Date'), qqq_prep, 'QQQ', TECH_FEATURES)
            else:
                feat_row = build_features_latest(group.set_index('Date'), spy_prep, 'SPY', NON_TECH_FEATURES)
        except Exception:
            continue

        if feat_row is None or feat_row.empty: continue
        if feat_row['Gap_Pct'].values[0] <= GAP_THRESHOLD: continue

        if is_tech:
            X = feat_row[BASE_FEATURES + TECH_FEATURES]
            prob = tech_model.predict_proba(X)[0][1]
        else:
            X = feat_row[BASE_FEATURES + NON_TECH_FEATURES]
            prob = non_tech_model.predict_proba(X)[0][1]

        if prob > 0.5:
            signals.append({
                'Ticker': ticker,
                'Sector': sector,
                'Gap_Pct': feat_row['Gap_Pct'].values[0],
                'Probability': prob,
                'Model': 'Tech' if is_tech else 'Non-Tech'
            })

    res_df = pd.DataFrame(signals).sort_values('Probability', ascending=False)
    print(f"\nGenerated {len(res_df)} signals.")
    print(res_df.head(10))

    out_path = os.path.join(MODEL_DIR, f"daily_plan_{latest_date.date()}.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(OUTPUT_DIR, 'production_daily_plan_v6_4.py'), 'w') as f:
        f.write(script_content)

# --- 3. Main ---

def main():
    print("=== EXP-13: Production Deployment (V6.4) ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)

    qqq_feats = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feats = prepare_benchmark_features(spy_raw, 'SPY')

    print("\n--- Training Tech Model (Base + QQQ) ---")
    tech_data = []
    for ticker, group in stock_raw.groupby('Ticker'):
        if sector_map.get(ticker) == 'Technology':
            df = group.set_index('Date').copy()
            fdf = build_features(df, qqq_feats, 'QQQ', TECH_FEATURES)
            if not fdf.empty and not fdf[fdf['Is_Signal']].empty:
                tech_data.append(fdf[fdf['Is_Signal']])

    if tech_data:
        full_tech = pd.concat(tech_data)
        X = full_tech[BASE_FEATURES + TECH_FEATURES]
        y = full_tech['Label']
        w = full_tech['Strategy_Ret'].abs() * 100

        tech_model = LGBMClassifier(**TECH_PARAMS)
        tech_model.fit(X, y, sample_weight=w)
        joblib.dump(tech_model, os.path.join(OUTPUT_DIR, 'v6.4_tech_model.joblib'))
        print("Tech Model Saved.")

    print("\n--- Training Non-Tech Model (Base + SPY) ---")
    non_tech_data = []
    for ticker, group in stock_raw.groupby('Ticker'):
        if sector_map.get(ticker) != 'Technology':
            df = group.set_index('Date').copy()
            fdf = build_features(df, spy_feats, 'SPY', NON_TECH_FEATURES)
            if not fdf.empty and not fdf[fdf['Is_Signal']].empty:
                non_tech_data.append(fdf[fdf['Is_Signal']])

    if non_tech_data:
        full_nt = pd.concat(non_tech_data)
        X = full_nt[BASE_FEATURES + NON_TECH_FEATURES]
        y = full_nt['Label']
        w = full_nt['Strategy_Ret'].abs() * 100

        nt_model = LGBMClassifier(**NON_TECH_PARAMS)
        nt_model.fit(X, y, sample_weight=w)
        joblib.dump(nt_model, os.path.join(OUTPUT_DIR, 'v6.4_non_tech_model.joblib'))
        print("Non-Tech Model Saved.")

    print("\n--- Generating Production Script ---")
    generate_production_script()
    print("Done.")

if __name__ == '__main__':
    main()
