import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from lightgbm import LGBMClassifier
import joblib
import time
import matplotlib.pyplot as plt

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# V6.2/exp/Sell_Model_Lab/03_Experiments/EXP_20.../02_Implementation.py
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
LAB_UTILS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '02_Lab_Utils'))

sys.path.append(LAB_UTILS_DIR)
# Try to import metrics, but fallback if fails
try:
    from metrics import LabMetrics
except ImportError:
    print("Warning: Could not import LabMetrics. Using local evaluation.")
    LabMetrics = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training/Testing Split
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
DATA_END = '2025-12-31' # Fetch up to now

# Strategy Parameters
GAP_THRESHOLD = 0.005
PROFIT_THRESHOLD = 0.002

# Feature Sets
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']

# Baseline Feature Sets (V6.2.4.RC)
TECH_BASE_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_BASE_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

# New Features
TECH_NEW_FEATURE = 'Rel_Gap_QQQ'
NON_TECH_NEW_FEATURE = 'Rel_Gap_SPY'

# Model Params (Fixed)
TECH_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 3,
    'num_leaves': 15,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': -1
}

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
    # Try to reuse a recent sector map to save time
    # Check EXP-18 or EXP-13 or just use a local one if exists
    cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    # Check if we can find one in a neighbor directory to bootstrap
    if not os.path.exists(cache_path):
        neighbor_dirs = ['EXP_19_Crypto_Pure_Play', 'EXP_18_Production_Script_Update', 'EXP_13_Production_Deployment']
        for d in neighbor_dirs:
            p = os.path.join(BASE_DIR, '..', d, '03_Output', 'sector_map.json')
            if os.path.exists(p):
                print(f"Found existing sector map in {d}")
                with open(p, 'r') as f:
                    existing = json.load(f)
                # We can use this, but need to check if we have all tickers
                # For simplicity, we'll just use it and fetch missing if any
                with open(cache_path, 'w') as f:
                    json.dump(existing, f)
                break

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    missing = [t for t in tickers if t not in data]
    if missing:
        print(f"Fetching sectors for {len(missing)} tickers...")
        for i, t in enumerate(missing):
            try:
                if i > 0 and i % 10 == 0: time.sleep(0.5)
                ticker_obj = yf.Ticker(t)
                data[t] = ticker_obj.info.get('sector', 'Unknown')
            except:
                data[t] = 'Unknown'

        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=4)

    return data

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # Download in chunks to avoid some yfinance issues
    data = yf.download(
        all_tickers, start=TRAIN_START, end=DATA_END,
        interval='1d', auto_adjust=True, progress=False, threads=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns:
            # This happens if only 1 ticker is downloaded
            pass
        data = data.reset_index()

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # Handle missing Close if Adj Close exists (auto_adjust=True usually handles this but safety first)
    if 'Close' in data.columns and data['Close'].isnull().all() and 'Adj Close' in data.columns:
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

        # Merge Benchmark Features
        common_idx = df.index.intersection(bm_df.index)
        if len(common_idx) == 0: return pd.DataFrame()

        df_sub = df.loc[common_idx].copy()
        bm_sub = bm_df.loc[common_idx]

        for f in bm_features:
            if 'Gap' in f or 'RSI' in f or 'Dist' in f:
                df_sub[f] = bm_sub[f]

        # Correlation
        aligned_close = pd.concat([df_sub['Close'], bm_sub['Close']], axis=1)
        corr = aligned_close.iloc[:,0].rolling(20).corr(aligned_close.iloc[:,1])
        corr_name = 'Sector_Corr' if bm_prefix == 'QQQ' else 'Market_Corr'
        df_sub[corr_name] = corr.shift(1)

        df = df_sub

        # Gap Calculation
        df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        # --- NEW: Relative Gap Feature ---
        # Note: bm_features already contains the index gap (e.g., QQQ_Gap_Pct)
        bm_gap_col = f'{bm_prefix}_Gap_Pct'
        rel_gap_col = f'Rel_Gap_{bm_prefix}'
        df[rel_gap_col] = df['Gap_Pct'] - df[bm_gap_col]

        # Targets
        df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
        df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
        df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

    except Exception as e:
        # print(f"Error building features: {e}")
        return pd.DataFrame()

    return df

def evaluate_model(model, X, df_meta):
    """
    Evaluates model and returns metrics + signal DataFrame
    """
    if X.empty:
        return {'Win_Rate': 0, 'Avg_Return': 0, 'Count': 0}, pd.DataFrame()

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    res_df = df_meta.copy()
    res_df['Probability'] = probs
    res_df['Prediction'] = preds

    # Filter for actual trades (Prediction == 1)
    trades = res_df[res_df['Prediction'] == 1]

    if len(trades) == 0:
        return {'Win_Rate': 0, 'Avg_Return': 0, 'Count': 0}, res_df

    win_rate = (trades['Strategy_Ret'] > 0).mean()
    avg_ret = trades['Strategy_Ret'].mean()

    return {
        'Win_Rate': win_rate,
        'Avg_Return': avg_ret,
        'Count': len(trades)
    }, res_df

# --- 3. Main Logic ---

def main():
    print("=== EXP-20: Relative Gap Features ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)

    qqq_feats = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feats = prepare_benchmark_features(spy_raw, 'SPY')

    # Prepare Datasets
    print("\nPreparing Datasets...")

    tech_data = []
    non_tech_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        sector = sector_map.get(ticker, 'Unknown')
        df = group.set_index('Date').copy()

        if sector == 'Technology':
            # Build with QQQ context
            fdf = build_features(df, qqq_feats, 'QQQ', TECH_BASE_FEATURES)
            if not fdf.empty:
                fdf['Ticker'] = ticker
                tech_data.append(fdf)
        else:
            # Build with SPY context
            fdf = build_features(df, spy_feats, 'SPY', NON_TECH_BASE_FEATURES)
            if not fdf.empty:
                fdf['Ticker'] = ticker
                non_tech_data.append(fdf)

    full_tech = pd.concat(tech_data) if tech_data else pd.DataFrame()
    full_nt = pd.concat(non_tech_data) if non_tech_data else pd.DataFrame()

    # Filter for Valid Signals (Gap > Threshold) for Training/Testing
    # Note: We filter strictly by Gap Threshold as defined in the strategy
    full_tech = full_tech[full_tech['Is_Signal']].copy()
    full_nt = full_nt[full_nt['Is_Signal']].copy()

    print(f"Tech Samples: {len(full_tech)}")
    print(f"Non-Tech Samples: {len(full_nt)}")

    # Split Train/Test
    tech_train = full_tech[full_tech.index <= TRAIN_END]
    tech_test = full_tech[full_tech.index >= TEST_START]

    nt_train = full_nt[full_nt.index <= TRAIN_END]
    nt_test = full_nt[full_nt.index >= TEST_START]

    results = []

    # --- Experiment A: Tech Sector ---
    print("\n--- Experiment A: Tech Sector ---")

    # A1. Baseline
    feat_base = BASE_FEATURES + TECH_BASE_FEATURES
    print(f"Training Baseline (Features: {len(feat_base)})...")

    model_tech_base = LGBMClassifier(**TECH_PARAMS)
    model_tech_base.fit(
        tech_train[feat_base],
        tech_train['Label'],
        sample_weight=tech_train['Strategy_Ret'].abs() * 100
    )

    metrics_tech_base, df_tech_base = evaluate_model(
        model_tech_base,
        tech_test[feat_base],
        tech_test[['Strategy_Ret', 'Label', 'Ticker']]
    )
    print(f"Baseline: WR={metrics_tech_base['Win_Rate']:.2%}, Ret={metrics_tech_base['Avg_Return']:.4f}, Count={metrics_tech_base['Count']}")

    # A2. Test (Relative Gap)
    feat_test = feat_base + [TECH_NEW_FEATURE]
    print(f"Training Test (Features: {len(feat_test)})...")

    model_tech_test = LGBMClassifier(**TECH_PARAMS)
    model_tech_test.fit(
        tech_train[feat_test],
        tech_train['Label'],
        sample_weight=tech_train['Strategy_Ret'].abs() * 100
    )

    metrics_tech_test, df_tech_test = evaluate_model(
        model_tech_test,
        tech_test[feat_test],
        tech_test[['Strategy_Ret', 'Label', 'Ticker']]
    )
    print(f"Test:     WR={metrics_tech_test['Win_Rate']:.2%}, Ret={metrics_tech_test['Avg_Return']:.4f}, Count={metrics_tech_test['Count']}")

    results.append({
        'Sector': 'Tech',
        'Model': 'Baseline',
        'Win_Rate': metrics_tech_base['Win_Rate'],
        'Avg_Return': metrics_tech_base['Avg_Return'],
        'Count': metrics_tech_base['Count']
    })
    results.append({
        'Sector': 'Tech',
        'Model': 'Test (+RelGap)',
        'Win_Rate': metrics_tech_test['Win_Rate'],
        'Avg_Return': metrics_tech_test['Avg_Return'],
        'Count': metrics_tech_test['Count']
    })

    # Feature Importance for Tech
    imp_df = pd.DataFrame({
        'Feature': feat_test,
        'Importance': model_tech_test.feature_importances_
    }).sort_values('Importance', ascending=False)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, 'tech_feature_importance.csv'), index=False)
    print("Top Tech Features:")
    print(imp_df.head(5))

    # --- Experiment B: Non-Tech Sector ---
    print("\n--- Experiment B: Non-Tech Sector ---")

    # B1. Baseline
    feat_base_nt = BASE_FEATURES + NON_TECH_BASE_FEATURES
    print(f"Training Baseline (Features: {len(feat_base_nt)})...")

    model_nt_base = LGBMClassifier(**NON_TECH_PARAMS)
    model_nt_base.fit(
        nt_train[feat_base_nt],
        nt_train['Label'],
        sample_weight=nt_train['Strategy_Ret'].abs() * 100
    )

    metrics_nt_base, df_nt_base = evaluate_model(
        model_nt_base,
        nt_test[feat_base_nt],
        nt_test[['Strategy_Ret', 'Label', 'Ticker']]
    )
    print(f"Baseline: WR={metrics_nt_base['Win_Rate']:.2%}, Ret={metrics_nt_base['Avg_Return']:.4f}, Count={metrics_nt_base['Count']}")

    # B2. Test (Relative Gap)
    feat_test_nt = feat_base_nt + [NON_TECH_NEW_FEATURE]
    print(f"Training Test (Features: {len(feat_test_nt)})...")

    model_nt_test = LGBMClassifier(**NON_TECH_PARAMS)
    model_nt_test.fit(
        nt_train[feat_test_nt],
        nt_train['Label'],
        sample_weight=nt_train['Strategy_Ret'].abs() * 100
    )

    metrics_nt_test, df_nt_test = evaluate_model(
        model_nt_test,
        nt_test[feat_test_nt],
        nt_test[['Strategy_Ret', 'Label', 'Ticker']]
    )
    print(f"Test:     WR={metrics_nt_test['Win_Rate']:.2%}, Ret={metrics_nt_test['Avg_Return']:.4f}, Count={metrics_nt_test['Count']}")

    results.append({
        'Sector': 'Non-Tech',
        'Model': 'Baseline',
        'Win_Rate': metrics_nt_base['Win_Rate'],
        'Avg_Return': metrics_nt_base['Avg_Return'],
        'Count': metrics_nt_base['Count']
    })
    results.append({
        'Sector': 'Non-Tech',
        'Model': 'Test (+RelGap)',
        'Win_Rate': metrics_nt_test['Win_Rate'],
        'Avg_Return': metrics_nt_test['Avg_Return'],
        'Count': metrics_nt_test['Count']
    })

    # Feature Importance for Non-Tech
    imp_df_nt = pd.DataFrame({
        'Feature': feat_test_nt,
        'Importance': model_nt_test.feature_importances_
    }).sort_values('Importance', ascending=False)
    imp_df_nt.to_csv(os.path.join(OUTPUT_DIR, 'non_tech_feature_importance.csv'), index=False)
    print("Top Non-Tech Features:")
    print(imp_df_nt.head(5))

    # Save Summary
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)
    print("\nSummary:")
    print(summary_df)

if __name__ == "__main__":
    main()
