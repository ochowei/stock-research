import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer, precision_score
import joblib
import time

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Date Range
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# Strategy Parameters
GAP_THRESHOLD = 0.005      # 0.5% Gap Threshold
PROFIT_THRESHOLD = 0.002   # 0.2% Profit Threshold

# Feature Sets
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']

# Tuning Parameters
N_ITER = 30 # Reduced for speed in this experiment
CV_SPLITS = 5
RANDOM_STATE = 42

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
    """Fetches sector information for tickers using yfinance with caching."""
    # Try to reuse cache from EXP-06 if possible, or create local one
    global_cache_path = os.path.join(BASE_DIR, '..', 'EXP_06_Hyperparameter_Tuning', '03_Output', 'sector_map.json')
    local_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    if os.path.exists(global_cache_path):
        print(f"Loading sectors from global cache ({global_cache_path})...")
        with open(global_cache_path, 'r') as f:
            return json.load(f)

    if os.path.exists(local_cache_path):
        print("Loading sectors from local cache...")
        with open(local_cache_path, 'r') as f:
            return json.load(f)

    print("Fetching sector information (this may take a moment)...")
    sector_map = {}

    for i, t in enumerate(tickers):
        try:
            if i > 0 and i % 20 == 0:
                print(f"Fetched {i}/{len(tickers)}...")
                time.sleep(1)

            ticker_obj = yf.Ticker(t)
            sec = ticker_obj.info.get('sector', 'Unknown')
            sector_map[t] = sec
        except Exception as e:
            sector_map[t] = 'Unknown'

    # Save cache
    with open(local_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)

    return sector_map

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY', '^VIX']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END,
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
             pass
        data = data.reset_index()

    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()
    if 'Date' not in data.columns:
        data = data.reset_index()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    if 'Close' in data.columns and data['Close'].isnull().all():
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
            print("WARNING: 'Close' is all NaN, using 'Adj Close' instead.")
            data['Close'] = data['Adj Close']

    # Separate benchmarks
    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(benchmarks)]

    return stock_df, qqq_df

def prepare_benchmark_features(qqq_df):
    """Calculates features for QQQ to be merged"""
    df = qqq_df.copy()

    # QQQ Base Indicators
    df['Prev_Close'] = df['Close'].shift(1)

    # QQQ Gap (Available at Open)
    df['QQQ_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # QQQ T-1 Indicators
    df['QQQ_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['QQQ_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Close']] # Keep Close for correlation

def build_features(df, qqq_df):
    """Feature Engineering - Base + Tech"""
    df = df.sort_index()

    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.index = pd.to_datetime(df.index).normalize()

    # Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    try:
        # Indicators (T-1)
        df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

        # Volume
        df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

        # Dist MA20
        close_filled = df['Close'].ffill()
        sum_prev_19 = close_filled.rolling(19).sum().shift(1)
        open_p = df['Open'].fillna(df['Close'])
        ma20_sim = (sum_prev_19 + open_p) / 20
        df['Dist_MA20'] = (open_p / ma20_sim) - 1

        # --- Tech Features Merge ---
        # Join with QQQ features on Index (Date)
        # Note: QQQ features are already shifted appropriately in prepare_benchmark_features
        # except 'Close' which is used for correlation

        # We need to do this carefully.
        # Since we are iterating per ticker, we can merge.
        # But merging inside the loop is slow if we do full join.
        # Better: Join index based.

        common_idx = df.index.intersection(qqq_df.index)
        if len(common_idx) == 0: return pd.DataFrame()

        # Subset both to common dates
        df_sub = df.loc[common_idx].copy()
        qqq_sub = qqq_df.loc[common_idx]

        df_sub['QQQ_Gap_Pct'] = qqq_sub['QQQ_Gap_Pct']
        df_sub['QQQ_RSI_14'] = qqq_sub['QQQ_RSI_14']
        df_sub['QQQ_Dist_MA20'] = qqq_sub['QQQ_Dist_MA20']

        # Correlation (Rolling 20 T-1)
        # We need unshifted closes aligned by date
        # Then calculate correlation, then shift result

        # Align closes
        aligned_close = pd.concat([df_sub['Close'], qqq_sub['Close']], axis=1)
        aligned_close.columns = ['Stock_Close', 'QQQ_Close']

        corr_series = aligned_close['Stock_Close'].rolling(20).corr(aligned_close['QQQ_Close'])
        df_sub['Sector_Corr'] = corr_series.shift(1)

        df = df_sub

    except Exception as e:
        # print(f"Err: {e}")
        return pd.DataFrame()

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Labeling
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

    df_clean = df.dropna(subset=BASE_FEATURES + TECH_FEATURES)
    return df_clean

def evaluate_metrics(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})
    if len(df) == 0: return 0, 0, 0

    model_df = df[df['Pred'] == 1]
    if len(model_df) == 0:
        return 0, 0, 0

    win_rate = (model_df['Return'] > 0).mean()
    avg_ret = model_df['Return'].mean()
    count = len(model_df)

    return win_rate, avg_ret, count

def tune_model(X, y, weights, name="Model"):
    print(f"\nTuning {name}...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.02, 0.05],
        'num_leaves': [15, 31, 63],
        'max_depth': [3, 5, 7, -1],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0],
    }

    model = LGBMClassifier(n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)
    scorer = make_scorer(precision_score, zero_division=0)
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring=scorer,
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X, y, sample_weight=weights)
    print(f"Best Params ({name}): {search.best_params_}")
    return search.best_estimator_

# --- 3. Main ---

def main():
    print(f"=== EXP-07: Tech-Specific Features ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, qqq_raw = fetch_data(tickers)

    print("Preparing QQQ features...")
    qqq_feats = prepare_benchmark_features(qqq_raw)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        sector = sector_map.get(ticker, 'Unknown')
        if sector != 'Technology':
            continue # We only care about Tech stocks for this experiment

        df = group.set_index('Date').copy()
        df = df.dropna(subset=['Close'])
        if df.empty: continue

        feat_df = build_features(df, qqq_feats)
        if feat_df.empty: continue

        feat_df['Ticker'] = ticker

        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No signals found for Tech Sector.")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Tech Signals: {len(full_df)}")

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    # --- Experiment ---

    # 1. Baseline: Base Features only
    print("\n--- 1. Baseline Tech Model (Base Features) ---")
    X_base = train_df[BASE_FEATURES]
    y_train = train_df['Label']
    w_train = train_df['Strategy_Ret'].abs() * 100

    # Using EXP-06 found params for Tech (Depth 3, LR 0.01) as starting point for fairness?
    # Or should we retune to be safe?
    # Let's retune both to be fair comparison of feature sets.
    model_base = tune_model(X_base, y_train, w_train, "Tech_Base_Feats")

    # 2. Experiment: Base + Tech Features
    print("\n--- 2. Experiment Tech Model (Base + Tech Features) ---")
    ALL_FEATS = BASE_FEATURES + TECH_FEATURES
    X_exp = train_df[ALL_FEATS]

    model_exp = tune_model(X_exp, y_train, w_train, "Tech_New_Feats")

    # --- Evaluation ---
    print("\nEvaluating on Test Set...")

    # Baseline Preds
    preds_base = model_base.predict(test_df[BASE_FEATURES])

    # Experiment Preds
    preds_exp = model_exp.predict(test_df[ALL_FEATS])

    # Metrics
    base_win, base_avg, base_cnt = evaluate_metrics(test_df['Label'], preds_base, test_df['Strategy_Ret'])
    exp_win, exp_avg, exp_cnt = evaluate_metrics(test_df['Label'], preds_exp, test_df['Strategy_Ret'])

    # Feature Importance (Exp Model)
    imp = pd.DataFrame({
        'Feature': ALL_FEATS,
        'Importance': model_exp.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nFeature Importance (New Model):")
    print(imp)
    imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    # --- Results ---
    print("\n" + "="*60)
    print("RESULTS Comparison (Tech Sector Only - Test Set 2024-2025)")
    print("="*60)
    print(f"{'Model':<20} | {'Win Rate':<10} | {'Avg Ret':<10} | {'Trades':<10}")
    print("-" * 60)
    print(f"{'Tech Baseline':<20} | {base_win:.2%}    | {base_avg:.4f}    | {base_cnt}")
    print(f"{'Tech + Features':<20} | {exp_win:.2%}    | {exp_avg:.4f}    | {exp_cnt}")

    # Save Results
    res_data = [
        {'Model': 'Tech_Baseline', 'Win_Rate': base_win, 'Avg_Ret': base_avg, 'Trades': base_cnt},
        {'Model': 'Tech_New_Feats', 'Win_Rate': exp_win, 'Avg_Ret': exp_avg, 'Trades': exp_cnt},
    ]
    pd.DataFrame(res_data).to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Save Models
    joblib.dump(model_base, os.path.join(OUTPUT_DIR, 'model_tech_base.joblib'))
    joblib.dump(model_exp, os.path.join(OUTPUT_DIR, 'model_tech_exp.joblib'))

if __name__ == '__main__':
    main()
