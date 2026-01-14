import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
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

# Feature Set (Winner of EXP-03)
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']

# Tuning Parameters
N_ITER = 50
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
    sector_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    if os.path.exists(sector_cache_path):
        print("Loading sectors from cache...")
        with open(sector_cache_path, 'r') as f:
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
            # print(f"Warning: Could not fetch sector for {t}: {e}")
            sector_map[t] = 'Unknown'

    # Save cache
    with open(sector_cache_path, 'w') as f:
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

    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    stock_df = data[~data['Ticker'].isin(benchmarks)]
    return stock_df, vix_df

def build_features(df):
    """Feature Engineering - Base Set Only"""
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

    except Exception as e:
        return pd.DataFrame()

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Labeling
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    # df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100 # We pass this separately if needed

    df_clean = df.dropna(subset=BASE_FEATURES)
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
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.02, 0.05, 0.1],
        'num_leaves': [15, 20, 31, 50, 63],
        'max_depth': [-1, 3, 5, 7, 10],
        'min_child_samples': [20, 50, 100],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    model = LGBMClassifier(n_jobs=1, random_state=RANDOM_STATE, verbosity=-1)

    # Use Precision (Class 1) as proxy for Win Rate
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
    print(f"Best Score (Precision): {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_

# --- 3. Main ---

def main():
    print(f"=== EXP-06: Hyperparameter Tuning ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, _ = fetch_data(tickers)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        df = df.dropna(subset=['Close'])
        if df.empty: continue

        feat_df = build_features(df)
        if feat_df.empty: continue

        feat_df['Ticker'] = ticker
        feat_df['Sector'] = sector_map.get(ticker, 'Unknown')
        feat_df['Is_Tech'] = (feat_df['Sector'] == 'Technology').astype(int)

        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No signals found.")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Gap Signals: {len(full_df)}")

    # Sort by date for TimeSeriesSplit
    full_df = full_df.sort_index()

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    # --- BASELINE PARAMS (EXP-05) ---
    baseline_params = {
        'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
        'n_jobs': -1, 'random_state': 42, 'verbosity': -1
    }

    # --- 1. Tune Global Model ---
    X_train = train_df[BASE_FEATURES]
    y_train = train_df['Label']
    w_train = train_df['Strategy_Ret'].abs() * 100 # Sample Weights

    global_opt, global_params = tune_model(X_train, y_train, w_train, "Global_Opt")

    # Train Baseline Global for comparison
    global_base = LGBMClassifier(**baseline_params)
    global_base.fit(X_train, y_train, sample_weight=w_train)

    # --- 2. Tune Tech Model ---
    print("\n--- Tuning Tech Sector ---")
    train_tech = train_df[train_df['Is_Tech'] == 1]
    X_tech = train_tech[BASE_FEATURES]
    y_tech = train_tech['Label']
    w_tech = train_tech['Strategy_Ret'].abs() * 100

    if len(train_tech) > 0:
        tech_opt, tech_params = tune_model(X_tech, y_tech, w_tech, "Tech_Opt")

        tech_base = LGBMClassifier(**baseline_params)
        tech_base.fit(X_tech, y_tech, sample_weight=w_tech)
    else:
        tech_opt, tech_params = None, {}
        tech_base = None

    # --- 3. Tune Non-Tech Model ---
    print("\n--- Tuning Non-Tech Sector ---")
    train_non = train_df[train_df['Is_Tech'] == 0]
    X_non = train_non[BASE_FEATURES]
    y_non = train_non['Label']
    w_non = train_non['Strategy_Ret'].abs() * 100

    if len(train_non) > 0:
        non_opt, non_params = tune_model(X_non, y_non, w_non, "NonTech_Opt")

        non_base = LGBMClassifier(**baseline_params)
        non_base.fit(X_non, y_non, sample_weight=w_non)
    else:
        non_opt, non_params = None, {}
        non_base = None

    # --- 4. Evaluation ---
    print("\nEvaluating on Test Set...")

    def get_preds(model, df):
        if not model: return np.zeros(len(df))
        return model.predict(df[BASE_FEATURES])

    # A. Global Baseline
    pred_g_base = get_preds(global_base, test_df)

    # B. Global Optimized
    pred_g_opt = get_preds(global_opt, test_df)

    # C. Ensemble Baseline
    pred_ens_base = []
    # D. Ensemble Optimized
    pred_ens_opt = []

    for idx, row in test_df.iterrows():
        feats = row[BASE_FEATURES].values.reshape(1, -1)
        is_tech = row['Is_Tech']

        # Baseline Ensemble
        if is_tech == 1 and tech_base:
            p_base = tech_base.predict(feats)[0]
        elif is_tech == 0 and non_base:
            p_base = non_base.predict(feats)[0]
        else:
            p_base = global_base.predict(feats)[0]
        pred_ens_base.append(p_base)

        # Optimized Ensemble
        if is_tech == 1 and tech_opt:
            p_opt = tech_opt.predict(feats)[0]
        elif is_tech == 0 and non_opt:
            p_opt = non_opt.predict(feats)[0]
        else:
            p_opt = global_opt.predict(feats)[0]
        pred_ens_opt.append(p_opt)

    pred_ens_base = np.array(pred_ens_base)
    pred_ens_opt = np.array(pred_ens_opt)

    # Metrics
    gb_win, gb_avg, gb_cnt = evaluate_metrics(test_df['Label'], pred_g_base, test_df['Strategy_Ret'])
    go_win, go_avg, go_cnt = evaluate_metrics(test_df['Label'], pred_g_opt, test_df['Strategy_Ret'])
    eb_win, eb_avg, eb_cnt = evaluate_metrics(test_df['Label'], pred_ens_base, test_df['Strategy_Ret'])
    eo_win, eo_avg, eo_cnt = evaluate_metrics(test_df['Label'], pred_ens_opt, test_df['Strategy_Ret'])

    # --- Results ---
    print("\n" + "="*60)
    print("RESULTS Comparison (Test Set 2024-2025)")
    print("="*60)
    print(f"{'Model':<20} | {'Win Rate':<10} | {'Avg Ret':<10} | {'Trades':<10}")
    print("-" * 60)
    print(f"{'Global Baseline':<20} | {gb_win:.2%}    | {gb_avg:.4f}    | {gb_cnt}")
    print(f"{'Global Optimized':<20} | {go_win:.2%}    | {go_avg:.4f}    | {go_cnt}")
    print(f"{'Ensemble Baseline':<20} | {eb_win:.2%}    | {eb_avg:.4f}    | {eb_cnt}")
    print(f"{'Ensemble Optimized':<20} | {eo_win:.2%}    | {eo_avg:.4f}    | {eo_cnt}")

    # Save Results
    res_data = [
        {'Model': 'Global_Baseline', 'Win_Rate': gb_win, 'Avg_Ret': gb_avg, 'Trades': gb_cnt},
        {'Model': 'Global_Optimized', 'Win_Rate': go_win, 'Avg_Ret': go_avg, 'Trades': go_cnt},
        {'Model': 'Ensemble_Baseline', 'Win_Rate': eb_win, 'Avg_Ret': eb_avg, 'Trades': eb_cnt},
        {'Model': 'Ensemble_Optimized', 'Win_Rate': eo_win, 'Avg_Ret': eo_avg, 'Trades': eo_cnt},
    ]
    pd.DataFrame(res_data).to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Save Params
    all_params = {
        'Global_Opt': global_params,
        'Tech_Opt': tech_params,
        'NonTech_Opt': non_params
    }
    with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w') as f:
        json.dump(all_params, f, indent=4)

    # Save Models
    joblib.dump(global_opt, os.path.join(OUTPUT_DIR, 'model_global_opt.joblib'))
    if tech_opt: joblib.dump(tech_opt, os.path.join(OUTPUT_DIR, 'model_tech_opt.joblib'))
    if non_opt: joblib.dump(non_opt, os.path.join(OUTPUT_DIR, 'model_non_tech_opt.joblib'))

if __name__ == '__main__':
    main()
