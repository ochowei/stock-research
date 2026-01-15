import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
import joblib
import optuna
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
GAP_THRESHOLD = 0.005
PROFIT_THRESHOLD = 0.002

# Feature Sets
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

# Optuna Settings
N_TRIALS = 50
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

def optimize_params(X, y, sample_weights, study_name="opt_study"):
    print(f"\nRunning Optimization for {study_name}...")

    # Sort by index (date) for TimeSeriesSplit
    # X and y should already be sorted by date index

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'verbosity': -1
        }

        # Penalize deep trees if max_depth > 10 (soft constraint guidance)
        # but we let the metric decide.

        tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr = sample_weights.iloc[train_idx]

            model = LGBMClassifier(**params)
            model.fit(X_tr, y_tr, sample_weight=w_tr)

            # Predict
            preds = model.predict(X_val)

            # Calculate Precision (Win Rate)
            # We want to maximize Win Rate, but also care about frequency.
            # Optuna only optimizes one float. Let's use Precision (Win Rate).
            score = precision_score(y_val, preds, zero_division=0)
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"Best Trial ({study_name}): {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    return study.best_params

def evaluate_metrics(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})
    if len(df) == 0: return 0, 0, 0, 0

    model_df = df[df['Pred'] == 1]
    if len(model_df) == 0:
        return 0, 0, 0, 0

    win_rate = (model_df['Return'] > 0).mean()
    avg_ret = model_df['Return'].mean()
    count = len(model_df)

    # Simple Sharpe (ignoring risk free rate, daily granularity assumption for std dev not perfect but sufficient)
    std_ret = model_df['Return'].std()
    sharpe = avg_ret / std_ret if std_ret > 0 else 0

    return win_rate, avg_ret, count, sharpe

# --- 3. Main ---

def main():
    print("=== EXP-22: Context-Aware Hyperparameter Optimization ===")

    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)

    qqq_feats = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feats = prepare_benchmark_features(spy_raw, 'SPY')

    # --- 1. Data Preparation ---
    print("\nPreparing Data...")
    tech_data = []
    non_tech_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        sector = sector_map.get(ticker, 'Unknown')

        if sector == 'Technology':
            df = group.set_index('Date').copy()
            fdf = build_features(df, qqq_feats, 'QQQ', TECH_FEATURES)
            if not fdf.empty and not fdf[fdf['Is_Signal']].empty:
                fdf['Ticker'] = ticker
                tech_data.append(fdf[fdf['Is_Signal']])
        else:
            df = group.set_index('Date').copy()
            fdf = build_features(df, spy_feats, 'SPY', NON_TECH_FEATURES)
            if not fdf.empty and not fdf[fdf['Is_Signal']].empty:
                fdf['Ticker'] = ticker
                non_tech_data.append(fdf[fdf['Is_Signal']])

    full_tech = pd.concat(tech_data).sort_index() if tech_data else pd.DataFrame()
    full_nt = pd.concat(non_tech_data).sort_index() if non_tech_data else pd.DataFrame()

    print(f"Tech Signals: {len(full_tech)}")
    print(f"Non-Tech Signals: {len(full_nt)}")

    # Split Train/Test
    tech_train = full_tech[full_tech.index <= TRAIN_END]
    tech_test = full_tech[(full_tech.index >= TEST_START) & (full_tech.index <= TEST_END)]

    nt_train = full_nt[full_nt.index <= TRAIN_END]
    nt_test = full_nt[(full_nt.index >= TEST_START) & (full_nt.index <= TEST_END)]

    # --- 2. Optimization (Tech) ---
    print("\n--- Tuning Tech Model (Base + QQQ) ---")
    if not tech_train.empty:
        X_tech = tech_train[BASE_FEATURES + TECH_FEATURES]
        y_tech = tech_train['Label']
        w_tech = tech_train['Strategy_Ret'].abs() * 100

        tech_best_params = optimize_params(X_tech, y_tech, w_tech, "Tech_Opt")

        # Train Final Model
        tech_model = LGBMClassifier(**tech_best_params, random_state=RANDOM_STATE, n_jobs=-1)
        tech_model.fit(X_tech, y_tech, sample_weight=w_tech)
    else:
        print("No Tech Training Data!")
        tech_model = None
        tech_best_params = {}

    # --- 3. Optimization (Non-Tech) ---
    print("\n--- Tuning Non-Tech Model (Base + SPY) ---")
    if not nt_train.empty:
        X_nt = nt_train[BASE_FEATURES + NON_TECH_FEATURES]
        y_nt = nt_train['Label']
        w_nt = nt_train['Strategy_Ret'].abs() * 100

        nt_best_params = optimize_params(X_nt, y_nt, w_nt, "NonTech_Opt")

        # Train Final Model
        nt_model = LGBMClassifier(**nt_best_params, random_state=RANDOM_STATE, n_jobs=-1)
        nt_model.fit(X_nt, y_nt, sample_weight=w_nt)
    else:
        print("No Non-Tech Training Data!")
        nt_model = None
        nt_best_params = {}

    # --- 4. Baseline Models (for comparison) ---
    # Tech: Depth 3, LR 0.01 (EXP-06 params)
    BASE_TECH_PARAMS = {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 15, 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}
    # Non-Tech: Unlimited Depth, LR 0.02 (EXP-06 params)
    BASE_NT_PARAMS = {'n_estimators': 500, 'learning_rate': 0.02, 'max_depth': -1, 'num_leaves': 31, 'random_state': 42, 'n_jobs': -1, 'verbosity': -1}

    print("\nTraining Baseline Models...")
    if not tech_train.empty:
        tech_base = LGBMClassifier(**BASE_TECH_PARAMS)
        tech_base.fit(tech_train[BASE_FEATURES + TECH_FEATURES], tech_train['Label'], sample_weight=tech_train['Strategy_Ret'].abs()*100)
    else: tech_base = None

    if not nt_train.empty:
        nt_base = LGBMClassifier(**BASE_NT_PARAMS)
        nt_base.fit(nt_train[BASE_FEATURES + NON_TECH_FEATURES], nt_train['Label'], sample_weight=nt_train['Strategy_Ret'].abs()*100)
    else: nt_base = None

    # --- 5. Evaluation on Test Set ---
    print("\nEvaluating on Test Set (2024-2025)...")

    results = []

    # Tech Evaluation
    if not tech_test.empty and tech_model and tech_base:
        X_test = tech_test[BASE_FEATURES + TECH_FEATURES]
        y_test = tech_test['Label']
        ret_test = tech_test['Strategy_Ret']

        # Baseline
        base_preds = tech_base.predict(X_test)
        bw, ba, bc, bs = evaluate_metrics(y_test, base_preds, ret_test)
        results.append({'Model': 'Tech_Baseline', 'Win_Rate': bw, 'Avg_Ret': ba, 'Trades': bc, 'Sharpe': bs})

        # Optimized
        opt_preds = tech_model.predict(X_test)
        ow, oa, oc, os = evaluate_metrics(y_test, opt_preds, ret_test)
        results.append({'Model': 'Tech_Optimized', 'Win_Rate': ow, 'Avg_Ret': oa, 'Trades': oc, 'Sharpe': os})

    # Non-Tech Evaluation
    if not nt_test.empty and nt_model and nt_base:
        X_test = nt_test[BASE_FEATURES + NON_TECH_FEATURES]
        y_test = nt_test['Label']
        ret_test = nt_test['Strategy_Ret']

        # Baseline
        base_preds = nt_base.predict(X_test)
        bw, ba, bc, bs = evaluate_metrics(y_test, base_preds, ret_test)
        results.append({'Model': 'NonTech_Baseline', 'Win_Rate': bw, 'Avg_Ret': ba, 'Trades': bc, 'Sharpe': bs})

        # Optimized
        opt_preds = nt_model.predict(X_test)
        ow, oa, oc, os = evaluate_metrics(y_test, opt_preds, ret_test)
        results.append({'Model': 'NonTech_Optimized', 'Win_Rate': ow, 'Avg_Ret': oa, 'Trades': oc, 'Sharpe': os})

    # Combined Ensemble Evaluation
    # Construct combined predictions for strict A/B test
    # Baseline Ensemble vs Optimized Ensemble

    ens_base_preds = []
    ens_base_rets = []
    ens_base_labels = []

    ens_opt_preds = []
    ens_opt_rets = []
    ens_opt_labels = []

    # Iterate Tech Test
    if not tech_test.empty and tech_base and tech_model:
        X = tech_test[BASE_FEATURES + TECH_FEATURES]

        p_base = tech_base.predict(X)
        p_opt = tech_model.predict(X)

        # Only where model predicts 1
        # To calculate ensemble stats, we need to collect all trades predicted by the system

        # Actually easier to append result rows
        # For Baseline
        for i, pred in enumerate(p_base):
            if pred == 1:
                ens_base_preds.append(1)
                ens_base_rets.append(tech_test.iloc[i]['Strategy_Ret'])
                ens_base_labels.append(tech_test.iloc[i]['Label'])

        # For Optimized
        for i, pred in enumerate(p_opt):
            if pred == 1:
                ens_opt_preds.append(1)
                ens_opt_rets.append(tech_test.iloc[i]['Strategy_Ret'])
                ens_opt_labels.append(tech_test.iloc[i]['Label'])

    # Iterate Non-Tech Test
    if not nt_test.empty and nt_base and nt_model:
        X = nt_test[BASE_FEATURES + NON_TECH_FEATURES]

        p_base = nt_base.predict(X)
        p_opt = nt_model.predict(X)

        for i, pred in enumerate(p_base):
            if pred == 1:
                ens_base_preds.append(1)
                ens_base_rets.append(nt_test.iloc[i]['Strategy_Ret'])
                ens_base_labels.append(nt_test.iloc[i]['Label'])

        for i, pred in enumerate(p_opt):
            if pred == 1:
                ens_opt_preds.append(1)
                ens_opt_rets.append(nt_test.iloc[i]['Strategy_Ret'])
                ens_opt_labels.append(nt_test.iloc[i]['Label'])

    # Calculate Ensemble Metrics
    if ens_base_rets:
        eb_win = np.mean([r > 0 for r in ens_base_rets])
        eb_avg = np.mean(ens_base_rets)
        eb_cnt = len(ens_base_rets)
        eb_std = np.std(ens_base_rets)
        eb_shp = eb_avg / eb_std if eb_std > 0 else 0
        results.append({'Model': 'Ensemble_Baseline', 'Win_Rate': eb_win, 'Avg_Ret': eb_avg, 'Trades': eb_cnt, 'Sharpe': eb_shp})

    if ens_opt_rets:
        eo_win = np.mean([r > 0 for r in ens_opt_rets])
        eo_avg = np.mean(ens_opt_rets)
        eo_cnt = len(ens_opt_rets)
        eo_std = np.std(ens_opt_rets)
        eo_shp = eo_avg / eo_std if eo_std > 0 else 0
        results.append({'Model': 'Ensemble_Optimized', 'Win_Rate': eo_win, 'Avg_Ret': eo_avg, 'Trades': eo_cnt, 'Sharpe': eo_shp})

    # Display Results
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("RESULTS Comparison (Test Set 2024-2025)")
    print("="*80)
    print(res_df.to_string(index=False))

    # Save Artifacts
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Save Params: convert float64 to float for json serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(os.path.join(OUTPUT_DIR, 'best_params.json'), 'w') as f:
        json.dump({'Tech': tech_best_params, 'NonTech': nt_best_params}, f, indent=4, default=convert_numpy)

    if tech_model: joblib.dump(tech_model, os.path.join(OUTPUT_DIR, 'model_tech_opt.joblib'))
    if nt_model: joblib.dump(nt_model, os.path.join(OUTPUT_DIR, 'model_non_tech_opt.joblib'))

if __name__ == '__main__':
    main()
