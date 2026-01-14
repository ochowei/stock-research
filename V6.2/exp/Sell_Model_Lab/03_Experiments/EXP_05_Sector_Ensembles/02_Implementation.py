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

    # Batch fetching not really supported for 'info' in yfinance in a simple way,
    # so we iterate.
    for i, t in enumerate(tickers):
        try:
            # We can sometimes guess sector or just fetch
            # To be robust, we fetch.
            # Adding a small delay to avoid rate limits
            if i > 0 and i % 10 == 0:
                print(f"Fetched {i}/{len(tickers)}...")
                time.sleep(1)

            ticker_obj = yf.Ticker(t)
            # Accessing info triggers the request
            sec = ticker_obj.info.get('sector', 'Unknown')
            sector_map[t] = sec
        except Exception as e:
            print(f"Warning: Could not fetch sector for {t}: {e}")
            sector_map[t] = 'Unknown'

    # Save cache
    with open(sector_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)

    return sector_map

def fetch_data(tickers):
    # Add Benchmarks
    benchmarks = ['QQQ', 'SPY', '^VIX']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END,
        interval='1d', auto_adjust=True, progress=True, threads=True
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
    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').copy()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').copy()

    # Calculate Benchmark Gaps (needed for potential features, though Base doesn't use them,
    # we might want to ensure feature consistency if we ever add them back,
    # but for EXP-05 we stick to Base features which don't include benchmark gaps strictly,
    # EXCEPT 'Gap_Pct' is needed).

    # Actually Base features are: 'Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20'.
    # None of these strictly require Benchmark data, but let's keep the pipeline standard.

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
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

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

# --- 3. Main ---

def main():
    print(f"=== EXP-05: Sector-Specific Ensembles ===")

    tickers = load_tickers()

    # Fetch Sectors
    sector_map = fetch_sectors(tickers)

    # Identify Tech vs Non-Tech
    tech_tickers = [t for t, s in sector_map.items() if s == 'Technology']
    non_tech_tickers = [t for t, s in sector_map.items() if s != 'Technology']

    print(f"Tech Tickers ({len(tech_tickers)}): {tech_tickers}")
    print(f"Non-Tech Tickers ({len(non_tech_tickers)}): {len(non_tech_tickers)}")

    # Fetch Data
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

    # Split Train/Test
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    # Define Model Params (Standard LGBM)
    lgbm_params = {
        'n_estimators': 200, 'learning_rate': 0.05, 'num_leaves': 31,
        'n_jobs': -1, 'random_state': 42, 'verbosity': -1
    }

    # --- 1. Train Global Model ---
    print("\nTraining Global Model...")
    global_model = LGBMClassifier(**lgbm_params)
    global_model.fit(train_df[BASE_FEATURES], train_df['Label'], sample_weight=train_df['Sample_Weight'])

    # --- 2. Train Tech Model ---
    print("Training Tech Model...")
    train_tech = train_df[train_df['Is_Tech'] == 1]
    if len(train_tech) > 0:
        tech_model = LGBMClassifier(**lgbm_params)
        tech_model.fit(train_tech[BASE_FEATURES], train_tech['Label'], sample_weight=train_tech['Sample_Weight'])
    else:
        print("Warning: No Tech training data!")
        tech_model = None

    # --- 3. Train Non-Tech Model ---
    print("Training Non-Tech Model...")
    train_non_tech = train_df[train_df['Is_Tech'] == 0]
    if len(train_non_tech) > 0:
        non_tech_model = LGBMClassifier(**lgbm_params)
        non_tech_model.fit(train_non_tech[BASE_FEATURES], train_non_tech['Label'], sample_weight=train_non_tech['Sample_Weight'])
    else:
        print("Warning: No Non-Tech training data!")
        non_tech_model = None

    # --- 4. Evaluate on Test Set ---
    print("\nEvaluating on Test Set...")

    # A. Global Baseline
    pred_global = global_model.predict(test_df[BASE_FEATURES])
    g_win, g_avg, g_count = evaluate_metrics(test_df['Label'], pred_global, test_df['Strategy_Ret'])

    # B. Ensemble
    # If Tech Model exists, use it for Tech stocks. Else use Global.
    # If Non-Tech Model exists, use it for Non-Tech stocks. Else use Global.

    pred_ensemble = []

    for idx, row in test_df.iterrows():
        is_tech = row['Is_Tech']
        feats = row[BASE_FEATURES].values.reshape(1, -1)

        if is_tech == 1 and tech_model:
            p = tech_model.predict(feats)[0]
        elif is_tech == 0 and non_tech_model:
            p = non_tech_model.predict(feats)[0]
        else:
            p = global_model.predict(feats)[0] # Fallback

        pred_ensemble.append(p)

    pred_ensemble = np.array(pred_ensemble)
    e_win, e_avg, e_count = evaluate_metrics(test_df['Label'], pred_ensemble, test_df['Strategy_Ret'])

    # --- Results ---
    print("\n" + "="*40)
    print("RESULTS Comparison (Test Set)")
    print("="*40)
    print(f"{'Metric':<15} | {'Global':<10} | {'Ensemble':<10} | {'Diff':<10}")
    print("-" * 55)
    print(f"{'Win Rate':<15} | {g_win:.2%}    | {e_win:.2%}    | {e_win-g_win:+.2%}")
    print(f"{'Avg Return':<15} | {g_avg:.4f}    | {e_avg:.4f}    | {e_avg-g_avg:+.4f}")
    print(f"{'Trades':<15} | {g_count:<10} | {e_count:<10} | {e_count-g_count:+d}")

    # Save Results
    results = {
        'Global_Win_Rate': g_win, 'Global_Avg_Ret': g_avg, 'Global_Count': g_count,
        'Ensemble_Win_Rate': e_win, 'Ensemble_Avg_Ret': e_avg, 'Ensemble_Count': e_count
    }
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Save Models
    joblib.dump(global_model, os.path.join(OUTPUT_DIR, 'model_global.joblib'))
    if tech_model: joblib.dump(tech_model, os.path.join(OUTPUT_DIR, 'model_tech.joblib'))
    if non_tech_model: joblib.dump(non_tech_model, os.path.join(OUTPUT_DIR, 'model_non_tech.joblib'))

    # Detailed Analysis by Sector (Optional but helpful)
    print("\n--- Sector Breakdown (Ensemble) ---")
    test_df['Ensemble_Pred'] = pred_ensemble

    for sector_type in ['Tech', 'Non-Tech']:
        flag = 1 if sector_type == 'Tech' else 0
        sub = test_df[test_df['Is_Tech'] == flag]
        if len(sub) == 0: continue

        sub_preds = sub['Ensemble_Pred']
        s_win, s_avg, s_count = evaluate_metrics(sub['Label'], sub_preds, sub['Strategy_Ret'])
        print(f"{sector_type}: Win={s_win:.2%}, Avg={s_avg:.4f}, Count={s_count}")

if __name__ == '__main__':
    main()
