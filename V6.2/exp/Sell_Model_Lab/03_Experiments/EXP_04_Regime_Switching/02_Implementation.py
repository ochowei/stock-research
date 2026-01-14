import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import joblib

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
VIX_THRESHOLD = 20.0       # Regime Split Threshold

# Features
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

def fetch_data(tickers):
    # Add Benchmarks and Crypto
    benchmarks = ['QQQ', 'SPY', '^VIX', 'BTC-USD', 'ETH-USD']
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

    # Handle Missing Close
    if 'Close' in data.columns and data['Close'].isnull().all():
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
            print("WARNING: 'Close' is all NaN, using 'Adj Close' instead.")
            data['Close'] = data['Adj Close']

    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    # We don't strictly need other benchmarks for features this time, but keeping the join logic consistent is safer

    stock_df = data[~data['Ticker'].isin(benchmarks)]
    return stock_df, vix_df

def build_features(df, vix_df):
    """Feature Engineering - Base Set Only + VIX for Regime"""
    df = df.sort_index()

    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.index = pd.to_datetime(df.index).normalize()

    if df['Close'].isnull().all():
        return pd.DataFrame()

    # Join VIX
    df = df.join(vix_df, how='left')
    # Shift VIX because we need T-1 VIX to decide regime for T open
    df['VIX'] = df['VIX'].shift(1).ffill().bfill().fillna(20.0)

    # Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

    try:
        # Stock Features (T-1)
        df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

        # Volume
        df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

        # Gap
        df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        # Dist MA20
        close_filled = df['Close'].ffill()
        sum_prev_19 = close_filled.rolling(19).sum().shift(1)
        open_p = df['Open'].fillna(df['Close'])
        ma20_sim = (sum_prev_19 + open_p) / 20
        df['Dist_MA20'] = (open_p / ma20_sim) - 1

    except Exception as e:
        return pd.DataFrame()

    # Labeling
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # Clean
    features_needed = BASE_FEATURES + ['VIX']
    df_clean = df.dropna(subset=features_needed)

    return df_clean

def train_model(X, y, w):
    model = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        n_jobs=-1, random_state=42, verbosity=-1
    )
    model.fit(X, y, sample_weight=w)
    return model

def evaluate_performance(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})

    # Baseline (Taking all valid gap signals)
    base_win = (df['Return'] > 0).mean()
    base_avg = df['Return'].mean()

    # Model (Taking signals predicted as 1)
    model_df = df[df['Pred'] == 1]

    if len(model_df) == 0:
        return 0, 0, 0, base_win, base_avg, df['Return'].sum()

    mod_win = (model_df['Return'] > 0).mean()
    mod_avg = model_df['Return'].mean()
    mod_tot = model_df['Return'].sum()

    return mod_win, mod_avg, mod_tot, base_win, base_avg, df['Return'].sum()

# --- 3. Main ---

def main():
    print(f"=== EXP-04: Regime-Switching Model ===")

    tickers = load_tickers()
    stock_raw, vix_raw = fetch_data(tickers)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        df = df.dropna(subset=['Close'])
        if df.empty: continue

        feat_df = build_features(df, vix_raw)

        if feat_df.empty: continue

        feat_df['Ticker'] = ticker
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Gap Signals: {len(full_df)}")

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    # --- Train Global Model (Control) ---
    print("\nTraining Global Model (Control)...")
    global_model = train_model(train_df[BASE_FEATURES], train_df['Label'], train_df['Sample_Weight'])

    # --- Train Regime Models ---
    print("\nTraining Regime Models...")

    train_low_vix = train_df[train_df['VIX'] <= VIX_THRESHOLD]
    train_high_vix = train_df[train_df['VIX'] > VIX_THRESHOLD]

    print(f"  Low VIX Samples : {len(train_low_vix)}")
    print(f"  High VIX Samples: {len(train_high_vix)}")

    model_low = train_model(train_low_vix[BASE_FEATURES], train_low_vix['Label'], train_low_vix['Sample_Weight'])
    model_high = train_model(train_high_vix[BASE_FEATURES], train_high_vix['Label'], train_high_vix['Sample_Weight'])

    # --- Evaluation ---
    print("\nEvaluating on OOS Test Data...")

    # 1. Global Model Preds
    y_pred_global = global_model.predict(test_df[BASE_FEATURES])

    # 2. Regime System Preds
    # We can vectorize this: create empty preds, fill with mask
    y_pred_regime = np.zeros(len(test_df), dtype=int)

    mask_low = (test_df['VIX'] <= VIX_THRESHOLD)
    mask_high = (test_df['VIX'] > VIX_THRESHOLD)

    if mask_low.any():
        y_pred_regime[mask_low] = model_low.predict(test_df.loc[mask_low, BASE_FEATURES])
    if mask_high.any():
        y_pred_regime[mask_high] = model_high.predict(test_df.loc[mask_high, BASE_FEATURES])

    # Metrics
    g_win, g_avg, g_tot, b_win, b_avg, b_tot = evaluate_performance(test_df['Label'], y_pred_global, test_df['Strategy_Ret'])
    r_win, r_avg, r_tot, _, _, _ = evaluate_performance(test_df['Label'], y_pred_regime, test_df['Strategy_Ret'])

    print(f"\nResults (2024-2025):")
    print(f"{'Metric':<20} | {'Global Model':<15} | {'Regime System':<15} | {'Diff':<10}")
    print("-" * 70)
    print(f"{'Win Rate':<20} | {g_win:.2%}        | {r_win:.2%}        | {r_win-g_win:+.2%}")
    print(f"{'Avg Return':<20} | {g_avg:.4f}         | {r_avg:.4f}         | {r_avg-g_avg:+.4f}")
    print(f"{'Total Return':<20} | {g_tot:.4f}         | {r_tot:.4f}         | {r_tot-g_tot:+.4f}")
    print(f"{'Signals':<20} | {sum(y_pred_global):<15} | {sum(y_pred_regime):<15} | {sum(y_pred_regime)-sum(y_pred_global):+}")

    # Save outputs
    results = {
        'Metric': ['Win Rate', 'Avg Return', 'Total Return', 'Signals'],
        'Global': [g_win, g_avg, g_tot, int(sum(y_pred_global))],
        'Regime': [r_win, r_avg, r_tot, int(sum(y_pred_regime))]
    }
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'performance_comparison.csv'), index=False)

    joblib.dump(global_model, os.path.join(OUTPUT_DIR, 'model_global.joblib'))
    joblib.dump(model_low, os.path.join(OUTPUT_DIR, 'model_low_vix.joblib'))
    joblib.dump(model_high, os.path.join(OUTPUT_DIR, 'model_high_vix.joblib'))

    # Save a detailed breakdown csv
    detailed = test_df.copy()
    detailed['Pred_Global'] = y_pred_global
    detailed['Pred_Regime'] = y_pred_regime
    detailed.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'))

if __name__ == '__main__':
    main()
