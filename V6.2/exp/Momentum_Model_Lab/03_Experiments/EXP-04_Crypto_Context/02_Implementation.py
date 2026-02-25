import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib

# --- 1. Settings & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path handling:
# BASE_DIR is .../V6.2/exp/Momentum_Model_Lab/03_Experiments/EXP-04_Crypto_Context
# Level 1 up: 03_Experiments, 2: Momentum_Model_Lab, 3: exp
EXP_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../..'))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

# V6.2 Root for resources (one level up from exp)
V6_2_ROOT = os.path.abspath(os.path.join(EXP_DIR, '..'))

print(f"Added {EXP_DIR} to sys.path")
print(f"V6.2 Root: {V6_2_ROOT}")

# Experiment Period
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# Strategy Parameters
GAP_THRESHOLD = 0.005      # 0.5% Gap
PROFIT_THRESHOLD = 0.002   # 0.2% Profit Target

# --- 2. Helper Functions ---

def load_tickers():
    path = os.path.join(V6_2_ROOT, 'resource', '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []

    with open(path, 'r') as f:
        raw = json.load(f)
    # Convert 'NYSE:MP' -> 'MP', 'NYSE:BRK.B' -> 'BRK-B'
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    # Add VIX and BTC-USD
    all_tickers = tickers + ['^VIX', 'BTC-USD']
    print(f"Downloading data for {len(all_tickers)} tickers...")

    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END,
        interval='1d', auto_adjust=True, progress=False, threads=True
    )

    # Handle MultiIndex Columns
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        # Single ticker case (unlikely but safe to handle)
        data['Ticker'] = all_tickers[0]
        data = data.reset_index()

    # Standardize Date
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # Extract Context Assets
    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    btc_df = data[data['Ticker'] == 'BTC-USD'].set_index('Date')[['Close']].rename(columns={'Close': 'BTC'})

    # Filter Stock Data
    stock_df = data[~data['Ticker'].isin(['^VIX', 'BTC-USD'])]

    print(f"  - Stock Data Rows: {len(stock_df)}")
    print(f"  - VIX Data Rows: {len(vix_df)}")
    print(f"  - BTC Data Rows: {len(btc_df)}")

    # Fill Context Data (Forward Fill then Backfill)
    vix_df = vix_df.resample('D').ffill().bfill()
    btc_df = btc_df.resample('D').ffill().bfill()

    # --- Pre-calculate BTC Features ---
    # This ensures we use all calendar days (weekends) for calculation
    btc_df['BTC_Change'] = btc_df['BTC'].pct_change()
    btc_df['BTC_MA20'] = btc_df['BTC'].rolling(20).mean()
    btc_df['BTC_Trend_Score'] = (btc_df['BTC'] > btc_df['BTC_MA20']).astype(int)
    btc_df['BTC_RSI'] = ta.rsi(btc_df['BTC'], length=14)

    # Drop intermediate column if needed, or keep for debugging
    # We only need the features

    return stock_df, vix_df, btc_df

def build_features(df, vix_df, btc_df, ticker_name="Unknown"):
    """Feature Engineering for EXP-04"""
    df = df.sort_index()

    # Ensure numeric
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaNs in OHLCV (Fix for failed tickers)
    df = df.dropna(subset=cols_to_numeric)

    # Join Context Data
    # Join aligns on index (Date). Since stock df is business days,
    # it will pick up the corresponding BTC values from that day.
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(vix_df[['VIX']], how='left')
    df = df.join(btc_df[['BTC_Change', 'BTC_Trend_Score', 'BTC_RSI']], how='left')

    # Fill Context NaNs (e.g. holidays misalignment)
    df['VIX'] = df['VIX'].ffill().bfill()
    cols_to_fill = ['BTC_Change', 'BTC_Trend_Score', 'BTC_RSI']
    for c in cols_to_fill:
        if c in df.columns:
             df[c] = df[c].ffill().bfill()

    # Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 25: return pd.DataFrame()

    try:
        # --- Baseline Features ---
        # Calculate on Close (Day T)
        df['RSI_14_Raw'] = ta.rsi(df['Close'], length=14)
        df['ATR_14_Raw'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # Shift to avoid Lookahead Bias (Use T-1)
        df['RSI_14'] = df['RSI_14_Raw'].shift(1)
        df['ATR_14'] = df['ATR_14_Raw'].shift(1)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

        # Vol_MA20 uses Close but it is shifted later
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        # Vol_Ratio uses shift(1) already
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)

        # Shift VIX (Use T-1)
        df['VIX'] = df['VIX'].shift(1)

        # Shift BTC Features (Use T-1)
        # BTC data joined at Date T is Close[T]. We must shift to get Close[T-1].
        df['BTC_Change'] = df['BTC_Change'].shift(1)
        df['BTC_Trend_Score'] = df['BTC_Trend_Score'].shift(1)
        df['BTC_RSI'] = df['BTC_RSI'].shift(1)

    except Exception as e:
        print(f"Error calculating indicators for {ticker_name}: {e}")
        return pd.DataFrame()

    # Gap & Target
    # Gap uses Open[T] vs Prev_Close[T-1] (Known at Open)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Target uses Close[T] vs Open[T] (Future)
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open']

    # Labeling
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # Debug print for specific ticker
    if ticker_name == 'NVDA':
        print(f"DEBUG: NVDA Gap_Pct stats: {df['Gap_Pct'].describe()}")
        print(f"DEBUG: NVDA Signals count: {(df['Gap_Pct'] > GAP_THRESHOLD).sum()}")

    # Final Clean
    features_to_check = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret', 'Vol_Ratio', 'BTC_Change', 'BTC_Trend_Score', 'BTC_RSI']
    # Check if columns exist before dropping (BTC ones might be missing if join failed completely)
    missing_cols = [c for c in features_to_check if c not in df.columns]
    if missing_cols:
        print(f"Missing columns for {ticker_name}: {missing_cols}")
        return pd.DataFrame()

    df = df.dropna(subset=features_to_check)

    return df

def train_and_evaluate(train_df, test_df, features, model_name="Model"):
    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    # Evaluate
    df_res = pd.DataFrame({'Label': y_test, 'Pred': y_pred, 'Return': r_test})
    model_trades = df_res[df_res['Pred'] == 1]

    if len(model_trades) == 0:
        return 0, 0, 0, model

    win_rate = (model_trades['Return'] > 0).mean()
    avg_ret = model_trades['Return'].mean()
    total_ret = model_trades['Return'].sum()

    return win_rate, avg_ret, total_ret, model

# --- 3. Main Execution ---

def main():
    print("=== EXP-04: Crypto Context Integration (Risk-On Regime) ===")

    tickers = load_tickers()
    if not tickers:
        print("[Error] No tickers found.")
        return

    stock_raw, vix_raw, btc_raw = fetch_data(tickers)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw, btc_raw, ticker_name=ticker)

        if feat_df.empty: continue
        feat_df['Ticker'] = ticker

        # Filter for signals only to save memory/processing
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Gap Signals Found: {len(full_df)}")

    # Split Data
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    if len(train_df) < 50:
        print("[Error] Not enough training data.")
        return

    # --- Baseline Model ---
    print("\nTraining Baseline Model (V6.1 Parity)...")
    base_features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX']
    b_win, b_avg, b_tot, base_model = train_and_evaluate(train_df, test_df, base_features, "Baseline")

    # --- EXP-04 Model ---
    print("\nTraining EXP-04 Model (Crypto Context)...")
    exp_features = base_features + ['BTC_Change', 'BTC_Trend_Score', 'BTC_RSI']
    e_win, e_avg, e_tot, exp_model = train_and_evaluate(train_df, test_df, exp_features, "EXP-04")

    # --- Comparison ---
    print("\n" + "="*80)
    print("EXP-04 RESULTS COMPARISON (OOS 2024-2025)")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline':<20} {'EXP-04':<20} {'Diff':<10}")
    print("-" * 75)
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {e_win*100:6.2f}%              {e_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {e_avg*100:6.3f}%              {e_avg-b_avg:+.3%}")
    print(f"{'Total Return':<20} {b_tot*100:6.2f}%              {e_tot*100:6.2f}%              {e_tot-b_tot:+.2%}")
    print("-" * 75)

    # Feature Importance
    imp = pd.Series(exp_model.feature_importances_, index=exp_features).sort_values(ascending=False)
    print("\n[EXP-04 Feature Importance]")
    print(imp)

    # Save Artifacts
    joblib.dump(exp_model, os.path.join(OUTPUT_DIR, 'exp04_model.joblib'))
    joblib.dump(base_model, os.path.join(OUTPUT_DIR, 'baseline_model.joblib'))

    # Save Performance Report
    report = pd.DataFrame({
        'Metric': ['Win Rate', 'Avg Return', 'Total Return'],
        'Baseline': [b_win, b_avg, b_tot],
        'EXP-04': [e_win, e_avg, e_tot],
        'Diff': [e_win-b_win, e_avg-b_avg, e_tot-b_tot]
    })
    report.to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    imp.plot(kind='barh')
    plt.title('EXP-04 Feature Importance (Crypto Context)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    print("\nSaved artifacts to 03_Output/")

if __name__ == '__main__':
    main()
