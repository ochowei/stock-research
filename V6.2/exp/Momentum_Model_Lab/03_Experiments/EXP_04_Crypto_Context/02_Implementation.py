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

# --- 1. Setup & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# V6.2/exp/Momentum_Model_Lab/03_Experiments/EXP_04_Crypto_Context/ -> V6.2/resource
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../../../resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training/Testing Periods
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# Strategy Parameters
GAP_THRESHOLD = 0.005      # 0.5% Gap
PROFIT_THRESHOLD = 0.002   # 0.2% Profit Target (Momentum)

# --- 2. Helper Functions ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    # Clean tickers: "NASDAQ:AAPL" -> "AAPL"
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    # Add Context Tickers (VIX, BTC, ETH)
    context_tickers = ['^VIX', 'BTC-USD', 'ETH-USD']
    all_tickers = tickers + context_tickers
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # Download in bulk
    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END,
        interval='1d', auto_adjust=True, progress=True, threads=True
    )

    # Handle MultiIndex Columns
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        # Should not happen with multiple tickers, but handle it just in case
        data['Ticker'] = all_tickers[0]
        data = data.reset_index()

    # Normalize Date
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # Separate Context Data
    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    btc_df = data[data['Ticker'] == 'BTC-USD'].set_index('Date')[['Close']].rename(columns={'Close': 'BTC_Close'})
    eth_df = data[data['Ticker'] == 'ETH-USD'].set_index('Date')[['Close']].rename(columns={'Close': 'ETH_Close'})

    # Filter out context tickers from stock data
    stock_df = data[~data['Ticker'].isin(context_tickers)]

    print(f"  - Stock Data Rows: {len(stock_df)}")
    print(f"  - VIX Data Rows: {len(vix_df)}")
    print(f"  - BTC Data Rows: {len(btc_df)}")

    return stock_df, vix_df, btc_df, eth_df

def build_crypto_features(btc_df, eth_df):
    """
    Generate Crypto Context Features.
    IMPORTANT: We use T-1 data for T prediction to avoid Look-Ahead Bias.
    Indices are dates.
    """
    # 1. BTC Features
    btc = btc_df.copy()
    btc['BTC_Ret'] = btc['BTC_Close'].pct_change()
    try:
        btc['BTC_RSI'] = ta.rsi(btc['BTC_Close'], length=14)
        btc['BTC_MA50'] = btc['BTC_Close'].rolling(50).mean()
    except Exception:
        btc['BTC_RSI'] = 50
        btc['BTC_MA50'] = btc['BTC_Close']

    btc['BTC_Trend'] = (btc['BTC_Close'] > btc['BTC_MA50']).astype(int)

    # 2. ETH Features
    eth = eth_df.copy()
    eth['ETH_Ret'] = eth['ETH_Close'].pct_change()

    # Combine
    context = btc[['BTC_Ret', 'BTC_RSI', 'BTC_Trend']].join(eth[['ETH_Ret']], how='outer')

    # Fill NaN
    context = context.ffill().bfill()

    return context

def build_features(df, vix_df, crypto_context):
    """
    Feature Engineering for Equity Momentum.
    """
    df = df.sort_index()

    # Ensure numeric
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Join Context Data (VIX, Crypto)
    df.index = pd.to_datetime(df.index).normalize()

    # VIX (T-1 due to join on Date, assuming VIX for Date T is available?
    # Actually VIX is an index. yfinance gives Daily Close.
    # For trading on T Open, we have T-1 Close of VIX.
    # IMPORTANT: When we join on 'Date', row T gets VIX of T.
    # We MUST shift VIX and Crypto by 1 to represent "Yesterday's Context".

    # Shift Context Data by 1 day to align T-1 context with T row
    vix_shifted = vix_df.shift(1)
    crypto_shifted = crypto_context.shift(1)

    df = df.join(vix_shifted, how='left')
    df = df.join(crypto_shifted, how='left')

    # Fill missing context
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)
    df['BTC_Ret'] = df['BTC_Ret'].fillna(0.0)
    df['BTC_RSI'] = df['BTC_RSI'].fillna(50.0)
    df['BTC_Trend'] = df['BTC_Trend'].fillna(0)
    df['ETH_Ret'] = df['ETH_Ret'].fillna(0.0)

    # Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    # Indicators
    if len(df) < 15: return pd.DataFrame()

    try:
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    except Exception:
        df['RSI_14'] = np.nan
        df['ATR_Pct'] = np.nan

    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)

    # Gap & Target
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open']

    # Labeling
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # Drop NaNs
    df = df.dropna(subset=['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret', 'BTC_RSI'])

    return df

def evaluate_model(X_train, y_train, w_train, X_test, y_test, r_test, model_name="Model"):
    print(f"\nTraining {model_name}...")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    # Metrics
    win_rate = (y_test[y_pred == 1] == 1).mean() if sum(y_pred) > 0 else 0
    avg_return = r_test[y_pred == 1].mean() if sum(y_pred) > 0 else 0
    total_signals = sum(y_pred)

    return model, win_rate, avg_return, total_signals, y_pred

# --- 3. Main Execution ---

def main():
    print(f"=== EXP-04: Crypto Context Integration ===")

    tickers = load_tickers()
    if not tickers: return

    stock_raw, vix_raw, btc_raw, eth_raw = fetch_data(tickers)

    print("\nProcessing Crypto Context...")
    crypto_context = build_crypto_features(btc_raw, eth_raw)

    print("\nBuilding Equity Features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw, crypto_context)

        if feat_df.empty: continue
        feat_df['Ticker'] = ticker

        # Filter Signals
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No signals found.")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Signals: {len(full_df)}")

    # Split Train/Test
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Train Size: {len(train_df)}")
    print(f"Test Size : {len(test_df)}")

    # Feature Sets
    base_feats = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX']
    exp_feats  = base_feats + ['BTC_Ret', 'BTC_RSI', 'BTC_Trend', 'ETH_Ret']

    # Train Baseline
    model_base, base_win, base_avg, base_cnt, base_pred = evaluate_model(
        train_df[base_feats], train_df['Label'], train_df['Sample_Weight'],
        test_df[base_feats], test_df['Label'], test_df['Strategy_Ret'],
        "Baseline Model"
    )

    # Train Experiment
    model_exp, exp_win, exp_avg, exp_cnt, exp_pred = evaluate_model(
        train_df[exp_feats], train_df['Label'], train_df['Sample_Weight'],
        test_df[exp_feats], test_df['Label'], test_df['Strategy_Ret'],
        "Crypto Context Model"
    )

    # Results
    print("\n" + "="*60)
    print("EXP-04 RESULTS COMPARISON (2024-2025)")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<20} {'Crypto Context':<20} {'Diff':<10}")
    print("-" * 75)
    print(f"{'Signals':<20} {base_cnt:<20} {exp_cnt:<20}")
    print(f"{'Win Rate':<20} {base_win*100:6.2f}%              {exp_win*100:6.2f}%              {exp_win-base_win:+.2%}")
    print(f"{'Avg Return':<20} {base_avg*100:6.3f}%              {exp_avg*100:6.3f}%              {exp_avg-base_avg:+.3%}")
    print("-" * 75)

    # Save Report
    with open(os.path.join(OUTPUT_DIR, 'performance_report.txt'), 'w') as f:
        f.write("EXP-04 RESULTS COMPARISON (2024-2025)\n")
        f.write(f"Baseline Win Rate: {base_win:.4f}\n")
        f.write(f"Crypto Win Rate  : {exp_win:.4f}\n")
        f.write(f"Diff Win Rate    : {exp_win-base_win:.4f}\n")
        f.write(f"Baseline Avg Ret : {base_avg:.4f}\n")
        f.write(f"Crypto Avg Ret   : {exp_avg:.4f}\n")

    # Feature Importance
    imp = pd.Series(model_exp.feature_importances_, index=exp_feats).sort_values(ascending=False)
    print("\n[Feature Importance - Crypto Model]")
    print(imp)

    plt.figure(figsize=(10, 6))
    imp.plot(kind='barh')
    plt.title('Feature Importance (Crypto Context Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))

    # Equity Curve
    test_df['Base_Pred'] = base_pred
    test_df['Exp_Pred'] = exp_pred

    daily_base = test_df[test_df['Base_Pred']==1].groupby(test_df[test_df['Base_Pred']==1].index)['Strategy_Ret'].mean()
    daily_exp = test_df[test_df['Exp_Pred']==1].groupby(test_df[test_df['Exp_Pred']==1].index)['Strategy_Ret'].mean()

    # Reindex to full test period to align dates
    full_dates = test_df.index.unique().sort_values()
    daily_base = daily_base.reindex(full_dates, fill_value=0)
    daily_exp = daily_exp.reindex(full_dates, fill_value=0)

    equity_base = (1 + daily_base).cumprod()
    equity_exp = (1 + daily_exp).cumprod()

    plt.figure(figsize=(12, 6))
    plt.plot(equity_base, label='Baseline (No Crypto)', color='gray', linestyle='--')
    plt.plot(equity_exp, label='Experiment (With Crypto)', color='orange', linewidth=2)
    plt.title('Equity Curve Comparison: Baseline vs Crypto Context')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'equity_curve_comparison.png'))
    print("\n[Saved] All artifacts saved to 03_Output/")

if __name__ == '__main__':
    main()
