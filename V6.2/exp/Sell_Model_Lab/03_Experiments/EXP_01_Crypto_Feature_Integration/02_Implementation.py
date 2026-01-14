import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Resource dir is ../../../../resource
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
        # If single ticker, but we asked for many, it should return multi-index or single level with ticker name.
        # But here we are downloading many. If only one fails or something, yfinance might behave differently.
        # Assuming we get data.
        if 'Ticker' not in data.columns:
             # This path is risky if yfinance format changes, but for many tickers it's usually MultiIndex
             pass
        data = data.reset_index()

    # Force Date column to be present and datetime
    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()

    # Force Date column to be present and datetime
    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()

    # Check if 'Date' exists now
    if 'Date' not in data.columns:
        # If still not present, maybe it's in the index but name is not 'Date' or it is MultiIndex
        data = data.reset_index()

    # Ensure Date is datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # Separate Benchmarks and Crypto
    # Note: If Ticker column is missing (e.g. single ticker download often returns simple DF),
    # we need to handle it. But we download multiple tickers, so 'Ticker' should be a column or index level.
    if 'Ticker' not in data.columns:
         # Attempt to find which column might be ticker if it was reset from index
         # For yfinance with multiple tickers, it usually returns MultiIndex columns (Price, Ticker) or Stacked.
         # We did stack().
         pass

    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').copy()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').copy()
    btc_df = data[data['Ticker'] == 'BTC-USD'].set_index('Date').copy()
    eth_df = data[data['Ticker'] == 'ETH-USD'].set_index('Date').copy()

    # Calculate Benchmark Gaps
    for df, name in [(qqq_df, 'QQQ'), (spy_df, 'SPY')]:
        df['Prev_Close'] = df['Close'].shift(1)
        df[f'{name}_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Keep only needed columns for benchmarks
    qqq_gap = qqq_df[[f'QQQ_Gap']]
    spy_gap = spy_df[[f'SPY_Gap']]

    # Process Crypto Data
    # We need Close for features.
    btc_close = btc_df[['Close']].rename(columns={'Close': 'BTC_Close'})
    eth_close = eth_df[['Close']].rename(columns={'Close': 'ETH_Close'})

    stock_df = data[~data['Ticker'].isin(benchmarks)]

    print(f"  - Stock Data Rows: {len(stock_df)}")

    return stock_df, vix_df, qqq_gap, spy_gap, btc_close, eth_close

def calculate_totm_features(dates):
    """Calculate TOTM (Time of The Month) features"""
    dates = sorted(list(set(dates)))
    df = pd.DataFrame({'Date': dates})
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    df['Days_From_Start'] = df.groupby(['Year', 'Month']).cumcount()
    df['Days_To_End'] = df.groupby(['Year', 'Month'])['Date'].transform('count') - df['Days_From_Start'] - 1

    return df.set_index('Date')[['Days_From_Start', 'Days_To_End']]

def build_features(df, vix_df, qqq_gap, spy_gap, btc_close, eth_close, totm_df):
    """Feature Engineering (Enhanced with Crypto)"""
    df = df.sort_index()

    # 1. Numeric Conversion
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Join External Data
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(vix_df, how='left')
    df = df.join(qqq_gap, how='left')
    df = df.join(spy_gap, how='left')
    df = df.join(btc_close, how='left') # Join BTC
    df = df.join(eth_close, how='left') # Join ETH
    df = df.join(totm_df, how='left')

    # Fill VIX and Crypto
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)
    df['BTC_Close'] = df['BTC_Close'].ffill().bfill()
    df['ETH_Close'] = df['ETH_Close'].ffill().bfill()

    # 3. Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame() # Need more history for SMA50

    # 4. Indicators
    try:
        # Stock Features (T-1)
        df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

        # Volume
        df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

        # Crypto Features (T-1)
        # BTC RSI
        df['BTC_RSI'] = ta.rsi(df['BTC_Close'], length=14).shift(1)

        # BTC Trend: Close / SMA(50) - 1
        btc_sma50 = df['BTC_Close'].rolling(50).mean()
        df['BTC_Trend'] = (df['BTC_Close'] / btc_sma50 - 1).shift(1)

        # Crypto Correlation (Stock Close vs BTC Close, 30d)
        # We want the correlation of the *movements*, or just prices?
        # Usually correlation of returns is more stationary.
        # Design said "Crypto_Corr: Rolling correlation (30d) between Stock Close and BTC Close."
        # Using returns is safer for correlation. Let's use Returns.
        # But if the design strictly said "Close", I should follow or improve.
        # Correlation of prices (Close vs Close) is often spurious due to trends.
        # I will calculate Correlation of Returns (pct_change) over 30 days.
        stock_ret = df['Close'].pct_change()
        btc_ret = df['BTC_Close'].pct_change()
        df['Crypto_Corr'] = stock_ret.rolling(30).corr(btc_ret).shift(1)

    except Exception as e:
        # print(f"Error calculating indicators: {e}")
        return pd.DataFrame() # Skip if calculation fails

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # --- Other Features ---
    # A. Dist_MA20
    sum_prev_19 = df['Close'].rolling(19).sum().shift(1)
    ma20_sim = (sum_prev_19 + df['Open']) / 20
    df['Dist_MA20'] = (df['Open'] / ma20_sim) - 1

    # B. Relative Strength
    df['Rel_Gap_QQQ'] = df['Gap_Pct'] - df['QQQ_Gap']
    df['Rel_Gap_SPY'] = df['Gap_Pct'] - df['SPY_Gap']

    # --- Labeling ---
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # Cleaning
    # Added crypto features to requirement
    req_cols = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret',
                'Dist_MA20', 'Rel_Gap_QQQ', 'Rel_Gap_SPY', 'Days_From_Start',
                'BTC_RSI', 'BTC_Trend', 'Crypto_Corr']
    df = df.dropna(subset=req_cols)

    return df

def evaluate_performance(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})

    base_win = (df['Return'] > 0).mean()
    base_avg = df['Return'].mean()
    base_tot = df['Return'].sum()

    model_df = df[df['Pred'] == 1]
    if len(model_df) == 0:
        return 0, 0, 0, base_win, base_avg, base_tot

    mod_win = (model_df['Return'] > 0).mean()
    mod_avg = model_df['Return'].mean()
    mod_tot = model_df['Return'].sum()

    return mod_win, mod_avg, mod_tot, base_win, base_avg, base_tot

# --- 3. Main ---

def main():
    print(f"=== EXP-01: Crypto Feature Integration ===")

    tickers = load_tickers()
    stock_raw, vix_raw, qqq_raw, spy_raw, btc_raw, eth_raw = fetch_data(tickers)

    # TOTM
    all_dates = stock_raw['Date'].unique()
    totm_df = calculate_totm_features(all_dates)

    print("\nBuilding features...")
    all_data = []

    # Iterate over stocks
    # Using groupby is safe but we need to ensure we don't mix tickers in calculations
    # build_features handles single ticker DF.

    # Debug: Check btc_raw availability
    if btc_raw.empty:
        print("Error: BTC data is empty!")
        return

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw, qqq_raw, spy_raw, btc_raw, eth_raw, totm_df)

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

    # Feature List
    features = [
        'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX',
        'Dist_MA20', 'Rel_Gap_QQQ', 'Rel_Gap_SPY',
        'Days_From_Start', 'Days_To_End',
        'BTC_RSI', 'BTC_Trend', 'Crypto_Corr' # New Features
    ]

    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    print(f"\nTraining XGBoost with {len(features)} features...")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    m_win, m_avg, m_tot, b_win, b_avg, b_tot = evaluate_performance(y_test, y_pred, r_test)

    print("\n" + "="*80)
    print("EXP-01 RESULTS (OOS 2024-2025)")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline':<20} {'Model (Crypto)':<20} {'Diff':<10}")
    print("-" * 80)
    print(f"{'Count':<20} {len(y_test):<20} {sum(y_pred):<20}")
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {m_win*100:6.2f}%              {m_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {m_avg*100:6.3f}%              {m_avg-b_avg:+.3%}")
    print(f"{'Total Return':<20} {b_tot*100:6.1f}%              {m_tot*100:6.1f}%")
    print("-" * 80)

    # Save Metrics to CSV
    metrics = {
        'Metric': ['Count', 'Win Rate', 'Avg Return', 'Total Return'],
        'Baseline': [len(y_test), b_win, b_avg, b_tot],
        'Model': [sum(y_pred), m_win, m_avg, m_tot],
        'Diff': [sum(y_pred)-len(y_test), m_win-b_win, m_avg-b_avg, m_tot-b_tot]
    }
    pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Feature Importance
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n[Feature Importance]")
    print(imp)

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    imp.plot(kind='barh')
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))

    # Save Model
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_01_model.joblib'))
    print(f"\nModel saved to {os.path.join(OUTPUT_DIR, 'exp_01_model.joblib')}")

if __name__ == '__main__':
    main()
