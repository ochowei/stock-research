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
        if 'Ticker' not in data.columns:
             pass
        data = data.reset_index()

    # Force Date column to be present and datetime
    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()

    # Check if 'Date' exists now
    if 'Date' not in data.columns:
        data = data.reset_index()

    # Ensure Date is datetime
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').copy()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').copy()
    btc_df = data[data['Ticker'] == 'BTC-USD'].set_index('Date').copy()
    eth_df = data[data['Ticker'] == 'ETH-USD'].set_index('Date').copy()

    print(f"DEBUG: VIX shape: {vix_df.shape}")
    print(f"DEBUG: BTC shape: {btc_df.shape}")
    print(f"DEBUG: ETH shape: {eth_df.shape}")
    print(f"DEBUG: QQQ shape: {qqq_df.shape}")

    # Calculate Benchmark Gaps
    for df_bench, name in [(qqq_df, 'QQQ'), (spy_df, 'SPY')]:
        if df_bench.empty:
            print(f"[WARNING] {name} data is empty!")
            df_bench[f'{name}_Gap'] = np.nan
        else:
            df_bench['Prev_Close'] = df_bench['Close'].shift(1)
            df_bench[f'{name}_Gap'] = (df_bench['Open'] - df_bench['Prev_Close']) / df_bench['Prev_Close']

    # Keep only needed columns for benchmarks
    if not qqq_df.empty:
        qqq_gap = qqq_df[[f'QQQ_Gap']]
    else:
        qqq_gap = pd.DataFrame(columns=['QQQ_Gap'])

    if not spy_df.empty:
        spy_gap = spy_df[[f'SPY_Gap']]
    else:
        spy_gap = pd.DataFrame(columns=['SPY_Gap'])

    # Process Crypto Data
    # We need Close for features.
    if btc_df.empty:
        print("[WARNING] BTC data is empty!")
        btc_close = pd.DataFrame(columns=['BTC_Close'])
    else:
        btc_close = btc_df[['Close']].rename(columns={'Close': 'BTC_Close'})

    if eth_df.empty:
         print("[WARNING] ETH data is empty!")
         eth_close = pd.DataFrame(columns=['ETH_Close'])
    else:
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

    # DEBUG: Check index overlap
    # if not vix_df.empty:
    #     vix_overlap = df.index.intersection(vix_df.index).size
    #     if vix_overlap == 0:
    #          print(f"DEBUG: No overlap with VIX for {df.index[0]} - {df.index[-1]}")

    df = df.join(vix_df, how='left')
    df = df.join(qqq_gap, how='left')
    df = df.join(spy_gap, how='left')
    df = df.join(btc_close, how='left') # Join BTC
    df = df.join(eth_close, how='left') # Join ETH
    df = df.join(totm_df, how='left')

    # Fill VIX and Crypto
    # FIX: Shift VIX to avoid lookahead bias. We need yesterday's VIX Close.
    df['VIX'] = df['VIX'].shift(1).ffill().bfill().fillna(20.0)

    if 'BTC_Close' in df.columns:
        # FIX: Shift Crypto closes too if we use them for correlation or trends
        # In build_features, we calculate BTC_RSI and BTC_Trend using T-1.
        # However, 'Crypto_Corr' uses rolling correlation of returns.
        # If we calculate returns from Unshifted BTC_Close, we get today's return.
        # We need T-1 returns for correlation?
        # Actually, let's look at how indicators are calculated.
        # BTC_RSI uses 'BTC_Close'. If 'BTC_Close' is today's close, ta.rsi().shift(1) is correct.
        # So we keep BTC_Close as is (aligned by date), but SHIFT the indicators derived from it.
        # But wait, Crypto markets are 24/7. Stock Open is 9:30.
        # If 'BTC_Close' is from yfinance (daily), it's usually UTC midnight or Close of stock day?
        # yfinance BTC-USD is usually UTC midnight to midnight.
        # If we trade at 9:30 ET, previous day's BTC close is safe.
        # Current day's BTC close (which happens at 7PM ET or midnight UTC next day?) is future.
        # So we must treat BTC_Close on row T as potentially future or concurrent.
        # Safest is to shift it or rely on shift(1) in feature engineering.

        # In build_features:
        # df['BTC_RSI'] = ta.rsi(df['BTC_Close'], length=14).shift(1) -> SAFE (uses T-1)
        # btc_sma50 ... shift(1) -> SAFE
        # Crypto_Corr = stock_ret.rolling(30).corr(btc_ret).shift(1) -> SAFE (uses T-1 correlations)

        # So BTC_Close itself can stay as is, AS LONG AS we verify usages are shifted.

        # However, VIX is used raw in feature list: 'VIX'
        # features = [ ... 'VIX', ... ]
        # So 'VIX' column MUST be shifted.

        df['BTC_Close'] = df['BTC_Close'].ffill().bfill()
    else:
        # Fallback if BTC missing
        df['BTC_Close'] = 0

    if 'ETH_Close' in df.columns:
        df['ETH_Close'] = df['ETH_Close'].ffill().bfill()
    else:
         df['ETH_Close'] = 0

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
        # Handle cases where BTC_Close might be 0 (if missing)
        btc_s = df['BTC_Close'].replace(0, np.nan)
        # Fix pct_change warnings by filling NA first or ignoring if acceptable
        stock_ret = df['Close'].ffill().pct_change()
        btc_ret = btc_s.ffill().pct_change()
        df['Crypto_Corr'] = stock_ret.rolling(30).corr(btc_ret).shift(1)

    except Exception as e:
        # print(f"Error calculating indicators: {e}")
        return pd.DataFrame() # Skip if calculation fails

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # --- Other Features ---
    # A. Dist_MA20
    # Use ffill to avoid NaNs if there are gaps in trading
    # We need to be careful not to introduce lookahead bias.
    # rolling(19).sum().shift(1) is correct.
    # But if there are NaNs in Close history, sum might be NaN.
    # Let's fill Close before calculating rolling
    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)

    # If open is missing, use Close (should be rare)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1

    # B. Relative Strength
    df['Rel_Gap_QQQ'] = df['Gap_Pct'] - df['QQQ_Gap']
    df['Rel_Gap_SPY'] = df['Gap_Pct'] - df['SPY_Gap']

    # --- Labeling ---
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # DEBUG: Check signal count before drop
    num_signals = df['Is_Signal'].sum()
    if num_signals > 0:
        if np.random.rand() < 0.05:
            print(f"DEBUG: Found {num_signals} signals before dropna for a ticker")

    # Cleaning
    # Added crypto features to requirement
    req_cols = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret',
                'Dist_MA20', 'Rel_Gap_QQQ', 'Rel_Gap_SPY', 'Days_From_Start',
                'BTC_RSI', 'BTC_Trend', 'Crypto_Corr']

    # Check for NaNs before drop
    if df.isnull().values.any():
        missing_counts = df[req_cols].isnull().sum()
        if missing_counts.sum() > 0:
            # Only print first few to avoid spam
             if np.random.rand() < 0.01: # Sample 1% of errors
                 print(f"DEBUG: Missing cols for a ticker: {missing_counts[missing_counts > 0].to_dict()}")

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
    print(f"=== EXP-02: LightGBM Migration ===")

    tickers = load_tickers()
    stock_raw, vix_raw, qqq_raw, spy_raw, btc_raw, eth_raw = fetch_data(tickers)

    # TOTM
    all_dates = stock_raw['Date'].unique()
    totm_df = calculate_totm_features(all_dates)

    print("\nBuilding features...")
    all_data = []

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

    print(f"\nTraining LightGBM with {len(features)} features...")
    # Using LGBMClassifier
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    m_win, m_avg, m_tot, b_win, b_avg, b_tot = evaluate_performance(y_test, y_pred, r_test)

    print("\n" + "="*80)
    print("EXP-02 RESULTS (OOS 2024-2025)")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline':<20} {'Model (LGBM)':<20} {'Diff':<10}")
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
    plt.title('Feature Importance (LightGBM)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))

    # Save Model
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_02_model.joblib'))
    print(f"\nModel saved to {os.path.join(OUTPUT_DIR, 'exp_02_model.joblib')}")

if __name__ == '__main__':
    main()
