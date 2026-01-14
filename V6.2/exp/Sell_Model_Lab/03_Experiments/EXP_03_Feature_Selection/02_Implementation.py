import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
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
        if 'Ticker' not in data.columns:
             pass
        data = data.reset_index()

    # Force Date column to be present and datetime
    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()
    if 'Date' not in data.columns:
        data = data.reset_index()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # DEBUG: Check what columns we actually have for FI
    # fi_df = data[data['Ticker'] == 'FI']
    # if not fi_df.empty:
    #     print(f"DEBUG: FI Data Sample:\n{fi_df.head()}")

    # Handle MultiIndex Columns if they persist (sometimes reset_index fails to flatten completely if levels are named oddly)
    # The debug output showed columns ['Date', 'Ticker', 'Adj Close', 'Close', ...] which looks flat.
    # But values were NaN. This implies stack() created the rows but values were missing.
    # This happens if yfinance returned empty DF for that ticker but it was in the columns MultiIndex?
    # Or if 'Adj Close' is the only column returned but we look for 'Close'.

    # FIX: If 'Close' is all NaN but 'Adj Close' is not, use Adj Close.
    # Check if Close is all NaN
    if 'Close' in data.columns and data['Close'].isnull().all():
        if 'Adj Close' in data.columns and not data['Adj Close'].isnull().all():
            print("WARNING: 'Close' is all NaN, using 'Adj Close' instead.")
            data['Close'] = data['Adj Close']

    # Also, some tickers might have ALL NaNs. We should filter them out from stock_df to avoid processing them.
    # But first, benchmarks.

    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').copy()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').copy()
    btc_df = data[data['Ticker'] == 'BTC-USD'].set_index('Date').copy()
    eth_df = data[data['Ticker'] == 'ETH-USD'].set_index('Date').copy()

    # Calculate Benchmark Gaps
    for df_bench, name in [(qqq_df, 'QQQ'), (spy_df, 'SPY')]:
        if df_bench.empty:
            df_bench[f'{name}_Gap'] = np.nan
        else:
            df_bench['Prev_Close'] = df_bench['Close'].shift(1)
            df_bench[f'{name}_Gap'] = (df_bench['Open'] - df_bench['Prev_Close']) / df_bench['Prev_Close']

            # DEBUG: Check if we have NaN gaps for benchmarks
            # print(f"DEBUG: {name} Gaps NaNs: {df_bench[f'{name}_Gap'].isnull().sum()} / {len(df_bench)}")

    if not qqq_df.empty:
        qqq_gap = qqq_df[[f'QQQ_Gap']]
    else:
        qqq_gap = pd.DataFrame(columns=['QQQ_Gap'])

    if not spy_df.empty:
        spy_gap = spy_df[[f'SPY_Gap']]
    else:
        spy_gap = pd.DataFrame(columns=['SPY_Gap'])

    if btc_df.empty:
        btc_close = pd.DataFrame(columns=['BTC_Close'])
    else:
        btc_close = btc_df[['Close']].rename(columns={'Close': 'BTC_Close'})

    if eth_df.empty:
         eth_close = pd.DataFrame(columns=['ETH_Close'])
    else:
        eth_close = eth_df[['Close']].rename(columns={'Close': 'ETH_Close'})

    stock_df = data[~data['Ticker'].isin(benchmarks)]
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
    """Feature Engineering"""
    df = df.sort_index()

    # 1. Numeric Conversion
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Join External Data
    df.index = pd.to_datetime(df.index).normalize()

    # DEBUG: Check if we have valid price data before joining
    if df['Close'].isnull().all():
        # print(f"DEBUG: All Close prices are NaN for {df['Ticker'].iloc[0] if 'Ticker' in df else 'Unknown'}")
        return pd.DataFrame()

    df = df.join(vix_df, how='left')
    df = df.join(qqq_gap, how='left')
    df = df.join(spy_gap, how='left')
    df = df.join(btc_close, how='left')
    df = df.join(eth_close, how='left')
    df = df.join(totm_df, how='left')

    # Fill VIX and Crypto
    df['VIX'] = df['VIX'].shift(1).ffill().bfill().fillna(20.0)
    if 'BTC_Close' in df.columns:
        df['BTC_Close'] = df['BTC_Close'].ffill().bfill()
    else:
        df['BTC_Close'] = 0
    if 'ETH_Close' in df.columns:
        df['ETH_Close'] = df['ETH_Close'].ffill().bfill()
    else:
         df['ETH_Close'] = 0

    # 3. Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

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
        df['BTC_RSI'] = ta.rsi(df['BTC_Close'], length=14).shift(1)
        btc_sma50 = df['BTC_Close'].rolling(50).mean()
        df['BTC_Trend'] = (df['BTC_Close'] / btc_sma50 - 1).shift(1)

        btc_s = df['BTC_Close'].replace(0, np.nan)
        stock_ret = df['Close'].ffill().pct_change()
        btc_ret = btc_s.ffill().pct_change()
        df['Crypto_Corr'] = stock_ret.rolling(30).corr(btc_ret).shift(1)

    except Exception as e:
        return pd.DataFrame()

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # --- Other Features ---
    # A. Dist_MA20
    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1

    # B. Relative Strength
    # FIX: If QQQ_Gap is all NaN, this makes Rel_Gap_QQQ all NaN
    # We should fill missing benchmark gaps with 0 (assuming market is flat if unknown)
    # OR we drop these features if benchmark is missing.
    # But dropping features later will kill the row.

    if df['QQQ_Gap'].isnull().all():
        df['QQQ_Gap'] = 0.0
    else:
        df['QQQ_Gap'] = df['QQQ_Gap'].fillna(0.0)

    if df['SPY_Gap'].isnull().all():
        df['SPY_Gap'] = 0.0
    else:
        df['SPY_Gap'] = df['SPY_Gap'].fillna(0.0)

    df['Rel_Gap_QQQ'] = df['Gap_Pct'] - df['QQQ_Gap']
    df['Rel_Gap_SPY'] = df['Gap_Pct'] - df['SPY_Gap']

    # --- Labeling ---
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # Cleaning
    # Ensure we keep rows where any of the POTENTIAL features are valid
    # To be safe, we dropNA on the largest superset of features
    all_feats = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20',
                 'Rel_Gap_QQQ', 'Rel_Gap_SPY', 'Days_From_Start', 'Days_To_End',
                 'BTC_RSI', 'BTC_Trend', 'Crypto_Corr', 'VIX']

    # We only drop if CRITICAL features are missing.
    # For ablation, some features might be missing if we didn't calculate them (but we did calculate all)

    # DEBUG: Check nulls before dropping
    before_len = len(df)

    # Special Handling: If Rel_Gap_QQQ/SPY is all NaN (due to benchmark mismatch?), we should investigate
    if 'Rel_Gap_QQQ' in df.columns and df['Rel_Gap_QQQ'].isnull().all():
        pass

    df_clean = df.dropna(subset=all_feats)
    if len(df_clean) == 0 and before_len > 0:
        # Check which column is causing the drop
        null_counts = df[all_feats].isnull().sum()
        # Print for the first ticker only to debug
        if df['Ticker'].iloc[0] == 'AAPL' or before_len > 1000: # heuristic to find a major stock
             print(f"DEBUG: {df['Ticker'].iloc[0]} Dropped all rows. Null counts:\n{null_counts[null_counts > 0]}")
             # Print head of problematic columns
             # print(df[all_feats].head())
        pass

    df = df_clean
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
    print(f"=== EXP-03: Feature Selection (Ablation Study) ===")

    tickers = load_tickers()
    stock_raw, vix_raw, qqq_raw, spy_raw, btc_raw, eth_raw = fetch_data(tickers)

    all_dates = stock_raw['Date'].unique()
    totm_df = calculate_totm_features(all_dates)

    print("\nBuilding features...")

    # DEBUG: Check Data Availability
    print(f"DEBUG: Stock Data: {len(stock_raw)} rows, Cols: {stock_raw.columns.tolist()}")
    print(f"DEBUG: BTC Data: {len(btc_raw)} rows")
    print(f"DEBUG: SPY Data: {len(spy_raw)} rows")
    print(f"DEBUG: QQQ Data: {len(qqq_raw)} rows")

    if not stock_raw.empty:
        print(f"DEBUG: Sample Stock Row:\n{stock_raw.iloc[0]}")

    all_data = []

    if btc_raw.empty:
        print("Error: BTC data is empty!")
        return

    # Debug counter
    processed_count = 0

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()

        # CLEANING: Drop rows where Close is missing immediately
        df = df.dropna(subset=['Close'])
        if df.empty:
            continue

        # DEBUG: Print first ticker processing
        if processed_count == 0:
            print(f"DEBUG: Processing {ticker}...")
            # Check Volume presence
            if 'Volume' in df.columns:
                 print(f"DEBUG: Volume NaNs: {df['Volume'].isnull().sum()} / {len(df)}")
            else:
                 print("DEBUG: Volume column MISSING!")

        feat_df = build_features(df, vix_raw, qqq_raw, spy_raw, btc_raw, eth_raw, totm_df)

        if feat_df.empty:
            if processed_count == 0:
                print(f"DEBUG: {ticker} feat_df is empty after build_features.")
            continue

        feat_df['Ticker'] = ticker
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

        processed_count += 1

    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Gap Signals: {len(full_df)}")

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    # --- Feature Subsets ---
    base_features = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']

    subsets = {
        'Base': base_features,
        'Base_TOTM': base_features + ['Days_From_Start', 'Days_To_End'],
        'Base_Crypto': base_features + ['BTC_RSI', 'BTC_Trend', 'Crypto_Corr'],
        'All': base_features + ['Days_From_Start', 'Days_To_End', 'BTC_RSI', 'BTC_Trend', 'Crypto_Corr', 'VIX', 'Rel_Gap_QQQ', 'Rel_Gap_SPY']
    }

    results = []

    # --- Loop through Subsets ---
    for name, feats in subsets.items():
        print(f"\n--- Testing Subset: {name} ({len(feats)} features) ---")

        X_train = train_df[feats]
        y_train = train_df['Label']
        w_train = train_df['Sample_Weight']

        X_test = test_df[feats]
        y_test = test_df['Label']
        r_test = test_df['Strategy_Ret']

        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            n_jobs=-1, random_state=42, verbosity=-1
        )
        model.fit(X_train, y_train, sample_weight=w_train)

        y_pred = model.predict(X_test)

        m_win, m_avg, m_tot, b_win, b_avg, b_tot = evaluate_performance(y_test, y_pred, r_test)

        print(f"Win Rate: {m_win:.2%} (Base: {b_win:.2%})")
        print(f"Avg Ret : {m_avg:.4f} (Base: {b_avg:.4f})")
        print(f"Signals : {sum(y_pred)}")

        results.append({
            'Subset': name,
            'Feature_Count': len(feats),
            'Signals': sum(y_pred),
            'Win_Rate': m_win,
            'Avg_Return': m_avg,
            'Total_Return': m_tot,
            'Win_Rate_Diff': m_win - b_win,
            'Avg_Return_Diff': m_avg - b_avg
        })

        # If this is the "All" model, run Permutation Importance
        if name == 'All':
            print("\nCalculating Permutation Importance for 'All' model...")
            # We use a validation set or the test set for permutation importance.
            # Ideally validation, but here we use Test to explain OOS performance.
            perm_result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )

            perm_sorted_idx = perm_result.importances_mean.argsort()

            plt.figure(figsize=(10, 8))
            plt.boxplot(
                perm_result.importances[perm_sorted_idx].T,
                vert=False,
                labels=np.array(feats)[perm_sorted_idx]
            )
            plt.title("Permutation Importance (Test Set)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'permutation_importance.png'))

            # Save raw importance data
            imp_df = pd.DataFrame({
                'Feature': np.array(feats)[perm_sorted_idx],
                'Importance_Mean': perm_result.importances_mean[perm_sorted_idx],
                'Importance_Std': perm_result.importances_std[perm_sorted_idx]
            }).sort_values('Importance_Mean', ascending=False)
            imp_df.to_csv(os.path.join(OUTPUT_DIR, 'permutation_importance.csv'), index=False)
            print("Permutation importance saved.")

            # Also save the "All" model
            joblib.dump(model, os.path.join(OUTPUT_DIR, 'model_all.joblib'))

    # Save Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'subsets_performance.csv'), index=False)

    print("\n" + "="*80)
    print("EXP-03 SUMMARY")
    print("="*80)
    print(res_df[['Subset', 'Signals', 'Win_Rate', 'Avg_Return']].to_string(index=False))

if __name__ == '__main__':
    main()
