
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import json
import os
import sys
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# --- Configuration ---
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "03_Output")
RESOURCE_DIR = os.path.join(EXPERIMENT_DIR, "../../../../resource")
CRYPTO_POOL_FILE = os.path.join(RESOURCE_DIR, "2025_final_crypto_sensitive_pool.json")

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
TEST_START_DATE = "2024-01-01"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def load_tickers():
    if not os.path.exists(CRYPTO_POOL_FILE):
        print(f"Error: {CRYPTO_POOL_FILE} not found.")
        sys.exit(1)
    with open(CRYPTO_POOL_FILE, 'r') as f:
        data = json.load(f)
    # Handle different json structures if necessary, assuming list of strings or dict
    if isinstance(data, list):
        tickers = data
    elif isinstance(data, dict):
        # Extract tickers from values or keys based on structure
        tickers = []
        for key, val in data.items():
            if isinstance(val, list):
                tickers.extend(val)
            else:
                tickers.append(key)
        tickers = list(set(tickers))
    else:
        tickers = []

    # Clean tickers (remove exchange prefix)
    cleaned_tickers = []
    for t in tickers:
        if ":" in t:
            cleaned_tickers.append(t.split(":")[1])
        else:
            cleaned_tickers.append(t)
    return cleaned_tickers

def download_data(tickers):
    print(f"Downloading data for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=True, progress=False, threads=True)
        if data.empty:
            print("No data downloaded.")
            return None

        # Flatten MultiIndex if necessary
        if isinstance(data.columns, pd.MultiIndex):
             # Stack to get Ticker as a column, then reset index
            data = data.stack(level=0, future_stack=True).reset_index()
            data.rename(columns={'level_1': 'Ticker', 'Date': 'Date'}, inplace=True)
            # Or depending on yfinance version, level 0 might be Ticker if group_by='ticker'
            # Let's check the structure in memory or assume standard stack
            pass
        else:
            # Single ticker
            data['Ticker'] = tickers[0]
            data.reset_index(inplace=True)

        return data
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def download_btc():
    print("Downloading BTC-USD data...")
    btc = yf.download("BTC-USD", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    btc.reset_index(inplace=True)
    return btc

def process_btc_features(btc_df):
    """
    Calculates BTC features.
    IMPORTANT: All features must be shifted to be available at Open.
    """
    df = btc_df.copy()

    # Standardize columns
    # Flatten MultiIndex if exists
    if isinstance(df.columns, pd.MultiIndex):
        # If columns are (Price, Ticker), drop level 1
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns and 'close' in df.columns:
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)

    # BTC_Change (T-1 Close vs T-2 Close) -> Available at T Open
    df['BTC_Prev_Close'] = df['Close'].shift(1)
    df['BTC_Prev_Prev_Close'] = df['Close'].shift(2)
    df['BTC_Change'] = (df['BTC_Prev_Close'] - df['BTC_Prev_Prev_Close']) / df['BTC_Prev_Prev_Close']

    # BTC_RSI (on T-1 Close)
    df['BTC_RSI'] = ta.rsi(df['Close'], length=14).shift(1)

    # BTC_Gap (T Open vs T-1 Close) -> Available at T Open
    # Wait: Strategy runs at Market Open (9:30 AM ET). BTC trades 24/7.
    # We can use BTC price at 9:30 AM ET?
    # yfinance gives daily data (UTC midnight usually).
    # Using BTC Open (UTC 00:00) vs Prev Close might be stale by 9:30 AM ET.
    # However, for simplicity and consistency with historical data, we use Daily candles.
    # BTC_Gap = (BTC_Open_Today - BTC_Close_Yesterday) / BTC_Close_Yesterday
    df['BTC_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Keep only date and features
    cols = ['Date', 'BTC_Change', 'BTC_RSI', 'BTC_Gap', 'Close']
    return df[cols].rename(columns={'Close': 'BTC_Close'})

def calculate_base_features(df):
    """
    Standard Base Features: Gap_Pct, RSI_14, ATR_Pct, Vol_Ratio, Dist_MA20
    """
    df = df.copy()
    # Sort just in case
    df.sort_values('Date', inplace=True)

    # Shifted Close for indicators to avoid lookahead
    prev_close = df['Close'].shift(1)

    # Gap_Pct (Open - PrevClose) / PrevClose
    df['Gap_Pct'] = (df['Open'] - prev_close) / prev_close

    # RSI 14 (using shifted close)
    # Using pandas-ta, we apply to the whole series but need to ensure we use valid inputs.
    # Actually, to prevent lookahead, we calculate RSI on the Close column, THEN shift the result by 1.
    df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    # ATR Pct
    # ATR requires High, Low, Close. We calculate on daily bars, then shift result by 1.
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_Pct'] = (atr.shift(1) / prev_close)

    # Vol_Ratio (Volume / MA_Volume)
    vol_ma = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = (df['Volume'].shift(1) / vol_ma.shift(1))

    # Dist_MA20 (Close - MA20) / MA20
    ma20 = df['Close'].rolling(window=20).mean()
    df['Dist_MA20'] = (prev_close - ma20.shift(1)) / ma20.shift(1)

    return df

def prepare_data(stock_data, btc_data):
    # Process BTC
    btc_feat = process_btc_features(btc_data)

    # Process Stocks
    processed_dfs = []

    # yfinance group_by='ticker' structure handling
    # If the df is already melted/stacked by download_data:
    tickers = stock_data['Ticker'].unique()

    for ticker in tickers:
        df = stock_data[stock_data['Ticker'] == ticker].copy()
        if len(df) < 50: continue

        # Base Features
        df = calculate_base_features(df)

        # Merge BTC (on Date)
        # Ensure dates match properly (tz-naive)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        btc_feat['Date'] = pd.to_datetime(btc_feat['Date']).dt.tz_localize(None)

        df = pd.merge(df, btc_feat, on='Date', how='left')

        # Crypto_Corr (Rolling correlation between Stock Close and BTC Close)
        # Using shifted closes for calculation?
        # Corr should be known at Open. So correlation of (Stock_Close_T-1...T-20) vs (BTC_Close_T-1...T-20)
        # We can calculate rolling corr of Close and BTC_Close, then shift result by 1.
        df['Crypto_Corr'] = df['Close'].rolling(window=20).corr(df['BTC_Close']).shift(1)

        # Label: (Open - Close) / Open > 0.002
        # BUT we execute Hold-to-Close (Sell Open, Buy Close).
        # Profit = (Open - Close) / Open
        # If Profit > 0.002, Label = 1
        df['Return'] = (df['Open'] - df['Close']) / df['Open']
        df['Target'] = (df['Return'] > 0.002).astype(int)

        # Gap Threshold Filter (Pre-filter for training? Or just a feature?)
        # In V6.2, we usually only train on gaps > 0.5% or > 1%.
        # Let's verify the standard practice. Usually we train on specific gaps.
        # But to be safe, we'll train on all data or at least Gaps > 0
        # Let's stick to the V6.2 standard: Gap > 0.5% (0.005)
        df = df[df['Gap_Pct'] > 0.005]

        processed_dfs.append(df)

    if not processed_dfs:
        return pd.DataFrame()

    full_df = pd.concat(processed_dfs, ignore_index=True)
    full_df.dropna(inplace=True)
    return full_df

def train_and_evaluate(df, features, model_name="Model"):
    print(f"\n--- Training {model_name} ---")
    print(f"Features: {features}")

    # Split
    train_df = df[df['Date'] < TEST_START_DATE]
    test_df = df[df['Date'] >= TEST_START_DATE]

    print(f"Train Size: {len(train_df)}, Test Size: {len(test_df)}")

    if len(train_df) == 0 or len(test_df) == 0:
        print("Insufficient data for split.")
        return None

    X_train = train_df[features]
    y_train = train_df['Target']
    X_test = test_df[features]
    y_test = test_df['Target']
    returns_test = test_df['Return']

    # Train LightGBM
    # Using parameters from EXP-06 (Optimized or default?)
    # Tech used strict regularization. Crypto is likely similar to Tech.
    # Params: max_depth=3, learning_rate=0.01, n_estimators=500
    model = lgb.LGBMClassifier(
        max_depth=3,
        learning_rate=0.01,
        n_estimators=500,
        random_state=42,
        verbosity=-1
    )

    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Evaluate
    # Filter by probability threshold? Default 0.5 for predict
    # Or optimize threshold? Standard is usually 0.5 or 0.55.
    # Let's use standard predict (0.5).

    # Calculate performance on executed trades
    executed_mask = preds == 1
    executed_returns = returns_test[executed_mask]

    win_rate = np.mean(executed_returns > 0)
    avg_return = np.mean(executed_returns)
    total_return = np.sum(executed_returns)
    count = len(executed_returns)

    print(f"Signals: {count}")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Avg Return: {avg_return:.4f}")

    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    importance.to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_importance.csv"), index=False)

    return {
        'Model': model_name,
        'Win Rate': win_rate,
        'Avg Return': avg_return,
        'Total Return': total_return,
        'Signals': count,
        'Object': model
    }

# --- Main Execution ---

def main():
    # 1. Load Tickers
    tickers = load_tickers()
    print(f"Loaded {len(tickers)} crypto-sensitive tickers: {tickers}")

    # 2. Download Data
    # Fix: yfinance might fail on some tickers, ensure list is clean
    stock_data = download_data(tickers)
    btc_data = download_btc()

    if stock_data is None or btc_data is None:
        print("Data download failed.")
        sys.exit(1)

    # 3. Prepare Data
    full_df = prepare_data(stock_data, btc_data)
    print(f"Prepared Data Shape: {full_df.shape}")

    if full_df.empty:
        print("No valid data after processing.")
        sys.exit(1)

    # 4. Define Feature Sets
    base_features = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
    crypto_features = base_features + ['BTC_Gap', 'BTC_RSI', 'BTC_Change', 'Crypto_Corr']

    # 5. Train & Evaluate
    baseline_res = train_and_evaluate(full_df, base_features, "Baseline (Base Only)")
    crypto_res = train_and_evaluate(full_df, crypto_features, "Crypto Model (Base + Crypto)")

    # 6. Save Results
    results = pd.DataFrame([baseline_res, crypto_res])
    results_path = os.path.join(OUTPUT_DIR, "performance_report.csv")
    results[['Model', 'Win Rate', 'Avg Return', 'Total Return', 'Signals']].to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # 7. Comparison Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results))
    width = 0.35

    plt.bar(x - width/2, results['Win Rate'], width, label='Win Rate')
    plt.bar(x + width/2, results['Avg Return'] * 10, width, label='Avg Return (x10)') # Scale for visibility

    plt.xticks(x, results['Model'])
    plt.legend()
    plt.title("Baseline vs Crypto Model Performance")
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_plot.png"))

    # 8. Save Models
    joblib.dump(baseline_res['Object'], os.path.join(OUTPUT_DIR, "baseline_model.joblib"))
    joblib.dump(crypto_res['Object'], os.path.join(OUTPUT_DIR, "crypto_model.joblib"))

if __name__ == "__main__":
    main()
