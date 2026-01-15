import sys
import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import lightgbm as lgb
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Add path to Lab Utils
sys.path.append(os.path.abspath("../../02_Lab_Utils"))

from metrics import LabMetrics

# ==========================================
# Configuration
# ==========================================
EXPERIMENT_NAME = "EXP_19_Crypto_Pure_Play"
OUTPUT_DIR = "03_Output"
STOCKS = ['COIN', 'MSTR', 'RIOT', 'MARA']
CRYPTO_TICKER = 'BTC-USD'
START_DATE = "2020-01-01"
# Training Split
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"

# ==========================================
# 1. Data Acquisition
# ==========================================
def fetch_data(tickers, start_date):
    print(f"Fetching data for {tickers}...")
    try:
        data = yf.download(tickers, start=start_date, progress=False, group_by='ticker')

        # Check structure
        if isinstance(data.columns, pd.MultiIndex):
            # Stack to get Ticker as a column (or index level)
            try:
                df = data.stack(level=0, future_stack=True).reset_index()
            except TypeError:
                df = data.stack(level=0).reset_index()

            df.rename(columns={'level_1': 'Ticker', 'Date': 'Date'}, inplace=True)
        else:
            # Single ticker case
            df = data.reset_index()
            df['Ticker'] = tickers[0]

        # Ensure columns are standard
        df.columns.name = None
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_btc_data(start_date):
    print(f"Fetching BTC data...")
    btc = yf.download(CRYPTO_TICKER, start=start_date, progress=False)
    btc = btc.reset_index()

    # Clean columns (remove MultiIndex if present)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]

    # Fix for yfinance returning 'Ticker' in columns sometimes
    if 'Ticker' in btc.columns:
        btc = btc.drop(columns=['Ticker'])

    # Calculate features
    # BTC_Gap: (Open - Prev_Close) / Prev_Close
    btc['BTC_Gap'] = (btc['Open'] - btc['Close'].shift(1)) / btc['Close'].shift(1)

    # BTC_RSI_14: RSI of Close, Shifted 1 (available at Open)
    try:
        btc['BTC_RSI_14'] = ta.rsi(btc['Close'], length=14).shift(1)
    except Exception as e:
        print(f"Error calculating BTC RSI: {e}")
        btc['BTC_RSI_14'] = 0

    # BTC_Change: (Close - Prev_Close) / Prev_Close (Prev Day Return)
    btc['BTC_Change'] = btc['Close'].pct_change().shift(1)

    # Select only needed columns
    btc_features = btc[['Date', 'BTC_Gap', 'BTC_RSI_14', 'BTC_Change']].copy()
    return btc_features

# ==========================================
# 2. Feature Engineering
# ==========================================
def process_stock_data(df, btc_features):
    processed_dfs = []

    # Group by Ticker to process independently
    for ticker, group in df.groupby('Ticker'):
        g = group.copy().sort_values('Date')
        g = g.set_index('Date')

        # 1. Target Label: (Open - Close) / Open > 0.002 (Profit > 0.2%)
        g['return'] = (g['Open'] - g['Close']) / g['Open']
        g['target'] = (g['return'] > 0.002).astype(int)

        # 2. Base Features
        # Gap_Pct: (Open - Prev_Close) / Prev_Close
        g['Gap_Pct'] = (g['Open'] - g['Close'].shift(1)) / g['Close'].shift(1)

        # RSI_14: Shift 1
        g['RSI_14'] = ta.rsi(g['Close'], length=14).shift(1)

        # ATR_Pct: ATR(14) / Close. Shift 1.
        atr = ta.atr(g['High'], g['Low'], g['Close'], length=14)
        g['ATR_Pct'] = (atr / g['Close']).shift(1)

        # Vol_Ratio: Vol / MA_Vol(20). Shift 1.
        vol_ma = g['Volume'].rolling(window=20).mean()
        g['Vol_Ratio'] = (g['Volume'] / vol_ma).shift(1)

        # Dist_MA20: (Close - MA20) / MA20. Shift 1.
        ma20 = g['Close'].rolling(window=20).mean()
        g['Dist_MA20'] = ((g['Close'] - ma20) / ma20).shift(1)

        # 3. Join BTC Features
        g = g.merge(btc_features, on='Date', how='left')

        # Drop NaNs
        g = g.dropna()

        g['Ticker'] = ticker
        processed_dfs.append(g)

    return pd.concat(processed_dfs)

# ==========================================
# 3. Model Training & Evaluation
# ==========================================
def train_and_evaluate(df):
    base_features = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
    crypto_features = base_features + ['BTC_Gap', 'BTC_RSI_14', 'BTC_Change']

    # Split
    train = df[(df['Date'] >= TRAIN_START) & (df['Date'] <= TRAIN_END)]
    test = df[(df['Date'] >= TEST_START)]

    print(f"Train Size: {len(train)}, Test Size: {len(test)}")

    results = {}

    # --- Control Model (Base) ---
    print("Training Control Model (Base Features)...")
    model_base = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    model_base.fit(train[base_features], train['target'])

    preds_base = model_base.predict(test[base_features])
    probs_base = model_base.predict_proba(test[base_features])[:, 1]

    # Evaluate Base
    res_base = test.copy()
    res_base['pred'] = preds_base
    res_base['prob'] = probs_base
    # Filter for signals (Prediction = 1)
    signals_base = res_base[res_base['pred'] == 1].copy()

    # Calculate real profit boolean for evaluation
    if 'is_profit' in signals_base.columns:
        signals_base = signals_base.drop(columns=['is_profit'])
    signals_base['is_profit'] = (signals_base['return'] > 0).astype(int)

    metrics_base = LabMetrics.evaluate_experiment(signals_base)
    results['Control'] = metrics_base

    # --- Test Model (Base + Crypto) ---
    print("Training Test Model (Base + Crypto Features)...")
    model_crypto = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    model_crypto.fit(train[crypto_features], train['target'])

    preds_crypto = model_crypto.predict(test[crypto_features])
    probs_crypto = model_crypto.predict_proba(test[crypto_features])[:, 1]

    # Evaluate Crypto
    res_crypto = test.copy()
    res_crypto['pred'] = preds_crypto
    res_crypto['prob'] = probs_crypto
    signals_crypto = res_crypto[res_crypto['pred'] == 1].copy()

    if 'is_profit' in signals_crypto.columns:
        signals_crypto = signals_crypto.drop(columns=['is_profit'])
    signals_crypto['is_profit'] = (signals_crypto['return'] > 0).astype(int)

    metrics_crypto = LabMetrics.evaluate_experiment(signals_crypto)
    results['Test'] = metrics_crypto

    # --- Feature Importance (Test Model) ---
    importance = pd.DataFrame({
        'Feature': crypto_features,
        'Importance': model_crypto.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Save Artifacts
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    importance.to_csv(f"{OUTPUT_DIR}/feature_importance.csv", index=False)
    joblib.dump(model_base, f"{OUTPUT_DIR}/control_model.joblib")
    joblib.dump(model_crypto, f"{OUTPUT_DIR}/test_model.joblib")
    signals_base.to_csv(f"{OUTPUT_DIR}/signals_control.csv")
    signals_crypto.to_csv(f"{OUTPUT_DIR}/signals_test.csv")

    return results, importance

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Fetch Data
    stock_data = fetch_data(STOCKS, START_DATE)
    btc_data = get_btc_data(START_DATE)

    if stock_data.empty or btc_data.empty:
        print("Data fetching failed.")
        sys.exit(1)

    # 2. Process
    full_data = process_stock_data(stock_data, btc_data)

    # 3. Train & Evaluate
    results, importance = train_and_evaluate(full_data)

    # 4. Report
    print("\n================ Results ================")
    print(f"Control (Base): {results['Control']['metrics']}")
    print(f"Test (Crypto):  {results['Test']['metrics']}")
    print("\nFeature Importance:")
    print(importance)

    # Write summary to file for Analyze step
    with open(f"{OUTPUT_DIR}/performance_report.txt", "w") as f:
        f.write(str(results))
