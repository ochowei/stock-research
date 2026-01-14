
import os
import sys
import json
import logging
import joblib
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Setup
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from production_daily_plan_v6_2 import clean_ticker
except ImportError:
    # If running from exp/
    sys.path.append(BASE_DIR)
    from production_daily_plan_v6_2 import clean_ticker

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, 'r') as f:
        tickers = json.load(f)
    return [clean_ticker(t) for t in tickers]

def download_data(tickers, start, end):
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, progress=True)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False) # For relative strength later, or benchmark
    return data, vix, spy

def prepare_features(data, vix, spy, tickers):
    all_rows = []

    # Pre-process VIX
    # VIX feature usually is the previous close.
    vix_close = vix['Close']

    for t in tickers:
        try:
            if len(tickers) > 1:
                df = data[t].copy()
            else:
                df = data.copy()

            if df.empty: continue
            df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])

            # Indicators (Calculated on Close, shifted by 1 to represent "Prior Day")
            # We want to predict for Day T based on Day T-1 info + Day T Open.

            # 1. RSI (14)
            df['RSI_14'] = ta.rsi(df['Close'], length=14)

            # 2. ATR Pct
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR_Pct'] = df['ATR'] / df['Close']

            # 3. Volume Ratio
            # Generator: df_daily['Volume'].iloc[-1] / df_daily['Volume'].rolling(20).mean().iloc[-2]
            # Comparison of Volume(T) vs MeanVolume(T-1...T-20)
            # We calculate this for T first.
            vol_ma_shifted = df['Volume'].rolling(20).mean().shift(1)
            df['Vol_Ratio'] = df['Volume'] / vol_ma_shifted

            # Shift these features because at Open of Day T, we only know Close of T-1
            # But wait. daily_gap_signal_generator logic:
            # rsi = ta.rsi(df_daily['Close'], length=14).iloc[-1]
            # If run before market, iloc[-1] is yesterday.
            # So yes, use shifted values.

            df['Prev_Close'] = df['Close'].shift(1)
            df['Prev_RSI_14'] = df['RSI_14'].shift(1)
            df['Prev_ATR_Pct'] = df['ATR_Pct'].shift(1)
            df['Prev_Vol_Ratio'] = df['Vol_Ratio'].shift(1)

            # VIX feature (Shifted)
            # Need to align dates.
            # df['VIX'] = vix_close.reindex(df.index).shift(1) # This is slow in loop

            # 4. Gap Pct (Open T - Close T-1) / Close T-1
            df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

            # 5. Label
            # Sell Model: Predict if price drops intraday.
            # (Open - Close) / Open > 0.002
            df['Intraday_Ret'] = (df['Open'] - df['Close']) / df['Open']
            df['Label'] = (df['Intraday_Ret'] > 0.002).astype(int)

            df['Ticker'] = t
            df['Date'] = df.index

            # Filter valid rows
            # We need Prev_Close to be valid, and Features.
            # Also align VIX

            all_rows.append(df[['Date', 'Ticker', 'Gap_Pct', 'Prev_RSI_14', 'Prev_ATR_Pct', 'Prev_Vol_Ratio', 'Intraday_Ret', 'Label']])

        except Exception as e:
            # print(f"Error processing {t}: {e}")
            continue

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Merge VIX efficiently
    vix_shifted = vix_close.shift(1)
    if isinstance(vix_shifted, pd.DataFrame):
         vix_shifted = vix_shifted.iloc[:, 0]

    vix_shifted = vix_shifted.rename('VIX')

    combined = combined.merge(vix_shifted, left_on='Date', right_index=True, how='left')

    return combined.dropna()

def train_baseline():
    tickers = load_tickers()
    start_date = "2019-10-01" # Buffer for indicators
    end_date = "2025-12-31"

    data, vix, spy = download_data(tickers, start_date, end_date)

    print("Processing features...")
    df = prepare_features(data, vix, spy, tickers)

    # Filter: Gap > 0 (Sell Model Assumption)
    # The prompt implies recreating EXP-07. Usually Sell models operate on Gap Up.
    df = df[df['Gap_Pct'] > 0.005]

    # Train / Test Split
    train_df = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]
    test_df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-12-31')]

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    feature_cols = ['Prev_RSI_14', 'Prev_ATR_Pct', 'Prev_Vol_Ratio', 'Gap_Pct', 'VIX']
    X_train = train_df[feature_cols]
    y_train = train_df['Label']
    w_train = train_df['Intraday_Ret'].abs() * 100

    X_test = test_df[feature_cols]
    y_test = test_df['Label']

    # Model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate
    for name, subset in [("Train", train_df), ("Test (OOS)", test_df)]:
        X = subset[feature_cols]
        y = subset['Label']
        ret = subset['Intraday_Ret']

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        # Metrics
        # Win Rate: Precision (When we predict Sell (1), how often is it right?)
        # Wait, if the model predicts 1, we Trade.
        # Win Rate = Accuracy on predicted positives. = Precision.

        subset['Pred'] = preds
        trades = subset[subset['Pred'] == 1]

        if len(trades) > 0:
            win_rate = trades['Label'].mean()
            avg_return = trades['Intraday_Ret'].mean() * 100 # %
            print(f"[{name}] Trades: {len(trades)} | Win Rate: {win_rate:.2%} | Avg Return: {avg_return:.3f}%")
        else:
            print(f"[{name}] No trades predicted.")

    # Save model
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_07_repro_baseline.joblib'))
    print("Model saved.")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    train_baseline()
