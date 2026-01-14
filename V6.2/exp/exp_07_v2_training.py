
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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

# Setup
logging.getLogger('yfinance').setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
SECTOR_FILE = os.path.join(RESOURCE_DIR, 'ticker_sectors.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from production_daily_plan_v6_2 import clean_ticker
except ImportError:
    sys.path.append(BASE_DIR)
    from production_daily_plan_v6_2 import clean_ticker

def load_tickers_and_sectors():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    with open(path, 'r') as f:
        tickers = json.load(f)

    cleaned = [clean_ticker(t) for t in tickers]

    sector_map = {}
    if os.path.exists(SECTOR_FILE):
        with open(SECTOR_FILE, 'r') as f:
            sector_map = json.load(f)

    # Fill missing sectors with 'Unknown'
    for t in cleaned:
        if t not in sector_map:
            sector_map[t] = 'Unknown'

    return cleaned, sector_map

def download_data(tickers, start, end):
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, progress=True)
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    return data, vix, spy

def prepare_features(data, vix, spy, tickers, sector_map):
    all_rows = []

    # Process VIX and SPY
    vix_close = vix['Close']
    if isinstance(vix_close, pd.DataFrame): vix_close = vix_close.iloc[:, 0]

    spy_open = spy['Open']
    spy_close = spy['Close']
    if isinstance(spy_open, pd.DataFrame): spy_open = spy_open.iloc[:, 0]
    if isinstance(spy_close, pd.DataFrame): spy_close = spy_close.iloc[:, 0]

    # Calculate SPY Gap
    spy_prev_close = spy_close.shift(1)
    spy_gap = (spy_open - spy_prev_close) / spy_prev_close
    spy_gap = spy_gap.rename("SPY_Gap")

    for t in tickers:
        try:
            if len(tickers) > 1:
                df = data[t].copy()
            else:
                df = data.copy()

            if df.empty: continue
            df = df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'])

            # --- Base Indicators ---

            # RSI 14
            df['RSI_14'] = ta.rsi(df['Close'], length=14)

            # ATR Pct
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR_Pct'] = df['ATR'] / df['Close']

            # Volume Ratio (match generator logic: Vol(T) / Mean(T-1..T-20))
            vol_ma_shifted = df['Volume'].rolling(20).mean().shift(1)
            df['Vol_Ratio'] = df['Volume'] / vol_ma_shifted

            # Dist MA20
            # Generator: ma20_sim = ((df_daily['Close'].tail(19).mean() * 19) + curr_price) / 20
            # Wait, generator simulates MA20 including current price?
            # "ma20_sim = ((df_daily['Close'].tail(19).mean() * 19) + curr_price) / 20"
            # df_daily['Close'].tail(19) implies previous 19 days.
            # So it calculates MA20 as if today's price is included.
            # Here for training, we can calculate rolling mean of 20 days.
            # Rolling(20).mean() includes today. So (Close(T) + Sum(Close(T-1..T-19)))/20.
            # Yes, standard rolling mean.
            df['MA20'] = df['Close'].rolling(20).mean()
            df['Dist_MA20'] = (df['Close'] / df['MA20']) - 1

            # --- Shift Features (Info available at Open) ---
            df['Prev_Close'] = df['Close'].shift(1)
            df['Prev_RSI_14'] = df['RSI_14'].shift(1)
            df['Prev_ATR_Pct'] = df['ATR_Pct'].shift(1)
            df['Prev_Vol_Ratio'] = df['Vol_Ratio'].shift(1)
            df['Prev_Dist_MA20'] = df['Dist_MA20'].shift(1)

            # Gap
            df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

            # --- New Features ---

            # Relative Strength (RS_Gap)
            # We need to merge with SPY Gap. We'll do it after concat to be efficient?
            # Or do it here via index alignment.
            # df has Date index.
            df = df.merge(spy_gap, left_index=True, right_index=True, how='left')
            df['RS_Gap'] = df['Gap_Pct'] - df['SPY_Gap']

            # Calendar
            df['Day'] = df.index.day
            df['Month_Start'] = df.index.is_month_start.astype(int)
            df['Month_End'] = df.index.is_month_end.astype(int)
            # TOTM (First 3 days or Last day)
            # A bit complex to get exactly "First 3 trading days".
            # Simpler proxy: Day <= 3 or Is_Month_End
            df['Is_TOTM'] = ((df['Day'] <= 3) | (df['Month_End'] == 1)).astype(int)

            # Sector
            sector = sector_map.get(t, 'Unknown')
            df['Sector'] = sector

            # Label: Sell Model (Intraday Drop > 0.2%)
            df['Intraday_Ret'] = (df['Open'] - df['Close']) / df['Open']
            df['Label'] = (df['Intraday_Ret'] > 0.002).astype(int)

            df['Ticker'] = t
            df['Date'] = df.index

            # Select columns
            cols = ['Date', 'Ticker', 'Gap_Pct', 'Prev_RSI_14', 'Prev_ATR_Pct', 'Prev_Vol_Ratio',
                    'Prev_Dist_MA20', 'RS_Gap', 'Is_TOTM', 'Sector', 'Intraday_Ret', 'Label']

            all_rows.append(df[cols])

        except Exception as e:
            continue

    if not all_rows:
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)

    # Merge VIX
    vix_shifted = vix_close.shift(1).rename('VIX')
    combined = combined.merge(vix_shifted, left_on='Date', right_index=True, how='left')

    return combined.dropna()

def train_v2():
    tickers, sector_map = load_tickers_and_sectors()
    start_date = "2019-10-01"
    end_date = "2025-12-31"

    data, vix, spy = download_data(tickers, start_date, end_date)

    print("Processing features...")
    df = prepare_features(data, vix, spy, tickers, sector_map)

    # Filter: Gap > 0.5%
    df = df[df['Gap_Pct'] > 0.005]

    # Encoding Sector
    print("Encoding sectors...")
    # Get top sectors to avoid sparse classes
    top_sectors = df['Sector'].value_counts().nlargest(10).index
    df['Sector_Clean'] = df['Sector'].apply(lambda x: x if x in top_sectors else 'Other')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sector_encoded = encoder.fit_transform(df[['Sector_Clean']])
    sector_cols = [f"Sec_{c}" for c in encoder.categories_[0]]
    sector_df = pd.DataFrame(sector_encoded, columns=sector_cols, index=df.index)

    df = pd.concat([df, sector_df], axis=1)

    # Feature Columns
    feature_cols = ['Prev_RSI_14', 'Prev_ATR_Pct', 'Prev_Vol_Ratio', 'Gap_Pct', 'VIX',
                    'Prev_Dist_MA20', 'RS_Gap', 'Is_TOTM'] + sector_cols

    # Train / Test Split
    train_df = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]
    test_df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-12-31')]

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    X_train = train_df[feature_cols]
    y_train = train_df['Label']
    w_train = train_df['Intraday_Ret'].abs() * 100

    X_test = test_df[feature_cols]
    y_test = test_df['Label']

    # Model
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate
    results = {}
    for name, subset in [("Train", train_df), ("Test (OOS)", test_df)]:
        X = subset[feature_cols]
        y = subset['Label']

        preds = model.predict(X)
        probs = model.predict_proba(X)[:, 1]

        subset['Pred'] = preds
        subset['Prob'] = probs

        trades = subset[subset['Pred'] == 1]

        win_rate = trades['Label'].mean() if len(trades) > 0 else 0
        avg_ret = trades['Intraday_Ret'].mean() * 100 if len(trades) > 0 else 0

        print(f"[{name}] Trades: {len(trades)} | Win Rate: {win_rate:.2%} | Avg Return: {avg_ret:.3f}%")
        results[name] = {'WinRate': win_rate, 'AvgRet': avg_ret}

    # Feature Importance
    print("\nFeature Importance:")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(10))

    # Save Model and Encoder
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_07_v2_model.joblib'))
    joblib.dump(encoder, os.path.join(OUTPUT_DIR, 'exp_07_v2_encoder.joblib'))
    joblib.dump(top_sectors, os.path.join(OUTPUT_DIR, 'exp_07_v2_sectors.joblib'))
    print("Model saved.")

if __name__ == '__main__':
    train_v2()
