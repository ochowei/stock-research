import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# --- Configuration ---
EXP_NAME = "EXP_15_Crypto_Specific_Ensemble"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "03_Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add Lab Utils to Path
LAB_UTILS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../02_Lab_Utils"))
sys.path.append(LAB_UTILS_PATH)

# Tickers
PURE_CRYPTO_POOL = ['COIN', 'MSTR', 'RIOT', 'MARA']
BTC_TICKER = "BTC-USD"

# Dates
START_DATE = "2023-01-01"
END_DATE = "2024-10-25" # Up to recent
SPLIT_DATE = "2024-07-01"

# Features
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
BTC_FEATURES = ['BTC_RSI_14', 'BTC_Trend', 'Crypto_Corr', 'BTC_Ret']

# Model Params (LightGBM)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'random_state': 42,
    'verbose': -1
}

def fetch_data(tickers, start, end):
    print(f"Fetching data for {tickers}...")
    try:
        # Use simple download loop to avoid MultiIndex complexity/variability across versions
        all_data = []
        for t in tickers:
            df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                continue

            # Reset index to get Date as column
            df = df.reset_index()

            # Flatten columns if MultiIndex (happens sometimes even with single ticker in new yf)
            if isinstance(df.columns, pd.MultiIndex):
                # Usually (Price, Ticker) or just Price
                # We just want the Price part
                # If we have ('Open', 'COIN'), we want 'Open'
                # Check if level 1 is the ticker
                new_cols = []
                for col in df.columns:
                    if isinstance(col, tuple):
                         # Assume the one that matches standard OHLC is the one we keep
                         # Or just take the first level
                         new_cols.append(col[0])
                    else:
                        new_cols.append(col)
                df.columns = new_cols

            df['Ticker'] = t
            all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        data = pd.concat(all_data)

        # Standardize columns: Capitalize first letter
        data.columns = [str(c).capitalize() for c in data.columns]

        # Ensure we have required cols
        required = {'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'}
        available = set(data.columns)
        if not required.issubset(available):
            print(f"Missing columns. Available: {available}")
            return pd.DataFrame()

        return data[list(required)]

    except Exception as e:
        print(f"Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_btc_features(btc_df):
    print("Processing BTC features...")
    df = btc_df.copy().sort_values('Date')

    # Calculate Indicators
    # RSI
    df['BTC_RSI_14'] = ta.rsi(df['Close'], length=14)
    # Trend (Dist to MA20)
    df['MA20'] = ta.sma(df['Close'], length=20)
    df['BTC_Trend'] = (df['Close'] - df['MA20']) / df['MA20']
    # Returns
    df['BTC_Ret'] = df['Close'].pct_change()

    # SHIFT ALL FEATURES BY 1 to prevent look-ahead
    feature_cols = ['BTC_RSI_14', 'BTC_Trend', 'BTC_Ret', 'Close']
    df[feature_cols] = df[feature_cols].shift(1)

    # Rename Close to BTC_Close for Correlation calculation later
    df.rename(columns={'Close': 'BTC_Close'}, inplace=True)

    return df[['Date', 'BTC_RSI_14', 'BTC_Trend', 'BTC_Ret', 'BTC_Close']]

def process_stock_features(stock_df):
    print("Processing Stock features...")
    df = stock_df.copy().sort_values(['Ticker', 'Date'])

    # Group by Ticker
    results = []
    for ticker, group in df.groupby('Ticker'):
        g = group.copy()

        # 1. Target: (Open - Close) / Open > 0.002
        g['Ret_Open_Close'] = (g['Open'] - g['Close']) / g['Open']
        g['Target'] = (g['Ret_Open_Close'] > 0.002).astype(int)

        # 2. Base Features
        # Gap_Pct: (Open - Prev_Close) / Prev_Close
        g['Prev_Close'] = g['Close'].shift(1)
        g['Gap_Pct'] = (g['Open'] - g['Prev_Close']) / g['Prev_Close']

        # RSI_14 (Shifted 1)
        rsi = ta.rsi(g['Close'], length=14)
        g['RSI_14'] = rsi.shift(1)

        # ATR_Pct (Shifted 1)
        atr = ta.atr(g['High'], g['Low'], g['Close'], length=14)
        g['ATR_Pct'] = (atr / g['Close']).shift(1)

        # Vol_Ratio (Volume / MA_Volume) (Shifted 1)
        vol_ma = ta.sma(g['Volume'], length=20)
        g['Vol_Ratio'] = (g['Volume'] / vol_ma).shift(1)

        # Dist_MA20 (Shifted 1)
        ma20 = ta.sma(g['Close'], length=20)
        g['Dist_MA20'] = ((g['Close'] - ma20) / ma20).shift(1)

        results.append(g)

    return pd.concat(results)

def calculate_crypto_corr(merged_df):
    print("Calculating Crypto Correlation...")

    results = []
    for ticker, group in merged_df.groupby('Ticker'):
        g = group.copy().sort_values('Date')

        # Stock Ret T-1
        g['Stock_Ret_T1'] = g['Prev_Close'].pct_change()

        # Rolling Corr
        g['Crypto_Corr'] = g['Stock_Ret_T1'].rolling(window=20).corr(g['BTC_Ret'])

        results.append(g)

    return pd.concat(results)

def main():
    # 1. Get Data
    stock_data = fetch_data(PURE_CRYPTO_POOL, START_DATE, END_DATE)
    btc_data = fetch_data([BTC_TICKER], START_DATE, END_DATE)

    if stock_data.empty or btc_data.empty:
        print("Failed to fetch data. Exiting.")
        return

    # 2. Process Features
    stock_features = process_stock_features(stock_data)
    btc_features = process_btc_features(btc_data)

    # 3. Merge
    merged = pd.merge(stock_features, btc_features, on='Date', how='left')

    # 4. Calculate Correlation (Requires both data)
    merged = calculate_crypto_corr(merged)

    # 5. Clean
    merged.dropna(inplace=True)

    print(f"Data ready. Shape: {merged.shape}")

    # 6. Train/Test Split
    train = merged[merged['Date'] < SPLIT_DATE]
    test = merged[merged['Date'] >= SPLIT_DATE]

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # 7. Experiment: Base vs Base+BTC
    experiments = {
        'Control_Base': BASE_FEATURES,
        'Test_Crypto': BASE_FEATURES + BTC_FEATURES
    }

    summary_metrics = []

    for name, features in experiments.items():
        print(f"\nRunning Experiment: {name}")
        X_train = train[features]
        y_train = train['Target']
        X_test = test[features]
        y_test = test['Target']

        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Save Model
        joblib.dump(model, os.path.join(OUTPUT_DIR, f"{name}_model.joblib"))

        # Evaluate
        test_df = test.copy()
        test_df['Pred'] = preds
        test_df['Prob'] = probs

        trades = test_df[test_df['Pred'] == 1]
        win_rate = len(trades[trades['Target'] == 1]) / len(trades) if len(trades) > 0 else 0
        avg_return = trades['Ret_Open_Close'].mean()
        total_return = trades['Ret_Open_Close'].sum()
        count = len(trades)

        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Return: {avg_return:.4%}")
        print(f"  Trade Count: {count}")

        summary_metrics.append({
            'Model': name,
            'Win Rate': win_rate,
            'Avg Return': avg_return,
            'Total Return': total_return,
            'Trade Count': count
        })

        # Feature Importance
        imp = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        imp.to_csv(os.path.join(OUTPUT_DIR, f"{name}_feature_importance.csv"), index=False)

    # Save Summary
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "performance_report.csv"), index=False)
    print("\nSummary:")
    print(summary_df)

if __name__ == "__main__":
    main()
