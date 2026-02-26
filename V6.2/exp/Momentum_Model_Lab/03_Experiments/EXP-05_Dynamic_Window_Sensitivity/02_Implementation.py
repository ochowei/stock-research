import os
import sys
import json
import time
import random
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Setup & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../../../'))
sys.path.append(ROOT_DIR)

RESOURCE_DIR = os.path.join(ROOT_DIR, 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Parameters ---
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

GAP_THRESHOLD = 0.005
PROFIT_THRESHOLD = 0.002
WINDOWS = [5, 10, 14, 20, 50]

# --- 3. Utilities ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    tickers = sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))
    return tickers

def fetch_data_batched(tickers, batch_size=15):
    all_tickers = tickers + ['^VIX', 'QQQ', 'SPY']
    all_tickers = sorted(list(set(all_tickers)))

    print(f"Downloading data for {len(all_tickers)} tickers in batches of {batch_size}...")

    combined_df = pd.DataFrame()
    failed_tickers = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        print(f"  Batch {i//batch_size + 1}: {batch}")

        try:
            time.sleep(random.uniform(1, 3))

            data = yf.download(
                batch, start=TRAIN_START, end=TEST_END,
                interval='1d', auto_adjust=True, progress=False, threads=True
            )

            if data.empty:
                print(f"    [Warning] Empty data for batch {batch}")
                failed_tickers.extend(batch)
                continue

            if isinstance(data.columns, pd.MultiIndex):
                try:
                    data = data.stack(level=1, future_stack=True)
                except TypeError:
                    data = data.stack(level=1)

                data = data.rename_axis(['Date', 'Ticker']).reset_index()
            else:
                if len(batch) == 1:
                    data['Ticker'] = batch[0]
                    data = data.reset_index()
                else:
                    print("    [Error] Unexpected data format for batch")
                    continue

            combined_df = pd.concat([combined_df, data], ignore_index=True)

        except Exception as e:
            print(f"    [Error] Batch failed: {e}")
            failed_tickers.extend(batch)

    if combined_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    combined_df['Date'] = pd.to_datetime(combined_df['Date']).dt.tz_localize(None).dt.normalize()

    vix_df = combined_df[combined_df['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    stock_df = combined_df[combined_df['Ticker'] != '^VIX']

    vix_df = vix_df.resample('D').ffill()

    print(f"  - Stock Data Rows: {len(stock_df)}")
    print(f"  - VIX Data Rows: {len(vix_df)}")

    return stock_df, vix_df

def build_features(df, vix_df, window):
    df = df.sort_index()

    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.join(vix_df, how='left')
    df['VIX'] = df['VIX'].ffill().bfill()

    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    try:
        rsi_col = f'RSI_{window}'
        df[rsi_col] = ta.rsi(df['Close'], length=window)
        df[rsi_col] = df[rsi_col].shift(1)

        atr_col = f'ATR_{window}'
        df[atr_col] = ta.atr(df['High'], df['Low'], df['Close'], length=window)
        df[atr_col] = df[atr_col].shift(1)

        df[f'ATR_Pct_{window}'] = df[atr_col] / df['Prev_Close']

        roc_col = f'ROC_{window}'
        df[roc_col] = df['Close'].pct_change(periods=window)
        df[roc_col] = df[roc_col].shift(1)

    except Exception:
        return pd.DataFrame()

    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_MA20_Prev'] = df['Vol_MA20'].shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20_Prev']

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open']

    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

    df['VIX'] = df['VIX'].shift(1)

    feat_cols = [f'RSI_{window}', f'ATR_Pct_{window}', f'ROC_{window}', 'Vol_Ratio', 'Gap_Pct', 'VIX']
    df = df.dropna(subset=feat_cols + ['Strategy_Ret', 'Is_Signal'])

    return df

def run_experiment_for_window(stock_df, vix_df, window):
    print(f"\n--- Processing Window: {window} ---")

    all_data = []
    for ticker, group in stock_df.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_df, window)

        if not feat_df.empty:
            signal_df = feat_df[feat_df['Is_Signal']].copy()
            if not signal_df.empty:
                signal_df['Ticker'] = ticker
                all_data.append(signal_df)

    if not all_data:
        print(f"[Warning] No signals for Window {window}")
        return None

    full_df = pd.concat(all_data).sort_index()

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    if len(train_df) < 50 or len(test_df) < 10:
        print("[Warning] Insufficient data")
        return None

    features = [f'RSI_{window}', f'ATR_Pct_{window}', f'ROC_{window}', 'Vol_Ratio', 'Gap_Pct', 'VIX']

    X_train = train_df[features]
    y_train = train_df['Label']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    model = XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        n_jobs=-1, random_state=42, eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    base_win = (r_test > 0).mean()
    base_avg = r_test.mean()

    model_trades = test_df[y_pred == 1]
    if len(model_trades) > 0:
        mod_win = (model_trades['Strategy_Ret'] > 0).mean()
        mod_avg = model_trades['Strategy_Ret'].mean()
        count = len(model_trades)
    else:
        mod_win = 0
        mod_avg = 0
        count = 0

    print(f"Window {window} Result: Win={mod_win:.1%}, Avg={mod_avg:.3%}, Count={count}")

    daily_ret = model_trades.groupby(model_trades.index)['Strategy_Ret'].mean()
    daily_ret = daily_ret.reindex(pd.date_range(start=TEST_START, end=TEST_END), fill_value=0)

    return {
        'window': window,
        'win_rate': mod_win,
        'avg_return': mod_avg,
        'count': count,
        'baseline_win': base_win,
        'baseline_avg': base_avg,
        'daily_returns': daily_ret
    }

def main():
    print("=== EXP-05: Dynamic Window Sensitivity ===")

    tickers = load_tickers()
    stock_raw, vix_raw = fetch_data_batched(tickers)

    if stock_raw.empty:
        print("[Error] Failed to fetch data.")
        return

    results = []
    equity_curves = pd.DataFrame()

    for w in WINDOWS:
        res = run_experiment_for_window(stock_raw, vix_raw, w)
        if res:
            results.append(res)
            equity_curves[f'W_{w}'] = (1 + res['daily_returns']).cumprod()

    res_df = pd.DataFrame(results)
    res_path = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
    res_df.to_csv(res_path, index=False)
    print(f"\n[Saved] Results to {res_path}")

    plt.figure(figsize=(12, 6))
    for col in equity_curves.columns:
        plt.plot(equity_curves.index, equity_curves[col], label=col)

    plt.title('Equity Curve by Lookback Window (2024-2025)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, 'window_comparison.png')
    plt.savefig(plot_path)
    print(f"[Saved] Plot to {plot_path}")

if __name__ == '__main__':
    main()
