import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib

# --- 1. 設定與參數 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# V6.2/exp/ -> V6.2/resource/
RESOURCE_DIR = os.path.join(BASE_DIR, '..', 'resource')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 資料區間
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# 策略參數
GAP_THRESHOLD = 0.005      # 0.5% 跳空門檻
PROFIT_THRESHOLD = 0.002   # 0.2% 獲利門檻

# --- 2. 工具函數 ---

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    # raw is a list of "EXCHANGE:TICKER" strings
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    all_tickers = tickers + ['^VIX']
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # 下載數據
    # Note: yfinance return MultiIndex columns if len(tickers) > 1
    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END,
        interval='1d', auto_adjust=True, progress=True, threads=True
    )

    # 處理 MultiIndex Column (將 Ticker 轉為 Column)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            # New pandas structure might require different handling,
            # but yfinance typically returns (Price, Ticker)
            # stack(level=1) moves Ticker to index
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)

        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        # If only one ticker (unlikely given the list), add Ticker column
        data['Ticker'] = all_tickers[0]
        data = data.reset_index()

    # 強制將 Date 轉為 datetime 並正規化
    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # 分離 VIX
    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    stock_df = data[data['Ticker'] != '^VIX']

    print(f"  - Stock Data Rows: {len(stock_df)}")
    print(f"  - VIX Data Rows: {len(vix_df)}")

    if len(vix_df) == 0:
        print("[Warning] VIX data is empty! Feature 'VIX' will be NaN.")

    return stock_df, vix_df

def build_features(df, vix_df):
    """特徵工程"""
    df = df.sort_index()

    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 合併 VIX
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(vix_df, how='left')
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # 3. 基礎特徵 (T-1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    # 4. 技術指標
    if len(df) < 20: return pd.DataFrame() # Increased min length check

    try:
        # [Corrected] Use T-1 for indicators to avoid look-ahead bias
        df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    except Exception:
        df['RSI_14'] = np.nan
        df['ATR_Pct'] = np.nan

    # [Corrected] Vol MA also needs to be shifted to use T-1
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    # 5. Gap 與 Target
    # Gap: (Open - Prev_Close) / Prev_Close
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Strategy Return: Sell at Open, Buy at Close -> (Open - Close) / Open
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']

    # 6. Labeling
    # Signal: Gap > 0.5%
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD

    # Label: Strategy_Ret > 0.2%
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

    # Sample Weight: abs(Strategy_Ret) * 100
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # 7. 清洗
    df = df.dropna(subset=['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret', 'Vol_Ratio'])

    return df

def evaluate_performance(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})

    # Baseline: All Gaps
    base_win = (df['Return'] > 0).mean()
    base_avg = df['Return'].mean()
    base_tot = df['Return'].sum()

    # Model: Selected Gaps (Pred == 1)
    model_df = df[df['Pred'] == 1]
    if len(model_df) == 0:
        return 0, 0, 0, base_win, base_avg, base_tot

    mod_win = (model_df['Return'] > 0).mean()
    mod_avg = model_df['Return'].mean()
    mod_tot = model_df['Return'].sum()

    return mod_win, mod_avg, mod_tot, base_win, base_avg, base_tot

# --- 3. 主程式 ---

def main():
    print(f"=== EXP-V6.2-07: Reproduction Baseline (Corrected) ===")

    tickers = load_tickers()
    stock_raw, vix_raw = fetch_data(tickers)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw)

        if feat_df.empty: continue

        feat_df['Ticker'] = ticker

        # 只取符合訊號的行 (Gap > 0.5%)
        signal_df = feat_df[feat_df['Is_Signal']].copy()

        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Gap Signals: {len(full_df)}")

    # 準備數據集
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)} ({TRAIN_START} to {TRAIN_END})")
    print(f"Testing Samples : {len(test_df)} ({TEST_START} to {TEST_END})")

    features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX']

    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    # 評估
    m_win, m_avg, m_tot, b_win, b_avg, b_tot = evaluate_performance(y_test, y_pred, r_test)

    print("\n" + "="*60)
    print("RESULTS COMPARISON (OOS 2024-2025)")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline (All)':<20} {'Model (Filter)':<20} {'Diff':<10}")
    print("-" * 75)
    print(f"{'Count':<20} {len(y_test):<20} {sum(y_pred):<20}")
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {m_win*100:6.2f}%              {m_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {m_avg*100:6.3f}%              {m_avg-b_avg:+.3%}")
    print("-" * 75)

    # Relaxed check because correction might lower score
    if abs(m_win - 0.5997) > 0.05:
         print("\n[NOTE] Deviation from original baseline is expected due to leakage fix.")

    # 儲存
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_07_repro_model.joblib'))

if __name__ == '__main__':
    main()
