import os
import sys
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
# 資源目錄在 V6.2/resource，相對路徑 ../../../../resource
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add V6.2/exp to sys.path
EXP_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
sys.path.append(EXP_DIR)

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
    # Extract ticker from "Sector: Ticker" format
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    all_tickers = tickers + ['^VIX']
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # 下載數據
    try:
        data = yf.download(
            all_tickers, start=TRAIN_START, end=TEST_END,
            interval='1d', auto_adjust=True, progress=False, threads=True
        )
    except Exception as e:
        print(f"[Error] yfinance download failed: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if data.empty:
        print("[Error] No data downloaded.")
        return pd.DataFrame(), pd.DataFrame()

    # 處理 MultiIndex Column
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        # 單一股票的情況
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
    """特徵工程 (針對 Momentum 優化 - With Volume Features)"""
    df = df.sort_index()

    # 1. 確保數值型態
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. 合併 VIX
    df = df.join(vix_df, how='left')
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # Fix Data Leak: Shift VIX to ensure we use T-1 value for decision at Open T
    df['VIX'] = df['VIX'].shift(1)

    # 3. 基礎特徵 (T-1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    # 4. 技術指標
    if len(df) < 25: return pd.DataFrame()

    try:
        close_series = df['Close'].astype(float)
        high_series = df['High'].astype(float)
        low_series = df['Low'].astype(float)
        vol_series = df['Volume'].astype(float)

        # Basic Indicators
        df['RSI_14'] = ta.rsi(close_series, length=14)
        df['ATR_14'] = ta.atr(high_series, low_series, close_series, length=14)

        # Fix Data Leak: Shift indicators to avoid using current day's Close
        df['RSI_14'] = df['RSI_14'].shift(1)
        # Use ATR from T-1 normalized by Close T-1 (Prev_Close)
        df['ATR_Pct'] = df['ATR_14'].shift(1) / df['Prev_Close']

        # Volume Features
        df['Vol_MA20'] = vol_series.rolling(20).mean()
        # Vol_Ratio: Pre-Gap Volume / Avg Volume (T-1 based)
        # Note: Pre-Gap Volume is simply Prev_Vol.
        # Vol_MA20 at T-1 needs to be calculated on T-1.
        # df['Vol_MA20'].shift(1) gives MA20 calculated at T-1.
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)

        # [NEW] Vol_MA5_Slope
        # We want the trend of volume leading up to the gap (T-1).
        df['Vol_MA5'] = vol_series.rolling(5).mean()
        # Slope over 5 days ending at T-1: (MA5[T-1] - MA5[T-6]) / MA5[T-6]
        df['Vol_MA5_Slope'] = (df['Vol_MA5'].shift(1) - df['Vol_MA5'].shift(6)) / df['Vol_MA5'].shift(6)

    except Exception as e:
        # print(f"Error calculating indicators: {e}")
        return pd.DataFrame() # Return empty on failure to ensure data quality

    # 5. Gap 與 Target
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open']

    # 6. Labeling
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # 7. 最終清洗
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows where we don't have enough history or features are NaN
    cols_to_check = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Vol_MA5_Slope', 'Strategy_Ret']
    df = df.dropna(subset=cols_to_check)

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

# --- 3. 主程式 ---

def main():
    print(f"=== EXP-03: Volume Microstructure (False Breakout Filter) ===")

    tickers = load_tickers()
    if not tickers:
        print("[Error] No tickers found.")
        return

    stock_raw, vix_raw = fetch_data(tickers)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw)

        if feat_df.empty: continue
        feat_df['Ticker'] = ticker

        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Gap Signals: {len(full_df)}")

    # Split Data
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    if len(train_df) < 50:
        print("[Error] Not enough training data.")
        return

    # [UPDATED] Feature Set
    features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Vol_MA5_Slope', 'Gap_Pct', 'VIX']

    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    print("\nTraining Momentum XGBoost Model...")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    # Evaluate
    m_win, m_avg, m_tot, b_win, b_avg, b_tot = evaluate_performance(y_test, y_pred, r_test)

    print("\n" + "="*60)
    print("MOMENTUM MODEL RESULTS (EXP-03: Volume Slope)")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<20} {'Model (Filter)':<20} {'Diff':<10}")
    print("-" * 75)
    print(f"{'Count':<20} {len(y_test):<20} {sum(y_pred):<20}")
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {m_win*100:6.2f}%              {m_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {m_avg*100:6.3f}%              {m_avg-b_avg:+.3%}")
    print("-" * 75)

    # Save Report
    report = {
        'Metric': ['Win Rate', 'Avg Return', 'Total Return', 'Count'],
        'Baseline': [b_win, b_avg, b_tot, len(y_test)],
        'Model': [m_win, m_avg, m_tot, int(sum(y_pred))],
        'Diff': [m_win-b_win, m_avg-b_avg, m_tot-b_tot, int(sum(y_pred))-len(y_test)]
    }
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Feature Importance
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n[Feature Importance]")
    print(imp)

    plt.figure(figsize=(10, 6))
    imp.plot(kind='barh')
    plt.title('Feature Importance (EXP-03)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()

    # Save Model
    model_path = os.path.join(OUTPUT_DIR, 'momentum_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n[Saved] Model saved to: {model_path}")

if __name__ == '__main__':
    main()
