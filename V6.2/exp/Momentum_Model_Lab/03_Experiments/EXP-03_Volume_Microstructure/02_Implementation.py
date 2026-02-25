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
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path handling:
# BASE_DIR is .../V6.2/exp/Momentum_Model_Lab/03_Experiments/EXP-03_Volume_Microstructure
# We want to add V6.2/exp to sys.path to import daily_gap_signal_generator_v6_2_6_rc
# Level 1 up: 03_Experiments, 2: Momentum_Model_Lab, 3: exp
EXP_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../..'))
if EXP_DIR not in sys.path:
    sys.path.append(EXP_DIR)

# V6.2 Root for resources (one level up from exp)
V6_2_ROOT = os.path.abspath(os.path.join(EXP_DIR, '..'))

print(f"Added {EXP_DIR} to sys.path")
print(f"V6.2 Root: {V6_2_ROOT}")

try:
    # Attempt to import to satisfy the requirement
    import daily_gap_signal_generator_v6_2_6_rc as signal_gen
    print("Successfully imported daily_gap_signal_generator_v6_2_6_rc")
except ImportError as e:
    print(f"Warning: Could not import daily_gap_signal_generator_v6_2_6_rc: {e}")
    # Try importing the base name if versioned one fails
    try:
        import daily_gap_signal_generator
        print("Successfully imported daily_gap_signal_generator")
    except ImportError:
        pass


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
    # Resource is at V6.2/resource/2025_final_asset_pool.json
    path = os.path.join(V6_2_ROOT, 'resource', '2025_final_asset_pool.json')

    if not os.path.exists(path):
        print(f"[Error] File not found: {path}")
        return []

    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    all_tickers = tickers + ['^VIX']
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # 下載數據
    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END,
        interval='1d', auto_adjust=True, progress=False, threads=True
    )

    # 處理 MultiIndex Column (將 Ticker 轉為 Column)
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

    # 填補 VIX 空值
    vix_df = vix_df.resample('D').ffill()

    return stock_df, vix_df

def build_features(df, vix_df):
    """特徵工程 (EXP-03 Enhanced)"""
    df = df.sort_index()

    # 1. 確保數值型態
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. 合併 VIX
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(vix_df, how='left')
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # 3. 基礎特徵 (T-1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    # 4. 技術指標
    if len(df) < 25: return pd.DataFrame()

    try:
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

        # [EXP-03] Volume Microstructure Features
        df['Vol_MA20'] = df['Volume'].rolling(20).mean()
        df['Vol_MA5']  = df['Volume'].rolling(5).mean()

        # Vol_Ratio: Today's (Prev Day for Open) Volume vs 20D Avg
        # Note: We use shift(1) because we are predicting at Open, so we only know yesterday's volume.
        # Vol_MA20[T-1] is the MA of Volume up to T-1.
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)

        # Vol_MA5_Slope: (MA5[T-1] - MA5[T-2]) / MA5[T-2]
        # Slope leading into the gap day
        df['Vol_MA5_Slope'] = (df['Vol_MA5'].shift(1) - df['Vol_MA5'].shift(2)) / df['Vol_MA5'].shift(2)

    except Exception as e:
        # print(f"Error calculating indicators: {e}")
        return pd.DataFrame()

    # 5. Gap 與 Target
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open']

    # 6. Labeling
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # 7. 最終清洗
    features_to_check = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret', 'Vol_Ratio', 'Vol_MA5_Slope']
    df = df.dropna(subset=features_to_check)

    return df

def train_and_evaluate(train_df, test_df, features, model_name="Model"):
    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    # Evaluate
    df = pd.DataFrame({'Label': y_test, 'Pred': y_pred, 'Return': r_test})
    model_trades = df[df['Pred'] == 1]

    if len(model_trades) == 0:
        return 0, 0, 0, model

    win_rate = (model_trades['Return'] > 0).mean()
    avg_ret = model_trades['Return'].mean()
    total_ret = model_trades['Return'].sum()

    return win_rate, avg_ret, total_ret, model

# --- 3. 主程式 ---

def main():
    print(f"=== EXP-03: Volume Microstructure (False Breakout Filter) ===")

    tickers = load_tickers()
    if not tickers:
        print("[Error] No tickers found in asset pool.")
        return

    stock_raw, vix_raw = fetch_data(tickers)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw)

        if feat_df.empty: continue
        feat_df['Ticker'] = ticker

        # 過濾訊號 (Gap Up)
        signal_df = feat_df[feat_df['Is_Signal']].copy()

        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Gap Signals Found: {len(full_df)}")

    # Split Data
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    if len(train_df) < 50:
        print("[Error] Not enough training data.")
        return

    # --- Baseline Model ---
    print("\nTraining Baseline Model (V6.1 Parity)...")
    base_features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX']
    b_win, b_avg, b_tot, base_model = train_and_evaluate(train_df, test_df, base_features, "Baseline")

    # --- EXP-03 Model ---
    print("\nTraining EXP-03 Model (Volume Microstructure)...")
    exp_features = base_features + ['Vol_MA5_Slope']
    # Note: Vol_Ratio is already in baseline, but we ensure it's calculated correctly.
    # We added Vol_MA5_Slope as the new feature.

    e_win, e_avg, e_tot, exp_model = train_and_evaluate(train_df, test_df, exp_features, "EXP-03")

    # --- Comparison ---
    print("\n" + "="*80)
    print("EXP-03 RESULTS COMPARISON (OOS 2024-2025)")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline':<20} {'EXP-03':<20} {'Diff':<10}")
    print("-" * 75)
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {e_win*100:6.2f}%              {e_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {e_avg*100:6.3f}%              {e_avg-b_avg:+.3%}")
    print(f"{'Total Return':<20} {b_tot*100:6.2f}%              {e_tot*100:6.2f}%              {e_tot-b_tot:+.2%}")
    print("-" * 75)

    # Feature Importance for EXP-03
    imp = pd.Series(exp_model.feature_importances_, index=exp_features).sort_values(ascending=False)
    print("\n[EXP-03 Feature Importance]")
    print(imp)

    # Save Artifacts
    joblib.dump(exp_model, os.path.join(OUTPUT_DIR, 'exp03_model.joblib'))
    joblib.dump(base_model, os.path.join(OUTPUT_DIR, 'baseline_model.joblib'))

    # Save Performance Report
    report = pd.DataFrame({
        'Metric': ['Win Rate', 'Avg Return', 'Total Return'],
        'Baseline': [b_win, b_avg, b_tot],
        'EXP-03': [e_win, e_avg, e_tot],
        'Diff': [e_win-b_win, e_avg-b_avg, e_tot-b_tot]
    })
    report.to_csv(os.path.join(OUTPUT_DIR, 'performance_report.csv'), index=False)

    # Plot Feature Importance
    plt.figure(figsize=(10, 6))
    imp.plot(kind='barh')
    plt.title('EXP-03 Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    print("\nSaved artifacts to 03_Output/")

if __name__ == '__main__':
    main()
