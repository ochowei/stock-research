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
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    # 加入 QQQ, SPY 作為 Benchmark
    benchmarks = ['QQQ', 'SPY', '^VIX']
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
        data['Ticker'] = all_tickers[0]
        data = data.reset_index()

    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # 分離 Benchmarks
    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').copy()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').copy()

    # 計算 Benchmark Gaps
    for df, name in [(qqq_df, 'QQQ'), (spy_df, 'SPY')]:
        df['Prev_Close'] = df['Close'].shift(1)
        df[f'{name}_Gap'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # 保留需要的欄位
    qqq_gap = qqq_df[[f'QQQ_Gap']]
    spy_gap = spy_df[[f'SPY_Gap']]

    stock_df = data[~data['Ticker'].isin(benchmarks)]

    print(f"  - Stock Data Rows: {len(stock_df)}")

    return stock_df, vix_df, qqq_gap, spy_gap

def calculate_totm_features(dates):
    """計算 TOTM (Time of The Month) 特徵"""
    # 建立一個只包含交易日的 DataFrame
    dates = sorted(list(set(dates)))
    df = pd.DataFrame({'Date': dates})
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # 計算每個月的交易日排序
    # Days_From_Start: 0, 1, 2...
    df['Days_From_Start'] = df.groupby(['Year', 'Month']).cumcount()

    # Days_To_End: 0, 1, 2... (倒數)
    df['Days_To_End'] = df.groupby(['Year', 'Month'])['Date'].transform('count') - df['Days_From_Start'] - 1

    return df.set_index('Date')[['Days_From_Start', 'Days_To_End']]

def build_features(df, vix_df, qqq_gap, spy_gap, totm_df):
    """特徵工程 (Enhanced)"""
    df = df.sort_index()

    # 1. 數值轉換
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. 合併外部數據
    df.index = pd.to_datetime(df.index).normalize()
    df = df.join(vix_df, how='left')
    df = df.join(qqq_gap, how='left')
    df = df.join(spy_gap, how='left')
    df = df.join(totm_df, how='left')

    # 填補 VIX
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # 3. 基礎特徵
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 20: return pd.DataFrame()

    # 4. 原始特徵
    try:
        # [Corrected] Use T-1 for indicators to avoid look-ahead bias
        df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    except Exception:
        df['RSI_14'] = np.nan
        df['ATR_Pct'] = np.nan

    # [Corrected] Shift Volume MA
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # --- 新增特徵 ---

    # A. Dist_MA20
    # 模擬當前 MA20: (Sum(Close[T-19...T-1]) + Open) / 20
    # Close.rolling(19).sum().shift(1) 是 T-1 到 T-19 的總和
    sum_prev_19 = df['Close'].rolling(19).sum().shift(1)
    ma20_sim = (sum_prev_19 + df['Open']) / 20
    df['Dist_MA20'] = (df['Open'] / ma20_sim) - 1

    # B. Relative Strength
    df['Rel_Gap_QQQ'] = df['Gap_Pct'] - df['QQQ_Gap']
    df['Rel_Gap_SPY'] = df['Gap_Pct'] - df['SPY_Gap']

    # C. TOTM (已在外部計算並 join)
    # Days_From_Start, Days_To_End 已經有了

    # --- Labeling ---
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # 清洗
    req_cols = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret',
                'Dist_MA20', 'Rel_Gap_QQQ', 'Rel_Gap_SPY', 'Days_From_Start']
    df = df.dropna(subset=req_cols)

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
    print(f"=== EXP-V6.2-07 v2: Optimized Training (Corrected) ===")

    tickers = load_tickers()
    stock_raw, vix_raw, qqq_raw, spy_raw = fetch_data(tickers)

    # 計算全域 TOTM
    all_dates = stock_raw['Date'].unique()
    totm_df = calculate_totm_features(all_dates)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw, qqq_raw, spy_raw, totm_df)

        if feat_df.empty: continue
        feat_df['Ticker'] = ticker

        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Gap Signals: {len(full_df)}")

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")

    # 新特徵列表
    features = [
        'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX',
        'Dist_MA20', 'Rel_Gap_QQQ', 'Rel_Gap_SPY',
        'Days_From_Start', 'Days_To_End'
    ]

    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    print(f"\nTraining XGBoost with {len(features)} features...")
    model = XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    y_pred = model.predict(X_test)

    m_win, m_avg, m_tot, b_win, b_avg, b_tot = evaluate_performance(y_test, y_pred, r_test)

    print("\n" + "="*80)
    print("OPTIMIZED RESULTS COMPARISON (OOS 2024-2025)")
    print("="*80)
    print(f"{'Metric':<20} {'Baseline (All)':<20} {'Model (Filter)':<20} {'Diff':<10}")
    print("-" * 80)
    print(f"{'Count':<20} {len(y_test):<20} {sum(y_pred):<20}")
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {m_win*100:6.2f}%              {m_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {m_avg*100:6.3f}%              {m_avg-b_avg:+.3%}")
    print(f"{'Total Return':<20} {b_tot*100:6.1f}%              {m_tot*100:6.1f}%")
    print("-" * 80)

    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n[Feature Importance]")
    print(imp)

    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_07_v2_model.joblib'))
    print(f"\nModel saved to {os.path.join(OUTPUT_DIR, 'exp_07_v2_model.joblib')}")

if __name__ == '__main__':
    main()
