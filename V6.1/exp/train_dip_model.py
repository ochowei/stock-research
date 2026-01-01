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
GAP_THRESHOLD = -0.03      # Gap Down < -0.5%
PROFIT_THRESHOLD = 0.002    # 目標: 當日反彈 > 0.2%

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
    all_tickers = tickers + ['^VIX']
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

    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})
    stock_df = data[data['Ticker'] != '^VIX']
    
    return stock_df, vix_df

def build_features_strict(df, vix_df):
    """
    嚴格版特徵工程：只使用 T-1 數據與 T_0 Open
    """
    df = df.sort_index()
    
    # 數值轉換
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
            
    df = df.join(vix_df, how='left')
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # T-1 數據 (昨日)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_High'] = df['High'].shift(1)
    df['Prev_Low'] = df['Low'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    
    # --- 1. Gap (訊號源) ---
    # (Open - Prev_Close) / Prev_Close
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    
    # --- 2. 技術指標 (嚴格使用 T-1) ---
    if len(df) < 20: return pd.DataFrame()
    
    try:
        # RSI: 使用昨日收盤計算
        df['RSI_14'] = ta.rsi(df['Prev_Close'], length=14)
        
        # ATR: 使用昨日數據計算
        df['ATR_14'] = ta.atr(df['Prev_High'], df['Prev_Low'], df['Prev_Close'], length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
        
        # Dist_MA20 (乖離率): 模擬開盤當下的 MA20
        # 邏輯: 昨天的 MA19 * 19 + 今天的 Open => 除以 20
        df['MA20_Prev'] = df['Prev_Close'].rolling(19).mean() 
        df['MA20_Sim'] = (df['MA20_Prev'] * 19 + df['Open']) / 20
        # 開盤價相對於模擬均線的乖離
        df['Dist_MA20'] = (df['Open'] / df['MA20_Sim']) - 1
        
    except Exception:
        df['RSI_14'] = np.nan
        df['ATR_Pct'] = np.nan
        df['Dist_MA20'] = np.nan

    # 成交量濾網 (T-1)
    df['Vol_MA20'] = df['Prev_Vol'].rolling(20).mean()
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']
    
    # --- 3. 目標變數 (Target) ---
    # 策略: Buy Open -> Sell Close
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open'] 
    
    # --- 4. Labeling ---
    df['Is_Signal'] = df['Gap_Pct'] < GAP_THRESHOLD
    
    # Label: 成功反彈 (獲利 > 0.2%)
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    
    # Sample Weight: 依然根據波動加權
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100
    
    # 清洗
    df = df.dropna(subset=['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Dist_MA20', 'Strategy_Ret'])
    
    return df

def evaluate_performance(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})
    
    base_win = (df['Return'] > 0).mean()
    base_avg = df['Return'].mean()
    
    model_df = df[df['Pred'] == 1]
    if len(model_df) == 0:
        return 0, 0, base_win, base_avg
        
    mod_win = (model_df['Return'] > 0).mean()
    mod_avg = model_df['Return'].mean()
    
    return mod_win, mod_avg, base_win, base_avg

# --- 3. 主程式 ---

def main():
    print(f"=== Train CORRECTED Buy Dip Model (No Leakage) ===")
    
    tickers = load_tickers()
    if not tickers:
        print("[Error] No tickers found.")
        return

    stock_raw, vix_raw = fetch_data(tickers)
    
    print("\nBuilding features (Strict T-1 & Open)...")
    all_data = []
    
    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features_strict(df, vix_raw)
        
        if feat_df.empty: continue
        feat_df['Ticker'] = ticker
        
        # 篩選 Gap Down 訊號
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)
            
    if not all_data:
        print("[Error] No valid signals found!")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Dip Signals: {len(full_df)}")
    
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]
    
    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")
    
    # 特徵列表 (完全對應 Backtest 用的特徵)
    features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Dist_MA20']
    
    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']
    
    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']
    
    print("\nTraining XGBoost Model...")
    model = XGBClassifier(
        n_estimators=300,        # 增加樹的數量
        learning_rate=0.02,      # 降低學習率以增加穩健性
        max_depth=3,             # 保持淺層樹防止過擬合
        subsample=0.7, 
        colsample_bytree=0.7,
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    
    y_pred = model.predict(X_test)
    
    m_win, m_avg, b_win, b_avg = evaluate_performance(y_test, y_pred, r_test)
    
    print("\n" + "="*60)
    print("CORRECTED MODEL RESULTS (OOS 2024-2025)")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':<20} {'Model':<20} {'Diff':<10}")
    print("-" * 75)
    print(f"{'Win Rate':<20} {b_win*100:6.2f}%              {m_win*100:6.2f}%              {m_win-b_win:+.2%}")
    print(f"{'Avg Return':<20} {b_avg*100:6.3f}%              {m_avg*100:6.3f}%              {m_avg-b_avg:+.3%}")
    print("-" * 75)
    
    # 儲存模型
    model_path = os.path.join(OUTPUT_DIR, 'dip_model.joblib')
    joblib.dump(model, model_path)
    print(f"\n[Saved] Model saved to: {model_path}")
    print("[Next Step] Please run 'backtest_dip_model.py' again.")

if __name__ == '__main__':
    main()