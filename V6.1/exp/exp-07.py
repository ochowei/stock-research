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
    all_tickers = tickers + ['^VIX']
    print(f"Downloading data for {len(all_tickers)} tickers...")
    
    # 下載數據
    data = yf.download(
        all_tickers, start=TRAIN_START, end=TEST_END, 
        interval='1d', auto_adjust=True, progress=True, threads=True
    )
    
    # 處理 MultiIndex Column (將 Ticker 轉為 Column)
    if isinstance(data.columns, pd.MultiIndex):
        # 嘗試堆疊數據 (相容不同 pandas 版本)
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        # 單一股票的情況 (通常不會發生，因為我們加了 ^VIX)
        data['Ticker'] = all_tickers[0]
        data = data.reset_index()

    # 強制將 Date 轉為 datetime 並正規化 (移除時區與時間)
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
    
    # 1. 確保數值型態 (防呆)
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 2. 合併 VIX (確保索引對齊)
    # 確保 df index 也是 normalized date
    df.index = pd.to_datetime(df.index).normalize()
    
    # 合併
    df = df.join(vix_df, how='left')
    
    # 填補 VIX 空值 (前後填充，若還是空則填 20)
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # 3. 基礎特徵 (T-1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    
    # 修正 pct_change 警告
    df['Ret_1d'] = df['Close'].pct_change(fill_method=None)
    
    # 4. 技術指標 (需處理運算錯誤)
    if len(df) < 15: return pd.DataFrame()
    
    try:
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    except Exception:
        # 若計算失敗，填入 NaN
        df['RSI_14'] = np.nan
        df['ATR_Pct'] = np.nan

    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)
    
    # 5. Gap 與 Target
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open'] # 做空
    
    # 6. Labeling
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100
    
    # 7. 最終清洗
    # 這裡我們只 drop 關鍵特徵缺失的行，而不是全部
    # Gap_Pct 和 RSI_14 必須要有值
    df = df.dropna(subset=['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret'])
    
    return df

def evaluate_performance(y_true, y_pred, returns):
    df = pd.DataFrame({'Label': y_true, 'Pred': y_pred, 'Return': returns})
    
    # Baseline: All Gaps
    base_win = (df['Return'] > 0).mean()
    base_avg = df['Return'].mean()
    base_tot = df['Return'].sum()
    
    # Model: Selected Gaps
    model_df = df[df['Pred'] == 1]
    if len(model_df) == 0:
        return 0, 0, 0, base_win, base_avg, base_tot
        
    mod_win = (model_df['Return'] > 0).mean()
    mod_avg = model_df['Return'].mean()
    mod_tot = model_df['Return'].sum()
    
    return mod_win, mod_avg, mod_tot, base_win, base_avg, base_tot

# --- 3. 主程式 ---

def main():
    print(f"=== EXP-V6.1-07: Next-Day Suitability Classifier (Robust) ===")
    
    tickers = load_tickers()
    stock_raw, vix_raw = fetch_data(tickers)
    
    print("\nBuilding features...")
    all_data = []
    
    # GroupBy Ticker 處理
    # 注意: stock_raw 是一個 DataFrame，包含了 Date, Ticker, OHLCV
    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        
        # 處理單一股票特徵
        feat_df = build_features(df, vix_raw)
        
        if feat_df.empty: continue
        
        feat_df['Ticker'] = ticker
        
        # 過濾訊號
        signal_df = feat_df[feat_df['Is_Signal']].copy()
        
        if not signal_df.empty:
            all_data.append(signal_df)
            
    if not all_data:
        print("[Error] No valid signals found! Check data quality or Gap Threshold.")
        return

    full_df = pd.concat(all_data).sort_index()
    print(f"Total Valid Gap Signals: {len(full_df)}")
    
    # 準備數據集
    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]
    
    print(f"Training Samples: {len(train_df)}")
    print(f"Testing Samples : {len(test_df)}")
    
    if len(train_df) < 50 or len(test_df) < 10:
        print("[Error] Not enough data for training/testing.")
        return

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
    
    # 特徵重要性
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n[Feature Importance]")
    print(imp)
    
    # 儲存
    joblib.dump(model, os.path.join(OUTPUT_DIR, 'exp_07_model.joblib'))
    
    # 繪圖
    test_df['Model_Pred'] = y_pred
    daily_base = test_df.groupby(test_df.index)['Strategy_Ret'].mean()
    daily_model = test_df[test_df['Model_Pred']==1].groupby(test_df[test_df['Model_Pred']==1].index)['Strategy_Ret'].mean()
    daily_model = daily_model.reindex(daily_base.index, fill_value=0)
    
    equity_base = (1 + daily_base).cumprod()
    equity_model = (1 + daily_model).cumprod()
    
    plt.figure(figsize=(10, 5))
    plt.plot(equity_base, label='Baseline', color='gray', alpha=0.5)
    plt.plot(equity_model, label='Model Filtered', color='red', linewidth=2)
    plt.title('Exp-07: Next-Day Suitability Classifier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp_07_equity.png'))
    print("\nChart saved.")

if __name__ == '__main__':
    main()