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

# Path to Sector Map
SECTOR_MAP_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'Sell_Model_Lab', '03_Experiments', 'EXP_18_Production_Script_Update', '03_Output', 'sector_map.json'))

# 資料區間
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# 策略參數
GAP_THRESHOLD = 0.005      # 0.5% 跳空門檻
PROFIT_THRESHOLD = 0.002   # 0.2% 獲利門檻

# Sector ETF Mapping (SPDR)
ETF_MAP = {
    "Technology": "XLK",
    "Consumer Cyclical": "XLY",
    "Communication Services": "XLC",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB"
}
DEFAULT_ETF = "SPY"

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

def load_sector_map():
    if not os.path.exists(SECTOR_MAP_PATH):
        print(f"[Warning] Sector map not found at {SECTOR_MAP_PATH}")
        return {}
    with open(SECTOR_MAP_PATH, 'r') as f:
        return json.load(f)

def fetch_data(stock_tickers):
    # Collect all unique ETFs needed
    etf_tickers = list(set(ETF_MAP.values())) + [DEFAULT_ETF]
    all_tickers = stock_tickers + etf_tickers + ['^VIX']

    # Remove duplicates
    all_tickers = sorted(list(set(all_tickers)))

    print(f"Downloading data for {len(all_tickers)} tickers (Stocks + ETFs + VIX)...")

    # 下載數據 (Try with threads=False to avoid timeouts)
    try:
        data = yf.download(
            all_tickers, start=TRAIN_START, end=TEST_END,
            interval='1d', auto_adjust=True, progress=True, threads=False
        )
    except Exception as e:
        print(f"[Error] yfinance download failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if data.empty:
        print("[Error] No data downloaded.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Check for critical ETFs
    missing_etfs = []
    # If MultiIndex, level 1 is Ticker
    if isinstance(data.columns, pd.MultiIndex):
        downloaded_tickers = data.columns.get_level_values(1).unique()
    else:
        # Single Ticker? Unlikely with list input
        downloaded_tickers = [all_tickers[0]]

    for etf in etf_tickers:
        if etf not in downloaded_tickers:
            missing_etfs.append(etf)

    # Determine the column structure of the main data
    if isinstance(data.columns, pd.MultiIndex):
        # Usually (Price, Ticker)
        pass
    else:
        # If single ticker, convert to MultiIndex (Price, Ticker)
        # But wait, we don't know the ticker here easily if passed as list?
        # Actually if only 1 ticker in list, yfinance returns simple columns.
        # But we passed a list of 96 tickers.
        pass

    # Simplified Logic:
    # We will process `data` as usual.
    # If ETFs are missing, we download them, convert to (Price, Ticker) format, and concat.

    if missing_etfs:
        print(f"[Warning] Missing ETFs: {missing_etfs}. Retrying individually...")
        extra_dfs = []
        for etf in missing_etfs:
            try:
                print(f"  - Retrying {etf}...")
                etf_data = yf.download(etf, start=TRAIN_START, end=TEST_END, interval='1d', auto_adjust=True, progress=False, threads=False)
                if not etf_data.empty:
                    # Convert to MultiIndex (Price, Ticker)
                    # etf_data columns: Open, High, ...
                    etf_data.columns = pd.MultiIndex.from_product([etf_data.columns, [etf]])
                    extra_dfs.append(etf_data)
            except Exception as e:
                print(f"  - Failed to download {etf}: {e}")

        if extra_dfs:
            # Concat along axis 1
            extra_df = pd.concat(extra_dfs, axis=1)
            data = pd.concat([data, extra_df], axis=1)

    # 處理 MultiIndex Column (將 Ticker 轉為 Column)
    if isinstance(data.columns, pd.MultiIndex):
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

    # 分離 VIX, ETFs, Stocks
    vix_df = data[data['Ticker'] == '^VIX'].set_index('Date')[['Close']].rename(columns={'Close': 'VIX'})

    etf_df = data[data['Ticker'].isin(etf_tickers)].set_index(['Date', 'Ticker']).sort_index()

    stock_df = data[~data['Ticker'].isin(etf_tickers + ['^VIX'])]

    print(f"  - Stock Data Rows: {len(stock_df)}")
    print(f"  - ETF Data Rows: {len(etf_df)}")
    print(f"  - VIX Data Rows: {len(vix_df)}")

    if len(vix_df) == 0:
        print("[Warning] VIX data is empty! Feature 'VIX' will be NaN.")

    return stock_df, etf_df, vix_df

def calculate_rsi(series, length=14):
    return ta.rsi(series, length=length)

def prepare_etf_indicators(etf_df):
    """Pre-calculate RSI for all ETFs to avoid re-calculating inside the loop."""
    etf_indicators = {}

    # etf_df is indexed by [Date, Ticker] or just Date if filtered.
    # It's better to pivot or iterate by ticker.
    # Let's pivot to have columns as Tickers for Close price

    # Reset index to work with pivot
    df = etf_df.reset_index()
    pivot_close = df.pivot(index='Date', columns='Ticker', values='Close')

    for ticker in pivot_close.columns:
        series = pivot_close[ticker].dropna()
        if len(series) > 15:
            rsi = calculate_rsi(series)
            etf_indicators[ticker] = rsi

    return etf_indicators

def build_features(df, vix_df, etf_indicators, ticker, sector_map):
    """特徵工程 (包含 Sector Features)"""
    df = df.sort_index()

    # 1. 確保數值型態
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. 合併 VIX
    df = df.join(vix_df, how='left')
    df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0)

    # 3. 基礎特徵 (T-1)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['Ret_1d'] = df['Close'].pct_change(fill_method=None)

    # 4. 技術指標
    if len(df) < 15: return pd.DataFrame()

    try:
        close_series = df['Close'].astype(float)
        high_series = df['High'].astype(float)
        low_series = df['Low'].astype(float)

        df['RSI_14'] = ta.rsi(close_series, length=14)
        df['ATR_14'] = ta.atr(high_series, low_series, close_series, length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    except Exception as e:
        df['RSI_14'] = np.nan
        df['ATR_Pct'] = np.nan

    df['Vol_MA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20'].shift(1)

    # 5. [NEW] Sector Features
    sector_name = sector_map.get(ticker, 'Unknown')
    etf_ticker = ETF_MAP.get(sector_name, DEFAULT_ETF)

    if etf_ticker in etf_indicators:
        sector_rsi = etf_indicators[etf_ticker]
        # Align index
        df['Sector_RSI'] = sector_rsi.reindex(df.index)
    else:
        df['Sector_RSI'] = np.nan

    # Relative Strength RSI (Stock RSI - Sector RSI)
    df['Rel_Strength_RSI'] = df['RSI_14'] - df['Sector_RSI']

    # 6. Gap 與 Target
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['Strategy_Ret'] = (df['Close'] - df['Open']) / df['Open']

    # 7. Labeling
    df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
    df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)
    df['Sample_Weight'] = df['Strategy_Ret'].abs() * 100

    # 8. 清洗
    # Drop rows with NaN in features
    # Note: Sector_RSI might be NaN if ETF data is missing (rare)
    features_to_check = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Strategy_Ret', 'Sector_RSI']
    df = df.dropna(subset=features_to_check)

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
    print(f"=== EXP-02: Sector Relative Strength (Momentum Model) ===")

    # Load Metadata
    tickers = load_tickers()
    sector_map = load_sector_map()

    if not tickers:
        print("[Error] No tickers found.")
        return

    # Fetch Data
    stock_raw, etf_raw, vix_raw = fetch_data(tickers)

    # Pre-calculate ETF Indicators
    print("Calculating Sector ETF Indicators...")
    etf_indicators = prepare_etf_indicators(etf_raw)

    print("\nBuilding features...")
    all_data = []

    for ticker, group in stock_raw.groupby('Ticker'):
        df = group.set_index('Date').copy()
        feat_df = build_features(df, vix_raw, etf_indicators, ticker, sector_map)

        if feat_df.empty: continue
        feat_df['Ticker'] = ticker

        # Filter Signals
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

    # Define Features
    # Baseline: ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX']
    # New: ['Sector_RSI', 'Rel_Strength_RSI']
    features = ['RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Gap_Pct', 'VIX', 'Sector_RSI', 'Rel_Strength_RSI']

    X_train = train_df[features]
    y_train = train_df['Label']
    w_train = train_df['Sample_Weight']

    X_test = test_df[features]
    y_test = test_df['Label']
    r_test = test_df['Strategy_Ret']

    print("\nTraining Momentum XGBoost Model (with Sector Features)...")
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
    print("MOMENTUM MODEL RESULTS (EXP-02 SECTOR STRENGTH)")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline (All)':<20} {'Model (Filter)':<20} {'Diff':<10}")
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

    # Plot
    plt.figure(figsize=(10, 6))
    imp.plot(kind='barh')
    plt.title('Feature Importance (Sector Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
    plt.close()

    # Save Model
    model_path = os.path.join(OUTPUT_DIR, 'momentum_model_sector.joblib')
    joblib.dump(model, model_path)
    print(f"\n[Saved] Model saved to: {model_path}")

    # Save Equity Curve
    test_df = test_df.copy()
    test_df['Model_Pred'] = y_pred
    daily_base = test_df.groupby(test_df.index)['Strategy_Ret'].mean()

    model_signals = test_df[test_df['Model_Pred']==1]
    if not model_signals.empty:
        daily_model = model_signals.groupby(model_signals.index)['Strategy_Ret'].mean()
        daily_model = daily_model.reindex(daily_base.index, fill_value=0)
    else:
        daily_model = pd.Series(0, index=daily_base.index)

    equity_base = (1 + daily_base).cumprod()
    equity_model = (1 + daily_model).cumprod()

    plt.figure(figsize=(10, 5))
    plt.plot(equity_base, label='Baseline', color='gray', alpha=0.5)
    plt.plot(equity_model, label='Model (Sector)', color='green', linewidth=2)
    plt.title('Momentum Model Equity Curve (Sector Features)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'momentum_equity.png'))
    print("Chart saved to momentum_equity.png")

if __name__ == '__main__':
    main()
