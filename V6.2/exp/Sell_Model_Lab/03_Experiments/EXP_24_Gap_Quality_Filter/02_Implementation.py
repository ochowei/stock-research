
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import sys
import time

# --- 1. Settings & Parameters ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lab Utils Path
LAB_UTILS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '02_Lab_Utils'))
sys.path.append(LAB_UTILS_PATH)

# Date Range
TRAIN_START = '2020-01-01'
TRAIN_END   = '2023-12-31'
TEST_START  = '2024-01-01'
TEST_END    = '2025-12-31'

# Strategy Parameters
GAP_THRESHOLD = 0.005      # 0.5% Gap Threshold
PROFIT_THRESHOLD = 0.002   # 0.2% Profit Threshold

# Feature Definitions
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_CONTEXT = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20']
NON_TECH_CONTEXT = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20']

# Model Params (V6.2.4.RC)
TECH_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.01, 'max_depth': 3, 'num_leaves': 8,
    'random_state': 42, 'verbosity': -1, 'n_jobs': 1
}
NON_TECH_PARAMS = {
    'n_estimators': 200, 'learning_rate': 0.02, 'max_depth': -1, 'num_leaves': 31,
    'random_state': 42, 'verbosity': -1, 'n_jobs': 1
}

# --- 2. Utility Functions ---

def load_tickers():
    # DEBUG: Use a smaller, reliable set to ensure pipeline works first
    debug_tickers = [
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'GOOGL', 'META', 'NFLX', 'INTC',
        'JPM', 'BAC', 'XOM', 'CVX', 'PG', 'KO', 'JNJ', 'PFE', 'PEP', 'CSCO',
        'WMT', 'DIS', 'V', 'MA', 'HD', 'ADBE', 'CRM', 'ABBV', 'MRK', 'AVGO'
    ]
    print(f"DEBUG: Using {len(debug_tickers)} reliable tickers for stability.")
    return sorted(debug_tickers)

def fetch_sectors(tickers):
    """Fetches sector information for tickers using yfinance with caching."""
    sector_cache_path = os.path.join(OUTPUT_DIR, 'sector_map.json')

    if os.path.exists(sector_cache_path):
        with open(sector_cache_path, 'r') as f:
            return json.load(f)

    sector_map = {}
    print("Fetching sector information...")
    for i, t in enumerate(tickers):
        try:
            ticker_obj = yf.Ticker(t)
            sec = ticker_obj.info.get('sector', 'Unknown')
            sector_map[t] = sec
        except Exception:
            sector_map[t] = 'Unknown'
        time.sleep(0.1)

    with open(sector_cache_path, 'w') as f:
        json.dump(sector_map, f, indent=4)
    return sector_map

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    max_retries = 3
    data = pd.DataFrame()

    for i in range(max_retries):
        try:
            data = yf.download(
                all_tickers, start=TRAIN_START, end=TEST_END,
                interval='1d', auto_adjust=True, progress=False, threads=True
            )
            if not data.empty:
                break
        except Exception as e:
            print(f"Download failed attempt {i+1}: {e}")
            time.sleep(2)

    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Flatten MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns: pass
        data = data.reset_index()

    if 'Date' not in data.columns and data.index.name == 'Date':
        data = data.reset_index()

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    # Separate
    qqq = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stocks = data[~data['Ticker'].isin(benchmarks)]

    return stocks, qqq, spy

def safe_convert_numeric(df):
    df = df.copy()
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in target_cols:
        if col in df.columns:
            # Handle potential DataFrame (if multiple columns with same name exist but weren't caught)
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]

            # Force numeric
            s = pd.to_numeric(df[col], errors='coerce')

            # Fill NaNs? No, leave them as NaNs, but ensure type is float
            df[col] = s.astype('float64')

    return df

def prepare_benchmark_features(df, prefix):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df = safe_convert_numeric(df)
    df['Close'] = df['Close'].ffill()

    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    if len(df) > 14:
        rsi = ta.rsi(df['Close'], length=14)
        if rsi is not None:
             df[f'{prefix}_RSI_14'] = rsi.shift(1)
        else:
             df[f'{prefix}_RSI_14'] = np.nan
    else:
        df[f'{prefix}_RSI_14'] = np.nan

    sum_prev_19 = df['Close'].rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[[f'{prefix}_Gap_Pct', f'{prefix}_RSI_14', f'{prefix}_Dist_MA20']]

def build_features(df, qqq_df, spy_df):
    # DEBUG: Print initial shape
    # print(f"DEBUG: Processing ticker, shape {df.shape}")

    df = df.sort_index().copy()
    df = safe_convert_numeric(df)

    # Explicit cast
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
         if col in df.columns:
             df[col] = df[col].astype(float)

    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50:
        # print("DEBUG: Insufficient history (<50)")
        return pd.DataFrame()

    try:
        # RSI 14
        close_series = df['Close']
        # Double check type
        if close_series.dtype == 'object':
             close_series = close_series.astype('float64')

        rsi = ta.rsi(close_series, length=14)
        if rsi is not None:
            df['RSI_14'] = rsi.shift(1)
        else:
            df['RSI_14'] = np.nan

        # Other Base Features
        high_s = df['High']
        low_s = df['Low']

        atr = ta.atr(high_s, low_s, close_series, length=14)
        if atr is not None:
             df['ATR_14'] = atr.shift(1)
        else:
             df['ATR_14'] = np.nan

        df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
        df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
        df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

        close_filled = df['Close'].ffill()
        sum_prev_19 = close_filled.rolling(19).sum().shift(1)
        open_p = df['Open'].fillna(df['Close'])
        ma20_sim = (sum_prev_19 + open_p) / 20
        df['Dist_MA20'] = (open_p / ma20_sim) - 1

        df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        # Additional feature for analysis: Gap in ATR units
        df['Gap_ATR_Ratio'] = df['Gap_Pct'] / df['ATR_Pct']

        # Label
        df['Strategy_Ret'] = (df['Open'] - df['Close']) / df['Open']
        df['Is_Signal'] = df['Gap_Pct'] > GAP_THRESHOLD
        df['Label'] = (df['Strategy_Ret'] > PROFIT_THRESHOLD).astype(int)

        # Join Context
        if not qqq_df.empty:
            df = df.join(qqq_df, how='left')
        if not spy_df.empty:
            df = df.join(spy_df, how='left')

    except Exception as e:
        print(f"Feature build error: {e}")
        return pd.DataFrame()

    df = df.dropna()
    # DEBUG: print signal count
    signals = df[df['Is_Signal']]
    if not signals.empty:
        # print(f"DEBUG: Generated {len(signals)} signals")
        pass

    return df

def analyze_filters(df):
    """
    Analyzes Win Rate and Return by various bins.
    """
    stats = []

    # 1. Gap Size Buckets
    bins_gap = [0.005, 0.01, 0.02, 1.0]
    labels_gap = ['0.5-1.0%', '1.0-2.0%', '>2.0%']
    df['Gap_Bin'] = pd.cut(df['Gap_Pct'], bins=bins_gap, labels=labels_gap)

    print("\n--- Analysis by Gap Size ---")
    grp_gap = df.groupby('Gap_Bin', observed=True).agg({
        'Label': ['mean', 'count'],
        'Strategy_Ret': 'mean'
    })
    # Flatten MultiIndex columns
    grp_gap.columns = ['Win_Rate', 'Count', 'Avg_Return']
    # Reorder
    grp_gap = grp_gap[['Win_Rate', 'Avg_Return', 'Count']]
    print(grp_gap)

    # 2. Volume Ratio Buckets
    bins_vol = [-1, 0.8, 1.2, 2.0, 100]
    labels_vol = ['<0.8', '0.8-1.2', '1.2-2.0', '>2.0']
    df['Vol_Bin'] = pd.cut(df['Vol_Ratio'], bins=bins_vol, labels=labels_vol)

    print("\n--- Analysis by Prev Volume Ratio ---")
    grp_vol = df.groupby('Vol_Bin', observed=True).agg({
        'Label': ['mean', 'count'],
        'Strategy_Ret': 'mean'
    })
    grp_vol.columns = ['Win_Rate', 'Count', 'Avg_Return']
    grp_vol = grp_vol[['Win_Rate', 'Avg_Return', 'Count']]
    print(grp_vol)

    # 3. Gap vs ATR Ratio Buckets
    # e.g. Gap is 1x ATR, 2x ATR etc.
    bins_atr = [0, 0.5, 1.0, 2.0, 100]
    labels_atr = ['<0.5x', '0.5-1.0x', '1.0-2.0x', '>2.0x']
    df['Gap_ATR_Bin'] = pd.cut(df['Gap_ATR_Ratio'], bins=bins_atr, labels=labels_atr)

    print("\n--- Analysis by Gap/ATR Ratio ---")
    grp_atr = df.groupby('Gap_ATR_Bin', observed=True).agg({
        'Label': ['mean', 'count'],
        'Strategy_Ret': 'mean'
    })
    grp_atr.columns = ['Win_Rate', 'Count', 'Avg_Return']
    grp_atr = grp_atr[['Win_Rate', 'Avg_Return', 'Count']]
    print(grp_atr)

    return grp_gap, grp_vol, grp_atr

def run_experiment(stock_raw, qqq_feat, spy_feat, sector_map):
    all_data = []

    # 1. Feature Engineering
    print("Building features...")
    for ticker, group in stock_raw.groupby('Ticker'):
        group = group.copy()

        # Handle Index
        if group.index.name != 'Date':
             if 'Date' in group.columns:
                 df = group.set_index('Date')
             else:
                 df = group.reset_index()
                 if 'index' in df.columns: df = df.rename(columns={'index': 'Date'})
                 if 'Date' in df.columns: df = df.set_index('Date')
                 else: continue
        else:
            df = group

        df = df[~df.index.duplicated(keep='first')]
        if df.empty: continue
        df.index = pd.to_datetime(df.index)

        feat_df = build_features(df, qqq_feat, spy_feat)
        if feat_df.empty: continue

        feat_df['Ticker'] = ticker
        feat_df['Sector'] = sector_map.get(ticker, 'Unknown')
        feat_df['Is_Tech'] = (feat_df['Sector'] == 'Technology').astype(int)

        signal_df = feat_df[feat_df['Is_Signal']].copy()
        if not signal_df.empty:
            all_data.append(signal_df)

    if not all_data:
        print("No signals generated.")
        return pd.DataFrame()

    full_df = pd.concat(all_data).sort_index()

    train_df = full_df[full_df.index <= TRAIN_END]
    test_df = full_df[(full_df.index >= TEST_START) & (full_df.index <= TEST_END)]

    print(f"Train Size: {len(train_df)}, Test Size: {len(test_df)}")

    # 2. Model Training & Prediction

    # Tech Model
    tech_cols = BASE_FEATURES + TECH_CONTEXT
    train_tech = train_df[train_df['Is_Tech'] == 1]
    test_tech = test_df[test_df['Is_Tech'] == 1]

    print(f"Training Tech Model (n={len(train_tech)})...")
    if len(train_tech) > 50:
        tech_model = LGBMClassifier(**TECH_PARAMS)
        tech_model.fit(train_tech[tech_cols], train_tech['Label'], sample_weight=train_tech['Strategy_Ret'].abs()*100)

        test_tech = test_tech.copy()
        if len(test_tech) > 0:
            test_tech['Pred_Prob'] = tech_model.predict_proba(test_tech[tech_cols])[:, 1]
        else:
            test_tech['Pred_Prob'] = 0
    else:
        print("Skipping Tech Model (Insufficient Data)")
        test_tech = test_tech.copy()
        test_tech['Pred_Prob'] = 0

    # Non-Tech Model
    non_tech_cols = BASE_FEATURES + NON_TECH_CONTEXT
    train_non = train_df[train_df['Is_Tech'] == 0]
    test_non = test_df[test_df['Is_Tech'] == 0]

    print(f"Training Non-Tech Model (n={len(train_non)})...")
    if len(train_non) > 50:
        non_tech_model = LGBMClassifier(**NON_TECH_PARAMS)
        non_tech_model.fit(train_non[non_tech_cols], train_non['Label'], sample_weight=train_non['Strategy_Ret'].abs()*100)

        test_non = test_non.copy()
        if len(test_non) > 0:
            test_non['Pred_Prob'] = non_tech_model.predict_proba(test_non[non_tech_cols])[:, 1]
        else:
            test_non['Pred_Prob'] = 0
    else:
        print("Skipping Non-Tech Model (Insufficient Data)")
        test_non = test_non.copy()
        test_non['Pred_Prob'] = 0

    combined_test = pd.concat([test_tech, test_non]).sort_index()

    # 3. Filter Analysis on High Probability Trades
    # We only care about trades the model WOULD have taken.
    predictions = combined_test[combined_test['Pred_Prob'] > 0.5].copy()

    print(f"\nTotal Baseline Trades: {len(predictions)}")
    if len(predictions) == 0:
        return pd.DataFrame()

    baseline_wr = predictions['Label'].mean()
    baseline_ret = predictions['Strategy_Ret'].mean()
    print(f"Baseline Win Rate: {baseline_wr:.2%}")
    print(f"Baseline Avg Return: {baseline_ret:.4f}")

    # Run Breakdown
    grp_gap, grp_vol, grp_atr = analyze_filters(predictions)

    # Save Breakdown
    grp_gap.to_csv(os.path.join(OUTPUT_DIR, 'analysis_gap_size.csv'))
    grp_vol.to_csv(os.path.join(OUTPUT_DIR, 'analysis_vol_ratio.csv'))
    grp_atr.to_csv(os.path.join(OUTPUT_DIR, 'analysis_gap_atr.csv'))

    # Save raw predictions for deeper manual check if needed
    predictions.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'))

    return predictions

def main():
    print("=== EXP-24: Gap Quality Filter ===")
    tickers = load_tickers()
    sector_map = fetch_sectors(tickers)
    stock_raw, qqq_raw, spy_raw = fetch_data(tickers)

    if stock_raw.empty:
        print("Data load failed.")
        return

    qqq_feat = prepare_benchmark_features(qqq_raw, 'QQQ')
    spy_feat = prepare_benchmark_features(spy_raw, 'SPY')

    run_experiment(stock_raw, qqq_feat, spy_feat, sector_map)
    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()
