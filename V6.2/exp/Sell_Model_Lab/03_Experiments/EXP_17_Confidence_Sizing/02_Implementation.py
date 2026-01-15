import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import time

# --- Path Setup ---
# Add V6.2 root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# --- Configuration ---
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, '03_Output')
EXP_13_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, '../EXP_13_Production_Deployment/03_Output'))
RESOURCE_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, '../../../../resource'))

NON_TECH_MODEL_PATH = os.path.join(EXP_13_DIR, 'v6.4_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(EXP_13_DIR, 'v6.4_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(EXP_13_DIR, 'sector_map.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

START_DATE = '2023-01-01'
GAP_THRESHOLD = 0.005

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    with open(path, 'r') as f:
        raw = json.load(f)
    # Filter out known bad tickers if necessary
    bad_tickers = ['FI'] # 'FI' is failing consistently
    tickers = sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))
    return [t for t in tickers if t not in bad_tickers]

def prepare_benchmark(bm_df, prefix):
    df = bm_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df[f'{prefix}_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1
    return df

def generate_features_for_ticker(group, bm_prep, prefix, bm_features):
    """
    Generates features for a whole history dataframe (vectorized).
    """
    df = group.sort_index().copy()

    # Ensure all columns are numeric
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=cols_to_numeric)

    # Basic Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Indicators (All shifted by 1 to avoid lookahead for Open execution, except Gap)
    # Note: production script calculates RSI on Close and shifts 1.
    df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1

    # Target (Sell Model: Profit if Price Drops)
    # Return = (Open - Close) / Open
    df['Return'] = (df['Open'] - df['Close']) / df['Open']

    # Join Benchmark Features
    # Join on Index (Date)
    df = df.join(bm_prep[bm_features], how='left')

    # Correlation
    # Rolling correlation between Ticker Close and BM Close (shifted 1)
    aligned_bm_close = bm_prep['Close'].reindex(df.index)
    corr = df['Close'].rolling(20).corr(aligned_bm_close).shift(1)

    corr_name = 'Sector_Corr' if prefix == 'QQQ' else 'Market_Corr'
    df[corr_name] = corr

    return df.dropna()

def download_with_retry(tickers, retries=3):
    for i in range(retries):
        try:
            print(f"Download attempt {i+1}...")
            # Use threads=False to be safer with rate limits
            data = yf.download(tickers, start=pd.to_datetime(START_DATE) - pd.Timedelta(days=60),
                               auto_adjust=True, progress=False, threads=False)
            if not data.empty:
                return data
        except Exception as e:
            print(f"Download failed: {e}")
            time.sleep(2)
    return pd.DataFrame()

def run_simulation():
    print("Loading Models...")
    try:
        non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
        tech_model = joblib.load(TECH_MODEL_PATH)
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    tickers = load_tickers()
    all_tickers = tickers + ['QQQ', 'SPY']
    print(f"Loaded {len(tickers)} tickers. Total to download: {len(all_tickers)}")

    print("Downloading Data...")
    data = download_with_retry(all_tickers)

    if data.empty:
        print("Failed to download any data.")
        return

    # Flatten MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        data = data.reset_index()

    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    if 'Ticker' not in data.columns:
        print("Error: 'Ticker' column missing after flattening. Columns:", data.columns)
        return

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()

    if qqq_df.empty or spy_df.empty:
        print("Critical: QQQ or SPY data missing.")
        return

    # Benchmark Prep
    qqq_prep = prepare_benchmark(qqq_df, 'QQQ')
    spy_prep = prepare_benchmark(spy_df, 'SPY')

    stock_df = data[~data['Ticker'].isin(['QQQ', 'SPY'])]

    results = []

    print("Processing Tickers...")
    unique_tickers = stock_df['Ticker'].unique()
    print(f"Found {len(unique_tickers)} unique tickers in downloaded data.")

    # Debug: Check Gap stats
    all_gaps = []

    for ticker in unique_tickers:
        group = stock_df[stock_df['Ticker'] == ticker].set_index('Date').sort_index()

        if len(group) < 50:
            continue

        sector = sector_map.get(ticker, 'Unknown')
        is_tech = (sector == 'Technology')

        try:
            if is_tech:
                feat_df = generate_features_for_ticker(group, qqq_prep, 'QQQ', [f for f in TECH_FEATURES if 'Corr' not in f])
                model_cols = BASE_FEATURES + TECH_FEATURES
                model = tech_model
            else:
                feat_df = generate_features_for_ticker(group, spy_prep, 'SPY', [f for f in NON_TECH_FEATURES if 'Corr' not in f])
                model_cols = BASE_FEATURES + NON_TECH_FEATURES
                model = non_tech_model

            # Filter Date > START_DATE
            feat_df = feat_df[feat_df.index >= pd.to_datetime(START_DATE)]

            if feat_df.empty: continue

            # Collect Gap stats
            all_gaps.extend(feat_df['Gap_Pct'].tolist())

            # Gap Filter
            feat_df = feat_df[feat_df['Gap_Pct'] > GAP_THRESHOLD]

            if feat_df.empty: continue

            X = feat_df[model_cols]
            probs = model.predict_proba(X)[:, 1]

            feat_df['Probability'] = probs
            feat_df['Ticker'] = ticker
            feat_df['Sector'] = sector

            results.append(feat_df[['Ticker', 'Sector', 'Probability', 'Return', 'Gap_Pct']])

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

    # Debug: Print Gap Stats
    if all_gaps:
        all_gaps = np.array(all_gaps)
        print(f"\nGap Statistics (All Tickers):")
        print(f"  Mean: {np.mean(all_gaps):.4f}")
        print(f"  Max: {np.max(all_gaps):.4f}")
        print(f"  > 0.005 count: {np.sum(all_gaps > 0.005)} / {len(all_gaps)}")

    if not results:
        print("No signals generated after processing.")
        return

    all_signals = pd.concat(results)
    all_signals = all_signals.sort_index()

    print(f"Total signals processed: {len(all_signals)}")
    print(f"Max Probability: {all_signals['Probability'].max()}")

    # Filter for signals > 0.5 (Buy Threshold)
    trades = all_signals[all_signals['Probability'] > 0.5].copy()

    print(f"Total Trades Generated (Prob > 0.5): {len(trades)}")

    if trades.empty:
        print("No trades meeting probability threshold.")
        return

    # --- Strategies ---

    # 1. Baseline: Fixed $10k
    trades['Size_Baseline'] = 10000.0

    # 2. Variant A: Tiered
    def tiered_size(prob):
        if prob >= 0.60: return 15000.0
        elif prob >= 0.55: return 10000.0
        else: return 5000.0

    trades['Size_VariantA'] = trades['Probability'].apply(tiered_size)

    # 3. Variant B: Linear
    def linear_size(prob):
        factor = 0.5 + (prob - 0.5) * 5
        return 10000.0 * factor

    trades['Size_VariantB'] = trades['Probability'].apply(linear_size)

    # Calculate PnL
    trades['PnL_Baseline'] = trades['Return'] * trades['Size_Baseline']
    trades['PnL_VariantA'] = trades['Return'] * trades['Size_VariantA']
    trades['PnL_VariantB'] = trades['Return'] * trades['Size_VariantB']

    # Save Trade Log
    trades.to_csv(os.path.join(OUTPUT_DIR, 'trade_log.csv'))

    # Daily Aggregation
    daily_res = trades.groupby(trades.index)[['PnL_Baseline', 'PnL_VariantA', 'PnL_VariantB']].sum()

    # Fill missing days with 0 (no trades)
    idx_range = pd.date_range(start=daily_res.index.min(), end=daily_res.index.max(), freq='B') # Business days
    daily_res = daily_res.reindex(idx_range).fillna(0)

    daily_res.to_csv(os.path.join(OUTPUT_DIR, 'daily_pnl.csv'))

    # Metrics
    metrics = []
    for strat in ['Baseline', 'VariantA', 'VariantB']:
        pnl = daily_res[f'PnL_{strat}']
        total_ret = pnl.sum()

        equity = pnl.cumsum()
        peak = equity.cummax()
        dd = equity - peak
        max_dd = dd.min()

        mean_ret = pnl.mean()
        std_ret = pnl.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0

        strat_trades = trades[trades[f'Size_{strat}'] > 0]
        wins = len(strat_trades[strat_trades['Return'] > 0])
        total = len(strat_trades)
        wr = wins / total if total > 0 else 0

        metrics.append({
            'Strategy': strat,
            'Total_Return_Dollar': total_ret,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown_Dollar': max_dd,
            'Win_Rate': wr,
            'Trade_Count': total,
            'Avg_Size': trades[f'Size_{strat}'].mean()
        })

    metrics_df = pd.DataFrame(metrics)
    print("\nResults:")
    print(metrics_df)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'performance_metrics.csv'), index=False)

    # Plot Equity Curves
    plt.figure(figsize=(12, 6))
    plt.plot(daily_res['PnL_Baseline'].cumsum(), label='Baseline (Equal)')
    plt.plot(daily_res['PnL_VariantA'].cumsum(), label='Variant A (Tiered)')
    plt.plot(daily_res['PnL_VariantB'].cumsum(), label='Variant B (Linear)')
    plt.title('Equity Curve Comparison: Position Sizing')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'equity_curves.png'))

if __name__ == "__main__":
    run_simulation()
