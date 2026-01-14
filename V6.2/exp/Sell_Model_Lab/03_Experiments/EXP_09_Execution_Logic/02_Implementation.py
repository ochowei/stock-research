
import os
import sys
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# --- Paths & Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
OUTPUT_DIR = os.path.join(BASE_DIR, '03_Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Re-use models from EXP-08
EXP_08_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'EXP_08_Production_Integration', '03_Output'))
NON_TECH_MODEL_PATH = os.path.join(EXP_08_DIR, 'v6.3_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(EXP_08_DIR, 'v6.3_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(EXP_08_DIR, 'sector_map.json')

# Parameters
TEST_START = '2024-01-01'
TEST_END = datetime.today().strftime('%Y-%m-%d')
GAP_THRESHOLD = 0.005 # 0.5%
SCORE_THRESHOLD = 0.5

# Feature Defs
BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']

# Grid Search Parameters
PT_LEVELS = [0.002, 0.005, 0.010, None]  # None means hold to close
SL_LEVELS = [0.005, 0.010, 0.020, None]  # None means hold to close (implicit SL)

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"[Error] Asset pool not found at {path}")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def fetch_data(tickers):
    benchmarks = ['QQQ', 'SPY']
    all_tickers = tickers + benchmarks
    print(f"Downloading data for {len(all_tickers)} tickers...")

    # Download extra history for indicators
    start_date = '2023-09-01'
    data = yf.download(all_tickers, start=start_date, end=None, interval='1d', auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns: pass
        data = data.reset_index()

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(benchmarks)]

    return stock_df, qqq_df

def prepare_benchmark_features(qqq_df):
    df = qqq_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['QQQ_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['QQQ_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['QQQ_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Close']]

def build_features(df, qqq_df, is_tech=False):
    df = df.sort_index().copy()

    # Calculate Base Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)

    if len(df) < 50: return pd.DataFrame()

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

    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Tech Features
    if is_tech and not qqq_df.empty:
        common_idx = df.index.intersection(qqq_df.index)
        if len(common_idx) == 0: return pd.DataFrame()

        df_sub = df.loc[common_idx].copy()
        qqq_sub = qqq_df.loc[common_idx]

        df_sub['QQQ_Gap_Pct'] = qqq_sub['QQQ_Gap_Pct']
        df_sub['QQQ_RSI_14'] = qqq_sub['QQQ_RSI_14']
        df_sub['QQQ_Dist_MA20'] = qqq_sub['QQQ_Dist_MA20']

        # Sector Corr
        aligned_close = pd.concat([df_sub['Close'], qqq_sub['Close']], axis=1)
        aligned_close.columns = ['Stock_Close', 'QQQ_Close']
        corr_series = aligned_close['Stock_Close'].rolling(20).corr(aligned_close['QQQ_Close'])
        df_sub['Sector_Corr'] = corr_series.shift(1)

        df = df_sub

    return df

def simulate_trade(row, pt_pct, sl_pct):
    """
    Simulates a short trade with daily OHLC.
    Entry: Open
    Target: Open * (1 - pt_pct)
    Stop: Open * (1 + sl_pct)
    """
    entry_price = row['Open']

    # Define Prices
    target_price = entry_price * (1 - pt_pct) if pt_pct else -99999
    stop_price = entry_price * (1 + sl_pct) if sl_pct else 99999

    day_high = row['High']
    day_low = row['Low']
    day_close = row['Close']

    # Logic:
    # 1. Check SL (Conservative: Assume SL hit first if High >= Stop)
    # Note: If we gap up above SL, we fill at Open (which is entry), so invalid.
    # But here we enter at Open.
    # If High > Stop, we likely hit it.

    sl_hit = False
    if sl_pct is not None:
        if day_high >= stop_price:
            sl_hit = True

    pt_hit = False
    if pt_pct is not None:
        if day_low <= target_price:
            pt_hit = True

    # Conflict Resolution (Both Hit)
    if sl_hit and pt_hit:
        # Conservative: SL hit first
        return -sl_pct

    if sl_hit:
        return -sl_pct

    if pt_hit:
        return pt_pct

    # Neither hit - Close position
    return (entry_price - day_close) / entry_price


def main():
    print("=== EXP-09: Execution Logic Refinement ===")

    # 1. Setup
    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    stock_df, qqq_df = fetch_data(tickers)
    qqq_feats = prepare_benchmark_features(qqq_df)

    # 2. Load Models
    if not os.path.exists(NON_TECH_MODEL_PATH) or not os.path.exists(TECH_MODEL_PATH):
        print("Models not found! Ensure EXP-08 ran successfully.")
        return

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    # 3. Generate Signals (Test Set)
    print("Generating Signals on Test Set...")
    signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        sector = sector_map.get(ticker, 'Unknown')
        is_tech = (sector == 'Technology')

        df = group.set_index('Date').copy()
        df = df[df.index >= '2023-11-01'] # Buffer for test start

        if df.empty: continue

        # Build Features
        try:
            feat_df = build_features(df, qqq_feats, is_tech=is_tech)
        except Exception as e:
            continue

        if feat_df.empty: continue

        # Filter Test Period
        feat_df = feat_df[feat_df.index >= TEST_START]

        # Filter Candidates (Gap > 0.5%)
        candidates = feat_df[feat_df['Gap_Pct'] > GAP_THRESHOLD].copy()

        if candidates.empty: continue

        # Predict
        if is_tech:
            # Tech features
            cols = BASE_FEATURES + TECH_FEATURES
            # Handle missing cols? Should be there.
            if not all(c in candidates.columns for c in cols): continue
            probs = tech_model.predict_proba(candidates[cols])[:, 1]
        else:
            cols = BASE_FEATURES
            if not all(c in candidates.columns for c in cols): continue
            probs = non_tech_model.predict_proba(candidates[cols])[:, 1]

        candidates['Probability'] = probs

        # Filter by Score
        trades = candidates[candidates['Probability'] > SCORE_THRESHOLD].copy()
        trades['Ticker'] = ticker
        trades['Sector'] = sector

        signals.append(trades)

    all_signals = pd.concat(signals)
    print(f"Total Signals Generated: {len(all_signals)}")

    # 4. Grid Search Simulation
    print("\nRunning Grid Search Simulation...")
    results = []

    # Add raw OHLC back to signals (index is Date)
    # signals df has OHLC because it came from build_features(df)

    for pt in PT_LEVELS:
        for sl in SL_LEVELS:
            config_name = f"PT={pt if pt else 'Close'}_SL={sl if sl else 'None'}"
            # print(f"Simulating {config_name}...")

            # Apply logic row by row
            # Vectorization is hard with custom logic, apply is fine for ~2000 signals
            outcomes = all_signals.apply(lambda row: simulate_trade(row, pt, sl), axis=1)

            # Metrics
            total_ret = outcomes.sum()
            avg_ret = outcomes.mean()
            win_rate = (outcomes > 0).mean()
            count = len(outcomes)

            # Drawdown (Cumulative Sum -> Max Drop)
            cum_ret = outcomes.cumsum()
            running_max = cum_ret.cummax()
            drawdown = cum_ret - running_max
            max_dd = drawdown.min()

            results.append({
                'PT': pt,
                'SL': sl,
                'Win_Rate': win_rate,
                'Avg_Return': avg_ret,
                'Total_Return': total_ret,
                'Max_Drawdown': max_dd,
                'Trade_Count': count
            })

    # 5. Save Results
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values('Total_Return', ascending=False)

    print("\n=== Top 5 Configurations ===")
    print(res_df.head(5).to_string())

    res_df.to_csv(os.path.join(OUTPUT_DIR, 'grid_search_results.csv'), index=False)

    # Save baseline (PT=0.002, SL=None) comparison
    baseline = res_df[(res_df['PT'] == 0.002) & (res_df['SL'].isna())]
    if not baseline.empty:
        print("\nBaseline (PT=0.2%, SL=None):")
        print(baseline.to_string())

    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
