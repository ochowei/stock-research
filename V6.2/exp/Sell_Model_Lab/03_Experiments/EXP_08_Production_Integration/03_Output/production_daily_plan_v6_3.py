"""
Production Daily Plan V6.3
--------------------------
Generates daily sell signals using a Heterogeneous Ensemble:
1. Non-Tech Model: Base Features (5) + LightGBM (Unlimited Depth, LR 0.02)
2. Tech Model: Base + QQQ Features (9) + LightGBM (Depth 3, LR 0.01)

Usage: python production_daily_plan_v6_3.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
from datetime import datetime

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for resource dir in standard V6.2 structure relative to this script
# If this script is in V6.2/exp/Sell_Model_Lab/03_Experiments/EXP_08.../03_Output
# Then resource is at ../../../../resource
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
MODEL_DIR = BASE_DIR # Models are in the same dir as the script in 03_Output

# Correction: The original Implementation put resource at ../../../../resource
# Let's double check where we are running from.
# If we run this script directly, __file__ is the script path.

NON_TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.3_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.3_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(MODEL_DIR, 'sector_map.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']

GAP_THRESHOLD = 0.005

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        # Try finding it relative to V6.2 root if path is wrong
        # Assuming we are deep in exp/Sell_Model_Lab...
        # Let's try to look for V6.2 root
        root_markers = ['V6.2', 'resource']
        parts = BASE_DIR.split(os.sep)
        if 'V6.2' in parts:
            idx = parts.index('V6.2')
            # V6.2/resource
            path = os.sep.join(parts[:idx+1] + ['resource', '2025_final_asset_pool.json'])
        else:
            print(f"[Error] Asset pool not found at {path}")
            return []

    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def get_sector(ticker, sector_map):
    return sector_map.get(ticker, 'Unknown')

def prepare_benchmark(qqq_df):
    df = qqq_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df['QQQ_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']
    df['QQQ_RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['QQQ_Dist_MA20'] = (open_p / ma20_sim) - 1
    return df

def build_features_latest(df, qqq_df, is_tech=False):
    """Builds features for the LAST row only (for prediction)"""
    # We need enough history for indicators (e.g. 50 days)
    if len(df) < 50: return None

    df = df.sort_index().copy()

    # Calculate Indicators on full history
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14).shift(1)
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']
    df['Vol_MA20'] = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / df['Vol_MA20']

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)

    # Use Open if available (Live), else use Close (Simulated Open for today?)
    # For daily plan generation, we typically run BEFORE market open or AT market open.
    # If Pre-market, we might fetch Pre-market open.
    # For now, we assume we have 'Open' (Live) or we use 'Close' of prev day as proxy if missing?
    # Actually, standard procedure:
    # We need today's Opening price to calculate Gap.
    # If generating plan BEFORE open, we can't calculate Gap yet.
    # This script likely generates "Potential Signals" or assumes we run it right after Open.
    # In V6.2/V6.3, we typically pass the 'Current Open' to the model.
    # Let's assume the data contains the latest candle with Open.

    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # Extract latest row
    last_idx = df.index[-1]
    row = df.iloc[[-1]].copy()

    if is_tech:
        # QQQ Features
        # Match by date? Or just take latest QQQ?
        # Assuming QQQ data is aligned
        if last_idx not in qqq_df.index:
            # Try to use latest available QQQ
            qqq_row = qqq_df.iloc[[-1]]
        else:
            qqq_row = qqq_df.loc[[last_idx]]

        row['QQQ_Gap_Pct'] = qqq_row['QQQ_Gap_Pct'].values[0]
        row['QQQ_RSI_14'] = qqq_row['QQQ_RSI_14'].values[0]
        row['QQQ_Dist_MA20'] = qqq_row['QQQ_Dist_MA20'].values[0]

        # Sector Corr (Needs rolling calculation aligned)
        # We need the last 20 days aligned
        common_idx = df.index.intersection(qqq_df.index)
        df_sub = df.loc[common_idx]
        qqq_sub = qqq_df.loc[common_idx]

        aligned_close = pd.concat([df_sub['Close'], qqq_sub['Close']], axis=1)
        aligned_close.columns = ['Stock_Close', 'QQQ_Close']
        corr = aligned_close['Stock_Close'].rolling(20).corr(aligned_close['QQQ_Close']).shift(1)

        if np.isnan(corr.iloc[-1]): return None
        row['Sector_Corr'] = corr.iloc[-1]

    return row

def main():
    print("=== V6.3 Production Signal Generator ===")

    # Load Resources
    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    # Fetch Data (Live/Recent)
    # Fetching last 60 days to ensure enough for indicators
    print("Fetching market data...")
    data = yf.download(tickers + ['QQQ'], period='3mo', interval='1d', auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
        if 'Ticker' not in data.columns: pass
        data = data.reset_index()

    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    stock_df = data[data['Ticker'] != 'QQQ']

    # Prepare QQQ
    qqq_prep = prepare_benchmark(qqq_df)

    # Identify Latest Date
    latest_date = stock_df['Date'].max()
    print(f"Generating signals for date: {latest_date.date()}")

    signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        # Filter group to last date
        if group['Date'].max() != latest_date:
            continue

        sector = get_sector(ticker, sector_map)
        is_tech = (sector == 'Technology')

        try:
            feat_row = build_features_latest(group.set_index('Date'), qqq_prep, is_tech)
        except Exception as e:
            continue

        if feat_row is None or feat_row.empty: continue

        # Check Gap Threshold
        if feat_row['Gap_Pct'].values[0] <= GAP_THRESHOLD:
            continue

        # Predict
        if is_tech:
            feats = TECH_FEATURES + BASE_FEATURES # Ensure order?
            # Actually Implementation used BASE + TECH order.
            # "ALL_FEATS = BASE_FEATURES + TECH_FEATURES"
            X = feat_row[BASE_FEATURES + TECH_FEATURES]
            prob = tech_model.predict_proba(X)[0][1]
        else:
            X = feat_row[BASE_FEATURES]
            prob = non_tech_model.predict_proba(X)[0][1]

        if prob > 0.5:
            signals.append({
                'Ticker': ticker,
                'Sector': sector,
                'Gap_Pct': feat_row['Gap_Pct'].values[0],
                'Probability': prob,
                'Model': 'Tech' if is_tech else 'Non-Tech'
            })

    # Output
    res_df = pd.DataFrame(signals).sort_values('Probability', ascending=False)
    print(f"\nGenerated {len(res_df)} signals.")
    print(res_df.head(10))

    out_path = os.path.join(MODEL_DIR, f"daily_plan_{latest_date.date()}.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
