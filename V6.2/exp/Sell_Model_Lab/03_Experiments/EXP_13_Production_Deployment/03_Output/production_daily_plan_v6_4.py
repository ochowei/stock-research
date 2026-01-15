"""
Production Daily Plan V6.4
--------------------------
Generates daily sell signals using a Heterogeneous Ensemble:
1. Non-Tech Model: Base Features + SPY Context + LightGBM (Unlimited Depth, LR 0.02)
2. Tech Model: Base + QQQ Features + LightGBM (Depth 3, LR 0.01)

Usage: python production_daily_plan_v6_4.py
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
RESOURCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..', 'resource'))
MODEL_DIR = BASE_DIR

NON_TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.4_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.4_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(MODEL_DIR, 'sector_map.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

GAP_THRESHOLD = 0.005

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        # Fallback search
        root_markers = ['V6.2', 'resource']
        parts = BASE_DIR.split(os.sep)
        if 'V6.2' in parts:
            idx = parts.index('V6.2')
            path = os.sep.join(parts[:idx+1] + ['resource', '2025_final_asset_pool.json'])
        else:
            print(f"[Error] Asset pool not found at {path}")
            return []

    with open(path, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def get_sector(ticker, sector_map):
    return sector_map.get(ticker, 'Unknown')

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

def build_features_latest(df, bm_df, prefix, bm_features):
    """Builds features for the LAST row only"""
    if len(df) < 50: return None

    df = df.sort_index().copy()

    # Indicators
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
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

    last_idx = df.index[-1]
    row = df.iloc[[-1]].copy()

    # BM Features
    if last_idx not in bm_df.index:
        bm_row = bm_df.iloc[[-1]]
    else:
        bm_row = bm_df.loc[[last_idx]]

    for f in bm_features:
        if 'Corr' not in f:
            row[f] = bm_row[f].values[0]

    # Correlation
    common_idx = df.index.intersection(bm_df.index)
    df_sub = df.loc[common_idx]
    bm_sub = bm_df.loc[common_idx]

    aligned_close = pd.concat([df_sub['Close'], bm_sub['Close']], axis=1)
    corr = aligned_close.iloc[:,0].rolling(20).corr(aligned_close.iloc[:,1]).shift(1)

    corr_name = 'Sector_Corr' if prefix == 'QQQ' else 'Market_Corr'
    if np.isnan(corr.iloc[-1]): return None
    row[corr_name] = corr.iloc[-1]

    return row

def main():
    print("=== V6.4 Production Signal Generator ===")

    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    print("Fetching market data...")
    data = yf.download(tickers + ['QQQ', 'SPY'], period='3mo', interval='1d', auto_adjust=True, progress=False, threads=True)

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
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(['QQQ', 'SPY'])]

    # Prepare Benchmarks
    qqq_prep = prepare_benchmark(qqq_df, 'QQQ')
    spy_prep = prepare_benchmark(spy_df, 'SPY')

    latest_date = stock_df['Date'].max()
    print(f"Generating signals for date: {latest_date.date()}")

    signals = []

    for ticker, group in stock_df.groupby('Ticker'):
        if group['Date'].max() != latest_date: continue

        sector = get_sector(ticker, sector_map)
        is_tech = (sector == 'Technology')

        try:
            if is_tech:
                feat_row = build_features_latest(group.set_index('Date'), qqq_prep, 'QQQ', TECH_FEATURES)
            else:
                feat_row = build_features_latest(group.set_index('Date'), spy_prep, 'SPY', NON_TECH_FEATURES)
        except Exception:
            continue

        if feat_row is None or feat_row.empty: continue
        if feat_row['Gap_Pct'].values[0] <= GAP_THRESHOLD: continue

        if is_tech:
            X = feat_row[BASE_FEATURES + TECH_FEATURES]
            prob = tech_model.predict_proba(X)[0][1]
        else:
            X = feat_row[BASE_FEATURES + NON_TECH_FEATURES]
            prob = non_tech_model.predict_proba(X)[0][1]

        if prob > 0.5:
            signals.append({
                'Ticker': ticker,
                'Sector': sector,
                'Gap_Pct': feat_row['Gap_Pct'].values[0],
                'Probability': prob,
                'Model': 'Tech' if is_tech else 'Non-Tech'
            })

    res_df = pd.DataFrame(signals).sort_values('Probability', ascending=False)
    print(f"\nGenerated {len(res_df)} signals.")
    print(res_df.head(10))

    out_path = os.path.join(MODEL_DIR, f"daily_plan_{latest_date.date()}.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
