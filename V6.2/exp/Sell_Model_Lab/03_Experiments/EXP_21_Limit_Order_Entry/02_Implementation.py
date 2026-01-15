import os
import sys
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
from datetime import datetime, timedelta

# --- Configuration ---
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXP_DIR, '03_Output')
RESOURCE_DIR = os.path.abspath(os.path.join(EXP_DIR, '../../../../resource'))

# Using models from EXP-18 (which are V6.2.4.RC)
EXP_18_OUTPUT = os.path.abspath(os.path.join(EXP_DIR, '../EXP_18_Production_Script_Update/03_Output'))
NON_TECH_MODEL_PATH = os.path.join(EXP_18_OUTPUT, 'v6.2.4_rc_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(EXP_18_OUTPUT, 'v6.2.4_rc_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(EXP_18_OUTPUT, 'sector_map.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

GAP_THRESHOLD = 0.005
START_DATE = '2024-01-01'

def load_tickers():
    path = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return []
    with open(path, 'r') as f:
        raw = json.load(f)
    # Clean tickers: remove exchange prefix (e.g., 'NASDAQ:TSLA' -> 'TSLA')
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def prepare_benchmark(bm_df, prefix):
    """Calculates benchmark features."""
    df = bm_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # RSI
    rsi = ta.rsi(df['Close'], length=14)
    df[f'{prefix}_RSI_14'] = rsi.shift(1) if rsi is not None else np.nan

    # Dist MA20
    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[[c for c in df.columns if prefix in c]]

def run_backtest():
    print("=== EXP-21: Limit Order Entry Optimization ===")

    # 1. Load Models and Resources
    print("Loading Models...")
    try:
        non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
        tech_model = joblib.load(TECH_MODEL_PATH)
        with open(SECTOR_MAP_PATH, 'r') as f:
            sector_map = json.load(f)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # 2. Fetch Data
    print("Fetching Data...")
    tickers = load_tickers()
    if not tickers:
        print("No tickers loaded.")
        return

    all_tickers = tickers + ['QQQ', 'SPY']
    # Buffer for indicators
    start_buffer = (pd.to_datetime(START_DATE) - timedelta(days=90)).strftime('%Y-%m-%d')

    print(f"Downloading data for {len(all_tickers)} tickers from {start_buffer}...")
    data = yf.download(all_tickers, start=start_buffer, interval='1d', auto_adjust=True, progress=False, threads=True)

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

    # 3. Process Benchmarks
    qqq_raw = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy_raw = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()

    qqq_feats = prepare_benchmark(qqq_raw, 'QQQ')
    spy_feats = prepare_benchmark(spy_raw, 'SPY')

    qqq_close = qqq_raw['Close']
    spy_close = spy_raw['Close']

    # 4. Generate Trades
    print("Generating Signals...")
    trades = []

    # Group by ticker and process
    grouped = data[~data['Ticker'].isin(['QQQ', 'SPY'])].groupby('Ticker')

    for ticker, df in grouped:
        if len(df) < 50: continue

        sector = sector_map.get(ticker, 'Unknown')
        is_tech = (sector == 'Technology')

        df = df.set_index('Date').sort_index()

        # Calculate Base Features
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Vol'] = df['Volume'].shift(1)

        # Avoid division by zero
        df = df[df['Prev_Close'] > 0]

        df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        try:
            # RSI
            df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)

            # ATR
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR_14'] = atr.shift(1)
            df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

            # Vol Ratio
            vol_ma = df['Volume'].rolling(20).mean().shift(1)
            df['Vol_Ratio'] = df['Prev_Vol'] / vol_ma

            # Dist MA20
            close_filled = df['Close'].ffill()
            sum_prev_19 = close_filled.rolling(19).sum().shift(1)
            open_p = df['Open'].fillna(df['Close'])
            ma20_sim = (sum_prev_19 + open_p) / 20
            df['Dist_MA20'] = (open_p / ma20_sim) - 1

        except Exception:
            continue

        # Join Benchmark
        if is_tech:
            bm_feats = qqq_feats
            bm_close = qqq_close
            corr_name = 'Sector_Corr'
            needed_feats = BASE_FEATURES + TECH_FEATURES
            model = tech_model
        else:
            bm_feats = spy_feats
            bm_close = spy_close
            corr_name = 'Market_Corr'
            needed_feats = BASE_FEATURES + NON_TECH_FEATURES
            model = non_tech_model

        df = df.join(bm_feats, how='left')

        # Correlation
        common = df.index.intersection(bm_close.index)
        if len(common) > 20:
             s_c = df.loc[common, 'Close']
             b_c = bm_close.loc[common]
             roll_corr = s_c.rolling(20).corr(b_c).shift(1)
             df.loc[common, corr_name] = roll_corr

        # Drop NaNs
        df_clean = df.dropna(subset=needed_feats + ['Gap_Pct'])

        # Filter for Gap
        candidates = df_clean[df_clean['Gap_Pct'] > GAP_THRESHOLD].copy()

        if candidates.empty: continue

        # Predict
        try:
            probs = model.predict_proba(candidates[needed_feats])[:, 1]
            candidates['Prob'] = probs
        except Exception:
            continue

        # Filter Signals
        signals = candidates[candidates['Prob'] > 0.5]

        for date, row in signals.iterrows():
            if date < pd.to_datetime(START_DATE): continue

            trades.append({
                'Date': date,
                'Ticker': ticker,
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Prob': row['Prob']
            })

    trades_df = pd.DataFrame(trades)
    print(f"Generated {len(trades_df)} signals.")
    trades_df.to_csv(os.path.join(OUTPUT_DIR, 'raw_signals.csv'), index=False)

    # 5. Simulate Limit Orders
    print("Simulating Limit Order Scenarios...")

    # Scenarios: Offset Pct
    scenarios = {
        'Baseline (Open)': 0.000,
        'Limit +0.5%': 0.005,
        'Limit +1.0%': 0.010,
        'Limit +1.5%': 0.015
    }

    results_summary = []

    for name, offset in scenarios.items():
        # Simulation Logic
        # Entry Price = Open * (1 + offset)
        # If High >= Entry Price: Filled
        # Exit = Close (MOC)
        # Profit = (Entry - Close) / Entry

        filled_trades = []

        for _, trade in trades_df.iterrows():
            limit_price = trade['Open'] * (1 + offset)

            # Check Fill
            if trade['High'] >= limit_price:
                # Filled
                pnl = (limit_price - trade['Close']) / limit_price
                filled_trades.append(pnl)
            else:
                # Missed
                pass

        # Calculate Metrics
        n_filled = len(filled_trades)
        fill_rate = n_filled / len(trades_df) if len(trades_df) > 0 else 0

        if n_filled > 0:
            returns = np.array(filled_trades)
            avg_ret = np.mean(returns)
            total_ret = np.sum(returns)
            win_rate = np.mean(returns > 0)

            # Sharpe (assuming daily returns distribution from these trades)
            # Simplification: calculating Sharpe of the trade series
            std_dev = np.std(returns)
            sharpe = (avg_ret / std_dev) * np.sqrt(252) if std_dev > 0 else 0
        else:
            avg_ret = 0
            total_ret = 0
            win_rate = 0
            sharpe = 0

        results_summary.append({
            'Scenario': name,
            'Fill Rate': f"{fill_rate:.2%}",
            'Signal Count': n_filled,
            'Win Rate': f"{win_rate:.2%}",
            'Avg Return': f"{avg_ret:.4f}",
            'Total Return': f"{total_ret:.4f}",
            'Sharpe': f"{sharpe:.4f}"
        })

    summary_df = pd.DataFrame(results_summary)
    print("\nSummary Results:")
    print(summary_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'limit_order_results.csv'), index=False)

if __name__ == "__main__":
    run_backtest()
