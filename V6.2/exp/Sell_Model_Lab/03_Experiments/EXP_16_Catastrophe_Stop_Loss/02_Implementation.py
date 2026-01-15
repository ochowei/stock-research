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
EXP_13_OUTPUT = os.path.abspath(os.path.join(EXP_DIR, '../EXP_13_Production_Deployment/03_Output'))

NON_TECH_MODEL_PATH = os.path.join(EXP_13_OUTPUT, 'v6.4_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(EXP_13_OUTPUT, 'v6.4_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(EXP_13_OUTPUT, 'sector_map.json')

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
    # Clean tickers
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def prepare_benchmark(bm_df, prefix):
    df = bm_df.copy()
    df['Prev_Close'] = df['Close'].shift(1)
    df[f'{prefix}_Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    # RSI
    # Handle potential NaN from start
    rsi = ta.rsi(df['Close'], length=14)
    if rsi is not None:
        df[f'{prefix}_RSI_14'] = rsi.shift(1)
    else:
        df[f'{prefix}_RSI_14'] = np.nan

    # Dist MA20
    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df[f'{prefix}_Dist_MA20'] = (open_p / ma20_sim) - 1

    return df[[c for c in df.columns if prefix in c]]

def prepare_stock_features(df, bm_df, bm_prefix):
    # Ensure sorted
    df = df.sort_index()

    # Base Features
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev_Vol'] = df['Volume'].shift(1)
    df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

    rsi = ta.rsi(df['Close'], length=14)
    df['RSI_14'] = rsi.shift(1) if rsi is not None else np.nan

    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR_14'] = atr.shift(1) if atr is not None else np.nan
    df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

    vol_ma = df['Volume'].rolling(20).mean().shift(1)
    df['Vol_Ratio'] = df['Prev_Vol'] / vol_ma

    close_filled = df['Close'].ffill()
    sum_prev_19 = close_filled.rolling(19).sum().shift(1)
    open_p = df['Open'].fillna(df['Close'])
    ma20_sim = (sum_prev_19 + open_p) / 20
    df['Dist_MA20'] = (open_p / ma20_sim) - 1

    # Merge Benchmark Features
    # We need to join on Date index
    df = df.join(bm_df, how='left')

    # Correlation
    # Rolling correlation between stock close and benchmark close
    # Shift 1 to avoid lookahead
    # Need to ensure alignment
    common_idx = df.index.intersection(bm_df.index)
    if len(common_idx) < 20:
        df['Corr'] = np.nan
    else:
        # Re-align to be safe
        s_close = df.loc[common_idx, 'Close']
        b_close = bm_df.loc[common_idx].iloc[:, 0] # Assuming original bm_df has Close, wait.
        # prepare_benchmark returns only features. We need original Close for correlation.
        # Let's pass the original BM dataframe or keep Close in the prepared one.
        pass

    return df

def run_backtest():
    print("Loading Models...")
    try:
        non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
        tech_model = joblib.load(TECH_MODEL_PATH)
        with open(SECTOR_MAP_PATH, 'r') as f:
            sector_map = json.load(f)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    print("Fetching Data...")
    tickers = load_tickers()
    # Fetch Data
    # Adding buffer for indicators
    start_buffer = (pd.to_datetime(START_DATE) - timedelta(days=60)).strftime('%Y-%m-%d')
    all_tickers = tickers + ['QQQ', 'SPY']

    # Chunking to avoid timeouts? Or just try all.
    # yfinance handles it reasonably well usually.
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

    # Split Benchmarks
    qqq_raw = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy_raw = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()

    # Prepare BM features
    qqq_feats = prepare_benchmark(qqq_raw, 'QQQ')
    spy_feats = prepare_benchmark(spy_raw, 'SPY')

    # Need Close for Correlation
    qqq_close = qqq_raw['Close']
    spy_close = spy_raw['Close']

    trades = []

    print("Processing Tickers...")
    grouped = data[~data['Ticker'].isin(['QQQ', 'SPY'])].groupby('Ticker')

    for ticker, df in grouped:
        if len(df) < 50: continue

        sector = sector_map.get(ticker, 'Unknown')
        is_tech = (sector == 'Technology')

        df = df.set_index('Date').sort_index()

        # Base Features
        df['Prev_Close'] = df['Close'].shift(1)
        df['Prev_Vol'] = df['Volume'].shift(1)
        df['Gap_Pct'] = (df['Open'] - df['Prev_Close']) / df['Prev_Close']

        # Filter early to save compute? No, need rolling windows.
        # Compute Indicators
        try:
            df['RSI_14'] = ta.rsi(df['Close'], length=14).shift(1)
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df['ATR_14'] = atr.shift(1) # Shifted for signal generation
            # Note: We need TODAY's ATR for calculating Stop Loss in some variants,
            # but usually Stop Loss is based on ATR at entry (which is T-1 ATR or T Open ATR).
            # "3x ATR" usually refers to the volatility at the time of entry.
            # Using shifted ATR (T-1) is safe.

            df['ATR_Pct'] = df['ATR_14'] / df['Prev_Close']

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

        # Join features
        df = df.join(bm_feats, how='left')

        # Compute Correlation
        # Align index
        common = df.index.intersection(bm_close.index)
        if len(common) > 20:
             # Fast correlation
             # We need rolling corr of T-1 Close.
             s_c = df.loc[common, 'Close']
             b_c = bm_close.loc[common]
             roll_corr = s_c.rolling(20).corr(b_c).shift(1)
             df.loc[common, corr_name] = roll_corr

        # Drop NaN
        df_clean = df.dropna(subset=needed_feats + ['Gap_Pct'])

        # Filter for Signal Candidates
        candidates = df_clean[df_clean['Gap_Pct'] > GAP_THRESHOLD].copy()

        if candidates.empty: continue

        # Predict
        try:
            probs = model.predict_proba(candidates[needed_feats])[:, 1]
            candidates['Prob'] = probs
        except Exception as e:
            continue

        # Select Trades
        signals = candidates[candidates['Prob'] > 0.5]

        for date, row in signals.iterrows():
            if date < pd.to_datetime(START_DATE): continue

            # Trade Data
            entry_price = row['Open']
            day_high = row['High']
            day_low = row['Low']
            day_close = row['Close']
            prev_close = row['Prev_Close']
            atr_val = row['ATR_14']

            trades.append({
                'Date': date,
                'Ticker': ticker,
                'Sector': sector,
                'Entry': entry_price,
                'High': day_high,
                'Low': day_low,
                'Close': day_close,
                'ATR': atr_val,
                'Gap_Pct': row['Gap_Pct'],
                'Prob': row['Prob']
            })

    # Save raw trades
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(os.path.join(OUTPUT_DIR, 'raw_trades.csv'), index=False)
    print(f"Generated {len(trades_df)} trades.")

    # === Simulation ===
    print("Simulating Stop Losses...")

    # Define Scenarios
    # Stop Logic: If High >= Entry * (1 + Stop_Pct), Exit at Stop Price.
    # Assuming Short position.

    results = []

    scenarios = [
        ('No_Stop', None),
        ('Fixed_3%', 0.03),
        ('Fixed_5%', 0.05),
        ('Fixed_10%', 0.10),
        ('ATR_2x', '2xATR'),
        ('ATR_3x', '3xATR')
    ]

    for name, threshold in scenarios:
        pnl_list = []
        stopped_count = 0

        for _, trade in trades_df.iterrows():
            entry = trade['Entry']
            high = trade['High']
            close = trade['Close']
            atr = trade['ATR']

            # Calculate Stop Price
            if threshold is None:
                stop_price = float('inf')
            elif isinstance(threshold, float):
                stop_price = entry * (1 + threshold)
            elif threshold == '2xATR':
                stop_price = entry + (2 * atr)
            elif threshold == '3xATR':
                stop_price = entry + (3 * atr)

            # Check Trigger
            # Note: For Short, Stop is ABOVE Entry.
            is_stopped = high >= stop_price

            if is_stopped:
                # Loss = Entry - Stop_Price (Negative)
                # Return = (Entry - Stop_Price) / Entry = - (Stop_Price - Entry) / Entry
                # If Fixed 5%, Return is -5% (plus slippage usually, but ignoring for now)
                # Actually, strictly speaking: PnL = (Entry - Exit) / Entry
                pnl = (entry - stop_price) / entry
                stopped_count += 1
            else:
                # Held to close
                pnl = (entry - close) / entry

            pnl_list.append(pnl)

        # Metrics
        pnl_series = pd.Series(pnl_list)
        total_return = pnl_series.sum()
        avg_return = pnl_series.mean()
        win_rate = (pnl_series > 0).mean()
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252) if pnl_series.std() > 0 else 0

        # Max Drawdown
        cum_ret = pnl_series.cumsum()
        running_max = cum_ret.cummax()
        drawdown = cum_ret - running_max
        max_dd = drawdown.min()

        results.append({
            'Scenario': name,
            'Total_Return': total_return,
            'Avg_Return': avg_return,
            'Win_Rate': win_rate,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_dd,
            'Stop_Trigger_Rate': stopped_count / len(trades_df)
        })

    res_df = pd.DataFrame(results)
    print("\nResults:")
    print(res_df)
    res_df.to_csv(os.path.join(OUTPUT_DIR, 'stop_loss_results.csv'), index=False)

if __name__ == "__main__":
    run_backtest()
