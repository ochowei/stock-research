import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import joblib
import json
import os
import sys
from datetime import datetime, timedelta

# --- Configuration ---
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, '03_Output')
# Correctly pointing to EXP-13 output based on relative path from EXP-14
MODEL_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, '..', 'EXP_13_Production_Deployment', '03_Output'))
RESOURCE_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, '..', '..', '..', '..', 'resource'))

NON_TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.4_non_tech_model.joblib')
TECH_MODEL_PATH = os.path.join(MODEL_DIR, 'v6.4_tech_model.joblib')
SECTOR_MAP_PATH = os.path.join(MODEL_DIR, 'sector_map.json')
ASSET_POOL_PATH = os.path.join(RESOURCE_DIR, '2025_final_asset_pool.json')

BASE_FEATURES = ['Gap_Pct', 'RSI_14', 'ATR_Pct', 'Vol_Ratio', 'Dist_MA20']
TECH_FEATURES = ['QQQ_Gap_Pct', 'QQQ_RSI_14', 'QQQ_Dist_MA20', 'Sector_Corr']
NON_TECH_FEATURES = ['SPY_Gap_Pct', 'SPY_RSI_14', 'SPY_Dist_MA20', 'Market_Corr']

GAP_THRESHOLD = 0.005
TEST_PERIOD_DAYS = 729 # Keep under 730 for hourly data

def load_tickers():
    with open(ASSET_POOL_PATH, 'r') as f:
        raw = json.load(f)
    return sorted(list(set([t.split(':')[-1].strip().replace('.', '-') for t in raw])))

def prepare_benchmark_full(bm_df, prefix):
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

def generate_signals_historical(tickers, sector_map, non_tech_model, tech_model, start_date):
    print(f"Fetching daily data for {len(tickers)} tickers since {start_date}...")
    # Fetch data slightly before start_date for indicator warmup
    warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")

    data = yf.download(tickers + ['QQQ', 'SPY'], start=warmup_start, interval='1d', auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data.stack(level=1, future_stack=True)
        except TypeError:
            data = data.stack(level=1)
        data = data.rename_axis(['Date', 'Ticker']).reset_index()
    else:
         # Single ticker handling
        data = data.reset_index()
        if 'Ticker' not in data.columns:
            # If only one ticker + spy/qqq, might be complex, but usually we have many tickers
            pass

    data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None).dt.normalize()

    qqq_df = data[data['Ticker'] == 'QQQ'].set_index('Date').sort_index()
    spy_df = data[data['Ticker'] == 'SPY'].set_index('Date').sort_index()
    stock_df = data[~data['Ticker'].isin(['QQQ', 'SPY'])]

    # Prepare Benchmarks
    qqq_prep = prepare_benchmark_full(qqq_df, 'QQQ')
    spy_prep = prepare_benchmark_full(spy_df, 'SPY')

    signals = []

    print("Generating features and predicting...")

    # Process each ticker
    for ticker, group in stock_df.groupby('Ticker'):
        group = group.sort_values('Date').set_index('Date')

        if len(group) < 30: continue

        # Calculate features vectorised
        group['Prev_Close'] = group['Close'].shift(1)
        group['Prev_Vol'] = group['Volume'].shift(1)
        group['RSI_14'] = ta.rsi(group['Close'], length=14).shift(1)

        # Ensure High/Low/Close are float for ATR calculation to avoid isnan type error
        h = group['High'].astype(float)
        l = group['Low'].astype(float)
        c = group['Close'].astype(float)

        # Additional safety check for pandas-ta
        try:
            atr_val = ta.atr(h, l, c, length=14)
            if atr_val is None:
                # If ATR fails, fill with NaN
                atr_val = pd.Series([np.nan] * len(group), index=group.index)
            group['ATR_14'] = atr_val.shift(1)
        except Exception:
             group['ATR_14'] = np.nan
        group['ATR_Pct'] = group['ATR_14'] / group['Prev_Close']
        group['Vol_MA20'] = group['Volume'].rolling(20).mean().shift(1)
        group['Vol_Ratio'] = group['Prev_Vol'] / group['Vol_MA20']

        close_filled = group['Close'].ffill()
        sum_prev_19 = close_filled.rolling(19).sum().shift(1)
        open_p = group['Open'].fillna(group['Close'])
        ma20_sim = (sum_prev_19 + open_p) / 20
        group['Dist_MA20'] = (open_p / ma20_sim) - 1
        group['Gap_Pct'] = (group['Open'] - group['Prev_Close']) / group['Prev_Close']

        # Sector and Model Selection
        sector = sector_map.get(ticker, 'Unknown')
        is_tech = (sector == 'Technology')

        target_dates = group.index[group.index >= start_date]

        for date in target_dates:
            row = group.loc[date]

            # Gap Threshold Check
            if row['Gap_Pct'] <= GAP_THRESHOLD: continue

            # Benchmark Features
            if is_tech:
                if date not in qqq_prep.index: continue
                bm_row = qqq_prep.loc[date]
                # Rolling Corr
                # Ideally should pre-calc, but doing row-by-row for simplicity in this loop
                # To optimise: pre-calculate rolling corr for the whole series
                # But rolling corr requires aligned series.
                pass
            else:
                if date not in spy_prep.index: continue
                bm_row = spy_prep.loc[date]

            # Construct Feature Vector
            feat = {
                'Gap_Pct': row['Gap_Pct'],
                'RSI_14': row['RSI_14'],
                'ATR_Pct': row['ATR_Pct'],
                'Vol_Ratio': row['Vol_Ratio'],
                'Dist_MA20': row['Dist_MA20']
            }

            # Correlation Calculation (Costly inside loop but safest for correctness)
            # Optimization: Calculate rolling corr on the full dataframe first
            # We'll do a simplified 20-day corr here:
            lookback = group.loc[:date].iloc[-21:-1] # T-1 to T-20
            if len(lookback) < 20: continue

            if is_tech:
                bm_lookback = qqq_df.loc[lookback.index]
                if len(bm_lookback) != len(lookback): continue
                corr = lookback['Close'].corr(bm_lookback['Close'])

                feat['QQQ_Gap_Pct'] = bm_row['QQQ_Gap_Pct']
                feat['QQQ_RSI_14'] = bm_row['QQQ_RSI_14']
                feat['QQQ_Dist_MA20'] = bm_row['QQQ_Dist_MA20']
                feat['Sector_Corr'] = corr

                X = pd.DataFrame([feat])[BASE_FEATURES + TECH_FEATURES]
                prob = tech_model.predict_proba(X)[0][1]

            else:
                bm_lookback = spy_df.loc[lookback.index]
                if len(bm_lookback) != len(lookback): continue
                corr = lookback['Close'].corr(bm_lookback['Close'])

                feat['SPY_Gap_Pct'] = bm_row['SPY_Gap_Pct']
                feat['SPY_RSI_14'] = bm_row['SPY_RSI_14']
                feat['SPY_Dist_MA20'] = bm_row['SPY_Dist_MA20']
                feat['Market_Corr'] = corr

                X = pd.DataFrame([feat])[BASE_FEATURES + NON_TECH_FEATURES]
                prob = non_tech_model.predict_proba(X)[0][1]

            if prob > 0.5:
                signals.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Sector': sector,
                    'Open': row['Open'],
                    'Close': row['Close'], # Daily Close (Approx)
                    'Prob': prob
                })

    return pd.DataFrame(signals)

def fetch_intraday_execution(signals_df):
    results = []
    print(f"Fetching intraday data for {len(signals_df)} signals...")

    # Group by Ticker to minimize API calls? No, usually yfinance caches well or we can bulk download.
    # But we need specific dates for specific tickers.
    # We can try to bulk download 1h data for all tickers for the last 730 days.
    # But that's huge. Better to iterate.

    unique_tickers = signals_df['Ticker'].unique()

    # Download 1h data for all unique tickers for the max range needed
    # To avoid timeout, chunk it.

    start_date = signals_df['Date'].min().strftime('%Y-%m-%d')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    print(f"Downloading hourly data from {start_date} to {end_date}...")

    # We process in chunks of tickers to avoid massive memory usage or timeouts
    chunk_size = 20
    ticker_chunks = [unique_tickers[i:i + chunk_size] for i in range(0, len(unique_tickers), chunk_size)]

    for chunk in ticker_chunks:
        try:
            # Interval 1h, period max 730d.
            # We can't specify start/end with 1h if it's > 730 days ago.
            # But our test period is last 729 days.
            data_1h = yf.download(list(chunk), start=start_date, end=end_date, interval='1h', auto_adjust=True, progress=False, threads=True)

            if isinstance(data_1h.columns, pd.MultiIndex):
                try:
                    data_1h = data_1h.stack(level=1, future_stack=True)
                except TypeError:
                    data_1h = data_1h.stack(level=1)
                data_1h = data_1h.rename_axis(['Datetime', 'Ticker']).reset_index()
            else:
                data_1h = data_1h.reset_index()
                # If single ticker
                if 'Ticker' not in data_1h.columns and len(chunk) == 1:
                    data_1h['Ticker'] = chunk[0]

            # Ensure Datetime is timezone naive for comparison
            data_1h['Datetime'] = pd.to_datetime(data_1h['Datetime']).dt.tz_localize(None)

            # Process signals for these tickers
            for _, row in signals_df[signals_df['Ticker'].isin(chunk)].iterrows():
                target_date = row['Date']
                ticker = row['Ticker']

                # Filter for the specific date
                # 1h bars for a day usually: 9:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30
                day_data = data_1h[(data_1h['Ticker'] == ticker) &
                                   (data_1h['Datetime'].dt.date == target_date.date())].sort_values('Datetime')

                if day_data.empty:
                    continue

                # Baseline: Open of first candle (9:30) to Close of last candle
                # Wait, "Close" in daily data is usually the last print.
                # In 1h data, the last candle (15:30) close is the market close.

                # Baseline Entry: 9:30 Open
                entry_baseline = day_data.iloc[0]['Open']

                # Delayed Entry: 10:30 Open (The 2nd candle)
                # If we assume rows are sorted by time:
                # 0: 9:30-10:30
                # 1: 10:30-11:30
                if len(day_data) > 1:
                    entry_delayed = day_data.iloc[1]['Open']
                    entry_time = day_data.iloc[1]['Datetime']
                else:
                    # Fallback if partial data
                    entry_delayed = entry_baseline
                    entry_time = day_data.iloc[0]['Datetime']

                # Exit: Market Close (Close of the last available candle)
                exit_price = day_data.iloc[-1]['Close']

                # Calculate Returns (Short Selling)
                ret_baseline = (entry_baseline - exit_price) / entry_baseline
                ret_delayed = (entry_delayed - exit_price) / entry_delayed

                results.append({
                    'Date': target_date,
                    'Ticker': ticker,
                    'Sector': row['Sector'],
                    'Entry_BL': entry_baseline,
                    'Entry_D': entry_delayed,
                    'Exit': exit_price,
                    'Return_BL': ret_baseline,
                    'Return_D': ret_delayed,
                    'Entry_Time_D': entry_time.time()
                })

        except Exception as e:
            print(f"Error processing chunk {chunk}: {e}")
            continue

    return pd.DataFrame(results)

def main():
    print("=== EXP-14: Delayed Entry Optimization ===")

    # 1. Load Resources
    tickers = load_tickers()
    with open(SECTOR_MAP_PATH, 'r') as f:
        sector_map = json.load(f)

    non_tech_model = joblib.load(NON_TECH_MODEL_PATH)
    tech_model = joblib.load(TECH_MODEL_PATH)

    # 2. Generate Signals
    start_date = (datetime.now() - timedelta(days=TEST_PERIOD_DAYS)).strftime("%Y-%m-%d")
    signals_df = generate_signals_historical(tickers, sector_map, non_tech_model, tech_model, start_date)

    if signals_df.empty:
        print("No signals generated.")
        return

    print(f"Generated {len(signals_df)} signals.")
    signals_df.to_csv(os.path.join(OUTPUT_DIR, 'generated_signals.csv'), index=False)

    # 3. Fetch Intraday and Simulate
    results_df = fetch_intraday_execution(signals_df)

    if results_df.empty:
        print("No intraday results.")
        return

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'simulation_results.csv'), index=False)

    # 4. Analysis
    print("\n=== Results Analysis ===")

    # Baseline Metrics
    bl_win_rate = (results_df['Return_BL'] > 0).mean()
    bl_avg_ret = results_df['Return_BL'].mean()

    # Delayed Metrics
    d_win_rate = (results_df['Return_D'] > 0).mean()
    d_avg_ret = results_df['Return_D'].mean()

    print(f"Baseline (Open-Close): Win Rate={bl_win_rate:.2%}, Avg Ret={bl_avg_ret:.4%}")
    print(f"Delayed (10:30-Close): Win Rate={d_win_rate:.2%}, Avg Ret={d_avg_ret:.4%}")

    # Save Report
    report = {
        'Metric': ['Win Rate', 'Avg Return'],
        'Baseline': [bl_win_rate, bl_avg_ret],
        'Delayed': [d_win_rate, d_avg_ret],
        'Diff': [d_win_rate - bl_win_rate, d_avg_ret - bl_avg_ret]
    }
    pd.DataFrame(report).to_csv(os.path.join(OUTPUT_DIR, 'performance_comparison.csv'), index=False)

    print("Done.")

if __name__ == "__main__":
    main()
