
import os
import json
import yfinance as yf
import pandas as pd

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def clean_ticker(ticker):
    """Removes exchange prefix and handles special cases."""
    if ":" in ticker:
        ticker = ticker.split(":")[1]
    return ticker.replace('.', '-')

def download_data(tickers, start_date, end_date):
    """Downloads 1d OHLCV data for a list of tickers."""
    return yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=False,
        timeout=30
    )

def main():
    """
    Downloads daily data for a custom asset pool and macro indicators,
    and saves them to the V5.2/data/custom/ directory.
    """
    # --- Configuration ---
    START_DATE = '2015-01-01'
    END_DATE = '2025-11-30'

    # Build paths relative to the script's location
    script_dir = get_script_dir()
    v5_2_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Input file
    asset_pool_path = os.path.join(v5_2_dir, 'ml_pipeline', 'asset_pool.json')

    # Output directories and files
    output_dir = os.path.join(v5_2_dir, 'data', 'custom')
    tickers_output_path = os.path.join(output_dir, 'raw_tickers.pkl')
    macro_output_path = os.path.join(output_dir, 'raw_macro.pkl')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and Process Tickers ---
    print(f"Loading asset pool from: {asset_pool_path}")
    with open(asset_pool_path, 'r') as f:
        asset_pool = json.load(f)

    custom_tickers = [clean_ticker(t) for t in asset_pool]
    print(f"Loaded and cleaned {len(custom_tickers)} custom tickers.")

    # --- Download Custom Ticker Data ---
    print(f"Downloading daily data for {len(custom_tickers)} custom tickers...")
    daily_tickers_df = download_data(custom_tickers, START_DATE, END_DATE)

    # --- CRUCIAL: Package into dictionary for downstream compatibility ---
    output_tickers_data = {'daily': daily_tickers_df}

    print(f"Saving custom ticker data to: {tickers_output_path}")
    pd.to_pickle(output_tickers_data, tickers_output_path)
    print("Custom ticker data saved successfully.")

    # --- Download Macro Data ---
    macro_tickers = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', '^TNX']
    print(f"Downloading daily data for {len(macro_tickers)} macro indicators...")
    macro_df = download_data(macro_tickers, START_DATE, END_DATE)

    print(f"Saving macro data to: {macro_output_path}")
    pd.to_pickle(macro_df, macro_output_path)
    print("Macro data saved successfully.")

if __name__ == '__main__':
    main()
