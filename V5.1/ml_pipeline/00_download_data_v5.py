import yfinance as yf
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time

def get_asset_tickers(file_path):
    """Reads the asset pool JSON and returns a list of yfinance-compatible tickers."""
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"Error: Asset pool file not found at {file_path}")
        return []
        
    with open(file_path, 'r') as f:
        asset_pool = json.load(f)

    tickers = [asset.split(':')[-1] for asset in asset_pool]
    # yfinance uses '-' for dots in tickers like BRK.B
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

def download_data(tickers, start_date, end_date, interval, prepost=False):
    """Downloads historical data for a list of tickers with retry logic."""
    print(f"Downloading {interval} data for {len(tickers)} tickers from {start_date} to {end_date}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # yfinance download
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                prepost=prepost,
                threads=True,
                timeout=30
            )
            # Check if the dataframe is empty or contains only NaNs
            if not df.empty and not df.isnull().all().all():
                # On success, print a message and return the dataframe
                print(f"Successfully downloaded data on attempt {attempt + 1}.")
                return df
            else:
                print(f"Attempt {attempt + 1} of {max_retries} failed: No data returned.")
        except Exception as e:
            print(f"Attempt {attempt + 1} of {max_retries} failed with error: {e}")

        # If not the last attempt, wait before retrying
        if attempt < max_retries - 1:
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)

    # If all retries fail, return an empty DataFrame
    print("All download attempts failed.")
    return pd.DataFrame()

def main():
    """Main function to download and save ticker, macro, and sector data (V5.1)."""
    # --- Setup Paths ---
    # Get the directory where the script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to the script directory
    # 假設 asset_pool.json 位於同層目錄
    asset_pool_path = os.path.join(SCRIPT_DIR, 'asset_pool.json')
    output_dir = os.path.join(SCRIPT_DIR, 'data', 'temp_raw')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    # Get tickers from asset pool
    tickers = get_asset_tickers(asset_pool_path)
    if not tickers:
        print("No tickers found. Exiting.")
        return

    # Define macro symbols (V5 Base)
    macro_symbols = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', '^TNX']
    
    # Define Sector ETFs (V5.1 New Requirement)
    # Technology, Financials, Health Care, Consumer Discretionary, 
    # Consumer Staples, Energy, Industrials, Materials, Utilities
    sector_symbols = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']

    # Define date ranges
    end_date = datetime.now()
    ten_years_ago = end_date - timedelta(days=10*365)
    two_years_ago = end_date - timedelta(days=729) # Use 729 days to be safe with yfinance API limits

    # --- 1. Download Ticker Data (Stocks) ---
    print("\n--- Starting Ticker Data Download ---")
    daily_tickers_df = download_data(tickers, ten_years_ago, end_date, interval='1d')
    hourly_tickers_df = download_data(tickers, two_years_ago, end_date, interval='60m', prepost=True)

    # Combine into a single dictionary
    raw_tickers_data = {
        'daily': daily_tickers_df,
        'hourly': hourly_tickers_df
    }

    # Save ticker data
    tickers_output_path = os.path.join(output_dir, 'raw_tickers_data.pkl')
    pd.to_pickle(raw_tickers_data, tickers_output_path)
    print(f"Ticker data saved to {tickers_output_path}")

    # --- 2. Download Macro Data (Market Context) ---
    print("\n--- Starting Macro Data Download ---")
    # For macro, we mainly need daily data for L1 features
    daily_macro_df = download_data(macro_symbols, ten_years_ago, end_date, interval='1d')

    # Save macro data
    macro_output_path = os.path.join(output_dir, 'raw_macro_data.pkl')
    pd.to_pickle(daily_macro_df, macro_output_path)
    print(f"Macro data saved to {macro_output_path}")
    
    # --- 3. Download Sector Data (V5.1 Orthogonal Features) ---
    print("\n--- Starting Sector ETF Download (V5.1) ---")
    # We download Daily data for sectors to build Relative Strength & Sector RSI context
    daily_sector_df = download_data(sector_symbols, ten_years_ago, end_date, interval='1d')
    
    # Save sector data
    sector_output_path = os.path.join(output_dir, 'raw_sector_data.pkl')
    pd.to_pickle(daily_sector_df, sector_output_path)
    print(f"Sector data saved to {sector_output_path}")

    print("\nV5.1 Data download process completed successfully.")

if __name__ == '__main__':
    main()
