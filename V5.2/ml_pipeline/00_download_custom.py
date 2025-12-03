
import json
import os
import pickle
from lib.utils import format_ticker, download_data

def main():
    """Main function to run the custom portfolio data download."""
    # Define paths
    output_dir = 'V5.2/data/custom/'
    asset_pool_path = 'V5.2/ml_pipeline/asset_pool.json'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load tickers from asset_pool.json
    with open(asset_pool_path, 'r') as f:
        raw_tickers = json.load(f)

    # Format tickers for yfinance
    tickers = [format_ticker(t) for t in raw_tickers]

    # Define macro tickers as per instructions
    macro_tickers = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', '^TNX']

    # Define date range
    start_date = '2015-01-01'
    end_date = '2025-11-30'

    print("--- Downloading Custom Tickers ---")
    tickers_data = download_data(tickers, start_date, end_date)

    print("\n--- Downloading Macro Tickers ---")
    macro_data = download_data(macro_tickers, start_date, end_date)

    # Save data to pickle files
    with open(os.path.join(output_dir, 'raw_tickers.pkl'), 'wb') as f:
        pickle.dump(tickers_data, f)

    with open(os.path.join(output_dir, 'raw_macro.pkl'), 'wb') as f:
        pickle.dump(macro_data, f)

    print("\nData download complete.")
    print(f"Custom tickers data saved to {os.path.join(output_dir, 'raw_tickers.pkl')}")
    print(f"Macro tickers data saved to {os.path.join(output_dir, 'raw_macro.pkl')}")

if __name__ == "__main__":
    main()
