
import os
import yfinance as yf
import pandas as pd
import requests

def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def clean_ticker(ticker):
    """Handles special cases for ticker symbols."""
    return ticker.replace('.', '-')

def scrape_index_tickers(url, table_index, ticker_column):
    """Scrapes tickers from a Wikipedia page."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        ticker_df = tables[table_index]
        return ticker_df[ticker_column].str.upper().tolist()
    except Exception as e:
        print(f"Failed to scrape tickers from {url}: {e}")
        return []

def get_fallback_tickers():
    """Returns a predefined list of tickers if scraping fails."""
    return [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'NVDA', 'META', 'TSLA',
        'BRK-B', 'JPM', 'JNJ', 'V', 'UNH', 'HD', 'PG', 'MA', 'AVGO', 'LLY',
        'XOM', 'CVX', 'BAC', 'KO', 'PFE', 'PEP', 'COST', 'WMT', 'DIS', 'CSCO',
        'ADBE', 'INTC', 'CMCSA', 'NFLX', 'TMO', 'ABT', 'CRM', 'ACN', 'MCD',
        'WFC', 'QCOM', 'TXN', 'HON', 'UNP', 'PM', 'AMGN', 'CAT', 'RTX', 'LOW'
    ]

def download_data(tickers, start_date, end_date):
    """Downloads 1d OHLCV data for a list of tickers."""
    return yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        timeout=30
    )

def main():
    """
    Downloads daily data for S&P 100 and Nasdaq 100 components
    and saves them to the V5.2/data/index/ directory.
    """
    # --- Configuration ---
    START_DATE = '2015-01-01'
    END_DATE = '2025-11-30'

    # Build paths relative to the script's location to get to V5.2 root
    script_dir = get_script_dir()
    v5_2_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Output directories and files (in V5.2/data/)
    output_dir = os.path.join(v5_2_dir, 'data', 'index')
    tickers_output_path = os.path.join(output_dir, 'raw_tickers.pkl')
    macro_output_path = os.path.join(output_dir, 'raw_macro.pkl')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Scrape and Process Tickers ---
    print("Scraping index component tickers from Wikipedia...")
    sp100_url = 'https://en.wikipedia.org/wiki/S%26P_100'
    nasdaq100_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'

    sp100_tickers = scrape_index_tickers(sp100_url, 2, 'Symbol')
    nasdaq100_tickers = scrape_index_tickers(nasdaq100_url, 4, 'Ticker')

    index_tickers = sorted(list(set(sp100_tickers + nasdaq100_tickers)))

    if not index_tickers:
        print("Scraping failed. Using fallback ticker list.")
        index_tickers = get_fallback_tickers()

    index_tickers = [clean_ticker(t) for t in index_tickers]
    print(f"Successfully sourced {len(index_tickers)} unique index tickers.")

    # --- Download Index Ticker Data ---
    print(f"Downloading daily data for {len(index_tickers)} index tickers...")
    daily_tickers_df = download_data(index_tickers, START_DATE, END_DATE)

    # --- CRUCIAL: Package into dictionary for downstream compatibility ---
    output_tickers_data = {'daily': daily_tickers_df}

    print(f"Saving index ticker data to: {tickers_output_path}")
    pd.to_pickle(output_tickers_data, tickers_output_path)
    print("Index ticker data saved successfully.")

    # --- Download Macro Data ---
    macro_tickers = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', 'TNX']
    print(f"Downloading daily data for {len(macro_tickers)} macro indicators...")
    macro_df = download_data(macro_tickers, START_DATE, END_DATE)

    print(f"Saving macro data to: {macro_output_path}")
    pd.to_pickle(macro_df, macro_output_path)
    print("Macro data saved successfully.")

if __name__ == '__main__':
    main()
