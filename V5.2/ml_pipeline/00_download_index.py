
import os
import pickle
import pandas as pd
import requests
from lib.utils import download_data

def get_fallback_tickers():
    """Returns a predefined list of S&P 100 and Nasdaq 100 tickers."""
    # Last known good list as a fallback
    sp100 = [
        'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
        'AVGO', 'AXP', 'BA', 'BAC', 'BIIB', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B',
        'C', 'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO',
        'CVS', 'CVX', 'DASH', 'DE', 'DHR', 'DIS', 'DUK', 'EMR', 'EXC', 'FANG',
        'FDX', 'GD', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON',
        'IBM', 'INTC', 'JNJ', 'JPM', 'KDP', 'KHC', 'KO', 'LIN', 'LLY', 'LMT',
        'LOW', 'MA', 'MCD', 'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK',
        'MS', 'MSFT', 'NEE', 'NFLX', 'NKE', 'NOW', 'NVDA', 'ORCL', 'PANW', 'PEP',
        'PFE', 'PG', 'PM', 'PYPL', 'QCOM', 'RTX', 'SCHW', 'SO', 'SPG', 'T',
        'TGT', 'TMO', 'TMUS', 'TRI', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'V',
        'VRTX', 'VZ', 'WBD', 'WFC', 'WMT', 'XOM'
    ]
    nasdaq100 = [
        'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'AMAT', 'AMGN', 'AMZN',
        'ANSS', 'ASML', 'AVGO', 'AZN', 'BIIB', 'BKNG', 'BKR', 'CDNS', 'CDW', 'CEG',
        'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS',
        'CTSH', 'DDOG', 'DLTR', 'DXCM', 'EA', 'EXC', 'FAST', 'FTNT', 'GEHC', 'GFS',
        'GILD', 'GOOG', 'GOOGL', 'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG',
        'KDP', 'KHC', 'KLAC', 'LRCX', 'LULU', 'MAR', 'MCHP', 'MDB', 'MDLZ', 'MELI',
        'META', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MU', 'NFLX', 'NVDA', 'NXPI',
        'ODFL', 'ON', 'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM',
        'REGN', 'ROP', 'ROST', 'SBUX', 'SCHW', 'SIRI', 'SNPS', 'TEAM', 'TMUS',
        'TSLA', 'TTD', 'TTWO', 'TXN', 'VRSK', 'VRTX', 'WBA', 'WBD', 'WDAY', 'XEL', 'ZS'
    ]
    return sorted(list(set(sp100 + nasdaq100)))

def scrape_index_tickers():
    """Scrapes S&P 100 and Nasdaq 100 tickers from Wikipedia with a user-agent."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    sp100_url = 'https://en.wikipedia.org/wiki/S%26P_100'
    nasdaq100_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'

    sp100_tickers = []
    nasdaq100_tickers = []

    try:
        response = requests.get(sp100_url, headers=headers)
        response.raise_for_status()
        sp100_table = pd.read_html(response.text, attrs={'id': 'constituents'})[0]
        sp100_tickers = sp100_table['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"Could not scrape S&P 100 tickers: {e}")

    try:
        response = requests.get(nasdaq100_url, headers=headers)
        response.raise_for_status()
        nasdaq100_table = pd.read_html(response.text, attrs={'id': 'constituents'})[0]
        nasdaq100_tickers = nasdaq100_table['Ticker'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"Could not scrape Nasdaq 100 tickers: {e}")

    all_tickers = sorted(list(set(sp100_tickers + nasdaq100_tickers)))
    return all_tickers

def main():
    """Main function to run the index data download."""
    output_dir = 'V5.2/data/index/'
    os.makedirs(output_dir, exist_ok=True)

    print("--- Scraping Index Tickers ---")
    tickers = scrape_index_tickers()

    # Fallback to predefined list if scraping fails
    if not tickers:
        print("Web scraping failed. Using predefined fallback ticker list.")
        tickers = get_fallback_tickers()

    if not tickers:
        print("Failed to get ticker lists from scraping and fallback. Aborting.")
        return

    print(f"Found {len(tickers)} unique tickers.")

    macro_tickers = ['SPY', 'QQQ', 'IWO', 'VTI', '^VIX', '^TNX']
    start_date = '2015-01-01'
    end_date = '2025-11-30'

    print("\n--- Downloading Index Tickers ---")
    tickers_data = download_data(tickers, start_date, end_date)

    print("\n--- Downloading Macro Tickers ---")
    macro_data = download_data(macro_tickers, start_date, end_date)

    with open(os.path.join(output_dir, 'raw_tickers.pkl'), 'wb') as f:
        pickle.dump(tickers_data, f)

    with open(os.path.join(output_dir, 'raw_macro.pkl'), 'wb') as f:
        pickle.dump(macro_data, f)

    print("\nData download complete.")
    print(f"Index tickers data saved to {os.path.join(output_dir, 'raw_tickers.pkl')}")
    print(f"Macro tickers data saved to {os.path.join(output_dir, 'raw_macro.pkl')}")

if __name__ == "__main__":
    main()
