
import yfinance as yf
import pandas as pd

def format_ticker(ticker_string):
    """Removes exchange prefix and replaces dots with dashes for yfinance."""
    if ':' in ticker_string:
        ticker = ticker_string.split(':')[-1]
    else:
        ticker = ticker_string
    return ticker.replace('.', '-')

def download_data(tickers, start_date, end_date):
    """Downloads daily and hourly data for a list of tickers and formats it."""
    data = {'daily': pd.DataFrame(), 'hourly': pd.DataFrame()}
    if not tickers:
        print("No tickers to download.")
        return data

    # --- Daily Data ---
    print(f"Downloading daily data for {len(tickers)} tickers...")
    daily_df = yf.download(tickers, start=start_date, end=end_date, interval='1d', auto_adjust=True, progress=False)

    # --- Hourly Data (in chunks) ---
    print(f"Downloading hourly data for {len(tickers)} tickers...")
    hourly_dfs = []
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    current_start = start
    while current_start < end:
        current_end = current_start + pd.DateOffset(days=729)
        if current_end > end:
            current_end = end
        print(f"  Fetching hourly data from {current_start.date()} to {current_end.date()}...")
        hourly_chunk = yf.download(tickers, start=current_start, end=current_end, interval='60m', auto_adjust=True, progress=False)
        if not hourly_chunk.empty:
            hourly_dfs.append(hourly_chunk)
        current_start = current_end + pd.DateOffset(days=1)

    if hourly_dfs:
        hourly_df = pd.concat(hourly_dfs)
    else:
        hourly_df = pd.DataFrame()

    # --- Formatting ---
    if not daily_df.empty:
        if len(tickers) > 1:
            daily_df = daily_df.stack(level=1).rename_axis(['Date', 'Ticker']).reorder_levels(['Ticker', 'Date'])
        else:
            daily_df['Ticker'] = tickers[0]
            daily_df = daily_df.set_index('Ticker', append=True).reorder_levels(['Ticker', 'Date'])
        data['daily'] = daily_df

    if not hourly_df.empty:
        hourly_df = hourly_df[~hourly_df.index.duplicated(keep='first')]
        if len(tickers) > 1:
            hourly_df = hourly_df.stack(level=1).rename_axis(['Date', 'Ticker']).reorder_levels(['Ticker', 'Date'])
        else:
            hourly_df['Ticker'] = tickers[0]
            hourly_df = hourly_df.set_index('Ticker', append=True).reorder_levels(['Ticker', 'Date'])
        data['hourly'] = hourly_df

    return data
