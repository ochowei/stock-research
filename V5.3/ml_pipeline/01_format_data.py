
import os
import pandas as pd

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def format_ticker_data(data_dict):
    """
    Formats raw ticker data into a MultiIndex DataFrame (timestamp, symbol).
    """
    df = data_dict['daily']
    df_stacked = df.stack(level=1).reset_index()
    df_stacked.columns = df_stacked.columns.str.lower()
    df_stacked = df_stacked.rename(columns={'date': 'timestamp', 'ticker': 'symbol'})

    if 'adj close' in df_stacked.columns:
        df_stacked = df_stacked.drop(columns=['adj close'])

    df_stacked['timestamp'] = pd.to_datetime(df_stacked['timestamp'])
    df_stacked = df_stacked.set_index(['timestamp', 'symbol']).sort_index()

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df_stacked.columns:
            df_stacked[col] = pd.to_numeric(df_stacked[col], errors='coerce')

    return df_stacked

def format_macro_data(df):
    """
    Formats raw macro data into a single-index DataFrame.
    """
    df_stacked = df.stack(level=1).reset_index()
    df_stacked.columns = df_stacked.columns.str.lower()
    df_stacked = df_stacked.rename(columns={'date': 'timestamp', 'ticker': 'symbol'})

    if 'adj close' in df_stacked.columns:
        df_stacked = df_stacked.drop(columns=['adj close'])

    df_stacked['timestamp'] = pd.to_datetime(df_stacked['timestamp'])
    df_stacked = df_stacked.set_index(['timestamp']).sort_index()

    return df_stacked

def main():
    """
    Processes raw data for both 'custom' and 'index' tracks,
    converting .pkl files to sorted, numeric .parquet files.
    """
    script_dir = get_script_dir()
    v5_2_dir = os.path.abspath(os.path.join(script_dir, '..'))

    data_tracks = ['custom', 'index']

    for track in data_tracks:
        print(f"--- Processing data for track: {track} ---")

        # Define paths relative to V5.2 root
        base_data_dir = os.path.join(v5_2_dir, 'data', track)
        raw_tickers_path = os.path.join(base_data_dir, 'raw_tickers.pkl')
        raw_macro_path = os.path.join(base_data_dir, 'raw_macro.pkl')

        universe_output_path = os.path.join(base_data_dir, 'universe_daily.parquet')
        market_output_path = os.path.join(base_data_dir, 'market_indicators.parquet')

        # Process Tickers
        if os.path.exists(raw_tickers_path):
            print(f"Loading raw tickers from: {raw_tickers_path}")
            raw_tickers_data = pd.read_pickle(raw_tickers_path)
            formatted_tickers = format_ticker_data(raw_tickers_data)

            print(f"Saving formatted universe data to: {universe_output_path}")
            formatted_tickers.to_parquet(universe_output_path)
            print("Universe data saved successfully.")
        else:
            print(f"Warning: Ticker data not found for track '{track}' at {raw_tickers_path}")

        # Process Macro Indicators
        if os.path.exists(raw_macro_path):
            print(f"Loading raw macro data from: {raw_macro_path}")
            raw_macro_df = pd.read_pickle(raw_macro_path)
            formatted_macro = format_macro_data(raw_macro_df)

            print(f"Saving formatted market indicators to: {market_output_path}")
            formatted_macro.to_parquet(market_output_path)
            print("Market indicators saved successfully.")
        else:
            print(f"Warning: Macro data not found for track '{track}' at {raw_macro_path}")

if __name__ == '__main__':
    main()
