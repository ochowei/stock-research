import os
import pandas as pd

def get_script_dir():
    """Returns the directory of the currently running script."""
    return os.path.dirname(os.path.abspath(__file__))

def robust_stack(df):
    """
    Robustly stacks a MultiIndex DataFrame to (timestamp, symbol, OHLCV).
    Automatically detects whether columns are (Price, Ticker) or (Ticker, Price).
    """
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker case or flat dataframe
        print("  - Detected flat DataFrame (Single Ticker or pre-processed).")
        return df.reset_index()

    # Detect which level contains Price data (e.g., 'Close')
    l0 = df.columns.get_level_values(0)
    l1 = df.columns.get_level_values(1)
    
    stack_level = 1 # Default assumption: (Price, Ticker)
    
    if 'Close' in l0 or 'Open' in l0:
        # Structure: (Price, Ticker) -> Stack Level 1 (Ticker)
        print("  - Detected structure: (Price, Ticker). Stacking level 1.")
        stack_level = 1
    elif 'Close' in l1 or 'Open' in l1:
        # Structure: (Ticker, Price) -> Stack Level 0 (Ticker)
        print("  - Detected structure: (Ticker, Price). Stacking level 0.")
        stack_level = 0
    else:
        print("  - Warning: Could not detect Price level. Defaulting to stack level 1.")

    # Stack and reset index
    # future_stack=True is recommended for pandas 2.1+
    return df.stack(level=stack_level, future_stack=True).reset_index()

def format_ticker_data(data_dict):
    """
    Formats raw ticker data into a MultiIndex DataFrame (timestamp, symbol).
    """
    df = data_dict['daily']
    print(f"  - Formatting Ticker Data. Shape: {df.shape}")
    
    df_stacked = robust_stack(df)
    
    # Standardize column names
    df_stacked.columns = df_stacked.columns.str.lower()
    
    # Rename typical columns
    # Yfinance index is usually 'Date'
    # The stacked level (Ticker) usually has name 'Ticker' or 'level_0'/'level_1'
    rename_map = {'date': 'timestamp', 'ticker': 'symbol'}
    
    # If columns are unnamed (level_0, level_1), try to identify symbol column
    # The symbol column is usually the one that is NOT timestamp and NOT price data
    if 'symbol' not in df_stacked.columns and 'ticker' not in df_stacked.columns:
        # Simple heuristic: find the object column that is not 'timestamp'
        for col in df_stacked.columns:
            if col not in ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj close']:
                rename_map[col] = 'symbol'
                break
    
    df_stacked = df_stacked.rename(columns=rename_map)

    if 'adj close' in df_stacked.columns:
        df_stacked = df_stacked.drop(columns=['adj close'])

    # Ensure timestamp is datetime
    if 'timestamp' in df_stacked.columns:
        df_stacked['timestamp'] = pd.to_datetime(df_stacked['timestamp'])
        
        # Verify we have 'symbol'
        if 'symbol' in df_stacked.columns:
            df_stacked = df_stacked.set_index(['timestamp', 'symbol']).sort_index()
        else:
            print("  - Error: 'symbol' column missing after formatting. Check input structure.")
            return pd.DataFrame() # Return empty on failure
    else:
        print("  - Error: 'timestamp' column missing. Check input structure.")
        return pd.DataFrame()

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df_stacked.columns:
            df_stacked[col] = pd.to_numeric(df_stacked[col], errors='coerce')

    return df_stacked

def format_macro_data(df):
    """
    Formats raw macro data into a single-index DataFrame (Long Format).
    We keep it in Long Format (timestamp, symbol) or pivoting will be done later.
    Actually, to match V5.2 pipeline, we often output a timestamp-indexed DF.
    BUT, if we have multiple macro tickers (SPY, HYG...), we cannot simply set index to timestamp 
    without losing symbol info or creating duplicates.
    
    V5.2 L1 Feature Engineering expects to be able to pivot this data.
    So we should save it as (timestamp, symbol) or Long format.
    """
    print(f"  - Formatting Macro Data. Shape: {df.shape}")
    
    df_stacked = robust_stack(df)
    
    df_stacked.columns = df_stacked.columns.str.lower()
    
    rename_map = {'date': 'timestamp', 'ticker': 'symbol'}
    # Heuristic for symbol column if missing
    if 'symbol' not in df_stacked.columns and 'ticker' not in df_stacked.columns:
        for col in df_stacked.columns:
            if col not in ['timestamp', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj close']:
                rename_map[col] = 'symbol'
                break
                
    df_stacked = df_stacked.rename(columns=rename_map)

    if 'adj close' in df_stacked.columns:
        df_stacked = df_stacked.drop(columns=['adj close'])

    if 'timestamp' in df_stacked.columns:
        df_stacked['timestamp'] = pd.to_datetime(df_stacked['timestamp'])
        # We set index to timestamp for consistency with V5.2, but keep symbol as column
        # OR we set MultiIndex (timestamp, symbol).
        # Let's verify Step 2 input. Step 2 reads parquet and pivots.
        # So having (timestamp, symbol) index is safest.
        if 'symbol' in df_stacked.columns:
            # We treat macro data same as universe data: (Timestamp, Symbol)
            df_stacked = df_stacked.set_index(['timestamp', 'symbol']).sort_index()
        else:
             # Fallback for single ticker macro (unlikely now)
            df_stacked = df_stacked.set_index(['timestamp']).sort_index()
            
    return df_stacked

def main():
    """
    Processes raw data for both 'custom' and 'index' tracks.
    """
    script_dir = get_script_dir()

    # V5.3 handles both tracks
    data_tracks = ['custom', 'index']

    for track in data_tracks:
        print(f"\n--- Processing data for track: {track} ---")

        # Define paths
        base_data_dir = os.path.join(script_dir, 'data', track)
        raw_tickers_path = os.path.join(base_data_dir, 'raw_tickers.pkl')
        raw_macro_path = os.path.join(base_data_dir, 'raw_macro.pkl')

        universe_output_path = os.path.join(base_data_dir, 'universe_daily.parquet')
        market_output_path = os.path.join(base_data_dir, 'market_indicators.parquet')

        # Process Tickers
        if os.path.exists(raw_tickers_path):
            print(f"Loading raw tickers from: {raw_tickers_path}")
            raw_tickers_data = pd.read_pickle(raw_tickers_path)
            formatted_tickers = format_ticker_data(raw_tickers_data)

            if not formatted_tickers.empty:
                print(f"Saving formatted universe data to: {universe_output_path}")
                formatted_tickers.to_parquet(universe_output_path)
                print("Universe data saved successfully.")
            else:
                print("Warning: Formatted ticker data is empty.")
        else:
            print(f"Warning: Ticker data not found for track '{track}' at {raw_tickers_path}")

        # Process Macro Indicators
        if os.path.exists(raw_macro_path):
            print(f"Loading raw macro data from: {raw_macro_path}")
            raw_macro_df = pd.read_pickle(raw_macro_path)
            formatted_macro = format_macro_data(raw_macro_df)

            if not formatted_macro.empty:
                print(f"Saving formatted market indicators to: {market_output_path}")
                formatted_macro.to_parquet(market_output_path)
                print("Market indicators saved successfully.")
            else:
                print("Warning: Formatted macro data is empty.")
        else:
            print(f"Warning: Macro data not found for track '{track}' at {raw_macro_path}")

if __name__ == '__main__':
    main()