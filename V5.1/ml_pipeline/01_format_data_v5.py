import pandas as pd
import os

def process_and_save(df, output_path, name):
    """
    Formats the yfinance multi-index DataFrame to a standard (symbol, timestamp) panel
    and saves it to parquet.
    """
    print(f"\nProcessing {name}...")
    
    if df is None or df.empty:
        print(f"Warning: Input DataFrame for {name} is empty. Skipping.")
        return

    # Check if columns are MultiIndex (typical for yf.download with multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        print(f"  - Initial shape: {df.shape}")
        
        # Stack the Ticker level to the index
        # This converts (Date) index -> (Date, Ticker) index
        # And (Price) columns -> (Price) columns
        df_stacked = df.stack(future_stack=True)
        
        # Rename index levels to our standard
        df_stacked.index.names = ['timestamp', 'symbol']
        
        # Reorder levels to (symbol, timestamp) for easier grouping later
        df_stacked = df_stacked.reorder_levels(['symbol', 'timestamp'])
        
        # Sort the index
        df_stacked = df_stacked.sort_index()
        
        print(f"  - Stacked shape: {df_stacked.shape}")
        print(f"  - Columns: {list(df_stacked.columns)}")
        
        # Save to Parquet
        df_stacked.to_parquet(output_path)
        print(f"  - Successfully saved to {output_path}")
        
    else:
        # Handle single ticker or flat DataFrame case if necessary
        # yfinance might return a simple index if only 1 ticker is downloaded, 
        # though usually our download script ensures list input.
        print(f"  - Info: DataFrame columns are not MultiIndex. Assuming pre-formatted or single ticker.")
        # Ensure index name is correct
        df.index.name = 'timestamp' 
        df.to_parquet(output_path)
        print(f"  - Saved flat DataFrame to {output_path}")

def main():
    # --- Setup Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(SCRIPT_DIR, 'data', 'temp_raw')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data')

    # Ensure output directory exists (though usually created in step 00)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Input Files (From Step 00)
    tickers_pkl_path = os.path.join(RAW_DATA_DIR, 'raw_tickers_data.pkl')
    macro_pkl_path = os.path.join(RAW_DATA_DIR, 'raw_macro_data.pkl')
    sector_pkl_path = os.path.join(RAW_DATA_DIR, 'raw_sector_data.pkl') # V5.1 New

    # Output Files (For Step 01 Feature Engineering)
    universe_daily_path = os.path.join(OUTPUT_DIR, 'universe_daily.parquet')
    universe_60m_path = os.path.join(OUTPUT_DIR, 'universe_60m.parquet')
    market_indicators_path = os.path.join(OUTPUT_DIR, 'market_indicators.parquet')
    sector_daily_path = os.path.join(OUTPUT_DIR, 'sector_daily.parquet') # V5.1 New

    # --- Load Data ---
    print("Loading raw pickle files...")
    
    # Check for essential files
    if not os.path.exists(tickers_pkl_path) or not os.path.exists(macro_pkl_path):
        print("Error: Essential raw data files (tickers/macro) not found. Please run 00_download_data_v5.py first.")
        return
        
    raw_tickers = pd.read_pickle(tickers_pkl_path)
    raw_macro = pd.read_pickle(macro_pkl_path)
    
    # Load Sector Data (V5.1)
    raw_sector = None
    if os.path.exists(sector_pkl_path):
        raw_sector = pd.read_pickle(sector_pkl_path)
    else:
        print("Warning: Sector data not found. V5.1 orthogonal features may not work.")

    print("Load complete.")

    # --- 1. Process Tickers (Daily) ---
    # raw_tickers is a dict {'daily': df, 'hourly': df}
    process_and_save(
        raw_tickers.get('daily'), 
        universe_daily_path, 
        "Ticker Data (Daily)"
    )

    # --- 2. Process Tickers (60m) ---
    process_and_save(
        raw_tickers.get('hourly'), 
        universe_60m_path, 
        "Ticker Data (60m)"
    )

    # --- 3. Process Macro (Daily) ---
    # raw_macro is a single DataFrame
    process_and_save(
        raw_macro, 
        market_indicators_path, 
        "Macro Data (Daily)"
    )
    
    # --- 4. Process Sector (Daily) - V5.1 New ---
    if raw_sector is not None:
        process_and_save(
            raw_sector,
            sector_daily_path,
            "Sector Data (Daily)"
        )

    print("\nStep 0-2: Data Formatting Complete.")

if __name__ == "__main__":
    main()